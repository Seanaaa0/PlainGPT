# train.py (drop-in, fixed)
from pathlib import Path as _P
import os
import math
import torch.nn.utils as U
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DecoderOnlyLM
from data import load_text, make_splits, get_batch, SPMTokenizer, CharTokenizer
from lora import apply_lora, lora_parameters, save_lora_adapter, load_lora_adapter

try:
    from torch.backends.cuda import sdp_kernel
    if torch.cuda.is_available():
        sdp_kernel(enable_flash=False,
                   enable_mem_efficient=False, enable_math=True)
        torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

IS_SFT = os.environ.get("SFT_MODE", "0") == "1"
# ==== injected: seed & ckpt utils ====


def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def save_ckpt(path, model, opt, step, best, cfg):
    obj = {"model": getattr(model, "state_dict")() if hasattr(model, "state_dict") else model,
           "config": cfg, "step": int(step), "best": float(best)}
    try:
        obj["optimizer"] = opt.state_dict()
    except Exception:
        pass
    import torch as _torch
    _torch.save(obj, path)


def try_resume(path, model, opt):
    import os
    import torch as _torch
    if not path or not os.path.exists(path):
        return 0, float("inf")
    obj = _torch.load(path, map_location="cpu")
    state = obj["model"] if (isinstance(obj, dict) and "model" in obj) else obj
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        # tolerate head-only or shape drift
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(state, strict=False)
    if isinstance(obj, dict) and "optimizer" in obj:
        try:
            opt.load_state_dict(obj["optimizer"])
        except Exception:
            pass
    step = int(obj.get("step", 0)) if isinstance(obj, dict) else 0
    best = float(obj.get("best", float("inf"))) if isinstance(
        obj, dict) else float("inf")
    print(f"[resume] resumed step={step}, best={best}")
    return step, best
# ==== end injected ====


# -------------------------------
# 小工具
# -------------------------------


def safe_load(path, map_location="cpu"):
    """相容未來 weights_only=True 的安全載入"""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


IGNORE_INDEX = -100


@torch.no_grad()
def evaluate_tok(model, val_ids, batch_size, seq_len, device, max_tokens=None):
    """
    回傳每 token 平均 loss（nats）與 ppl。
    只用 val_ids 的前 max_tokens（若提供）。不 pad，連續切片。
    做法：逐批算 mean loss，再用 token 數做加權平均（更穩）。
    """
    model.eval()
    loss_sum = 0.0
    tok_sum = 0

    with torch.no_grad():
        ids = torch.as_tensor(val_ids, dtype=torch.long)
        if max_tokens is not None:
            ids = ids[: max_tokens + 1]

        T = seq_len + 1
        N = ids.numel()
        stride = batch_size * seq_len

        for i in range(0, N - T, stride):
            Bs = min(batch_size, (N - T - i) // seq_len)
            if Bs <= 0:
                break

            xs, ys = [], []
            for b in range(Bs):
                s = i + b * seq_len
                seg = ids[s: s + T]                 # (T,)
                xs.append(seg[:-1])
                ys.append(seg[1:])

            x = torch.stack(xs, dim=0).to(device)   # (B, T)
            y = torch.stack(ys, dim=0).to(device)   # (B, T)

            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits.float()

            # 每批用 mean，比 sum/手動除更不容易踩坑
            batch_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="mean",
            )
            ntoks = y.numel()
            loss_sum += float(batch_loss.item()) * ntoks
            tok_sum += int(ntoks)

    model.train()
    loss_per_tok = loss_sum / max(1, tok_sum)
    ppl = math.exp(loss_per_tok)
    return loss_per_tok, ppl


def sanity_eval_once(model, ids, seq_len, device, tag="val"):
    model.eval()
    with torch.no_grad():
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        T = seq_len + 1
        if ids_t.numel() <= T:
            print(f"[sanity/{tag}] not enough tokens.")
            return
        i = torch.randint(0, ids_t.numel() - T, (1,)).item()
        seg = ids_t[i:i+T].to(device)               # (T,)
        x = seg[:-1].unsqueeze(0)                   # (1, T)
        y = seg[1:].unsqueeze(0)                    # (1, T)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        logits = logits.float()
        loss_mean = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="mean",
        )
    model.train()
    print(
        f"[sanity/{tag}] mean_loss_per_tok={loss_mean.item():.4f}, ppl~{math.exp(loss_mean.item()):.1f}")


def sample(model, tok, cfg, device, prompt="Hello", max_new_tokens=100, temperature=0.7):
    model.to(device).eval()  # 確保裝置一致
    ids = torch.tensor(tok.encode(prompt), dtype=torch.long,
                       device=device)[None, :]
    for _ in range(max_new_tokens):
        with torch.no_grad(), torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
            logits, _ = model(ids[:, -cfg["seq_len"]:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())


# -------------------------------
# 預設設定（可被環境變數覆寫）
# -------------------------------
cfg = {



    "data_path": "./data/en_only.clean.txt",  # 會被 DATA_PATH 覆寫
    "tokenizer": "spm",                              # "spm" 或 "char"
    "spm_model": "./data/spm_en8k.model",
    "n_kv_head": 0,
    "d_model": 384,
    "n_head": 6,
    "n_layer": 6,
    "seq_len": 256,
    "dropout": 0.10,
    "batch_size": 32,
    "max_steps": 10000,
    "eval_every": 200,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "ckpt_path": "./pth/ckpt_minGPT.pth"}

# 覆寫路徑（環境變數）
cfg["data_path"] = os.getenv("DATA_PATH", cfg["data_path"])
cfg["spm_model"] = os.getenv("SPM_MODEL", cfg["spm_model"])
cfg["ckpt_path"] = os.getenv("CKPT_PATH", cfg["ckpt_path"])
# __GQA_PATCH_BEGIN__
# 強制決定 n_kv_head：環境變數優先；其次讀 BASE_CKPT 的 config；最後退回 n_head
_cfg_nkv = os.getenv("N_KV_HEAD")
if _cfg_nkv is not None:
    cfg["n_kv_head"] = int(_cfg_nkv)
else:
    _base = os.getenv("BASE_CKPT", "")
    if _base and _P(_base).exists():
        try:
            _ck = torch.load(_base, map_location="cpu")
            nk = (_ck.get("config", {}) or {}).get("n_kv_head", None)
            cfg["n_kv_head"] = int(
                nk) if nk is not None else int(cfg["n_head"])
        except Exception:
            cfg["n_kv_head"] = int(cfg["n_head"])
    else:
        cfg["n_kv_head"] = int(cfg["n_head"])
print(f"[DEBUG] use n_kv_head = {cfg['n_kv_head']}")
# __GQA_PATCH_END__
cfg["adapter_path"] = os.getenv("ADAPTER_PATH", cfg.get(
    "adapter_path", "./adapters/lora_en.pt"))
cfg["max_steps"] = int(os.getenv("MAX_STEPS", str(cfg["max_steps"])))
cfg["lr"] = float(os.getenv("LR", str(cfg.get("lr", 3e-4))))
cfg["warmup"] = int(os.getenv("WARMUP", "400"))

# Cosine（舊）/ Cosine 重啟（新）
cfg["cosine"] = int(os.getenv("COSINE", str(cfg.get("cosine", 0))))
cfg["cosine_restarts"] = int(os.getenv("COSINE_RESTARTS", "1"))
cfg["min_lr"] = float(os.getenv("MIN_LR", "1e-5"))
cfg["T0"] = int(os.getenv("T0", "3000"))   # 第一個週期步數
cfg["Tmult"] = int(os.getenv("TMULT", "2"))   # 週期倍增

# 梯度累積
cfg["grad_accum"] = int(os.getenv("GRAD_ACCUM", "3"))

# EMA
cfg["ema"] = int(os.getenv("EMA", "1"))
cfg["ema_decay"] = float(os.getenv("EMA_DECAY", "0.999"))

# Label smoothing
cfg["label_smooth"] = float(os.getenv("LABEL_SMOOTH", "0.0"))

cfg["adapter_path"] = os.getenv("ADAPTER_PATH", cfg.get(
    "adapter_path", "./adapters/lora_en.pt"))
BASE_CKPT = os.getenv("BASE_CKPT", "")      # 若提供，先載入當 base
USE_LORA = int(os.getenv("USE_LORA", "0"))  # 0=全參預訓, 1=只訓 LoRA
if USE_LORA:
    from pathlib import Path
    Path(os.path.dirname(cfg["adapter_path"])).mkdir(
        parents=True, exist_ok=True)
# -------------------------------
# 主程式
# -------------------------------


def lr_with_warmup_cos(step, base_lr, warmup, total_steps, min_lr):
    if step < warmup:
        return base_lr * float(step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def lr_with_warmup_cos_restarts(step, base_lr, warmup, min_lr, T0, Tmult):
    if step < warmup:
        return base_lr * float(step + 1) / max(1, warmup)
    t = step - warmup
    Ti = T0
    while t >= Ti:
        t -= Ti
        Ti = int(Ti * Tmult)

    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t / Ti))


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(
                    p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}


def main():
    import os
    import math
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F

    # ---------- config ----------
    cfg = dict(
        data_path=os.environ.get("DATA_PATH", "./data/en_only.clean.txt"),
        tokenizer="spm",
        spm_model=os.environ.get("SPM_MODEL", "./data/spm_en8k.model"),
        ckpt_path=os.environ.get("CKPT_PATH", "./pth/ckpt_minGPT.pth"),

        # model
        d_model=int(os.environ.get("D_MODEL", 384)),
        n_layer=int(os.environ.get("N_LAYER", 6)),
        n_head=int(os.environ.get("N_HEAD", 6)),
        n_kv_head=int(os.environ.get("N_KV_HEAD", 0)),   # 0 表示等於 n_head
        seq_len=int(os.environ.get("SEQ_LEN", 256)),
        dropout=float(os.environ.get("DROPOUT", 0.1)),

        # train
        batch_size=int(os.environ.get("BATCH_SIZE", 32)),
        grad_accum=int(os.environ.get("GRAD_ACCUM", 4)),
        lr=float(os.environ.get("LR", 3e-4)),
        min_lr=float(os.environ.get("MIN_LR", 3e-5)),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", 0.01)),
        warmup=int(os.environ.get("WARMUP", 1000)),
        max_steps=int(os.environ.get("MAX_STEPS", 10000)),
        eval_every=int(os.environ.get("EVAL_EVERY", 500)),
        save_every=int(os.environ.get("SAVE_EVERY", 2000)),

        cosine=bool(int(os.environ.get("COSINE", 1))),
        cosine_restarts=bool(int(os.environ.get("COSINE_RESTARTS", 0))),
        T0=int(os.environ.get("T0", 2000)),
        Tmult=int(os.environ.get("TMULT", 2)),
        label_smooth=float(os.environ.get("LABEL_SMOOTH", 0.0)),
        clip_grad_norm=float(os.environ.get("CLIP_NORM", 1.0)),

        ema=bool(int(os.environ.get("EMA", 0))),
        ema_decay=float(os.environ.get("EMA_DECAY", 0.999)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # ---------- tokenizer ----------
    assert cfg["tokenizer"] == "spm", "此版本只支援 SPMTokenizer"
    tok = SPMTokenizer(cfg["spm_model"])
    print("vocab_size =", tok.vocab_size)

    # ---------- ids: 優先走 token 快取 (numpy memmap) ----------
    cache = f"{cfg['data_path']}.tok.{os.path.basename(cfg['spm_model'])}.npy"
    if os.path.exists(cache):
        print(f"[fast] load token cache: {cache}")
        ids = np.load(cache, mmap_mode="r")          # numpy.memmap，不吃 RAM
    else:
        text = load_text(cfg["data_path"])           # 只有無快取才讀大檔
        ids = tok.encode(text)                       # 會花時間；之後可另存快取

    # ---------- <unk> 率（抽樣，不整塊 materialize） ----------
    if hasattr(tok, "unk_id"):
        import numpy as _np
        _uid = tok.unk_id
        try:
            _n = len(ids)
            _limit = min(_n, 5_000_000)
            _step = 1_000_000
            _cnt = 0
            _tot = 0
            _i = 0
            while _i < _limit:
                _sl = ids[_i:_i+_step]
                if hasattr(_sl, "shape"):
                    _cnt += _np.count_nonzero(_sl == _uid)
                    _tot += _sl.shape[0]
                else:
                    _arr = _np.array(_sl)
                    _cnt += _np.count_nonzero(_arr == _uid)
                    _tot += _arr.size
                _i += _step
            _unk = (_cnt / max(1, _tot)) if _tot else 0.0
            print(f"[diag] <unk> rate = {_unk:.3%} (est on {_tot} tokens)")
        except Exception as _e:
            print(f"[diag] <unk> rate = n/a (reason: {_e})")

    # ---------- split ----------
    train_ids, val_ids = make_splits(ids)
    print(
        f"[diag] len(train_ids)={len(train_ids):,}, len(val_ids)={len(val_ids):,}")

    # ---------- model ----------
    model = DecoderOnlyLM(
        vocab_size=tok.vocab_size,
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        max_seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
        n_kv_head=(cfg["n_kv_head"] or cfg["n_head"]),
    ).to(device)

    # 可選：用 BASE_CKPT 當初始化（只載權重、不帶 optimizer）
    base_ckpt = os.environ.get("BASE_CKPT", "")
    if base_ckpt and os.path.exists(base_ckpt):
        try:
            _ck = torch.load(base_ckpt, map_location="cpu")
            _st = _ck.get("model", _ck)
            model.load_state_dict(_st, strict=False)
            print(f"[init] BASE_CKPT loaded: {base_ckpt}")
        except Exception as e:
            print("[warn] BASE_CKPT load failed:", e)

    # ---------- optim / scaler / ema ----------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            betas=(0.9, 0.95), weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda" if device == "cuda" else "cpu")
    ema = EMA(model, decay=cfg["ema_decay"]) if cfg["ema"] else None

    # 續訓（會把 step 與 best 讀回來），若檔案不存在就從 0 開始
    start_step, best = try_resume(cfg["ckpt_path"], model, opt)
    # ===== debug: 印參數可訓練數 =====
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[param] trainable={trainable:,} / total={total:,}")
    frozen_samples = [n for n, p in model.named_parameters()
                      if not p.requires_grad][:5]
    if frozen_samples:
        print("[param] sample frozen:", frozen_samples)
# =================================

    # ---------- helper: lr schedule ----------
    def set_lr(step):
        if cfg["cosine_restarts"]:
            cur = lr_with_warmup_cos_restarts(step, cfg["lr"], cfg["warmup"],
                                              cfg["min_lr"], cfg["T0"], cfg["Tmult"])
        elif cfg["cosine"]:
            cur = lr_with_warmup_cos(step, cfg["lr"], cfg["warmup"],
                                     cfg["max_steps"], cfg["min_lr"])
        else:
            cur = cfg["lr"]
        for g in opt.param_groups:
            g["lr"] = cur
        return cur

    def evaluate_mc(model, ids, seq_len, device, windows=512, batch_size=16):
        """
        Monte-Carlo 評估：隨機抽 'windows' 個長度 (seq_len+1) 的連續片段，
        批次化前向；回傳每 token 平均 loss（nats）與 ppl。
        """
        model.eval()
        loss_sum = 0.0
        tok_sum = 0
        with torch.no_grad():
            ids = torch.as_tensor(ids, dtype=torch.long)
            T = seq_len + 1
            N = ids.numel()
            if N <= T:
                return float("inf"), float("inf")
            idx = torch.randint(0, N - T, (windows,))
            for i in range(0, windows, batch_size):
                j = idx[i:i+batch_size]
                xs, ys = [], []
                for s in j.tolist():
                    seg = ids[s:s+T]
                    xs.append(seg[:-1])
                    ys.append(seg[1:])
                x = torch.stack(xs, dim=0).to(device)
                y = torch.stack(ys, dim=0).to(device)

                out = model(x)
                logits = out[0] if isinstance(out, (tuple, list)) else out
                logits = logits.float()

                batch_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="mean",
                )
                ntoks = y.numel()
                loss_sum += float(batch_loss.item()) * ntoks
                tok_sum += int(ntoks)
        model.train()
        loss_per_tok = loss_sum / max(1, tok_sum)
        ppl = math.exp(loss_per_tok)
        return loss_per_tok, ppl

    # ---------- train loop ----------
    t0 = time.time()
    acc = cfg["grad_accum"]
    model.train()
    opt.zero_grad(set_to_none=True)

    for step in range(start_step + 1, cfg["max_steps"] + 1):
        cur_lr = set_lr(step - 1)

        # batch
        x, y = get_batch(train_ids, cfg["batch_size"], cfg["seq_len"], device)

        # forward with autocast
        with torch.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"),
            dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        ):
            # 先做前向拿 logits
            # --- forward ---
            out = model(x)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits.float()

            if IS_SFT:
                # === SFT path: 只學 "### Response:" 之後（robust 版）===
                # 準備分隔符（只建一次）
                if not hasattr(model, "_sft_mask_ready"):

                    tok = SPMTokenizer(cfg["spm_model"])
                    model._resp_tok_A = torch.tensor(tok.encode("### Response:\n"),
                                                     dtype=torch.long, device=logits.device)
                    model._resp_tok_B = torch.tensor(tok.encode("### Response:"),
                                                     dtype=torch.long, device=logits.device)
                    model._sft_mask_ready = True
                respA, respB = model._resp_tok_A, model._resp_tok_B
                R1, R2 = respA.numel(), respB.numel()

                y_mask = torch.zeros_like(y, dtype=torch.bool)  # 只學分隔符之後
                for b in range(y.size(0)):
                    row = x[b]
                    pos = -1
                    match_len = 0
                    for i in range(0, row.numel() - R1 + 1):
                        if torch.equal(row[i:i+R1], respA):
                            pos, match_len = i, R1
                    if pos < 0:
                        for i in range(0, row.numel() - R2 + 1):
                            if torch.equal(row[i:i+R2], respB):
                                pos, match_len = i, R2
                    if pos >= 0:
                        start = min(pos + match_len, y.size(1))
                        y_mask[b, start:] = True

                y_masked = y.clone()
                y_masked[~y_mask] = -100
                valid = int((y_masked != -100).sum().item())
                if valid == 0:
                    # SFT 批次沒有 response，直接跳過這步（不學壞）
                    loss = logits.sum() * 0.0
                    skip_backprop = True
                else:
                    skip_backprop = False
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y_masked.reshape(-1),
                        ignore_index=-100,
                        reduction="mean",
                        label_smoothing=0.05,
                    )
                # 可選：只在 SFT 模式下列印 mask
                if step == 1 or step % 200 == 0:
                    total = int(y.numel())
                    print(
                        f"[sft-mask] resp_tokens={valid}/{total} ({valid/max(1,total):.2%})")

            else:
                # === LM path: 一般語言模型（不要做任何遮罩/掃 delimiter）===
                skip_backprop = False
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    reduction="mean",
                )

            # （可選）每 200 步印一次比例

            if IS_SFT and (step == 1 or step % 200 == 0):
                total_tokens = int(y_masked.numel())
                print(f"[sft-mask] resp_tokens={valid}/{total_tokens} (...)")

       # backward (accumulate)
        scaler.scale(loss / acc).backward()

        if step % acc == 0:
            if cfg["clip_grad_norm"] and cfg["clip_grad_norm"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["clip_grad_norm"])

            # >>> 在這裡印梯度（還沒 step / zero_grad） <<<
            if step == 1 or step % 200 == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.get("clip_norm", 1.0))
                gsum = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        gsum += float(p.grad.detach().abs().sum().item())
                print(f"[grad] norm={float(gn):.3f}  sum|grad|={gsum:.3e}")

            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            if ema:
                ema.update(model)

        # log
        if step == 1 or step % 50 == 0:
            dt = time.time() - t0
            print(
                f"step {step:5d}/{cfg['max_steps']}  loss={loss.item():.3f}  lr={cur_lr:.2e}  ({dt:.2f}s)")
            t0 = time.time()

        # eval / save best —— Monte-Carlo 抽窗評估（快、數字穩）
        if cfg["eval_every"] and step % cfg["eval_every"] == 0:
            import os  # 放這裡免改檔案開頭

            # 快速 sanity 看一下一個隨機窗的 mean loss（應 ≈ 6~10）
            sanity_eval_once(model, val_ids, cfg["seq_len"], device, "val")

            # 控制抽窗數量與 eval 批量（可用環境變數覆寫）
            eval_windows = int(os.environ.get("EVAL_WINDOWS", "512"))
            eval_bs = int(os.environ.get(
                "EVAL_BATCH_SIZE", str(cfg["batch_size"])))
            eval_seqlen = int(os.environ.get(
                "EVAL_SEQ_LEN",  str(cfg["seq_len"])))
            ppl_cutoff = float(os.environ.get("PPL_CUTOFF",  "20"))
            over_mode = os.environ.get("PPL_OVER_CUTOFF", "skip")

            if ema:
                ema.apply_shadow(model)
            val_loss, ppl = evaluate_mc(model, val_ids, eval_seqlen, device,
                                        windows=eval_windows, batch_size=eval_bs)
            if ema:
                ema.restore(model)

            ppl_str = f"{ppl:.2f}" if val_loss < ppl_cutoff else (
                "—" if over_mode == "skip" else "inf")
            print(
                f" -> (MC {eval_windows}x, bs={eval_bs}, L={eval_seqlen}) val_loss/tok={val_loss:.4f}, ppl={ppl_str}")

            if val_loss < best:
                best = val_loss
                torch.save(
                    {"model": model.state_dict(), "config": cfg},
                    os.path.join(os.path.dirname(
                        cfg["ckpt_path"]) or ".", "ckpt_best.pth"),
                )
                print("[best] saved ckpt_best.pth")

        # periodic save
        if cfg["save_every"] and (step % cfg["save_every"] == 0):
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(),
                 "config": cfg, "step": step, "best": best},
                cfg["ckpt_path"],
            )
            print(f"[save] {cfg['ckpt_path']} (step={step})")

    # final save
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "config": cfg, "step": cfg["max_steps"], "best": best}, cfg["ckpt_path"])
    print("[done] training finished")


if __name__ == "__main__":
    # 讓 SDPA 走比較穩定的 math kernel（如果你之前有用到）
    try:
        from torch.backends.cuda import sdpa_kernel
        sdpa_kernel(enable_math=True, enable_flash=False,
                    enable_mem_efficient=False)
    except Exception:
        pass
    main()
