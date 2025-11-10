import os, math, argparse
import torch
import torch.nn.functional as F
import sentencepiece as spm
from model import DecoderOnlyLM

def safe_load(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location)
    return ck

def softmax_temp(x, temp: float):
    if temp <= 0:
        # greedy
        probs = torch.zeros_like(x).float()
        probs[x.argmax()] = 1.0
        return probs
    x = x.float() / max(1e-6, temp)
    x = x - x.max()
    p = torch.exp(x)
    p = p / p.sum()
    return p

def top_p_filter(probs, top_p: float):
    if top_p >= 1.0:
        return probs
    vals, idx = torch.sort(probs, descending=True)
    c = torch.cumsum(vals, dim=0)
    mask = c > top_p
    if mask.any():
        first = torch.nonzero(mask, as_tuple=False)[0].item()
        keep = idx[:first+1]
        out = torch.zeros_like(probs)
        out[keep] = probs[keep]
        s = out.sum()
        if s > 0:
            out = out / s
        else:
            out = probs  # fallback
        return out
    return probs

def clean_visible(s: str) -> str:
    # 簡單清理：移除 C0 控制碼，保留一般標點與非 ASCII
    return "".join(ch for ch in s if (ord(ch) >= 32 or ch in "\n\t"))

def build_model(cfg, sp, device):
    model = DecoderOnlyLM(
        vocab_size=sp.vocab_size(),
        d_model=cfg["d_model"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
        max_seq_len=cfg["seq_len"],
        dropout=cfg["dropout"],
        n_kv_head=(cfg.get("n_kv_head") or cfg["n_head"]),
    ).to(device)
    return model

def step_logits_fn(model, ids):
    out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    return logits[:, -1, :]  # [B=1, V]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--best_ckpt", default="")
    ap.add_argument("--spm", required=True)
    ap.add_argument("--prompt", action="append", default=[], help="inline prompt(s)")
    ap.add_argument("--prompts_file", default="")
    ap.add_argument("--min_new", type=int, default=40)
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--no_repeat", type=int, default=5)   # 簡化：禁止最近 N token
    ap.add_argument("--rep_penalty", type=float, default=1.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ck = safe_load(args.ckpt, map_location="cpu")
    cfg = ck.get("config")
    assert cfg is not None, "ckpt 需要包含 config（請用訓練時存的 ckpt_minGPT.pth）"

    sp = spm.SentencePieceProcessor(model_file=args.spm)
    try:
        eos_id = sp.eos_id()
    except Exception:
        eos_id = -1

    # SPM 一致性提示（不阻擋執行）
    cfg_spm = cfg.get("spm_model")
    if cfg_spm and os.path.basename(cfg_spm) != os.path.basename(args.spm):
        print(f"[warn] ckpt was trained with {cfg_spm}, but you passed {args.spm}.")

    # 建模 + 載權重
    model = build_model(cfg, sp, device)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state, strict=False)

    # 如有 best 覆寫
    if args.best_ckpt:
        ck2 = safe_load(args.best_ckpt, map_location="cpu")
        st2 = ck2.get("model", ck2)
        model.load_state_dict(st2, strict=False)
        print(f"[info] loaded best weights from {args.best_ckpt}")

    # ban 列表（控制碼、@、成對引號）；並在前 min_new 步禁止 EOS
    base_ban = set()
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        if ("@" in piece) or (piece in ("``", "''")) or any(ord(ch) < 32 for ch in piece):
            base_ban.add(i)

    # 讀 prompts：有空白行→分段；否則逐行一題
    prompts = list(args.prompt)
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            raw = [ln.rstrip("\n") for ln in f]
        if any(ln.strip() == "" for ln in raw):
            buf, segs = [], []
            for ln in raw:
                if ln.strip() == "":
                    if buf: segs.append("\n".join(buf)); buf = []
                else:
                    buf.append(ln)
            if buf: segs.append("\n".join(buf))
            prompts.extend(segs)
        else:
            prompts.extend([ln for ln in raw if ln.strip()])

    for i, prompt in enumerate(prompts, 1):
        ids = torch.tensor([sp.encode(prompt, out_type=int)],
                           device=device, dtype=torch.long)  # [1, L]
        seen_counts = {}

        for step in range(args.max_new):
            logits = step_logits_fn(model, ids).squeeze(0)  # [V]

            # 重複懲罰：對已出現 token 減 log(rep_penalty)
            if args.rep_penalty and args.rep_penalty != 1.0:
                for t in set(ids[0].tolist()):
                    cnt = (ids[0] == t).sum().item()
                    if cnt > 0:
                        logits[t] -= math.log(args.rep_penalty) * cnt

            probs = torch.softmax(logits, dim=-1)
            probs = softmax_temp(torch.log(probs + 1e-9), args.temp)
            probs = top_p_filter(probs, args.top_p)

            # 動態 ban：基礎禁字 + 最近 N token + 前 min_new 步禁止 EOS
            ban = set(base_ban)
            if args.no_repeat > 0:
                recent = ids[0].tolist()[-args.no_repeat:]
                ban.update(recent)
            if eos_id >= 0 and step < args.min_new:
                ban.add(eos_id)

            if ban:
                ban_idx = torch.tensor(list(ban), device=probs.device, dtype=torch.long)
                probs[ban_idx] = 0.0
                s = probs.sum()
                if s.item() == 0:
                    # fallback：只排除 ban 後做 greedy
                    logits2 = logits.clone()
                    logits2[ban_idx] = -1e30
                    next_id = int(torch.argmax(logits2).item())
                else:
                    probs = probs / s
                    next_id = int(torch.multinomial(probs, 1).item())
            else:
                next_id = int(torch.multinomial(probs, 1).item())

            # append
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
            if eos_id >= 0 and next_id == eos_id and step >= args.min_new:
                break

        out = sp.decode(ids[0].tolist())
        out = clean_visible(out)

        print(f"### Prompt {i}\n{prompt}\n---")
        print(out.strip(), "\n")

if __name__ == "__main__":
    main()
