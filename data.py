import torch
import numpy as np
from pathlib import Path


class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)

    def encode(self, s):  # str -> List[int]
        return [self.stoi[c] for c in s]

    def decode(self, ids):  # List[int] -> str
        return "".join(self.itos[i] for i in ids)


# --- 子詞級：SentencePiece tokenizer ---
try:
    import sentencepiece as spm
except Exception:
    spm = None  # 未安裝就維持 None，使用時再丟錯


class SPMTokenizer:
    def __init__(self, model_path: str):
        if spm is None:
            raise ImportError("請先安裝 sentencepiece：pip install sentencepiece")
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.vocab_size = self.sp.get_piece_size()
        self.unk_id = self.sp.unk_id()

    def encode(self, s: str):
        return self.sp.encode(s, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)


def load_text(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到資料 {p}. 請放一個純文字檔，例如 tiny_shakespeare.txt")
    return p.read_text(encoding="utf-8")


def make_splits(data_ids, split=0.9):
    n = len(data_ids)
    n_train = int(n * split)
    train_ids = torch.tensor(data_ids[:n_train], dtype=torch.long)
    val_ids = torch.tensor(data_ids[n_train:], dtype=torch.long)
    return train_ids, val_ids


def get_batch(data_ids, batch_size, seq_len, device):
    # 隨機抽連續片段
    ix = torch.randint(0, len(data_ids) - seq_len - 1,
                       (batch_size,), device=device)
    x = torch.stack([data_ids[i:i+seq_len] for i in ix])
    y = torch.stack([data_ids[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


def _to_tensor_slice(arr, start, length):
    # 支援 numpy memmap 或 torch tensor
    if isinstance(arr, np.ndarray):
        view = arr[start:start+length]                # 這是零拷貝 view
        t = torch.from_numpy(view.astype(np.int64, copy=False))
    elif torch.is_tensor(arr):
        t = arr[start:start+length].to(dtype=torch.long)
    else:
        t = torch.tensor(arr[start:start+length], dtype=torch.long)
    return t

def get_batch(data_ids, batch_size, seq_len, device):
    # 與原版相同介面，但可吃 numpy memmap
    max_i = len(data_ids) - seq_len - 1
    ix = torch.randint(max_i, (batch_size,))
    x = torch.stack([_to_tensor_slice(data_ids, int(i),     seq_len) for i in ix])
    y = torch.stack([_to_tensor_slice(data_ids, int(i) + 1, seq_len) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
