# lora.py
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # 原權重/偏置（clone 會保留原本裝置，這裡 OK）
        self.weight = nn.Parameter(
            linear.weight.data.clone(), requires_grad=False)
        self.bias = None
        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.data.clone(), requires_grad=False)

        # ---- 這兩行加上 device，和原 Linear 同裝置 ----
        dev = linear.weight.device
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(
            torch.empty(self.in_features, r, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(
            r, self.out_features, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(
            dropout) if dropout and dropout > 0 else nn.Identity()
        self.merged = False

    def forward(self, x):
        base = x @ self.weight.T
        if self.bias is not None:
            base = base + self.bias
        if self.merged:
            return base
        lora = self.dropout(x) @ self.lora_A @ self.lora_B
        return base + self.scaling * lora

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        delta = (self.lora_A @ self.lora_B).T * self.scaling  # [out,in]
        self.weight += delta
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        delta = (self.lora_A @ self.lora_B).T * self.scaling
        self.weight -= delta
        self.merged = False


def apply_lora(module: nn.Module, names=("q_proj", "v_proj"), r=8, alpha=16, dropout=0.05):
    """把符合名字的 nn.Linear 換成 LoRALinear（遞迴地對整個模型套用）"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in names:
            setattr(module, name, LoRALinear(
                child, r=r, alpha=alpha, dropout=dropout))
        else:
            apply_lora(child, names, r, alpha, dropout)


def lora_parameters(module: nn.Module):
    """收集所有 LoRA 的可訓參數（A/B）"""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield m.lora_A
            yield m.lora_B


@torch.no_grad()
def save_lora_adapter(module: nn.Module, path: str):
    """只存 A/B 參數，方便 adapter 方式載入"""
    state = {}
    for n, m in module.named_modules():
        if isinstance(m, LoRALinear):
            state[f"{n}.lora_A"] = m.lora_A.detach().cpu()
            state[f"{n}.lora_B"] = m.lora_B.detach().cpu()
            state[f"{n}.r"] = torch.tensor([m.r])
            state[f"{n}.alpha"] = torch.tensor([m.alpha])
    torch.save(state, path)


@torch.no_grad()
def load_lora_adapter(module: nn.Module, path: str, strict=True):
    """把 A/B 權重載回 LoRALinear"""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:

        state = torch.load(path, map_location="cpu")

    for n, m in module.named_modules():
        if isinstance(m, LoRALinear):
            m.lora_A.copy_(state[f"{n}.lora_A"])
            m.lora_B.copy_(state[f"{n}.lora_B"])
