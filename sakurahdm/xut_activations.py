import torch
from torch import nn
from torch.nn import functional as F


def powlu_gate(gate: torch.Tensor, m: float) -> torch.Tensor:
    """Two-projection PowLU (arXiv:2605.25704 §3.1)."""
    positive = gate > 0
    # where evaluates both branches; avoid a singular sqrt backward at zero
    # even when its result would be masked out.
    g = torch.where(positive, gate.float(), 1.0)
    pow_gate = g.pow(m / (g.sqrt() + 1)) * g.sigmoid()
    return torch.where(positive, pow_gate.to(gate.dtype), F.silu(gate))


class PowLUGatedMLP(nn.Module):
    def __init__(self, swiglu: nn.Module, m: float):
        super().__init__()
        self.m = m
        self.w12 = swiglu.w12
        self.w3 = swiglu.w3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(powlu_gate(gate, self.m) * value)


def use_powlu_in_xut(dit: nn.Module, m: float) -> None:
    from xut.modules.transformer import TransformerBlock

    for block in dit.modules():
        if isinstance(block, TransformerBlock):
            block.mlp = PowLUGatedMLP(block.mlp, m)
