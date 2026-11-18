"""复数张量工具。

全部复数用最后一维长度 2 的实数张量表示，顺序为实部、虚部。
不使用 torch 的原生复数类型，原因是部署侧没有复数算子。
"""

import torch

EPS = 1e-8


def cmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐元素复数乘法。a 与 b 的最后一维为 2。"""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br - ai * bi, ar * bi + ai * br], dim=-1)


def cmul_conj(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """返回 a 乘 b 的共轭。"""
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    return torch.stack([ar * br + ai * bi, ai * br - ar * bi], dim=-1)


def cconj(a: torch.Tensor) -> torch.Tensor:
    return torch.stack([a[..., 0], -a[..., 1]], dim=-1)


def cabs(a: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2 + EPS)


def cangle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a[..., 1], a[..., 0])


def from_polar(mag: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)
