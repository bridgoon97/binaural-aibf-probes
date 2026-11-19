"""频带压缩与解压矩阵。

低频逐 bin 恒等，高频三角带。这一设计沿用麦阵 AIBF 的「低于某频率恒等、高频三角 ERB」。
恒等区保证低频跨耳相位特征不被频带平均抹掉。

压缩矩阵 B 形状 [n_bands, n_bins]，每行和为 1。
解压矩阵 E 形状 [n_bins, n_bands]，每行和为 1。
B 与 E 互不为逆，使用时不能互换。
"""

import torch


def build_band_matrices(n_bins: int, n_linear: int, n_tri: int, device=None):
    assert n_linear < n_bins, "恒等区不能覆盖全部频点"
    n_bands = n_linear + n_tri
    comp = torch.zeros(n_bands, n_bins, device=device)
    expa = torch.zeros(n_bins, n_bands, device=device)

    # 恒等区
    for k in range(n_linear):
        comp[k, k] = 1.0
        expa[k, k] = 1.0

    # 三角区，边界在 bin 索引上按对数间隔
    lo, hi = n_linear, n_bins - 1
    edges = torch.logspace(
        torch.log10(torch.tensor(float(lo))),
        torch.log10(torch.tensor(float(hi))),
        steps=n_tri + 2,
    )
    for j in range(n_tri):
        a, c, b = edges[j].item(), edges[j + 1].item(), edges[j + 2].item()
        for k in range(lo, n_bins):
            if a <= k <= c and c > a:
                w = (k - a) / (c - a)
            elif c < k <= b and b > c:
                w = (b - k) / (b - c)
            else:
                w = 0.0
            if w > 0.0:
                comp[n_linear + j, k] = w
                expa[k, n_linear + j] = w

    comp = comp / comp.sum(dim=1, keepdim=True).clamp_min(1e-8)
    # 三角区有些 bin 可能没有任何带覆盖，退回最近带
    row_sum = expa.sum(dim=1)
    for k in range(n_bins):
        if row_sum[k] < 1e-8:
            positions = torch.arange(n_bins, device=comp.device).float()
            centers = (comp * positions).sum(-1)
            distance = (centers - k).abs()
            distance[comp.sum(-1) == 0] = float("inf")
            nearest = distance.argmin().item()
            expa[k, nearest] = 1.0
    expa = expa / expa.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return comp, expa


def band_center_hz(comp: torch.Tensor, bin_hz: float) -> torch.Tensor:
    """按压缩矩阵的权重重心给出每个带的中心频率。"""
    n_bins = comp.shape[1]
    freqs = torch.arange(n_bins, device=comp.device, dtype=comp.dtype) * bin_hz
    return comp @ freqs


def compress_complex(spec: torch.Tensor, comp: torch.Tensor) -> torch.Tensor:
    """spec [..., n_bins, 2] 到 [..., n_bands, 2]。实部虚部分别线性加权。"""
    comp_t = comp.transpose(0, 1)
    return torch.stack(
        [
            torch.matmul(spec[..., 0], comp_t),
            torch.matmul(spec[..., 1], comp_t),
        ],
        dim=-1,
    )


def expand_complex(band: torch.Tensor, expa: torch.Tensor) -> torch.Tensor:
    """band [..., n_bands, 2] 到 [..., n_bins, 2]。"""
    expa_t = expa.transpose(0, 1)
    return torch.stack(
        [
            torch.matmul(band[..., 0], expa_t),
            torch.matmul(band[..., 1], expa_t),
        ],
        dim=-1,
    )
