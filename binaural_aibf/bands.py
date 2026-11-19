"""频带压缩与解压矩阵，以及基于冻结 STFT 接口的去直流辅助工具。

低频逐 bin 恒等保相位，高频三角带采用频点功率池化避免复数相消干涉。
"""

import torch


def strip_dc(spec: torch.Tensor) -> torch.Tensor:
    """去掉直流 bin 0，将 [..., n_bins, 2] 裁为 [..., n_bins - 1, 2]。"""
    return spec[..., 1:, :]


def pad_dc_zero(spec: torch.Tensor) -> torch.Tensor:
    """在频点维前端补回全零直流 bin 0，将 [..., n_bins - 1, 2] 扩为 [..., n_bins, 2]。"""
    dc = spec.new_zeros(*spec.shape[:-2], 1, 2)
    return torch.cat([dc, spec], dim=-2)


def model_bin_freqs(stft, device=None) -> torch.Tensor:
    """仅读取冻结成员 stft.n_bins 与 stft.bin_hz，返回非直流频点频率（62.5..8000 Hz）。"""
    return torch.arange(1, stft.n_bins, device=device, dtype=torch.float32) * stft.bin_hz


def remove_dc_waveform(wav: torch.Tensor, stft) -> torch.Tensor:
    """仅调用冻结 stft.analyze 与 stft.synthesize，返回去除直流 bin 0 的波形。"""
    spec_ac = strip_dc(stft.analyze(wav))
    return stft.synthesize(pad_dc_zero(spec_ac), length=wav.shape[-1])


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


def band_center_hz(comp: torch.Tensor, bin_hz: float, start_bin: int = 1) -> torch.Tensor:
    """按压缩矩阵的权重重心给出每个带的中心频率。start_bin=1 表示跳过直流 bin 0。"""
    n_bins = comp.shape[1]
    freqs = (torch.arange(n_bins, device=comp.device, dtype=comp.dtype) + start_bin) * bin_hz
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


def compress_power(spec: torch.Tensor, comp: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """频点功率池化。[..., n_bins, 2] 到 [..., n_bands] 幅度，消除带内复数相消干涉。"""
    power = spec[..., 0].square() + spec[..., 1].square()
    band_power = torch.matmul(power, comp.transpose(0, 1))
    return torch.sqrt(band_power.clamp_min(0.0) + eps)


def compress_band(
    spec: torch.Tensor, comp: torch.Tensor, n_linear: int, eps: float = 1e-12
) -> torch.Tensor:
    """混合频带压缩：低频恒等区保复数，高频三角区用功率池化幅度写入实部（虚部置零）。"""
    lin = spec[..., :n_linear, :]
    tri_mag = compress_power(spec, comp[n_linear:], eps=eps)
    tri = torch.stack([tri_mag, torch.zeros_like(tri_mag)], dim=-1)
    return torch.cat([lin, tri], dim=-2)


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
