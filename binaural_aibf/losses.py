"""损失函数。

提供：
- training_loss_batch(est, target, mixture, stft, loss_cfg=None)：批量入口 [B, 2, S] -> (total, parts)
- training_loss(est, target, mixture, stft, loss_cfg=None)：单样本入口 [2, S] -> total
- binaural_loss(est_l, est_r, ref_l, ref_r, weights=None, ...)

所有 STFT 运算仅调用冻结接口，通过 bands 助手跳过直流 bin 0（128 频点）。
"""

import torch

from .bands import model_bin_freqs, pad_dc_zero, remove_dc_waveform, strip_dc
from .complexops import cabs, cmul_conj

EPS = 1e-8

DEFAULT_LOSS_CFG = {
    "spatial_weight": 0.5,
    "spec_weight": 1.0,
    "ild_weight": 0.1,
    "ipd_weight": 0.1,
    "active_floor_db": -50.0,
    "energy_weighted": True,
    "energy_power": 0.5,
    "ipd_freq_range": [125.0, 1500.0],
    "ild_freq_range": [500.0, 8000.0],
}


def spec_loss(est: torch.Tensor, ref: torch.Tensor, power: float = 0.3) -> torch.Tensor:
    """复数谱与压缩幅度谱联合误差。est 与 ref 形状 [B, T, F, 2]。"""
    me, mr = cabs(est), cabs(ref)
    ce = est * (me.clamp_min(EPS) ** (power - 1)).unsqueeze(-1)
    cr = ref * (mr.clamp_min(EPS) ** (power - 1)).unsqueeze(-1)
    return (ce - cr).pow(2).sum(-1).mean() + (me ** power - mr ** power).pow(2).mean()


def wave_loss(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """保留绝对增益的归一化波形误差。不使用尺度不变形式。"""
    return (est - ref).pow(2).mean() / (ref.pow(2).mean() + EPS)


def ild_loss(el, er, rl, rr, mask: torch.Tensor) -> torch.Tensor:
    """耳间电平差误差。mask 标出目标活跃频点或分频能量权重。"""
    de = 20 * torch.log10((cabs(el) + EPS) / (cabs(er) + EPS))
    dr = 20 * torch.log10((cabs(rl) + EPS) / (cabs(rr) + EPS))
    num = ((de - dr).abs() * mask).sum()
    w_sum = mask.sum()
    return torch.where(w_sum > EPS, num / w_sum.clamp_min(EPS), num.new_zeros(()))


def ipd_loss(el, er, rl, rr, mask: torch.Tensor) -> torch.Tensor:
    """耳间相位差的圆周误差。用互谱的单位向量差，避免解缠。"""
    ce = cmul_conj(el, er)
    cr = cmul_conj(rl, rr)
    ue = ce / cabs(ce).unsqueeze(-1).clamp_min(EPS)
    ur = cr / cabs(cr).unsqueeze(-1).clamp_min(EPS)
    d = (ue - ur).pow(2).sum(-1)
    num = (d * mask).sum()
    w_sum = mask.sum()
    return torch.where(w_sum > EPS, num / w_sum.clamp_min(EPS), num.new_zeros(()))


def coherence_penalty(el, er, rl, rr, mask: torch.Tensor) -> torch.Tensor:
    """残余耳间相干度惩罚。输出相干度不应超过输入相干度。"""
    def coh(a, b):
        cross = cmul_conj(a, b).mean(dim=1)
        pa = a.square().sum(-1).mean(dim=1)
        pb = b.square().sum(-1).mean(dim=1)
        return (torch.linalg.vector_norm(cross, dim=-1) / (pa * pb).sqrt().clamp_min(EPS)).clamp(0, 1)

    excess = (coh(el, er) - coh(rl, rr)).clamp_min(0.0)
    valid = mask.any(dim=1).float()
    return (excess * valid).sum() / valid.sum().clamp_min(1.0)


def active_mask(rl, rr, floor_db: float = -50.0) -> torch.Tensor:
    """目标活跃频点掩码。低能量频点屏蔽，防止静音处除零。"""
    p = 10 * torch.log10((cabs(rl) ** 2 + cabs(rr) ** 2) / 2 + EPS)
    return (p > p.amax(dim=(1, 2), keepdim=True) + floor_db).float()


def spatial_weights(
    ref_l: torch.Tensor,
    ref_r: torch.Tensor,
    freqs_hz: torch.Tensor | None = None,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按目标能量与心理声学频段构造 ILD 与 IPD 权重掩码。"""
    lc = loss_cfg or {}
    floor_db = float(lc.get("active_floor_db", -50.0))
    m = active_mask(ref_l, ref_r, floor_db=floor_db)
    if lc.get("energy_weighted", False):
        power_exp = float(lc.get("energy_power", 0.5))
        amp = ((cabs(ref_l).square() + cabs(ref_r).square()) * 0.5).sqrt()
        base_w = (amp.clamp_min(EPS) ** power_exp) * m
    else:
        base_w = m
    w_ild = base_w
    w_ipd = base_w
    if freqs_hz is not None:
        f = freqs_hz.to(device=ref_l.device, dtype=ref_l.dtype).view(1, 1, -1)
        if "ild_freq_range" in lc:
            lo, hi = lc["ild_freq_range"]
            w_ild = w_ild * ((f >= float(lo)) & (f <= float(hi))).to(dtype=ref_l.dtype)
        if "ipd_freq_range" in lc:
            lo, hi = lc["ipd_freq_range"]
            w_ipd = w_ipd * ((f >= float(lo)) & (f <= float(hi))).to(dtype=ref_l.dtype)
    return w_ild, w_ipd


def binaural_loss(
    est_l,
    est_r,
    ref_l,
    ref_r,
    weights: dict | None = None,
    freqs_hz: torch.Tensor | None = None,
    loss_cfg: dict | None = None,
):
    """返回总损失与分项。est 与 ref 均为复数谱 [B, T, F, 2]。"""
    lc = loss_cfg or {}
    w = {
        "wave": 0.0,
        "spec": float(lc.get("spec_weight", 1.0)),
        "ild": float(lc.get("ild_weight", 0.1)),
        "ipd": float(lc.get("ipd_weight", 0.1)),
        "coh": 0.0,
    }
    if weights:
        w.update(weights)
    if w.get("coh", 0) != 0:
        raise ValueError("总损失禁用相干度项：未定义残余噪声监督")
    w_ild, w_ipd = spatial_weights(ref_l, ref_r, freqs_hz=freqs_hz, loss_cfg=lc)
    parts = {
        "spec": spec_loss(est_l, ref_l) + spec_loss(est_r, ref_r),
        "ild": ild_loss(est_l, est_r, ref_l, ref_r, w_ild),
        "ipd": ipd_loss(est_l, est_r, ref_l, ref_r, w_ipd),
    }
    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def _mean_or_zero(vals: list[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    return torch.stack(vals).mean() if vals else ref.new_zeros(())


def training_loss_batch(
    est: torch.Tensor,
    target: torch.Tensor,
    mixture: torch.Tensor,
    stft,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """批量损失入口。输入 [B, 2, S]（第 0 通道左耳），返回 (total, detached parts)。

    静音目标样本按样本单独处理，不改变批内非静音样本的波形、谱与空间分项损失。
    """
    if est.dim() != 3 or est.shape[1] != 2 or est.shape != target.shape or est.shape != mixture.shape:
        raise ValueError("training_loss_batch 要求 est、target、mixture 均为同形状 [B, 2, S]")

    lc = DEFAULT_LOSS_CFG if loss_cfg is None else {**DEFAULT_LOSS_CFG, **loss_cfg}
    b, _, s = est.shape
    es_all = strip_dc(stft.analyze(est.reshape(b * 2, s)))
    rs_all = strip_dc(stft.analyze(target.reshape(b * 2, s)))
    est_ac = stft.synthesize(pad_dc_zero(es_all), length=s).reshape(b, 2, s)
    target_ac = stft.synthesize(pad_dc_zero(rs_all), length=s).reshape(b, 2, s)
    mix_ac = remove_dc_waveform(mixture.reshape(b * 2, s), stft).reshape(b, 2, s)

    es_b = es_all.reshape(b, 2, es_all.shape[1], es_all.shape[2], 2)
    rs_b = rs_all.reshape(b, 2, rs_all.shape[1], rs_all.shape[2], 2)
    freqs_hz = model_bin_freqs(stft, device=est.device)
    spatial_weight = float(lc.get("spatial_weight", 0.5))

    sample_losses = []
    wave_vals, spec_vals, ild_vals, ipd_vals, empty_vals = [], [], [], [], []
    for i in range(b):
        ref_power = target_ac[i].square().mean()
        mix_power = mix_ac[i].square().mean().clamp_min(1e-6)
        if ref_power <= 1e-10:
            e_i = est_ac[i].square().mean() / mix_power
            sample_losses.append(e_i)
            empty_vals.append(e_i.detach())
        else:
            w_i = (est_ac[i] - target_ac[i]).square().mean() / ref_power.clamp_min(1e-6)
            sp_total, sp_parts = binaural_loss(
                es_b[i : i + 1, 0],
                es_b[i : i + 1, 1],
                rs_b[i : i + 1, 0],
                rs_b[i : i + 1, 1],
                freqs_hz=freqs_hz,
                loss_cfg=lc,
            )
            sample_losses.append(w_i + spatial_weight * sp_total)
            wave_vals.append(w_i.detach())
            spec_vals.append(sp_parts["spec"].detach())
            ild_vals.append(sp_parts["ild"].detach())
            ipd_vals.append(sp_parts["ipd"].detach())

    total = torch.stack(sample_losses).mean()
    parts = {
        "wave": _mean_or_zero(wave_vals, total),
        "spec": _mean_or_zero(spec_vals, total),
        "ild": _mean_or_zero(ild_vals, total),
        "ipd": _mean_or_zero(ipd_vals, total),
        "empty": _mean_or_zero(empty_vals, total),
    }
    return total, parts


def training_loss(est, target, mixture, stft, loss_cfg: dict | None = None):
    """单样本入口 [2, S]（兼容 [B, 2, S]），返回标量损失。"""
    if est.dim() == 2:
        total, _ = training_loss_batch(
            est.unsqueeze(0),
            target.unsqueeze(0),
            mixture.unsqueeze(0),
            stft,
            loss_cfg=loss_cfg,
        )
        return total
    total, _ = training_loss_batch(est, target, mixture, stft, loss_cfg=loss_cfg)
    return total
