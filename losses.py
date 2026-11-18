"""损失函数。

架构本身不提供无失真约束，所以两耳一致性只能靠损失约束。ILD 与 IPD 两项不是
加分项，去掉它们网络没有任何理由保持左右一致。
"""

import torch

from complexops import cabs, cmul_conj

EPS = 1e-8


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
    """耳间电平差误差。mask 标出目标活跃频点。"""
    de = 20 * torch.log10((cabs(el) + EPS) / (cabs(er) + EPS))
    dr = 20 * torch.log10((cabs(rl) + EPS) / (cabs(rr) + EPS))
    return (((de - dr) * mask).abs().sum() / mask.sum().clamp_min(1.0))


def ipd_loss(el, er, rl, rr, mask: torch.Tensor) -> torch.Tensor:
    """耳间相位差的圆周误差。用互谱的单位向量差，避免解缠。"""
    ce = cmul_conj(el, er)
    cr = cmul_conj(rl, rr)
    ue = ce / cabs(ce).unsqueeze(-1).clamp_min(EPS)
    ur = cr / cabs(cr).unsqueeze(-1).clamp_min(EPS)
    d = (ue - ur).pow(2).sum(-1)
    return (d * mask).sum() / mask.sum().clamp_min(1.0)


def coherence_penalty(el, er, rl, rr, mask: torch.Tensor) -> torch.Tensor:
    """残余耳间相干度惩罚。输出相干度不应超过输入相干度。

    作用是防止两耳残余噪声被压成完全相关，听感上噪声从扩散塌成居中。
    """
    # 先沿时间平均互谱与自谱，再取模。仅供显式残余谱调用。
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


def binaural_loss(est_l, est_r, ref_l, ref_r, weights: dict | None = None):
    """返回总损失与分项。est 与 ref 均为复数谱 [B, T, F, 2]。"""
    w = {"wave": 0.0, "spec": 1.0, "ild": 0.1, "ipd": 0.1, "coh": 0.0}
    if weights:
        w.update(weights)
    if w.get("coh", 0) != 0:
        raise ValueError("总损失禁用相干度项：未定义残余噪声监督")
    m = active_mask(ref_l, ref_r)
    parts = {
        "spec": spec_loss(est_l, ref_l) + spec_loss(est_r, ref_r),
        "ild": ild_loss(est_l, est_r, ref_l, ref_r, m),
        "ipd": ipd_loss(est_l, est_r, ref_l, ref_r, m),

    }
    total = sum(w[k] * v for k, v in parts.items())
    return total, parts


def training_loss(est, target, mixture, stft):
    """输入 [2, 样本]。静音目标单列输出能量，不除以静音目标能量。"""
    ref_power = target.square().mean()
    mix_power = mixture.square().mean().clamp_min(1e-6)
    if ref_power <= 1e-10:
        return est.square().mean() / mix_power
    wave = (est - target).square().mean() / ref_power.clamp_min(1e-6)
    es, rs = stft.analyze(est), stft.analyze(target)
    spatial, _ = binaural_loss(es[0:1], es[1:2], rs[0:1], rs[1:2])
    return wave + 0.5 * spatial
