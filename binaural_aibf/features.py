"""CRF 估计器的输入特征与当前帧谱归一化。

跨耳特征仅使用本侧第 0 参考麦与对耳消息。
当前帧快支路谱支持多麦联合单帧 RMS 归一化，保留耳内电平与相位关系。
"""

import torch

from .complexops import cabs, cmul_conj

EPS = 1e-8


def log_mag(band_spec: torch.Tensor) -> torch.Tensor:
    """[B, T, n_bands, 2] 到 [B, T, n_bands]，归一到大致 [-1, 1]。"""
    db = 20.0 * torch.log10(cabs(band_spec) + EPS)
    return (db.clamp(-80.0, 20.0) + 30.0) / 50.0


def cross_ipd(local: torch.Tensor, remote: torch.Tensor):
    """返回余弦与正弦相位差，各为 [B, T, n_bands]。"""
    cross = cmul_conj(local, remote)
    denom = (cabs(local) * cabs(remote)).clamp_min(EPS)
    return cross[..., 0] / denom, cross[..., 1] / denom


def cross_ild(local: torch.Tensor, remote: torch.Tensor) -> torch.Tensor:
    """返回耳间电平差，单位 dB，裁剪并归一到大致 [-1, 1]。"""
    ratio = 20.0 * torch.log10((cabs(local) + EPS) / (cabs(remote) + EPS))
    return ratio.clamp(-20.0, 20.0) / 20.0


def compress_current_spec(
    band_spec: torch.Tensor,
    power: float = 0.3,
    eps: float = 1e-8,
    has_mic_dim: bool = False,
) -> torch.Tensor:
    """对当前帧频带谱做幂律压缩并按单帧联合 RMS 归一化，严格逐帧因果。

    - has_mic_dim=False：输入 [..., n_bands, 2]，沿 (-2, -1) 求 RMS。
    - has_mic_dim=True：输入 [..., n_mic, n_bands, 2]，沿 (-3, -2, -1) 求联合 RMS，
      保留同耳多麦间电平差与相位差。
    """
    mag = cabs(band_spec)
    comp = band_spec * (mag.clamp_min(eps) ** (power - 1.0)).unsqueeze(-1)
    reduce_dims = (-3, -2, -1) if has_mic_dim else (-2, -1)
    rms = comp.square().mean(dim=reduce_dims, keepdim=True).add(eps).sqrt()
    return comp / rms


def frontal_features(
    cos_ipd: torch.Tensor,
    sin_ipd: torch.Tensor,
    ild_norm: torch.Tensor,
    phase_gate: torch.Tensor,
    mag_gate: torch.Tensor,
    sigma_db: float = 6.0,
) -> torch.Tensor:
    """构造 0° 正前方物理方向先验特征，共 n_bands + 3 维。"""
    p_denom = phase_gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
    d_front = (cos_ipd * phase_gate).sum(dim=-1, keepdim=True) / p_denom
    s_front = (sin_ipd * phase_gate).sum(dim=-1, keepdim=True) / p_denom
    b_ild = torch.exp(-20.0 * ild_norm.abs() / sigma_db) * mag_gate
    m_denom = mag_gate.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_b_ild = b_ild.sum(dim=-1, keepdim=True) / m_denom
    return torch.cat([d_front, s_front, mean_b_ild, b_ild], dim=-1)


def build_crf_input(
    cfg: dict,
    local_band_now: torch.Tensor,
    local_band_matched: torch.Tensor,
    remote_band: torch.Tensor,
    valid: torch.Tensor,
    age_frames: torch.Tensor,
    valid_band: torch.Tensor,
    valid_band_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """拼出 CRF 全带估计器的输入向量 [B, T, D]。"""
    c = cfg["crf"]
    b, t, n_bands = local_band_now.shape[0], local_band_now.shape[1], local_band_now.shape[2]
    device = local_band_now.device
    parts = []

    if c["use_local_logmag"]:
        parts.append(log_mag(local_band_now))

    vb = valid_band.to(device).view(1, 1, n_bands)
    vb_phase = (valid_band if valid_band_phase is None else valid_band_phase).to(device).view(
        1, 1, n_bands
    )
    mag_gate = vb * valid
    phase_gate = vb_phase * valid

    use_frontal = bool(c.get("use_frontal_features", False))
    need_ipd = bool(c["use_cross_ipd"]) or use_frontal
    need_ild = bool(c["use_cross_ild"]) or use_frontal

    if need_ipd:
        cos_ipd, sin_ipd = cross_ipd(local_band_matched, remote_band)
        if cfg["message"]["payload"] == "logmag":
            cos_ipd = torch.zeros_like(cos_ipd)
            sin_ipd = torch.zeros_like(sin_ipd)
    if need_ild:
        ild_norm = cross_ild(local_band_matched, remote_band)

    if c["use_cross_ipd"]:
        parts.append(cos_ipd * phase_gate)
        parts.append(sin_ipd * phase_gate)

    if c["use_cross_ild"]:
        parts.append(ild_norm * mag_gate)

    if use_frontal:
        sigma_db = float(c.get("frontal_ild_sigma_db", 6.0))
        parts.append(
            frontal_features(cos_ipd, sin_ipd, ild_norm, phase_gate, mag_gate, sigma_db)
        )

    feat = torch.cat(parts, dim=-1)

    if c["use_age_feature"]:
        max_age = float(c.get("age_scale_frames", 20))
        age = (age_frames.float() / max_age).view(1, t, 1).expand(b, t, 1)
        feat = torch.cat([feat, age, valid], dim=-1)

    return feat


def build_crf_input_step(
    cfg: dict,
    local_band_now: torch.Tensor,
    local_band_matched: torch.Tensor,
    remote_band: torch.Tensor,
    valid: torch.Tensor,
    age_frames: torch.Tensor,
    valid_band: torch.Tensor,
    valid_band_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """拼出单帧流式推理的 CRF 输入向量 [B, D]。"""
    c = cfg["crf"]
    parts = []

    if c["use_local_logmag"]:
        parts.append(log_mag(local_band_now))

    vb = valid_band.to(device=local_band_now.device, dtype=local_band_now.dtype).unsqueeze(0)
    vb_phase = (
        (valid_band if valid_band_phase is None else valid_band_phase)
        .to(device=local_band_now.device, dtype=local_band_now.dtype)
        .unsqueeze(0)
    )
    mag_gate = vb * valid
    phase_gate = vb_phase * valid

    use_frontal = bool(c.get("use_frontal_features", False))
    need_ipd = bool(c["use_cross_ipd"]) or use_frontal
    need_ild = bool(c["use_cross_ild"]) or use_frontal

    if need_ipd:
        cos_ipd, sin_ipd = cross_ipd(local_band_matched, remote_band)
        if cfg["message"]["payload"] == "logmag":
            cos_ipd = torch.zeros_like(cos_ipd)
            sin_ipd = torch.zeros_like(sin_ipd)
    if need_ild:
        ild_norm = cross_ild(local_band_matched, remote_band)

    if c["use_cross_ipd"]:
        parts.append(cos_ipd * phase_gate)
        parts.append(sin_ipd * phase_gate)

    if c["use_cross_ild"]:
        parts.append(ild_norm * mag_gate)

    if use_frontal:
        sigma_db = float(c.get("frontal_ild_sigma_db", 6.0))
        parts.append(
            frontal_features(cos_ipd, sin_ipd, ild_norm, phase_gate, mag_gate, sigma_db)
        )

    feat = torch.cat(parts, dim=-1)

    if c["use_age_feature"]:
        max_age = float(c.get("age_scale_frames", 20))
        age = age_frames.to(dtype=local_band_now.dtype) / max_age
        feat = torch.cat([feat, age, valid.to(dtype=local_band_now.dtype)], dim=-1)

    return feat


def crf_input_dim(cfg: dict, n_bands: int) -> int:
    c = cfg["crf"]
    d = 0
    if c["use_local_logmag"]:
        d += n_bands
    if c["use_cross_ipd"]:
        d += 2 * n_bands
    if c["use_cross_ild"]:
        d += n_bands
    if c.get("use_frontal_features", False):
        d += n_bands + 3
    if c["use_age_feature"]:
        d += 2
    return d
