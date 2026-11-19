"""CRF 估计器的输入特征。

特征分两类，对应 AIBF 的五维空间特征在双耳场景下的拆分。

本侧当前帧特征
- 对数幅度谱，按频带。

跨耳历史帧特征
- 余弦相位差与正弦相位差。
- 电平差。
- 消息年龄与有效位。

跨耳特征必须在同时间戳的两帧之间计算。本侧谱因此要按 matched_idx 取配对帧。
这一步只影响特征支路，合成支路仍然用当前帧，所以不产生额外延迟。
"""

import torch

from complexops import cabs, cmul_conj

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


def build_crf_input(
    cfg: dict,
    local_band_now: torch.Tensor,
    local_band_matched: torch.Tensor,
    remote_band: torch.Tensor,
    valid: torch.Tensor,
    age_frames: torch.Tensor,
    valid_band: torch.Tensor,
) -> torch.Tensor:
    """拼出 CRF 全带估计器的输入向量 [B, T, D]。

    local_band_now      当前帧本侧频带谱
    local_band_matched  与对耳消息同时间戳的本侧频带谱
    remote_band         接收侧实际持有的对耳频带谱
    valid               [B, T, 1] 是否已收到任何消息
    age_frames          [T] 消息年龄
    valid_band          [n_bands] 哪些频带有对耳信息
    """
    c = cfg["crf"]
    b, t, n_bands = local_band_now.shape[0], local_band_now.shape[1], local_band_now.shape[2]
    device = local_band_now.device
    parts = []

    if c["use_local_logmag"]:
        parts.append(log_mag(local_band_now))

    vb = valid_band.to(device).view(1, 1, n_bands)
    gate = vb * valid  # 频带无对耳信息或尚未收到消息时置零

    if c["use_cross_ipd"]:
        cos_ipd, sin_ipd = cross_ipd(local_band_matched, remote_band)
        if cfg["message"]["payload"] == "logmag":
            cos_ipd = torch.zeros_like(cos_ipd)
            sin_ipd = torch.zeros_like(sin_ipd)
        parts.append(cos_ipd * gate)
        parts.append(sin_ipd * gate)

    if c["use_cross_ild"]:
        parts.append(cross_ild(local_band_matched, remote_band) * gate)

    feat = torch.cat(parts, dim=-1)

    if c["use_age_feature"]:
        max_age = float(cfg["crf"].get("age_scale_frames", 20))
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
) -> torch.Tensor:
    """拼出单帧流式推理的 CRF 输入向量 [B, D]。

    local_band_now      [B, n_bands, 2] 当前帧本侧频带谱
    local_band_matched  [B, n_bands, 2] 与对耳消息同时间戳的本侧频带谱
    remote_band         [B, n_bands, 2] 接收侧当前持有的对耳频带谱
    valid               [B, 1] 是否已收到任何消息
    age_frames          [B, 1] 当前消息年龄（单位帧）
    valid_band          [n_bands] 哪些频带有对耳信息
    """
    c = cfg["crf"]
    parts = []

    if c["use_local_logmag"]:
        parts.append(log_mag(local_band_now))

    vb = valid_band.to(device=local_band_now.device, dtype=local_band_now.dtype).unsqueeze(0)
    gate = vb * valid  # [B, n_bands]

    if c["use_cross_ipd"]:
        cos_ipd, sin_ipd = cross_ipd(local_band_matched, remote_band)
        if cfg["message"]["payload"] == "logmag":
            cos_ipd = torch.zeros_like(cos_ipd)
            sin_ipd = torch.zeros_like(sin_ipd)
        parts.append(cos_ipd * gate)
        parts.append(sin_ipd * gate)

    if c["use_cross_ild"]:
        parts.append(cross_ild(local_band_matched, remote_band) * gate)

    feat = torch.cat(parts, dim=-1)

    if c["use_age_feature"]:
        max_age = float(cfg["crf"].get("age_scale_frames", 20))
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
    if c["use_age_feature"]:
        d += 2
    return d
