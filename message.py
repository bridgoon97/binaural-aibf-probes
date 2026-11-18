"""跨耳消息通道。

这个模块是两个探针实验的全部可调面。

实验一改 delay_frames。实验二改 payload、band_range、n_msg_bands、bits_mag、
bits_phase、update_period。

语义约定
- 发送侧在帧索引能被 update_period 整除时发一条消息。
- 接收侧在帧 t 只能用满足 t_e + delay_frames <= t 的最新一条消息。
- 两次消息之间接收侧保持上一条，消息年龄随之增长。
- 消息内容由输入信号直接导出，不含网络状态。因此 delay_frames 取 0 也不构成环路。

这里的向量化实现与逐帧流式实现语义相同。t_e 由闭式给出，不做任何未来索引。
"""

from dataclasses import dataclass

import torch

from complexops import cabs, cangle, from_polar

EPS = 1e-8


@dataclass
class BitrateReport:
    values_per_message: int
    bits_per_message: int | None
    messages_per_sec: float
    payload_kbps_one_way: float | None
    app_kbps_one_way: float | None
    app_kbps_two_way: float | None

    def describe(self) -> str:
        if self.bits_per_message is None:
            return "浮点参照：不报告量化通信码率"
        return (
            f"每条消息 {self.values_per_message} 个量、{self.bits_per_message} bit，"
            f"每秒 {self.messages_per_sec:.1f} 条\n"
            f"单向净载荷 {self.payload_kbps_one_way:.2f} kbps，"
            f"单向应用层 {self.app_kbps_one_way:.2f} kbps，"
            f"双向应用层 {self.app_kbps_two_way:.2f} kbps"
        )


def quantize_ste(x: torch.Tensor, lo: float, hi: float, bits: int) -> torch.Tensor:
    """固定范围均匀标量量化，反向用直通估计。

    bits 取 0 表示不量化，用于拿浮点上界。
    """
    if bits <= 0:
        return x
    levels = 2 ** bits - 1
    xc = x.clamp(lo, hi)
    step = (hi - lo) / levels
    q = torch.round((xc - lo) / step) * step + lo
    return xc + (q - xc).detach()


class CrossEarMessage:
    """把一侧的频带复数谱编成消息，再按延迟与更新周期交付到另一侧。"""

    def __init__(self, cfg: dict, band_centers_hz: torch.Tensor, frame_rate: float):
        m = cfg["message"]
        self.payload = m["payload"]
        assert self.payload in ("complex", "logmag"), "payload 只支持 complex 与 logmag"
        self.band_range = tuple(m["band_range"])
        self.n_msg_bands = int(m["n_msg_bands"])
        self.bits_mag = int(m["bits_mag"])
        self.bits_phase = int(m["bits_phase"])
        self.mag_lo, self.mag_hi = m["mag_range_db"]
        self.update_period = int(m["update_period"])
        self.delay_frames = int(m["delay_frames"])
        self.sync_error_us = float(m["sync_error_us"])
        self.header_bits = int(m["header_bits"])
        self.frame_rate = frame_rate

        self.band_centers = band_centers_hz
        n_bands = band_centers_hz.numel()
        in_range = (band_centers_hz >= self.band_range[0]) & (
            band_centers_hz <= self.band_range[1]
        )
        self.in_range_idx = torch.nonzero(in_range, as_tuple=False).flatten()
        assert self.in_range_idx.numel() > 0, "band_range 内没有任何频带"
        assert self.n_msg_bands <= self.in_range_idx.numel(), (
            f"n_msg_bands={self.n_msg_bands} 超过 band_range 内的频带数 "
            f"{self.in_range_idx.numel()}"
        )

        assert self.update_period >= 1 and self.delay_frames >= 0
        assert self.bits_mag >= 0 and self.bits_phase >= 0
        assert self.header_bits >= 0
        assert self.mag_hi > self.mag_lo
        # 固定选择模型频带，不用数据依赖的峰值相位，不广播到不同频点。
        positions = torch.linspace(0, self.in_range_idx.numel() - 1, self.n_msg_bands).round().long()
        self.selected_idx = self.in_range_idx[positions]
        self.valid_band = torch.zeros(n_bands)
        self.valid_band[self.selected_idx] = 1.0
        self.n_bands = n_bands

    def bitrate(self) -> BitrateReport:
        if self.payload == "complex":
            vals = self.n_msg_bands * 2
            bits = self.n_msg_bands * (self.bits_mag + self.bits_phase)
        else:
            vals = self.n_msg_bands
            bits = self.n_msg_bands * self.bits_mag
        mps = self.frame_rate / self.update_period
        if self.bits_mag == 0 or (self.payload == "complex" and self.bits_phase == 0):
            return BitrateReport(vals, None, mps, None, None, None)
        payload_kbps = bits * mps / 1000.0
        app_kbps = ((bits + self.header_bits + 7) // 8 * 8) * mps / 1000.0
        return BitrateReport(
            values_per_message=vals,
            bits_per_message=bits,
            messages_per_sec=mps,
            payload_kbps_one_way=payload_kbps,
            app_kbps_one_way=app_kbps,
            app_kbps_two_way=2 * app_kbps,
        )

    def _encode(self, band_spec: torch.Tensor) -> torch.Tensor:
        """band_spec [B, T, n_bands, 2] 到消息域 [B, T, n_msg_bands, 2]。

        消息域的第二个分量在 logmag 模式下恒为零。
        """
        sel = band_spec[:, :, self.selected_idx.to(band_spec.device), :]
        out_mag = cabs(sel)
        out_pha = cangle(sel)

        mag_db = 20.0 * torch.log10(out_mag + EPS)
        mag_db = quantize_ste(mag_db, self.mag_lo, self.mag_hi, self.bits_mag)
        if self.payload == "complex":
            pha_q = quantize_ste(out_pha, -torch.pi, torch.pi, self.bits_phase)
        else:
            pha_q = torch.zeros_like(out_pha)
        return torch.stack([mag_db, pha_q], dim=-1)

    def _decode(self, msg: torch.Tensor, device) -> torch.Tensor:
        """消息域 [B, T, n_msg_bands, 2] 回到模型频带 [B, T, n_bands, 2]。

        band_range 以外的频带填零。logmag 模式下相位填零，此时跨耳相位特征无意义。
        """
        mag = 10.0 ** (msg[..., 0] / 20.0)
        pha = msg[..., 1]
        cplx = from_polar(mag, pha)  # [B, T, n_msg_bands, 2]

        b, t = cplx.shape[0], cplx.shape[1]
        out = torch.zeros(b, t, self.n_bands, 2, device=device)
        idx = self.selected_idx.to(device)
        out[:, :, idx, :] = cplx
        return out

    def _apply_sync_error(self, band_spec: torch.Tensor, band_centers: torch.Tensor):
        """给对耳谱加一个恒定时间偏移对应的相位旋转。"""
        if abs(self.sync_error_us) < 1e-9:
            return band_spec
        tau = self.sync_error_us * 1e-6
        ang = -2 * torch.pi * band_centers.to(band_spec.device) * tau
        rot = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        from complexops import cmul

        return cmul(band_spec, rot.view(1, 1, -1, 2))

    def deliver(self, remote_band_spec: torch.Tensor):
        """把对耳频带谱编码、延迟、解码。

        返回
        - received: [B, T, n_bands, 2] 接收侧在每帧实际持有的对耳频带谱
        - matched_idx: [T] 与 received 同时间戳的帧索引，用于取本侧配对帧
        - valid: [B, T, 1] 该帧是否已经收到过任何消息
        - age_frames: [T] 消息年龄，单位帧
        - decoded_all: [B, T, n_bands, 2] 按发送帧索引排列的解码结果

        decoded_all 供 FIR 取多个抽头使用。只有索引能被 update_period 整除的位置
        对应真实发出的消息，其余位置不要直接读。
        """
        b, t = remote_band_spec.shape[0], remote_band_spec.shape[1]
        device = remote_band_spec.device

        spec = self._apply_sync_error(remote_band_spec, self.band_centers)
        msg = self._encode(spec)

        frames = torch.arange(t, device=device)
        avail = frames - self.delay_frames
        emit = torch.div(avail, self.update_period, rounding_mode="floor") * self.update_period
        valid_t = emit >= 0
        emit_clamped = emit.clamp_min(0)
        age = frames - emit_clamped

        decoded_all = self._decode(msg, device)
        received = decoded_all[:, emit_clamped]
        valid = valid_t.float().view(1, t, 1).expand(b, t, 1)
        received = received * valid.unsqueeze(-1)
        return received, emit_clamped, valid, age, decoded_all
