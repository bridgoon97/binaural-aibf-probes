"""跨耳消息通道。

支持三种载荷模式：
- complex：对数幅度 + 圆周相位。
- logmag：仅送对数幅度（屏蔽伪 IPD）。
- mixed：64 bit 混合载荷（6 bit 宽带参考 + 6 低频圆周相位 + 7 高频相对幅度）。
"""

from dataclasses import dataclass

import torch

from .complexops import cabs, cangle, cmul, from_polar

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
    """固定范围均匀标量量化，反向用直通估计。bits 取 0 表示不量化。"""
    if bits <= 0:
        return x
    levels = 2 ** bits - 1
    xc = x.clamp(lo, hi)
    step = (hi - lo) / levels
    q = torch.round((xc - lo) / step) * step + lo
    return xc + (q - xc).detach()


def quantize_phase_ste(phase: torch.Tensor, bits: int) -> torch.Tensor:
    """在 [-π, π) 圆周上做 2^b 等分均匀量化，消除 ±π 重复码字，反向用直通估计。"""
    if bits <= 0:
        return phase
    n_levels = 2 ** bits
    step = (2.0 * torch.pi) / n_levels
    wrapped = torch.remainder(phase + torch.pi, 2.0 * torch.pi)
    idx = torch.remainder(torch.round(wrapped / step), float(n_levels))
    q = idx * step - torch.pi
    return phase + (q - phase).detach()


class CrossEarMessage:
    """把一侧参考麦的频带复数谱编成消息，再按延迟与更新周期交付到另一侧。"""

    def __init__(
        self,
        cfg: dict,
        band_centers_hz: torch.Tensor,
        frame_rate: float,
        n_linear: int | None = None,
    ):
        m = cfg["message"]
        self.payload = m["payload"]
        assert self.payload in ("complex", "logmag", "mixed"), (
            "payload 只支持 complex、logmag 与 mixed"
        )
        self.band_range = tuple(m.get("band_range", [125, 1000]))
        self.n_msg_bands = int(m.get("n_msg_bands", 8))
        self.bits_mag = int(m["bits_mag"])
        self.bits_phase = int(m["bits_phase"])
        self.mag_lo, self.mag_hi = m.get("mag_range_db", [-60, 20])
        self.update_period = int(m["update_period"])
        self.delay_frames = int(m["delay_frames"])
        self.sync_error_us = float(m.get("sync_error_us", 0.0))
        self.header_bits = int(m.get("header_bits", 32))
        self.frame_rate = frame_rate

        self.band_centers = band_centers_hz
        n_bands = band_centers_hz.numel()
        n_lin = int(cfg["bands"]["n_linear"]) if n_linear is None else int(n_linear)

        assert self.update_period >= 1 and self.delay_frames >= 0
        assert self.bits_mag >= 0 and self.bits_phase >= 0
        assert self.header_bits >= 0
        assert self.mag_hi > self.mag_lo

        self.valid_band = torch.zeros(n_bands)
        self.valid_band_phase = torch.zeros(n_bands)

        if self.payload == "mixed":
            self.bits_ref = int(m.get("bits_ref", 6))
            self.ref_lo, self.ref_hi = [float(v) for v in m.get("ref_range_db", [-60.0, 0.0])]
            self.phase_band_range = tuple(
                float(v) for v in m.get("phase_band_range", [125.0, 1375.0])
            )
            self.n_phase_bands = int(m.get("n_phase_bands", 6))
            self.mag_band_range = tuple(
                float(v) for v in m.get("mag_band_range", [1500.0, 7700.0])
            )
            self.n_mag_bands = int(m.get("n_mag_bands", 7))
            self.rel_mag_lo, self.rel_mag_hi = [
                float(v) for v in m.get("rel_mag_range_db", [-24.0, 12.0])
            ]
            assert self.bits_ref >= 0 and self.ref_hi > self.ref_lo
            assert self.rel_mag_hi > self.rel_mag_lo
            assert self.n_phase_bands >= 1 and self.n_mag_bands >= 1

            in_phase = (band_centers_hz >= self.phase_band_range[0]) & (
                band_centers_hz <= self.phase_band_range[1]
            )
            in_phase_idx = torch.nonzero(in_phase, as_tuple=False).flatten()
            assert in_phase_idx.numel() >= self.n_phase_bands, (
                f"phase_band_range 内频带不足：{in_phase_idx.numel()} < {self.n_phase_bands}"
            )
            pos_p = torch.linspace(0, in_phase_idx.numel() - 1, self.n_phase_bands).round().long()
            self.phase_idx = in_phase_idx[pos_p]
            assert int(self.phase_idx.max().item()) < n_lin, "mixed 相位频带必须全部位于低频恒等区"

            in_mag = (band_centers_hz >= self.mag_band_range[0]) & (
                band_centers_hz <= self.mag_band_range[1]
            )
            in_mag_idx = torch.nonzero(in_mag, as_tuple=False).flatten()
            assert in_mag_idx.numel() >= self.n_mag_bands, (
                f"mag_band_range 内频带不足：{in_mag_idx.numel()} < {self.n_mag_bands}"
            )
            pos_m = torch.linspace(0, in_mag_idx.numel() - 1, self.n_mag_bands).round().long()
            self.mag_idx = in_mag_idx[pos_m]
            assert set(self.phase_idx.tolist()).isdisjoint(set(self.mag_idx.tolist())), (
                "mixed 相位频带与幅度频带不得重叠"
            )

            self.selected_idx = torch.cat([self.phase_idx, self.mag_idx])
            self.valid_band[self.mag_idx] = 1.0
            self.valid_band_phase[self.phase_idx] = 1.0
        else:
            in_range = (band_centers_hz >= self.band_range[0]) & (
                band_centers_hz <= self.band_range[1]
            )
            self.in_range_idx = torch.nonzero(in_range, as_tuple=False).flatten()
            assert self.in_range_idx.numel() > 0, "band_range 内没有任何频带"
            assert self.n_msg_bands <= self.in_range_idx.numel(), (
                f"n_msg_bands={self.n_msg_bands} 超过 band_range 内的频带数 "
                f"{self.in_range_idx.numel()}"
            )
            positions = (
                torch.linspace(0, self.in_range_idx.numel() - 1, self.n_msg_bands).round().long()
            )
            self.selected_idx = self.in_range_idx[positions]
            self.valid_band[self.selected_idx] = 1.0
            if self.payload == "complex":
                phase_idx = self.selected_idx[self.selected_idx < n_lin]
                self.valid_band_phase[phase_idx] = 1.0

        self.mag_only_band = self.valid_band * (1.0 - self.valid_band_phase)
        self.n_bands = n_bands

    def bitrate(self) -> BitrateReport:
        mps = self.frame_rate / self.update_period
        if self.payload == "mixed":
            vals = 1 + self.n_phase_bands + self.n_mag_bands
            if self.bits_ref == 0 or self.bits_phase == 0 or self.bits_mag == 0:
                return BitrateReport(vals, None, mps, None, None, None)
            bits = (
                self.bits_ref
                + self.n_phase_bands * self.bits_phase
                + self.n_mag_bands * self.bits_mag
            )
        elif self.payload == "complex":
            vals = self.n_msg_bands * 2
            if self.bits_mag == 0 or self.bits_phase == 0:
                return BitrateReport(vals, None, mps, None, None, None)
            bits = self.n_msg_bands * (self.bits_mag + self.bits_phase)
        else:
            vals = self.n_msg_bands
            if self.bits_mag == 0:
                return BitrateReport(vals, None, mps, None, None, None)
            bits = self.n_msg_bands * self.bits_mag
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
        """band_spec [B, T, n_bands, 2] 到消息域。"""
        if self.payload == "mixed":
            sel_pha = band_spec[:, :, self.phase_idx.to(band_spec.device), :]
            sel_mag = band_spec[:, :, self.mag_idx.to(band_spec.device), :]
            mag_db = 20.0 * torch.log10(cabs(sel_mag) + EPS)
            ref_db = mag_db.mean(dim=-1, keepdim=True)
            ref_q = quantize_ste(ref_db, self.ref_lo, self.ref_hi, self.bits_ref)
            rel_db = mag_db - ref_q
            rel_q = quantize_ste(rel_db, self.rel_mag_lo, self.rel_mag_hi, self.bits_mag)
            pha_q = quantize_phase_ste(cangle(sel_pha), self.bits_phase)
            return torch.cat([ref_q, pha_q, rel_q], dim=-1)

        sel = band_spec[:, :, self.selected_idx.to(band_spec.device), :]
        out_mag = cabs(sel)
        out_pha = cangle(sel)

        mag_db = 20.0 * torch.log10(out_mag + EPS)
        mag_db = quantize_ste(mag_db, self.mag_lo, self.mag_hi, self.bits_mag)
        if self.payload == "complex":
            pha_q = quantize_phase_ste(out_pha, self.bits_phase)
        else:
            pha_q = torch.zeros_like(out_pha)
        return torch.stack([mag_db, pha_q], dim=-1)

    def _decode(self, msg: torch.Tensor, device) -> torch.Tensor:
        """消息域回到模型频带 [B, T, n_bands, 2]。"""
        if self.payload == "mixed":
            ref_q = msg[..., :1]
            pha_q = msg[..., 1 : 1 + self.n_phase_bands]
            rel_q = msg[..., 1 + self.n_phase_bands :]
            b, t = ref_q.shape[0], ref_q.shape[1]
            out = torch.zeros(b, t, self.n_bands, 2, device=device, dtype=msg.dtype)
            ref_mag = 10.0 ** (ref_q / 20.0)
            pha_cplx = from_polar(ref_mag.expand(-1, -1, self.n_phase_bands), pha_q)
            mag_lin = 10.0 ** ((ref_q + rel_q) / 20.0)
            mag_cplx = torch.stack([mag_lin, torch.zeros_like(mag_lin)], dim=-1)
            out[:, :, self.phase_idx.to(device), :] = pha_cplx
            out[:, :, self.mag_idx.to(device), :] = mag_cplx
            return out

        mag = 10.0 ** (msg[..., 0] / 20.0)
        pha = msg[..., 1]
        cplx = from_polar(mag, pha)

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
        return cmul(band_spec, rot.view(1, 1, -1, 2))

    def deliver(self, remote_band_spec: torch.Tensor):
        """把对耳参考麦频带谱编码、延迟、解码。"""
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

    def encode_decode_step(self, band_spec_frame: torch.Tensor) -> torch.Tensor:
        """单帧流式消息编解码。[B, n_bands, 2] -> [B, n_bands, 2]。"""
        x = band_spec_frame.unsqueeze(1)
        spec = self._apply_sync_error(x, self.band_centers)
        msg = self._encode(spec)
        return self._decode(msg, band_spec_frame.device).squeeze(1)
