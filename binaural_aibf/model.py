"""双耳 AIBF 主模型。

支持每耳 1 麦及多麦（local.n_mic >= 1）：
- 跨耳消息仅编码发送侧第 0 参考麦的频带谱，码率与每耳麦数无关。
- CRF 全带估计器仅读取本侧第 0 参考麦与跨耳历史特征。
- NeuralSCM 通道数为 n_ch = n_mic + 1（本侧全部 M 麦 + 对侧第 0 麦）。
- MVDRBandEstimator 每带输出 2 * n_mic 维复数滤波器系数，对全部本侧麦共轭加权求和。
- 模型严格仅通过 9 个冻结 STFT 接口成员访问前端，输入输出跳过直流 bin 0（128 频点）。
"""

import copy

import torch
import torch.nn as nn

from .bands import (
    band_center_hz,
    build_band_matrices,
    compress_band,
    expand_complex,
    pad_dc_zero,
    strip_dc,
)
from .complexops import cabs, cmul_conj
from .features import build_crf_input, compress_current_spec, crf_input_dim
from .message import CrossEarMessage
from .modules import (
    CRFEstimator,
    MVDRBandEstimator,
    NeuralSCM,
    convert_legacy_cell_state_dict,
)
from .stft import CausalSTFT


def build_delayed_taps(x: torch.Tensor, n_taps: int, tap_stride: int) -> torch.Tensor:
    """将时序张量 [B, T, ...] 并行拼接为因果延迟抽头流 [B, T, n_taps, ...]。"""
    if n_taps == 1:
        return x.unsqueeze(2)
    t = x.shape[1]
    max_delay = (n_taps - 1) * tap_stride
    pad_shape = (x.shape[0], max_delay, *x.shape[2:])
    padded = torch.cat([x.new_zeros(pad_shape), x], dim=1)
    taps = [
        padded[:, max_delay - k * tap_stride : max_delay - k * tap_stride + t]
        for k in range(n_taps)
    ]
    return torch.stack(taps, dim=2)


class BinauralAIBF(nn.Module):
    def __init__(self, cfg: dict, device=None):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        s = self.cfg["stft"]
        self.stft = CausalSTFT(
            s["sample_rate"],
            s["hop"],
            s["win"],
            s["n_fft"],
            device,
            synthesis_window_size=s.get("synthesis_window_size"),
        )

        bcfg = self.cfg["bands"]
        self.n_bins = self.stft.n_bins - 1
        self.n_linear = int(bcfg["n_linear"])
        comp, expa = build_band_matrices(
            self.n_bins, self.n_linear, bcfg["n_tri"], device
        )
        self.register_buffer("comp", comp)
        self.register_buffer("expa", expa)
        self.n_bands = comp.shape[0]
        centers = band_center_hz(comp, self.stft.bin_hz, start_bin=1)
        self.register_buffer("band_centers", centers)

        self.n_mic = int(self.cfg["local"].get("n_mic", 1))
        if self.n_mic < 1:
            raise ValueError("local.n_mic 必须 >= 1")
        self.communication = bool(self.cfg["message"].get("enabled", True))
        self.n_ch = self.n_mic + 1
        self.n_taps = int(self.cfg["crf"]["n_taps"])
        self.tap_stride = int(self.cfg["crf"].get("tap_stride", 4))

        self.msg = CrossEarMessage(
            self.cfg, centers.detach().cpu(), self.stft.frame_rate, n_linear=self.n_linear
        )

        d_in = crf_input_dim(self.cfg, self.n_bands)
        self.crf = CRFEstimator(
            d_in,
            self.cfg["crf"]["hidden"],
            self.cfg["crf"]["groups"],
            self.n_bands,
            self.n_taps,
        )
        self.scm = NeuralSCM(self.n_ch, self.n_bands, self.cfg["scm"]["instance_norm"])

        mcfg = self.cfg["mvdr"]
        self.normalize_current_spec = bool(mcfg.get("normalize_current_spec", True))
        self.current_spec_power = float(mcfg.get("current_spec_power", 0.3))
        d_mvdr = self.scm.dim + (2 * self.n_mic if mcfg["use_current_spec"] else 0)
        self.mvdr = MVDRBandEstimator(
            d_mvdr,
            mcfg["hidden"],
            mcfg["groups"],
            mcfg["n_layers"],
            self.n_bands,
            n_mic=self.n_mic,
            use_film=bool(mcfg.get("use_film", False)),
            crf_hidden=int(self.cfg["crf"]["hidden"]),
            film_dim=int(mcfg.get("film_dim", 8)),
            age_scale_frames=float(self.cfg["crf"].get("age_scale_frames", 20)),
            use_cross_band=bool(mcfg.get("use_cross_band", False)),
            cross_band_kernel=int(mcfg.get("cross_band_kernel", 5)),
            cross_band_bottleneck=int(mcfg.get("cross_band_bottleneck", 16)),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """兼容加载旧版逐帧 GroupedGRUCell 保存的权重字典。"""
        state_dict = convert_legacy_cell_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def bitrate_report(self):
        if not self.communication:
            from .message import BitrateReport

            return BitrateReport(0, 0, 0.0, 0.0, 0.0, 0.0)
        return self.msg.bitrate()

    def build_streaming_model(self):
        from .streaming import StreamingSideModel

        return StreamingSideModel(self)

    def build_streaming_runner(self, step_fn=None):
        from .streaming import BinauralStreamingRunner

        return BinauralStreamingRunner(self, step_fn=step_fn)

    def _run_side(
        self,
        local_bin: torch.Tensor,
        local_band: torch.Tensor,
        remote_band_raw: torch.Tensor,
    ) -> torch.Tensor:
        """全帧并行执行一侧网络。

        - local_bin: [B, T, n_bins, 2] (n_mic=1) 或 [B, T, n_mic, n_bins, 2]
        - local_band: [B, T, n_bands, 2] (n_mic=1) 或 [B, T, n_mic, n_bands, 2]
        - remote_band_raw: [B, T, n_bands, 2]（对侧第 0 参考麦频带谱）
        返回增强复数谱 [B, T, n_bins, 2]。
        """
        local_bin_m = local_bin.unsqueeze(2) if local_bin.dim() == 4 else local_bin
        local_band_m = local_band.unsqueeze(2) if local_band.dim() == 4 else local_band
        b, t, m = local_band_m.shape[0], local_band_m.shape[1], local_band_m.shape[2]
        if m != self.n_mic:
            raise ValueError(f"输入麦数 {m} 与配置 n_mic={self.n_mic} 不一致")
        device = local_band_m.device

        if self.communication:
            received, matched_idx, valid, age, _ = self.msg.deliver(remote_band_raw)
        else:
            received = torch.zeros_like(remote_band_raw)
            matched_idx = torch.arange(t, device=device)
            valid = torch.zeros(b, t, 1, device=device, dtype=local_band_m.dtype)
            age = torch.zeros(t, device=device)

        if self.cfg["scm"]["local_frame"] == "delayed":
            local_matched_m = local_band_m[:, matched_idx]
        else:
            local_matched_m = local_band_m

        # CRF 特征仅使用本侧第 0 参考麦
        local_ref_now = local_band_m[:, :, 0]
        local_ref_matched = local_matched_m[:, :, 0]
        feat = build_crf_input(
            self.cfg,
            local_band_now=local_ref_now,
            local_band_matched=local_ref_matched,
            remote_band=received,
            valid=valid,
            age_frames=age,
            valid_band=self.msg.valid_band,
            valid_band_phase=self.msg.valid_band_phase,
        )

        # SCM 通道向量为 [本侧 0..M-1 麦配对帧, 对侧第 0 麦]，共 n_ch = n_mic + 1 路
        pair = torch.cat([local_matched_m, received.unsqueeze(2)], dim=2)
        chan_all = build_delayed_taps(pair, self.n_taps, self.tap_stride).transpose(2, 3)
        if self.cfg["message"]["payload"] == "logmag":
            chan_all = torch.stack(
                [cabs(chan_all), torch.zeros_like(chan_all[..., 0])], dim=-1
            )
        elif self.cfg["message"]["payload"] == "mixed":
            mag_only = self.msg.mag_only_band.to(device).view(1, 1, 1, 1, self.n_bands, 1)
            mag_only_chan = torch.stack(
                [cabs(chan_all), torch.zeros_like(chan_all[..., 0])], dim=-1
            )
            chan_all = torch.where(mag_only > 0.5, mag_only_chan, chan_all)

        if self.mvdr.film_bridge is not None:
            fir, _, crf_h = self.crf(feat, return_hidden=True)
        else:
            fir, _ = self.crf(feat)
            crf_h = None

        scm = self.scm(chan_all, fir)
        if self.cfg["mvdr"]["use_current_spec"]:
            cur = (
                compress_current_spec(local_band_m, self.current_spec_power, has_mic_dim=True)
                if self.normalize_current_spec
                else local_band_m
            )
            cur_flat = cur.transpose(2, 3).reshape(b, t, self.n_bands, 2 * self.n_mic)
            scm = torch.cat([scm, cur_flat], dim=-1)

        g_band_flat, _ = self.mvdr(scm, crf_h=crf_h, age_frames=age, valid=valid)
        g_band = g_band_flat.reshape(b, t, self.n_bands, self.n_mic, 2).transpose(2, 3)
        g_bin = expand_complex(g_band, self.expa)  # [B, T, n_mic, n_bins, 2]

        limit = float(self.cfg["mvdr"].get("gain_limit", 2.0))
        norm = torch.sqrt(g_bin.square().sum(dim=(2, 4), keepdim=True) + 1e-8)
        g_bin = g_bin * (limit / norm.clamp_min(1e-8)).clamp_max(1.0)
        return cmul_conj(local_bin_m, g_bin).sum(dim=2)

    def _normalize_wav(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 2:
            if self.n_mic != 1:
                raise ValueError(f"n_mic={self.n_mic} 时输入波形必须为 [B, n_mic, S]")
            return wav.unsqueeze(1)
        if wav.dim() == 3:
            if wav.shape[1] != self.n_mic:
                raise ValueError(f"输入通道数 {wav.shape[1]} 与配置 n_mic={self.n_mic} 不一致")
            return wav
        raise ValueError("波形维度必须为 [B, S] 或 [B, n_mic, S]")

    def forward(
        self, wav_left: torch.Tensor, wav_right: torch.Tensor, shuffle_remote: bool = False
    ):
        """wav_* 形状 [B, S]（n_mic=1）或 [B, n_mic, S]。返回 (yl [B, S], yr [B, S], info)。"""
        wl = self._normalize_wav(wav_left)
        wr = self._normalize_wav(wav_right)
        if wl.shape != wr.shape:
            raise ValueError("左右波形必须同形状")

        b, m, n = wl.shape
        wav_both = torch.cat([wl, wr], dim=0)  # [2B, n_mic, S]
        spec_flat = strip_dc(self.stft.analyze(wav_both.reshape(2 * b * m, n)))
        t = spec_flat.shape[1]
        x_both = spec_flat.reshape(2 * b, m, t, self.n_bins, 2).transpose(1, 2)
        b_both = compress_band(x_both, self.comp, self.n_linear)

        # 跨耳消息仅传发送侧第 0 参考麦，码率与 n_mic 无关
        ref_b_both = b_both[:, :, 0]
        bl, br = ref_b_both.chunk(2, dim=0)
        remote_l = torch.roll(bl, bl.shape[1] // 2, dims=1) if shuffle_remote else bl
        remote_r = torch.roll(br, br.shape[1] // 2, dims=1) if shuffle_remote else br
        remote_both = torch.cat([remote_r, remote_l], dim=0)

        y_spec_both = self._run_side(x_both, b_both, remote_both)
        y_both = self.stft.synthesize(pad_dc_zero(y_spec_both), length=n)
        yl, yr = y_both.chunk(2, dim=0)

        info = {
            "bitrate": self.bitrate_report(),
            "n_bands": self.n_bands,
            "n_bins": self.n_bins,
            "n_mic": self.n_mic,
            "frame_rate": self.stft.frame_rate,
            "frontend_delay_ms": 1000.0 * self.stft.win / self.stft.sample_rate,
            "link_delay_ms": 1000.0
            * self.cfg["message"]["delay_frames"]
            * self.stft.hop
            / self.stft.sample_rate,
        }
        return yl, yr, info
