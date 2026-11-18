"""双耳 AIBF 主模型。

数据流，每侧各跑一遍，权重共享、状态独立。

1. STFT 得到本侧复数谱。
2. 频带压缩得到频带谱。
3. 对侧频带谱经跨耳消息通道，按配置量化、延迟、交付。
4. CRF 全带估计器读「当前帧本侧对数幅度谱」加「同时间戳的跨耳特征」，
   输出语音与噪声两类因果复数 FIR。
5. FIR 作用到通道向量 [本侧配对帧, 对侧] 上，取外积得到神经 SCM。
6. 逐带估计器读 SCM 加当前帧本侧复数谱，输出本侧复数滤波器。
7. 滤波器解压到频点，共轭乘当前帧本侧原谱，iSTFT 输出。

延迟只作用在第 3 步与第 5 步。第 6、7 步始终使用当前帧，所以链路延迟不进声到声。

mask_channel_idx 固定为本侧。对侧只进条件化支路，不进合成支路。这一点和麦阵
AIBF 现有实现一致，也是 S1 与 S2 的分界。要改成 S2，需要让滤波头输出两路系数
并相加，那时对侧必须是当前帧。
"""

import torch
import torch.nn as nn

from bands import band_center_hz, build_band_matrices, compress_complex, expand_complex
from complexops import cmul_conj, cabs
from features import build_crf_input, crf_input_dim
from message import CrossEarMessage
from modules import CRFEstimator, MVDRBandEstimator, NeuralSCM
from stft import CausalSTFT


class BinauralAIBF(nn.Module):
    def __init__(self, cfg: dict, device=None):
        super().__init__()
        self.cfg = cfg
        s = cfg["stft"]
        self.stft = CausalSTFT(s["sample_rate"], s["hop"], s["win"], s["n_fft"], device)

        bcfg = cfg["bands"]
        comp, expa = build_band_matrices(
            self.stft.n_bins, bcfg["n_linear"], bcfg["n_tri"], device
        )
        self.register_buffer("comp", comp)
        self.register_buffer("expa", expa)
        self.n_bands = comp.shape[0]
        centers = band_center_hz(comp, self.stft.bin_hz)
        self.register_buffer("band_centers", centers)

        self.n_mic = int(cfg["local"]["n_mic"])
        if self.n_mic != 1:
            raise ValueError("探针版本仅支持每耳 1 麦")
        self.communication = cfg["message"].get("enabled", True)
        self.n_ch = self.n_mic + 1  # 本侧麦数加一路对侧
        self.n_taps = int(cfg["crf"]["n_taps"])
        self.tap_stride = int(cfg["crf"].get("tap_stride", 4))

        self.msg = CrossEarMessage(cfg, centers.detach().cpu(), self.stft.frame_rate)

        d_in = crf_input_dim(cfg, self.n_bands)
        self.crf = CRFEstimator(
            d_in, cfg["crf"]["hidden"], cfg["crf"]["groups"], self.n_bands, self.n_taps
        )
        self.scm = NeuralSCM(self.n_ch, self.n_bands, cfg["scm"]["instance_norm"])

        d_mvdr = self.scm.dim + (2 if cfg["mvdr"]["use_current_spec"] else 0)
        self.mvdr = MVDRBandEstimator(
            d_mvdr,
            cfg["mvdr"]["hidden"],
            cfg["mvdr"]["groups"],
            cfg["mvdr"]["n_layers"],
            self.n_bands,
        )

    def bitrate_report(self):
        if not self.communication:
            from message import BitrateReport
            return BitrateReport(0, 0, 0., 0., 0., 0.)
        return self.msg.bitrate()

    def _tap_indices(self, matched_idx: torch.Tensor, valid_t: torch.Tensor):
        """返回每个抽头的索引与有效位。抽头步长等于消息更新周期。"""
        idx, val = [], []
        for k in range(self.n_taps):
            raw = matched_idx - k * self.tap_stride
            idx.append(raw.clamp_min(0))
            val.append(((raw >= 0).float() * valid_t).view(1, -1, 1, 1))
        return idx, val

    def _run_side(
        self,
        local_bin: torch.Tensor,
        local_band: torch.Tensor,
        remote_band_raw: torch.Tensor,
    ):
        """跑一侧。返回该侧输出复数谱与码率无关的诊断量。"""
        b, t = local_band.shape[0], local_band.shape[1]
        device = local_band.device

        if self.communication:
            received, matched_idx, valid, age, decoded_all = self.msg.deliver(remote_band_raw)
        else:
            received = decoded_all = torch.zeros_like(local_band)
            matched_idx = torch.arange(t, device=device)
            valid = torch.zeros(b, t, 1, device=device)
            age = torch.zeros(t, device=device)

        valid_t = valid[0, :, 0]

        # 配对帧的本侧谱。只给特征与 SCM 支路使用
        if self.cfg["scm"]["local_frame"] == "delayed":
            local_matched_src = matched_idx
        else:
            local_matched_src = torch.arange(t, device=device)
        local_matched = local_band[:, local_matched_src]

        feat = build_crf_input(
            self.cfg,
            local_band_now=local_band,
            local_band_matched=local_matched,
            remote_band=received,
            valid=valid,
            age_frames=age,
            valid_band=self.msg.valid_band,
        )

        # 固定墙钟间隔的历史接收缓存。消息更新周期不改变滤波器时间跨度。
        clock = torch.arange(t, device=device)
        tap_idx, tap_val = self._tap_indices(clock, valid_t)
        history_emit = [matched_idx[index] for index in tap_idx]

        local_tap_idx = [
            (history_emit[k] if self.cfg["scm"]["local_frame"] == "delayed" else tap_idx[k])
            for k in range(self.n_taps)
        ]
        local_taps = torch.stack(
            [local_band[:, local_tap_idx[k]] * (clock >= k * self.tap_stride).view(1, -1, 1, 1)
             for k in range(self.n_taps)], dim=1
        )  # [B, n_taps, T, n_bands, 2]
        remote_taps = torch.stack(
            [received[:, tap_idx[k]] * tap_val[k] for k in range(self.n_taps)],
            dim=1,
        )
        # chan_all [B, n_ch, n_taps, T, n_bands, 2]，通道轴在抽头轴之前
        if self.cfg["message"]["payload"] == "logmag":
            # 幅度分支不把虚构的零相位与本侧真实相位相减。
            local_taps = torch.stack([cabs(local_taps), torch.zeros_like(local_taps[..., 0])], -1)
            remote_taps = torch.stack([cabs(remote_taps), torch.zeros_like(remote_taps[..., 0])], -1)
        chan_all = torch.stack([local_taps, remote_taps], dim=1)

        crf_state = self.crf.init_state(b, device)
        mvdr_state = self.mvdr.init_state(b, device)
        outs = []
        for i in range(t):
            fir, crf_state = self.crf(feat[:, i], crf_state)
            scm = self.scm(chan_all[:, :, :, i], fir)
            if self.cfg["mvdr"]["use_current_spec"]:
                scm = torch.cat([scm, local_band[:, i]], dim=-1)
            g_band, mvdr_state = self.mvdr(scm, mvdr_state)
            g_bin = expand_complex(g_band, self.expa)
            limit = float(self.cfg["mvdr"].get("gain_limit", 2.0))
            g_bin = g_bin * (limit / cabs(g_bin).clamp_min(1e-8)).clamp_max(1).unsqueeze(-1)
            outs.append(cmul_conj(local_bin[:, i], g_bin))
        return torch.stack(outs, dim=1)

    def forward(self, wav_left: torch.Tensor, wav_right: torch.Tensor, shuffle_remote=False):
        """wav_* 形状 [B, S]。返回左右输出波形与诊断字典。"""
        xl = self.stft.analyze(wav_left)
        xr = self.stft.analyze(wav_right)
        bl = compress_complex(xl, self.comp)
        br = compress_complex(xr, self.comp)

        if wav_left.shape != wav_right.shape:
            raise ValueError("左右波形必须等长")
        # 仅作离线消息依赖诊断，不代表因果部署条件。
        remote_l = torch.roll(bl, bl.shape[1] // 2, dims=1) if shuffle_remote else bl
        remote_r = torch.roll(br, br.shape[1] // 2, dims=1) if shuffle_remote else br
        yl_spec = self._run_side(xl, bl, remote_r)
        yr_spec = self._run_side(xr, br, remote_l)

        n = min(wav_left.shape[1], wav_right.shape[1])
        yl = self.stft.synthesize(yl_spec, length=n)
        yr = self.stft.synthesize(yr_spec, length=n)

        info = {
            "bitrate": self.bitrate_report(),
            "n_bands": self.n_bands,
            "n_bins": self.stft.n_bins,
            "frame_rate": self.stft.frame_rate,
            "frontend_delay_ms": 1000.0 * self.stft.win / self.stft.sample_rate,
            "link_delay_ms": 1000.0
            * self.cfg["message"]["delay_frames"]
            * self.stft.hop
            / self.stft.sample_rate,
        }
        return yl, yr, info
