"""双耳 AIBF 主模型。

数据流，每侧各跑一遍，权重共享、状态独立。

1. STFT 得到本侧复数谱。
2. 频带压缩得到频带谱。
3. 对侧频带谱经跨耳消息通道，按配置量化、延迟、交付。
4. CRF 全带估计器读「当前帧本侧对数幅度谱」加「同时间戳的跨耳特征」，
   输出语音与噪声两类因果复数 FIR。
5. FIR 作用到并行拼接的延迟通道流 [本侧配对帧, 对侧] 上，取外积得到神经 SCM。
6. 逐带估计器读 SCM 加当前帧本侧复数谱，输出本侧复数滤波器。
7. 滤波器解压到频点，共轭乘当前帧本侧原谱，iSTFT 输出。

延迟只作用在第 3 步与第 5 步。第 6、7 步始终使用当前帧，所以链路延迟不进声到声。

训练与离线前向全程按时间维并行（延迟抽头用零填充移位后 stack，循环层用融合
GroupedGRU，左右耳沿批次维合并），消除逐帧 Python 循环与逐帧反向传播。
导出与单帧部署通过 build_streaming_model() 按单帧流式结构重新构造网络，
与训练模型严格数值等价。

mask_channel_idx 固定为本侧。对侧只进条件化支路，不进合成支路。这一点和麦阵
AIBF 现有实现一致，也是 S1 与 S2 的分界。要改成 S2，需要让滤波头输出两路系数
并相加，那时对侧必须是当前帧。
"""

import torch
import torch.nn as nn

from bands import band_center_hz, build_band_matrices, compress_complex, expand_complex
from complexops import cabs, cmul_conj
from features import build_crf_input, crf_input_dim
from message import CrossEarMessage
from modules import (
    CRFEstimator,
    MVDRBandEstimator,
    NeuralSCM,
    convert_legacy_cell_state_dict,
)
from stft import CausalSTFT


def build_delayed_taps(x: torch.Tensor, n_taps: int, tap_stride: int) -> torch.Tensor:
    """将时序张量 [B, T, ...] 并行拼接为因果延迟抽头流 [B, T, n_taps, ...]。

    抽头 k 对应延迟 k * tap_stride 帧（k=0 为当前帧，k>0 为历史延迟流）。
    早于序列起点的历史位置严格补零，等价于因果移位（roll + 前缀清零），
    且在 T < (n_taps - 1) * tap_stride 的极短片段上也不会回绕泄漏未来帧。
    """
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """兼容加载旧版逐帧 GroupedGRUCell 保存的权重字典。"""
        state_dict = convert_legacy_cell_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def bitrate_report(self):
        if not self.communication:
            from message import BitrateReport

            return BitrateReport(0, 0, 0.0, 0.0, 0.0, 0.0)
        return self.msg.bitrate()

    def build_streaming_model(self):
        """按单帧流式结构构造等价网络，可直接用于逐帧推理与 TorchScript / ONNX 导出。"""
        from streaming import StreamingSideModel

        return StreamingSideModel(self)

    def build_streaming_runner(self, step_fn=None):
        """构造带跨耳延迟队列与本侧历史缓存的双耳逐帧流式运行器。"""
        from streaming import BinauralStreamingRunner

        return BinauralStreamingRunner(self, step_fn=step_fn)

    def _run_side(
        self,
        local_bin: torch.Tensor,
        local_band: torch.Tensor,
        remote_band_raw: torch.Tensor,
    ) -> torch.Tensor:
        """全帧并行执行一侧网络。输入均为 [B, T, ...]，返回增强复数谱 [B, T, n_bins, 2]。"""
        b, t = local_band.shape[0], local_band.shape[1]
        device = local_band.device

        if self.communication:
            received, matched_idx, valid, age, _ = self.msg.deliver(remote_band_raw)
        else:
            received = torch.zeros_like(local_band)
            matched_idx = torch.arange(t, device=device)
            valid = torch.zeros(b, t, 1, device=device, dtype=local_band.dtype)
            age = torch.zeros(t, device=device)

        # 配对帧的本侧谱。只给特征与 SCM 支路使用，不进入当前帧滤波合成支路。
        if self.cfg["scm"]["local_frame"] == "delayed":
            local_matched = local_band[:, matched_idx]
        else:
            local_matched = local_band

        feat = build_crf_input(
            self.cfg,
            local_band_now=local_band,
            local_band_matched=local_matched,
            remote_band=received,
            valid=valid,
            age_frames=age,
            valid_band=self.msg.valid_band,
        )

        # 并行拼接当前通道对与多路固定墙钟延迟流：[B, T, n_ch, n_taps, n_bands, 2]
        pair = torch.stack([local_matched, received], dim=2)  # [B, T, 2, n_bands, 2]
        chan_all = build_delayed_taps(pair, self.n_taps, self.tap_stride).transpose(2, 3)
        if self.cfg["message"]["payload"] == "logmag":
            # 幅度分支不把虚构的零相位与本侧真实相位相减。
            chan_all = torch.stack(
                [cabs(chan_all), torch.zeros_like(chan_all[..., 0])], dim=-1
            )

        # 全序列并行前向与反向传播，无逐帧 Python 循环。
        fir, _ = self.crf(feat)  # [B, T, n_bands, n_taps, 2, 2]
        scm = self.scm(chan_all, fir)  # [B, T, n_bands, dim]
        if self.cfg["mvdr"]["use_current_spec"]:
            scm = torch.cat([scm, local_band], dim=-1)
        g_band, _ = self.mvdr(scm)  # [B, T, n_bands, 2]

        g_bin = expand_complex(g_band, self.expa)  # [B, T, n_bins, 2]
        limit = float(self.cfg["mvdr"].get("gain_limit", 2.0))
        g_bin = g_bin * (limit / cabs(g_bin).clamp_min(1e-8)).clamp_max(1).unsqueeze(-1)
        return cmul_conj(local_bin, g_bin)

    def forward(
        self, wav_left: torch.Tensor, wav_right: torch.Tensor, shuffle_remote=False
    ):
        """wav_* 形状 [B, S]。返回左右输出波形与诊断字典。"""
        if wav_left.shape != wav_right.shape:
            raise ValueError("左右波形必须等长")

        # 左右耳共享权重、状态独立，沿批次维合并做一次并行分析、前向与合成。
        wav_both = torch.cat([wav_left, wav_right], dim=0)
        x_both = self.stft.analyze(wav_both)
        b_both = compress_complex(x_both, self.comp)
        bl, br = b_both.chunk(2, dim=0)

        # 仅作离线消息依赖诊断，不代表因果部署条件。
        remote_l = torch.roll(bl, bl.shape[1] // 2, dims=1) if shuffle_remote else bl
        remote_r = torch.roll(br, br.shape[1] // 2, dims=1) if shuffle_remote else br
        remote_both = torch.cat([remote_r, remote_l], dim=0)

        y_spec_both = self._run_side(x_both, b_both, remote_both)
        n = wav_left.shape[1]
        y_both = self.stft.synthesize(y_spec_both, length=n)
        yl, yr = y_both.chunk(2, dim=0)

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
