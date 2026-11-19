"""因果 STFT 与 WOLA 合成。

对外保持 9 个冻结接口成员：
sample_rate、hop、win、n_fft、n_bins、frame_rate、bin_hz、analyze(x)、synthesize(spec, length)。

手写分帧，不用 torch.stft。原因是 torch.stft 在 win 小于 n_fft 时把窗居中补零，
相位参考和流式实现不一致。这里的约定是取 win 个样本、加窗、末端零填充到 n_fft。
"""

import torch

from .complexops import cmul_conj


def sqrt_hann(win: int, device=None) -> torch.Tensor:
    n = torch.arange(win, device=device, dtype=torch.float32)
    hann = 0.5 - 0.5 * torch.cos(2 * torch.pi * n / win)
    return torch.sqrt(hann + 1e-12)


class CausalSTFT:
    """分析与合成使用同一个平方根汉宁窗，合成端显式做窗平方归一化。"""

    def __init__(
        self,
        sample_rate: int,
        hop: int,
        win: int,
        n_fft: int,
        device=None,
        synthesis_window_size: int | None = None,
    ):
        assert win <= n_fft, "窗长不能超过 FFT 长度"
        if synthesis_window_size is not None and int(synthesis_window_size) != int(win):
            raise NotImplementedError(
                "CausalSTFT 仅支持等长分析/合成窗；如需不等长合成窗，请使用主仓 STFT 实现"
            )
        self.sample_rate = sample_rate
        self.hop = hop
        self.win = win
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.synthesis_window_size = synthesis_window_size
        self.window = sqrt_hann(win, device=device)

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    @property
    def bin_hz(self) -> float:
        return self.sample_rate / self.n_fft

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        """x 形状 [B, S]，返回 [B, T, n_bins, 2]。"""
        if x.ndim != 2 or x.shape[1] < 1:
            raise ValueError("波形必须为非空 [批次, 样本] 张量")
        # 左补 win-hop，右补到完整帧。时间戳从原始样本零点计。
        left = self.win - self.hop
        right = self.win - 1
        padded = torch.nn.functional.pad(x, (left, right))
        frames = padded.unfold(dimension=1, size=self.win, step=self.hop)
        frames = frames * self.window.to(x.device)
        pad = self.n_fft - self.win
        if pad > 0:
            frames = torch.nn.functional.pad(frames, (0, pad))
        spec = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        return torch.stack([spec.real, spec.imag], dim=-1)

    def synthesize(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """spec 形状 [B, T, n_bins, 2]，返回 [B, S]。"""
        t = spec.shape[1]
        cplx = torch.complex(spec[..., 0], spec[..., 1])
        frames = torch.fft.irfft(cplx, n=self.n_fft, dim=-1)[..., : self.win]
        w = self.window.to(spec.device)
        frames = frames * w

        total = (t - 1) * self.hop + self.win
        # 用 fold 向量化 WOLA 叠加，避免在训练反向图中展开 T 个时间步节点。
        out = (
            torch.nn.functional.fold(
                frames.transpose(1, 2).contiguous(),
                output_size=(1, total),
                kernel_size=(1, self.win),
                stride=(1, self.hop),
            )
            .squeeze(1)
            .squeeze(1)
        )
        wsq_cols = (w ** 2).view(1, self.win, 1).expand(1, self.win, t).contiguous()
        norm = (
            torch.nn.functional.fold(
                wsq_cols,
                output_size=(1, total),
                kernel_size=(1, self.win),
                stride=(1, self.hop),
            )
            .squeeze(1)
            .squeeze(1)
        )
        out = out / norm.clamp_min(1e-8)
        out = out[:, self.win - self.hop :]
        if length is not None:
            out = out[:, :length]
        return out


def apply_filter(spec: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """把复数滤波器的共轭乘到谱上，与 AIBF 的 conj(G) 乘参考谱同向。"""
    return cmul_conj(spec, gain)
