"""网络部件。

命名与麦阵 AIBF 对齐：
- GroupedLinear 对应 AIBF 文档里的 GL。
- CRFEstimator 对应全带估计器，输出语音与噪声两类因果复数 FIR（可选输出隐特征）。
- NeuralSCM 对应神经 SCM，即过滤后谱的外积。
- MVDRBandEstimator 对应逐带估计器（内置可选 SpatialFiLMBridge 与 CrossBandBlock），
  输出本侧 n_mic 个麦克风的复数滤波器系数（每带 2 * n_mic 维）。
"""

import torch
import torch.nn as nn

from .complexops import cmul, cmul_conj


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """沿特征维做组间通道混洗，使用动态批次维 -1 以兼容动态批次流式导出。"""
    if groups <= 1:
        return x
    d = x.shape[-1]
    if x.dim() == 2:
        return x.reshape(-1, groups, d // groups).transpose(1, 2).reshape(-1, d)
    if x.dim() == 3:
        t = x.shape[1]
        return x.reshape(-1, t, groups, d // groups).transpose(2, 3).reshape(-1, t, d)
    return x.reshape(-1, groups, d // groups).transpose(1, 2).reshape(x.shape)


class GroupedLinear(nn.Module):
    """分组线性层，用 1x1 卷积实现。输入输出维度都要能被 groups 整除。"""

    def __init__(self, d_in: int, d_out: int, groups: int = 1):
        super().__init__()
        assert d_in % groups == 0 and d_out % groups == 0, (
            f"分组线性层维度不匹配: {d_in}/{d_out}/g={groups}"
        )
        self.d_in = d_in
        self.d_out = d_out
        self.groups = groups
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., D]，最后一维是特征维。"""
        if x.dim() == 2:
            return self.conv(x.unsqueeze(-1)).squeeze(-1)
        if x.dim() == 3:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)
        b = x.shape[0]
        orig = x.shape[:-1]
        d = x.shape[-1]
        y = self.conv(x.reshape(b, -1, d).transpose(1, 2)).transpose(1, 2)
        return y.reshape(*orig, self.d_out)


class GroupedGRU(nn.Module):
    """全序列并行分组 GRU。各组状态独立，组间通过 channel_shuffle 交互。"""

    def __init__(self, d_in: int, hidden: int, groups: int = 1):
        super().__init__()
        assert d_in % groups == 0 and hidden % groups == 0
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.grus = nn.ModuleList(
            [
                nn.GRU(d_in // groups, hidden // groups, num_layers=1, batch_first=True)
                for _ in range(groups)
            ]
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xs = x.chunk(self.groups, dim=-1)
        if h is None:
            hs = [None] * self.groups
        else:
            hs = [hi.unsqueeze(0).contiguous() for hi in h.chunk(self.groups, dim=-1)]
        ys, next_hs = [], []
        for g, xi, hi in zip(self.grus, xs, hs):
            yi, h_last = g(xi.contiguous(), hi)
            ys.append(yi)
            next_hs.append(h_last.squeeze(0))
        return torch.cat(ys, dim=-1), torch.cat(next_hs, dim=-1)

    def init_state(
        self, n: int, device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device, dtype=dtype)


class GroupedGRUCell(nn.Module):
    """单帧流式分组 GRU 单元，与 GroupedGRU 权重严格一一对应。"""

    def __init__(self, d_in: int, hidden: int, groups: int = 1):
        super().__init__()
        assert d_in % groups == 0 and hidden % groups == 0
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.cells = nn.ModuleList(
            [nn.GRUCell(d_in // groups, hidden // groups) for _ in range(groups)]
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        xs = x.chunk(self.groups, dim=-1)
        hs = h.chunk(self.groups, dim=-1)
        return torch.cat([c(xi, hi) for c, xi, hi in zip(self.cells, xs, hs)], dim=-1)

    def init_state(
        self, n: int, device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device, dtype=dtype)

    @torch.no_grad()
    def copy_from_grouped_gru(self, src: GroupedGRU) -> "GroupedGRUCell":
        assert (
            self.d_in == src.d_in
            and self.hidden == src.hidden
            and self.groups == src.groups
        )
        for cell, gru in zip(self.cells, src.grus):
            cell.weight_ih.copy_(gru.weight_ih_l0)
            cell.weight_hh.copy_(gru.weight_hh_l0)
            cell.bias_ih.copy_(gru.bias_ih_l0)
            cell.bias_hh.copy_(gru.bias_hh_l0)
        return self


def convert_legacy_cell_state_dict(state_dict: dict) -> dict:
    """将旧版 GroupedGRUCell 权重键名映射到新版 GroupedGRU 键名。"""
    if not any(".cells." in k for k in state_dict):
        return state_dict
    converted = {}
    for key, val in state_dict.items():
        if ".cells." in key:
            new_key = key.replace(".cells.", ".grus.")
            for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                if new_key.endswith("." + suffix):
                    new_key = new_key + "_l0"
                    break
            converted[new_key] = val
        else:
            converted[key] = val
    return converted


class CRFEstimatorStep(nn.Module):
    """单帧流式 CRF 全带估计器，用于流式推理与单帧模型导出。"""

    def __init__(self, d_in: int, hidden: int, groups: int, n_bands: int, n_taps: int):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.n_bands = n_bands
        self.n_taps = n_taps
        d_out = n_bands * n_taps * 2 * 2
        self.pad = (-d_in) % groups
        self.inp = GroupedLinear(d_in + self.pad, hidden, groups)
        self.gru1 = GroupedGRUCell(hidden, hidden, groups)
        self.gru2 = GroupedGRUCell(hidden, hidden, groups)
        self.out = GroupedLinear(hidden, d_out, groups)

    def forward_with_hidden(
        self, feat: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.pad:
            feat = torch.nn.functional.pad(feat, (0, self.pad))
        x = channel_shuffle(torch.relu(self.inp(feat)), self.groups)
        next_h1 = self.gru1(x, h1)
        x1 = channel_shuffle(next_h1, self.groups)
        next_h2 = self.gru2(x1, h2)
        x2 = channel_shuffle(next_h2, self.groups)
        y = self.out(x2)
        fir = y.view(-1, self.n_bands, self.n_taps, 2, 2)
        return fir, next_h1, next_h2, x2

    def forward(
        self, feat: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fir, next_h1, next_h2, _ = self.forward_with_hidden(feat, h1, h2)
        return fir, next_h1, next_h2


class CRFEstimator(nn.Module):
    """全带估计器（训练/整段并行版）。每帧读全带特征，输出逐带语音与噪声复数 FIR。"""

    def __init__(self, d_in: int, hidden: int, groups: int, n_bands: int, n_taps: int):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.n_bands = n_bands
        self.n_taps = n_taps
        d_out = n_bands * n_taps * 2 * 2
        pad = (-d_in) % groups
        self.pad = pad
        self.inp = GroupedLinear(d_in + pad, hidden, groups)
        self.gru1 = GroupedGRU(hidden, hidden, groups)
        self.gru2 = GroupedGRU(hidden, hidden, groups)
        self.out = GroupedLinear(hidden, d_out, groups)
        nn.init.zeros_(self.out.conv.bias)
        nn.init.normal_(self.out.conv.weight, std=1e-3)

    def init_state(self, b: int, device, dtype: torch.dtype = torch.float32):
        return [
            self.gru1.init_state(b, device, dtype=dtype),
            self.gru2.init_state(b, device, dtype=dtype),
        ]

    def forward(self, feat: torch.Tensor, state=None, return_hidden: bool = False):
        single_frame = feat.dim() == 2
        if single_frame:
            feat = feat.unsqueeze(1)
        b, t, _ = feat.shape
        if self.pad:
            feat = torch.nn.functional.pad(feat, (0, self.pad))
        x = channel_shuffle(torch.relu(self.inp(feat)), self.groups)
        h1_in = None if state is None else state[0]
        h2_in = None if state is None else state[1]
        x, h1_last = self.gru1(x, h1_in)
        x = channel_shuffle(x, self.groups)
        x, h2_last = self.gru2(x, h2_in)
        x = channel_shuffle(x, self.groups)
        y = self.out(x)
        fir = y.view(b, t, self.n_bands, self.n_taps, 2, 2)
        if single_frame:
            fir = fir.squeeze(1)
            x_out = x.squeeze(1)
        else:
            x_out = x
        if return_hidden:
            return fir, [h1_last, h2_last], x_out
        return fir, [h1_last, h2_last]

    @torch.no_grad()
    def to_streaming(self) -> CRFEstimatorStep:
        step = CRFEstimatorStep(
            self.d_in, self.hidden, self.groups, self.n_bands, self.n_taps
        )
        step.inp.load_state_dict(self.inp.state_dict())
        step.gru1.copy_from_grouped_gru(self.gru1)
        step.gru2.copy_from_grouped_gru(self.gru2)
        step.out.load_state_dict(self.out.state_dict())
        step.to(device=self.out.conv.weight.device, dtype=self.out.conv.weight.dtype)
        step.eval()
        return step


class NeuralSCM(nn.Module):
    """神经 SCM。把 FIR 作用到通道向量 [本侧 M 麦, 对侧第 0 麦] 上，再取外积。"""

    def __init__(self, n_ch: int, n_bands: int, instance_norm: bool = True):
        super().__init__()
        self.n_ch = n_ch
        self.n_bands = n_bands
        self.instance_norm = instance_norm
        self.dim = 2 * n_ch * n_ch * 2
        self.norm = (
            nn.InstanceNorm1d(self.dim, affine=True) if instance_norm else nn.Identity()
        )

    def _compute(self, chan_hist: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
        chan = chan_hist.unsqueeze(2)
        f = fir.permute(0, 3, 2, 1, 4).unsqueeze(1)
        filtered = cmul(chan, f).sum(dim=3)

        a = filtered.unsqueeze(1)
        bb = filtered.unsqueeze(2)
        scm = cmul_conj(a, bb)
        scm = scm.permute(0, 4, 1, 2, 3, 5).reshape(-1, self.n_bands, self.dim)
        scm = self.norm(scm.transpose(1, 2)).transpose(1, 2)
        return scm

    def forward(self, chan_hist: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
        if chan_hist.dim() == 6:
            b, t, n_ch, n_taps, n_bands, _ = chan_hist.shape
            out = self._compute(
                chan_hist.reshape(b * t, n_ch, n_taps, n_bands, 2),
                fir.reshape(b * t, n_bands, n_taps, 2, 2),
            )
            return out.reshape(b, t, n_bands, self.dim)
        return self._compute(chan_hist, fir)

    @torch.no_grad()
    def to_streaming(self) -> "NeuralSCM":
        step = NeuralSCM(self.n_ch, self.n_bands, self.instance_norm)
        step.load_state_dict(self.state_dict())
        ref = next(self.parameters(), None)
        if ref is not None:
            step.to(device=ref.device, dtype=ref.dtype)
        step.eval()
        return step


class SpatialFiLMBridge(nn.Module):
    """CRF 空间估计器到逐带 MVDR 的全 64 带年龄门控 FiLM 条件桥（无循环状态）。"""

    def __init__(
        self,
        crf_hidden: int,
        mvdr_hidden: int,
        n_bands: int,
        film_dim: int = 8,
        age_scale_frames: float = 20.0,
    ):
        super().__init__()
        self.crf_hidden = crf_hidden
        self.mvdr_hidden = mvdr_hidden
        self.n_bands = n_bands
        self.film_dim = film_dim
        self.age_scale_frames = float(age_scale_frames)
        self.crf_to_band = nn.Linear(crf_hidden, n_bands * film_dim)
        self.band_act = nn.SiLU()
        self.band_to_film = nn.Linear(film_dim, 4 * mvdr_hidden)
        nn.init.normal_(self.band_to_film.weight, std=1e-3)
        nn.init.zeros_(self.band_to_film.bias)
        self.age_gate = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(
        self,
        crf_hidden_feat: torch.Tensor,
        age_frames: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dtype = crf_hidden_feat.dtype
        v = valid.to(dtype=dtype)
        if crf_hidden_feat.dim() == 3:
            b, t, _ = crf_hidden_feat.shape
            z = self.crf_to_band(crf_hidden_feat).view(b, t, self.n_bands, self.film_dim)
            age_n = age_frames.to(dtype=dtype) / self.age_scale_frames
            if age_n.dim() == 1:
                age_n = age_n.view(1, t, 1).expand(b, t, 1)
            gate_in = torch.cat([age_n, v], dim=-1)
            gate = (v * torch.sigmoid(self.age_gate(gate_in))).unsqueeze(-1)
        else:
            z = self.crf_to_band(crf_hidden_feat).reshape(-1, self.n_bands, self.film_dim)
            age_n = age_frames.to(dtype=dtype) / self.age_scale_frames
            gate_in = torch.cat([age_n, v], dim=-1)
            gate = (v * torch.sigmoid(self.age_gate(gate_in))).unsqueeze(-1)
        raw = self.band_to_film(self.band_act(z))
        g1, b1, g2, b2 = raw.chunk(4, dim=-1)
        return gate * torch.tanh(g1), gate * b1, gate * torch.tanh(g2), gate * b2

    @torch.no_grad()
    def to_streaming(self) -> "SpatialFiLMBridge":
        step = SpatialFiLMBridge(
            self.crf_hidden,
            self.mvdr_hidden,
            self.n_bands,
            self.film_dim,
            self.age_scale_frames,
        )
        step.load_state_dict(self.state_dict())
        ref = next(self.parameters(), None)
        if ref is not None:
            step.to(device=ref.device, dtype=ref.dtype)
        step.eval()
        return step


class CrossBandBlock(nn.Module):
    """快支路逐帧轻量跨频带交互模块（无循环状态，100% 因果且动态批次安全）。"""

    def __init__(
        self,
        hidden: int,
        n_bands: int,
        kernel_size: int = 5,
        bottleneck: int = 16,
        ch_mid: int = 8,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_bands = n_bands
        self.kernel_size = kernel_size
        self.bottleneck = bottleneck
        self.ch_mid = ch_mid
        self.norm = nn.LayerNorm(hidden)
        self.dw_conv = nn.Conv1d(
            hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden
        )
        self.dw_act = nn.PReLU(hidden)
        self.pw_conv = nn.Conv1d(hidden, hidden, kernel_size=1)
        nn.init.normal_(self.pw_conv.weight, std=1e-2)
        nn.init.zeros_(self.pw_conv.bias)

        self.ch_down = nn.Linear(hidden, ch_mid)
        self.f_down = nn.Linear(n_bands, bottleneck)
        self.f_act = nn.SiLU()
        self.f_up = nn.Linear(bottleneck, n_bands)
        self.ch_up = nn.Linear(ch_mid, hidden)
        nn.init.normal_(self.ch_up.weight, std=1e-2)
        nn.init.zeros_(self.ch_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x 形状 [N, n_bands, hidden]，返回同形状张量。"""
        h = self.norm(x)
        loc = self.pw_conv(self.dw_act(self.dw_conv(h.transpose(1, 2)))).transpose(1, 2)
        g = self.ch_down(h).transpose(1, 2)
        g = self.f_up(self.f_act(self.f_down(g))).transpose(1, 2)
        glb = self.ch_up(g)
        return x + loc + glb


class MVDRBandEstimatorStep(nn.Module):
    """单帧流式逐带估计器（含可选内置 FiLM 条件桥与跨频带模块），输出每带 2 * n_mic 实数。"""

    def __init__(
        self,
        d_in: int,
        hidden: int,
        groups: int,
        n_layers: int,
        n_bands: int,
        n_mic: int = 1,
        use_film: bool = False,
        crf_hidden: int = 128,
        film_dim: int = 8,
        age_scale_frames: float = 20.0,
        use_cross_band: bool = False,
        cross_band_kernel: int = 5,
        cross_band_bottleneck: int = 16,
    ):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.n_layers = n_layers
        self.n_bands = n_bands
        self.n_mic = n_mic
        self.use_film = use_film
        self.crf_hidden = crf_hidden
        self.film_dim = film_dim
        self.age_scale_frames = age_scale_frames
        self.use_cross_band = use_cross_band
        self.cross_band_kernel = cross_band_kernel
        self.cross_band_bottleneck = cross_band_bottleneck

        self.inp = GroupedLinear(d_in, hidden, 1)
        self.in1 = nn.InstanceNorm1d(hidden, affine=True)
        self.act1 = nn.PReLU(hidden)
        self.grus = nn.ModuleList(
            [GroupedGRUCell(hidden, hidden, groups) for _ in range(n_layers)]
        )
        self.film_bridge = (
            SpatialFiLMBridge(
                crf_hidden, hidden, n_bands, film_dim=film_dim, age_scale_frames=age_scale_frames
            )
            if use_film
            else None
        )
        self.cross_band = (
            CrossBandBlock(hidden, n_bands, cross_band_kernel, cross_band_bottleneck)
            if use_cross_band
            else None
        )
        self.mid = GroupedLinear(hidden, hidden, groups)
        self.in2 = nn.InstanceNorm1d(hidden, affine=True)
        self.act2 = nn.PReLU(hidden)
        self.out = nn.Linear(hidden, 2 * n_mic)

    def forward(
        self,
        feat: torch.Tensor,
        h_in: torch.Tensor,
        crf_h: torch.Tensor | None = None,
        age_frames: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        film: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if film is None and self.film_bridge is not None and crf_h is not None:
            film = self.film_bridge(crf_h, age_frames, valid)

        x = self.inp(feat)
        x = self.act1(self.in1(x.transpose(1, 2))).transpose(1, 2)
        if film is not None:
            x = (1.0 + film[0]) * x + film[1]
        x = x.reshape(-1, self.hidden)
        next_hs = []
        for idx, g in enumerate(self.grus):
            hi = h_in[:, idx].reshape(-1, self.hidden)
            h_new = g(x, hi)
            next_hs.append(h_new.reshape(-1, self.n_bands, self.hidden))
            x = channel_shuffle(h_new, self.groups)
            if idx == 0 and (self.cross_band is not None or film is not None):
                x_band = x.reshape(-1, self.n_bands, self.hidden)
                if self.cross_band is not None:
                    x_band = self.cross_band(x_band)
                if film is not None:
                    x_band = (1.0 + film[2]) * x_band + film[3]
                x = x_band.reshape(-1, self.hidden)
        h_out = torch.stack(next_hs, dim=1)
        x = x.reshape(-1, self.n_bands, self.hidden)
        x = self.mid(x)
        x = self.act2(self.in2(x.transpose(1, 2))).transpose(1, 2)
        return self.out(x), h_out


class MVDRBandEstimator(nn.Module):
    """逐带估计器（训练/整段并行版）。内置可选 FiLM 条件桥与跨频带模块。"""

    def __init__(
        self,
        d_in: int,
        hidden: int,
        groups: int,
        n_layers: int,
        n_bands: int,
        n_mic: int = 1,
        use_film: bool = False,
        crf_hidden: int = 128,
        film_dim: int = 8,
        age_scale_frames: float = 20.0,
        use_cross_band: bool = False,
        cross_band_kernel: int = 5,
        cross_band_bottleneck: int = 16,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_bands = n_bands
        self.hidden = hidden
        self.groups = groups
        self.n_layers = n_layers
        self.n_mic = n_mic
        self.use_film = use_film
        self.crf_hidden = crf_hidden
        self.film_dim = film_dim
        self.age_scale_frames = age_scale_frames
        self.use_cross_band = use_cross_band
        self.cross_band_kernel = cross_band_kernel
        self.cross_band_bottleneck = cross_band_bottleneck

        self.inp = GroupedLinear(d_in, hidden, 1)
        self.in1 = nn.InstanceNorm1d(hidden, affine=True)
        self.act1 = nn.PReLU(hidden)
        self.grus = nn.ModuleList(
            [GroupedGRU(hidden, hidden, groups) for _ in range(n_layers)]
        )
        self.film_bridge = (
            SpatialFiLMBridge(
                crf_hidden, hidden, n_bands, film_dim=film_dim, age_scale_frames=age_scale_frames
            )
            if use_film
            else None
        )
        self.cross_band = (
            CrossBandBlock(hidden, n_bands, cross_band_kernel, cross_band_bottleneck)
            if use_cross_band
            else None
        )
        self.mid = GroupedLinear(hidden, hidden, groups)
        self.in2 = nn.InstanceNorm1d(hidden, affine=True)
        self.act2 = nn.PReLU(hidden)
        self.out = nn.Linear(hidden, 2 * n_mic)
        nn.init.normal_(self.out.weight, std=1e-3)
        with torch.no_grad():
            bias = torch.zeros(2 * n_mic)
            bias[0] = 0.3
            self.out.bias.copy_(bias)

    def init_state(self, b: int, device, dtype: torch.dtype = torch.float32):
        return [g.init_state(b * self.n_bands, device, dtype=dtype) for g in self.grus]

    def forward(
        self,
        feat: torch.Tensor,
        state=None,
        crf_h: torch.Tensor | None = None,
        age_frames: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        film: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        if film is None and self.film_bridge is not None and crf_h is not None:
            film = self.film_bridge(crf_h, age_frames, valid)

        single_frame = feat.dim() == 3
        if single_frame:
            feat = feat.unsqueeze(1)
        b, t, n_bands, _ = feat.shape
        x = self.inp(feat).reshape(b * t, n_bands, self.hidden)
        x = self.act1(self.in1(x.transpose(1, 2))).transpose(1, 2)
        if film is not None:
            g1, b1, g2, b2 = film
            if single_frame and g1.dim() == 3:
                g1 = g1.unsqueeze(1)
                b1 = b1.unsqueeze(1)
                g2 = g2.unsqueeze(1)
                b2 = b2.unsqueeze(1)
            x = (1.0 + g1.reshape(b * t, n_bands, self.hidden)) * x + b1.reshape(
                b * t, n_bands, self.hidden
            )
        x = (
            x.reshape(b, t, n_bands, self.hidden)
            .transpose(1, 2)
            .reshape(b * n_bands, t, self.hidden)
        )
        new_state = []
        for idx, g in enumerate(self.grus):
            h0 = None if state is None else state[idx]
            x, h_last = g(x, h0)
            new_state.append(h_last)
            x = channel_shuffle(x, self.groups)
            if idx == 0 and (self.cross_band is not None or film is not None):
                x_band = (
                    x.reshape(b, n_bands, t, self.hidden)
                    .transpose(1, 2)
                    .reshape(b * t, n_bands, self.hidden)
                )
                if self.cross_band is not None:
                    x_band = self.cross_band(x_band)
                if film is not None:
                    x_band = (1.0 + g2.reshape(b * t, n_bands, self.hidden)) * x_band + b2.reshape(
                        b * t, n_bands, self.hidden
                    )
                x = (
                    x_band.reshape(b, t, n_bands, self.hidden)
                    .transpose(1, 2)
                    .reshape(b * n_bands, t, self.hidden)
                )
        x = (
            x.reshape(b, n_bands, t, self.hidden)
            .transpose(1, 2)
            .reshape(b * t, n_bands, self.hidden)
        )
        x = self.mid(x)
        x = self.act2(self.in2(x.transpose(1, 2))).transpose(1, 2)
        y = self.out(x).reshape(b, t, n_bands, 2 * self.n_mic)
        if single_frame:
            y = y.squeeze(1)
        return y, new_state

    @torch.no_grad()
    def to_streaming(self) -> MVDRBandEstimatorStep:
        step = MVDRBandEstimatorStep(
            self.d_in,
            self.hidden,
            self.groups,
            self.n_layers,
            self.n_bands,
            n_mic=self.n_mic,
            use_film=self.use_film,
            crf_hidden=self.crf_hidden,
            film_dim=self.film_dim,
            age_scale_frames=self.age_scale_frames,
            use_cross_band=self.use_cross_band,
            cross_band_kernel=self.cross_band_kernel,
            cross_band_bottleneck=self.cross_band_bottleneck,
        )
        step.inp.load_state_dict(self.inp.state_dict())
        step.in1.load_state_dict(self.in1.state_dict())
        step.act1.load_state_dict(self.act1.state_dict())
        for dst, src in zip(step.grus, self.grus):
            dst.copy_from_grouped_gru(src)
        if self.film_bridge is not None and step.film_bridge is not None:
            step.film_bridge.load_state_dict(self.film_bridge.state_dict())
        if self.cross_band is not None and step.cross_band is not None:
            step.cross_band.load_state_dict(self.cross_band.state_dict())
        step.mid.load_state_dict(self.mid.state_dict())
        step.in2.load_state_dict(self.in2.state_dict())
        step.act2.load_state_dict(self.act2.state_dict())
        step.out.load_state_dict(self.out.state_dict())
        step.to(device=self.out.weight.device, dtype=self.out.weight.dtype)
        step.eval()
        return step
