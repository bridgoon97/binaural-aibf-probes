"""网络部件。

命名与麦阵 AIBF 对齐
- GroupedLinear 对应 AIBF 文档里的 GL。
- CRFEstimator 对应全带估计器，输出语音与噪声两类因果复数 FIR。
- NeuralSCM 对应神经 SCM，即过滤后谱的外积。
- MVDRBandEstimator 对应逐带估计器，输出本侧复数滤波器。

训练主路径（CRFEstimator / NeuralSCM / MVDRBandEstimator）采用全序列并行计算，
使用基于 nn.GRU 的 GroupedGRU 避免逐帧反向传播。
流式导出路径（CRFEstimatorStep / MVDRBandEstimatorStep）使用 GroupedGRUCell
按单帧递推构造网络，与训练主路径权重严格一一对应、数值等价。

这不是经典 MVDR 闭式解，不默认具有无失真约束。声像保真只能靠损失约束。
"""

import torch
import torch.nn as nn

from complexops import cmul, cmul_conj


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
    """全序列并行分组 GRU。各组状态独立，组间不交换信息。

    训练与整段推理使用 nn.GRU 沿时间轴融合前向与反向，避免 Python 逐帧循环。
    """

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
        """x [N, T, d_in]，h [N, hidden] 或 None。返回 (y [N, T, hidden], h_last [N, hidden])。"""
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
    """单帧流式分组 GRU 单元。各组状态独立，组间不交换信息。

    总 hidden 为 hidden，分成 groups 组，每组 hidden/groups。
    权重与 GroupedGRU 严格一一对应，用于单帧流式构造与模型导出。
    """

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
        """x [N, d_in]，h [N, hidden]。返回新的 h [N, hidden]。"""
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

    def forward(
        self, feat: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """feat [B, D]，h1/h2 [B, hidden]。返回 (FIR [B, n_bands, n_taps, 2, 2], next_h1, next_h2)。"""
        if self.pad:
            feat = torch.nn.functional.pad(feat, (0, self.pad))
        x = torch.relu(self.inp(feat))
        next_h1 = self.gru1(x, h1)
        next_h2 = self.gru2(next_h1, h2)
        y = self.out(next_h2)
        fir = y.view(-1, self.n_bands, self.n_taps, 2, 2)
        return fir, next_h1, next_h2


class CRFEstimator(nn.Module):
    """全带估计器（训练/整段并行版）。每帧读全带特征，输出逐带语音与噪声复数 FIR。

    输出维度 n_bands * n_taps * 2 类 * 2 实虚，对应 AIBF 的 1536 那一项。
    FIR 系数跨通道共享。共享是必要条件，否则外积得到的 SCM 不是同一个空间量。
    """

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

    def forward(self, feat: torch.Tensor, state=None):
        """feat [B, T, D] 或 [B, D]。返回 FIR 与末帧状态 [h1, h2]。"""
        single_frame = feat.dim() == 2
        if single_frame:
            feat = feat.unsqueeze(1)
        b, t, _ = feat.shape
        if self.pad:
            feat = torch.nn.functional.pad(feat, (0, self.pad))
        x = torch.relu(self.inp(feat))
        h1_in = None if state is None else state[0]
        h2_in = None if state is None else state[1]
        x, h1_last = self.gru1(x, h1_in)
        x, h2_last = self.gru2(x, h2_in)
        y = self.out(x)
        fir = y.view(b, t, self.n_bands, self.n_taps, 2, 2)
        if single_frame:
            fir = fir.squeeze(1)
        return fir, [h1_last, h2_last]

    @torch.no_grad()
    def to_streaming(self) -> CRFEstimatorStep:
        """构造权重完全一致的单帧流式估计器。"""
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
    """神经 SCM。把 FIR 作用到通道向量上，再取外积。

    通道向量为 [本侧配对帧, 对侧]，共 n_ch 路。
    每类输出 n_ch x n_ch 复数矩阵，展平为 2 * n_ch^2 个实数。
    支持单帧 [B, n_ch, n_taps, n_bands, 2] 与全序列 [B, T, n_ch, n_taps, n_bands, 2]。
    """

    def __init__(self, n_ch: int, n_bands: int, instance_norm: bool = True):
        super().__init__()
        self.n_ch = n_ch
        self.n_bands = n_bands
        self.instance_norm = instance_norm
        self.dim = 2 * n_ch * n_ch * 2  # 两类，各 n_ch^2 复数
        self.norm = (
            nn.InstanceNorm1d(self.dim, affine=True) if instance_norm else nn.Identity()
        )

    def _compute(self, chan_hist: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
        """chan_hist [N, n_ch, n_taps, n_bands, 2]，fir [N, n_bands, n_taps, 2, 2]。"""
        # 对齐广播：chan [N, n_ch, 1, n_taps, n_bands, 2]，fir [N, 1, 2, n_taps, n_bands, 2]
        chan = chan_hist.unsqueeze(2)
        f = fir.permute(0, 3, 2, 1, 4).unsqueeze(1)  # [N, 1, 2类, n_taps, n_bands, 2]
        filtered = cmul(chan, f).sum(dim=3)  # [N, n_ch, 2类, n_bands, 2]

        a = filtered.unsqueeze(1)  # [N, 1, n_ch, 2, n_bands, 2]
        bb = filtered.unsqueeze(2)  # [N, n_ch, 1, 2, n_bands, 2]
        scm = cmul_conj(a, bb)  # [N, n_ch, n_ch, 2类, n_bands, 2]
        scm = scm.permute(0, 4, 1, 2, 3, 5).reshape(-1, self.n_bands, self.dim)
        # InstanceNorm1d 严格沿每帧的 n_bands 轴归一化，不跨时间步。
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


class MVDRBandEstimatorStep(nn.Module):
    """单帧流式逐带估计器，用于流式推理与单帧模型导出。"""

    def __init__(self, d_in: int, hidden: int, groups: int, n_layers: int, n_bands: int):
        super().__init__()
        self.d_in = d_in
        self.hidden = hidden
        self.groups = groups
        self.n_layers = n_layers
        self.n_bands = n_bands
        self.inp = GroupedLinear(d_in, hidden, 1)
        self.in1 = nn.InstanceNorm1d(hidden, affine=True)
        self.act1 = nn.PReLU(hidden)
        self.grus = nn.ModuleList(
            [GroupedGRUCell(hidden, hidden, groups) for _ in range(n_layers)]
        )
        self.mid = GroupedLinear(hidden, hidden, groups)
        self.in2 = nn.InstanceNorm1d(hidden, affine=True)
        self.act2 = nn.PReLU(hidden)
        self.out = nn.Linear(hidden, 2)

    def forward(
        self, feat: torch.Tensor, h_in: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """feat [B, n_bands, d_in]，h_in [B, n_layers, n_bands, hidden]。

        返回 (滤波器 [B, n_bands, 2], h_out [B, n_layers, n_bands, hidden])。
        """
        x = self.inp(feat)
        x = self.act1(self.in1(x.transpose(1, 2))).transpose(1, 2)
        x = x.reshape(-1, self.hidden)
        next_hs = []
        for idx, g in enumerate(self.grus):
            hi = h_in[:, idx].reshape(-1, self.hidden)
            x = g(x, hi)
            next_hs.append(x.reshape(-1, self.n_bands, self.hidden))
        h_out = torch.stack(next_hs, dim=1)
        x = x.reshape(-1, self.n_bands, self.hidden)
        x = self.mid(x)
        x = self.act2(self.in2(x.transpose(1, 2))).transpose(1, 2)
        return self.out(x), h_out


class MVDRBandEstimator(nn.Module):
    """逐带估计器（训练/整段并行版）。各频带共享权重、状态独立，输出本侧复数滤波器。"""

    def __init__(self, d_in: int, hidden: int, groups: int, n_layers: int, n_bands: int):
        super().__init__()
        self.d_in = d_in
        self.n_bands = n_bands
        self.hidden = hidden
        self.groups = groups
        self.n_layers = n_layers
        self.inp = GroupedLinear(d_in, hidden, 1)
        self.in1 = nn.InstanceNorm1d(hidden, affine=True)
        self.act1 = nn.PReLU(hidden)
        self.grus = nn.ModuleList(
            [GroupedGRU(hidden, hidden, groups) for _ in range(n_layers)]
        )
        self.mid = GroupedLinear(hidden, hidden, groups)
        self.in2 = nn.InstanceNorm1d(hidden, affine=True)
        self.act2 = nn.PReLU(hidden)
        self.out = nn.Linear(hidden, 2)
        nn.init.normal_(self.out.weight, std=1e-3)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.3, 0.0]))

    def init_state(self, b: int, device, dtype: torch.dtype = torch.float32):
        return [g.init_state(b * self.n_bands, device, dtype=dtype) for g in self.grus]

    def forward(self, feat: torch.Tensor, state=None):
        """feat [B, T, n_bands, d_in] 或 [B, n_bands, d_in]。返回滤波器与末帧状态。"""
        single_frame = feat.dim() == 3
        if single_frame:
            feat = feat.unsqueeze(1)
        b, t, n_bands, _ = feat.shape
        x = self.inp(feat).reshape(b * t, n_bands, self.hidden)
        # InstanceNorm1d 按每帧的 n_bands 轴归一化，保证与单帧流式严格等价。
        x = self.act1(self.in1(x.transpose(1, 2))).transpose(1, 2)
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
        x = (
            x.reshape(b, n_bands, t, self.hidden)
            .transpose(1, 2)
            .reshape(b * t, n_bands, self.hidden)
        )
        x = self.mid(x)
        x = self.act2(self.in2(x.transpose(1, 2))).transpose(1, 2)
        y = self.out(x).reshape(b, t, n_bands, 2)
        if single_frame:
            y = y.squeeze(1)
        return y, new_state

    @torch.no_grad()
    def to_streaming(self) -> MVDRBandEstimatorStep:
        """构造权重完全一致的单帧流式逐带估计器。"""
        step = MVDRBandEstimatorStep(
            self.d_in, self.hidden, self.groups, self.n_layers, self.n_bands
        )
        step.inp.load_state_dict(self.inp.state_dict())
        step.in1.load_state_dict(self.in1.state_dict())
        step.act1.load_state_dict(self.act1.state_dict())
        for dst, src in zip(step.grus, self.grus):
            dst.copy_from_grouped_gru(src)
        step.mid.load_state_dict(self.mid.state_dict())
        step.in2.load_state_dict(self.in2.state_dict())
        step.act2.load_state_dict(self.act2.state_dict())
        step.out.load_state_dict(self.out.state_dict())
        step.to(device=self.out.weight.device, dtype=self.out.weight.dtype)
        step.eval()
        return step
