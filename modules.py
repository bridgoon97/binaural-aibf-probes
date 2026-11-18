"""网络部件。

命名与麦阵 AIBF 对齐
- GroupedLinear 对应 AIBF 文档里的 GL。
- CRFEstimator 对应全带估计器，输出语音与噪声两类因果复数 FIR。
- NeuralSCM 对应神经 SCM，即过滤后谱的外积。
- MVDRBandEstimator 对应逐带估计器，输出本侧复数滤波器。

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
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=1, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, D] 或 [B, N, D]。最后一维是特征维。"""
        if x.dim() == 2:
            return self.conv(x.unsqueeze(-1)).squeeze(-1)
        b, n, d = x.shape
        y = self.conv(x.reshape(b * n, d).unsqueeze(-1)).squeeze(-1)
        return y.reshape(b, n, -1)


class GroupedGRUCell(nn.Module):
    """分组 GRU 单元。各组状态独立，组间不交换信息。

    总 hidden 为 hidden，分成 groups 组，每组 hidden/groups。
    """

    def __init__(self, d_in: int, hidden: int, groups: int = 1):
        super().__init__()
        assert d_in % groups == 0 and hidden % groups == 0
        self.groups = groups
        self.hidden = hidden
        self.cells = nn.ModuleList(
            [nn.GRUCell(d_in // groups, hidden // groups) for _ in range(groups)]
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """x [N, d_in]，h [N, hidden]。返回新的 h。"""
        xs = x.chunk(self.groups, dim=-1)
        hs = h.chunk(self.groups, dim=-1)
        return torch.cat([c(xi, hi) for c, xi, hi in zip(self.cells, xs, hs)], dim=-1)

    def init_state(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device)


class CRFEstimator(nn.Module):
    """全带估计器。每帧读一次全带特征，输出逐带的语音与噪声复数 FIR 系数。

    输出维度 n_bands * n_taps * 2 类 * 2 实虚，对应 AIBF 的 1536 那一项。
    FIR 系数跨通道共享。共享是必要条件，否则外积得到的 SCM 不是同一个空间量。
    """

    def __init__(self, d_in: int, hidden: int, groups: int, n_bands: int, n_taps: int):
        super().__init__()
        self.n_bands, self.n_taps = n_bands, n_taps
        d_out = n_bands * n_taps * 2 * 2
        pad = (-d_in) % groups
        self.pad = pad
        self.inp = GroupedLinear(d_in + pad, hidden, groups)
        self.gru1 = GroupedGRUCell(hidden, hidden, groups)
        self.gru2 = GroupedGRUCell(hidden, hidden, groups)
        self.out = GroupedLinear(hidden, d_out, groups)
        nn.init.zeros_(self.out.conv.bias)
        nn.init.normal_(self.out.conv.weight, std=1e-3)

    def init_state(self, b: int, device):
        return [self.gru1.init_state(b, device), self.gru2.init_state(b, device)]

    def forward(self, feat: torch.Tensor, state):
        """feat [B, D]。返回 FIR [B, n_bands, n_taps, 2, 2] 与新状态。"""
        if self.pad:
            feat = torch.nn.functional.pad(feat, (0, self.pad))
        x = torch.relu(self.inp(feat))
        h1 = self.gru1(x, state[0])
        h2 = self.gru2(h1, state[1])
        y = self.out(h2)
        fir = y.view(-1, self.n_bands, self.n_taps, 2, 2)
        return fir, [h1, h2]


class NeuralSCM(nn.Module):
    """神经 SCM。把 FIR 作用到通道向量上，再取外积。

    通道向量为 [本侧配对帧, 对侧], 共 n_ch 路。
    每类输出 n_ch x n_ch 复数矩阵，展平为 2 * n_ch^2 个实数。
    """

    def __init__(self, n_ch: int, n_bands: int, instance_norm: bool = True):
        super().__init__()
        self.n_ch = n_ch
        self.dim = 2 * n_ch * n_ch * 2  # 两类，各 n_ch^2 复数
        self.norm = (
            nn.InstanceNorm1d(self.dim, affine=True) if instance_norm else nn.Identity()
        )

    def forward(self, chan_hist: torch.Tensor, fir: torch.Tensor) -> torch.Tensor:
        """chan_hist [B, n_ch, n_taps, n_bands, 2]，抽头按时间从新到旧。

        fir [B, n_bands, n_taps, 2, 2]。返回 [B, n_bands, dim]。
        """
        b, n_ch, n_taps, n_bands, _ = chan_hist.shape
        # 对齐广播：chan [B, n_ch, 1, n_taps, n_bands, 2]，fir [B, 1, 2, n_taps, n_bands, 2]
        chan = chan_hist.unsqueeze(2)
        f = fir.permute(0, 3, 2, 1, 4).unsqueeze(1)  # [B, 1, 2类, n_taps, n_bands, 2]
        filtered = cmul(chan, f).sum(dim=3)  # [B, n_ch, 2类, n_bands, 2]

        a = filtered.unsqueeze(1)  # [B, 1, n_ch, 2, n_bands, 2]
        bb = filtered.unsqueeze(2)  # [B, n_ch, 1, 2, n_bands, 2]
        scm = cmul_conj(a, bb)  # [B, n_ch, n_ch, 2类, n_bands, 2]
        scm = scm.permute(0, 4, 1, 2, 3, 5).reshape(b, n_bands, -1)
        scm = self.norm(scm.transpose(1, 2)).transpose(1, 2)
        return scm


class MVDRBandEstimator(nn.Module):
    """逐带估计器。各频带共享权重、状态独立，输出本侧复数滤波器。"""

    def __init__(self, d_in: int, hidden: int, groups: int, n_layers: int, n_bands: int):
        super().__init__()
        self.n_bands, self.hidden, self.n_layers = n_bands, hidden, n_layers
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
        nn.init.normal_(self.out.weight, std=1e-3)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.3, 0.0]))

    def init_state(self, b: int, device):
        return [g.init_state(b * self.n_bands, device) for g in self.grus]

    def forward(self, feat: torch.Tensor, state):
        """feat [B, n_bands, d_in]。返回滤波器 [B, n_bands, 2] 与新状态。"""
        b, n_bands, _ = feat.shape
        x = self.inp(feat)
        x = self.act1(self.in1(x.transpose(1, 2))).transpose(1, 2)
        x = x.reshape(b * n_bands, -1)
        new_state = []
        for g, h in zip(self.grus, state):
            x = g(x, h)
            new_state.append(x)
        x = x.reshape(b, n_bands, -1)
        x = self.mid(x)
        x = self.act2(self.in2(x.transpose(1, 2))).transpose(1, 2)
        return self.out(x), new_state
