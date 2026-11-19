"""单帧流式网络构造与导出（支持 n_mic >= 1）。

- StreamingSideModel：单侧单帧纯张量网络，无动态控制流与时间循环，支持 TorchScript 与 ONNX。
- BinauralStreamingRunner：带跨耳延迟队列与本侧历史缓存的双耳逐帧流式运行器。
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .bands import compress_band, expand_complex, pad_dc_zero, strip_dc
from .complexops import cabs, cmul_conj
from .features import build_crf_input_step, compress_current_spec


class StreamingSideModel(nn.Module):
    """按流式单帧构造的单耳核心网络，可直接导出为 TorchScript / ONNX。

    n_mic=1 时 local_bin 为 [B, n_bins, 2]、local_band_* 为 [B, n_bands, 2]；
    n_mic>1 时 local_bin 为 [B, n_mic, n_bins, 2]、local_band_* 为 [B, n_mic, n_bands, 2]。
    """

    INPUT_NAMES = (
        "local_bin",
        "local_band_matched",
        "remote_band",
        "valid",
        "age_frames",
        "pair_hist",
        "crf_h1",
        "crf_h2",
        "mvdr_h",
    )
    OUTPUT_NAMES = (
        "out_bin",
        "local_band_now",
        "next_pair_hist",
        "next_crf_h1",
        "next_crf_h2",
        "next_mvdr_h",
    )

    def __init__(self, train_model):
        super().__init__()
        self.cfg = train_model.cfg
        self.n_bins = train_model.n_bins
        self.n_linear = train_model.n_linear
        self.n_bands = train_model.n_bands
        self.n_mic = train_model.n_mic
        self.n_ch = train_model.n_ch
        self.n_taps = train_model.n_taps
        self.tap_stride = train_model.tap_stride
        self.hist_len = (self.n_taps - 1) * self.tap_stride
        self.buf_len = max(self.hist_len, 1)
        self.local_frame = self.cfg["scm"]["local_frame"]
        self.payload = self.cfg["message"]["payload"]
        self.use_current_spec = bool(self.cfg["mvdr"]["use_current_spec"])
        self.normalize_current_spec = train_model.normalize_current_spec
        self.current_spec_power = train_model.current_spec_power
        self.gain_limit = float(self.cfg["mvdr"].get("gain_limit", 2.0))
        self.crf_hidden = int(self.cfg["crf"]["hidden"])
        self.mvdr_hidden = int(self.cfg["mvdr"]["hidden"])
        self.n_mvdr_layers = int(self.cfg["mvdr"]["n_layers"])

        self.register_buffer("comp", train_model.comp.detach().clone())
        self.register_buffer("expa", train_model.expa.detach().clone())
        self.register_buffer(
            "valid_band",
            train_model.msg.valid_band.detach().clone().to(
                device=train_model.comp.device, dtype=train_model.comp.dtype
            ),
        )
        self.register_buffer(
            "valid_band_phase",
            train_model.msg.valid_band_phase.detach().clone().to(
                device=train_model.comp.device, dtype=train_model.comp.dtype
            ),
        )
        self.register_buffer(
            "mag_only_band",
            train_model.msg.mag_only_band.detach().clone().to(
                device=train_model.comp.device, dtype=train_model.comp.dtype
            ),
        )

        self.crf = train_model.crf.to_streaming()
        self.scm = train_model.scm.to_streaming()
        self.mvdr = train_model.mvdr.to_streaming()
        self.eval()

    def init_state(
        self, batch_size: int = 1, device=None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if device is None:
            device = self.comp.device
        if dtype is None:
            dtype = self.comp.dtype
        pair_hist = torch.zeros(
            batch_size, self.n_ch, self.buf_len, self.n_bands, 2, device=device, dtype=dtype
        )
        crf_h1 = torch.zeros(batch_size, self.crf_hidden, device=device, dtype=dtype)
        crf_h2 = torch.zeros(batch_size, self.crf_hidden, device=device, dtype=dtype)
        mvdr_h = torch.zeros(
            batch_size,
            self.n_mvdr_layers,
            self.n_bands,
            self.mvdr_hidden,
            device=device,
            dtype=dtype,
        )
        return pair_hist, crf_h1, crf_h2, mvdr_h

    def dummy_inputs(
        self, batch_size: int = 1, device=None, dtype: torch.dtype | None = None
    ) -> tuple[torch.Tensor, ...]:
        if device is None:
            device = self.comp.device
        if dtype is None:
            dtype = self.comp.dtype
        if self.n_mic == 1:
            local_bin = torch.zeros(batch_size, self.n_bins, 2, device=device, dtype=dtype)
            local_band_matched = torch.zeros(
                batch_size, self.n_bands, 2, device=device, dtype=dtype
            )
        else:
            local_bin = torch.zeros(
                batch_size, self.n_mic, self.n_bins, 2, device=device, dtype=dtype
            )
            local_band_matched = torch.zeros(
                batch_size, self.n_mic, self.n_bands, 2, device=device, dtype=dtype
            )
        remote_band = torch.zeros(
            batch_size, self.n_bands, 2, device=device, dtype=dtype
        )
        valid = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        age_frames = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        pair_hist, crf_h1, crf_h2, mvdr_h = self.init_state(
            batch_size=batch_size, device=device, dtype=dtype
        )
        return (
            local_bin,
            local_band_matched,
            remote_band,
            valid,
            age_frames,
            pair_hist,
            crf_h1,
            crf_h2,
            mvdr_h,
        )

    def forward(
        self,
        local_bin: torch.Tensor,
        local_band_matched: torch.Tensor,
        remote_band: torch.Tensor,
        valid: torch.Tensor,
        age_frames: torch.Tensor,
        pair_hist: torch.Tensor,
        crf_h1: torch.Tensor,
        crf_h2: torch.Tensor,
        mvdr_h: torch.Tensor,
    ):
        local_band_now = compress_band(local_bin, self.comp, self.n_linear)
        if self.n_mic == 1:
            local_bin_m = local_bin.unsqueeze(1)
            local_band_m = local_band_now.unsqueeze(1)
            matched_raw = local_band_matched if self.local_frame == "delayed" else local_band_now
            matched_m = matched_raw.unsqueeze(1)
        else:
            local_bin_m = local_bin
            local_band_m = local_band_now
            matched_m = local_band_matched if self.local_frame == "delayed" else local_band_now

        feat = build_crf_input_step(
            self.cfg,
            local_band_now=local_band_m[:, 0],
            local_band_matched=matched_m[:, 0],
            remote_band=remote_band,
            valid=valid,
            age_frames=age_frames,
            valid_band=self.valid_band,
            valid_band_phase=self.valid_band_phase,
        )

        pair_now = torch.cat([matched_m, remote_band.unsqueeze(1)], dim=1)
        if self.n_taps == 1:
            chan_taps = pair_now.unsqueeze(2)
            next_pair_hist = pair_hist
        else:
            full_hist = torch.cat([pair_now.unsqueeze(2), pair_hist], dim=2)
            chan_taps = full_hist[:, :, :: self.tap_stride]
            next_pair_hist = full_hist[:, :, :-1]

        if self.payload == "logmag":
            chan_taps = torch.stack(
                [cabs(chan_taps), torch.zeros_like(chan_taps[..., 0])], dim=-1
            )
        elif self.payload == "mixed":
            mag_only = self.mag_only_band.view(1, 1, 1, self.n_bands, 1)
            mag_only_chan = torch.stack(
                [cabs(chan_taps), torch.zeros_like(chan_taps[..., 0])], dim=-1
            )
            chan_taps = torch.where(mag_only > 0.5, mag_only_chan, chan_taps)

        if self.mvdr.film_bridge is not None:
            fir, next_crf_h1, next_crf_h2, crf_h = self.crf.forward_with_hidden(
                feat, crf_h1, crf_h2
            )
        else:
            fir, next_crf_h1, next_crf_h2 = self.crf(feat, crf_h1, crf_h2)
            crf_h = None

        scm = self.scm(chan_taps, fir)
        if self.use_current_spec:
            cur = (
                compress_current_spec(local_band_m, self.current_spec_power, has_mic_dim=True)
                if self.normalize_current_spec
                else local_band_m
            )
            cur_flat = cur.transpose(1, 2).reshape(-1, self.n_bands, 2 * self.n_mic)
            scm = torch.cat([scm, cur_flat], dim=-1)

        g_band_flat, next_mvdr_h = self.mvdr(
            scm, mvdr_h, crf_h=crf_h, age_frames=age_frames, valid=valid
        )
        g_band = g_band_flat.reshape(-1, self.n_bands, self.n_mic, 2).transpose(1, 2)
        g_bin = expand_complex(g_band, self.expa)
        norm = torch.sqrt(g_bin.square().sum(dim=(1, 3), keepdim=True) + 1e-8)
        scale = (self.gain_limit / norm.clamp_min(1e-8)).clamp_max(1.0)
        g_bin = g_bin * scale
        out_bin = cmul_conj(local_bin_m, g_bin).sum(dim=1)
        return (
            out_bin,
            local_band_now,
            next_pair_hist,
            next_crf_h1,
            next_crf_h2,
            next_mvdr_h,
        )


@dataclass
class _SideRuntime:
    pair_hist: torch.Tensor
    crf_h1: torch.Tensor
    crf_h2: torch.Tensor
    mvdr_h: torch.Tensor
    local_hist: dict
    queue: deque
    held_remote: torch.Tensor | None
    held_emit: int
    has_valid: bool


class BinauralStreamingRunner:
    """双耳逐帧流式运行器（支持 n_mic >= 1）。"""

    def __init__(
        self,
        train_model,
        step_fn: Callable | None = None,
    ):
        self.cfg = train_model.cfg
        self.stft = train_model.stft
        self.msg = train_model.msg
        self.communication = train_model.communication
        self.local_frame = self.cfg["scm"]["local_frame"]
        self.n_mic = train_model.n_mic
        self.n_bins = train_model.n_bins
        self.n_linear = train_model.n_linear
        self.comp = train_model.comp.detach().clone()
        self.streaming_model = train_model.build_streaming_model()
        self.step_fn = self.streaming_model if step_fn is None else step_fn
        self.frame_idx = 0
        self.left: _SideRuntime | None = None
        self.right: _SideRuntime | None = None

    def reset(self, batch_size: int = 1, device=None, dtype: torch.dtype | None = None):
        if device is None:
            device = self.comp.device
        if dtype is None:
            dtype = self.comp.dtype
        self.frame_idx = 0
        self.left = self._new_side(batch_size, device, dtype)
        self.right = self._new_side(batch_size, device, dtype)

    def _new_side(self, batch_size: int, device, dtype: torch.dtype) -> _SideRuntime:
        pair_hist, crf_h1, crf_h2, mvdr_h = self.streaming_model.init_state(
            batch_size=batch_size, device=device, dtype=dtype
        )
        return _SideRuntime(
            pair_hist=pair_hist,
            crf_h1=crf_h1,
            crf_h2=crf_h2,
            mvdr_h=mvdr_h,
            local_hist={},
            queue=deque(),
            held_remote=None,
            held_emit=0,
            has_valid=False,
        )

    def _step_one_side(
        self,
        rt: _SideRuntime,
        local_bin: torch.Tensor,
        local_band_now: torch.Tensor,
        remote_band_now: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        b = local_bin.shape[0]
        device, dtype = local_bin.device, local_bin.dtype
        rt.local_hist[t] = local_band_now

        if self.communication:
            if t % self.msg.update_period == 0:
                decoded = self.msg.encode_decode_step(remote_band_now)
                rt.queue.append((t + self.msg.delay_frames, t, decoded))
            while rt.queue and rt.queue[0][0] <= t:
                _, emit_idx, decoded = rt.queue.popleft()
                rt.held_remote = decoded
                rt.held_emit = emit_idx
                rt.has_valid = True
                stale = [k for k in rt.local_hist if k < rt.held_emit]
                for k in stale:
                    del rt.local_hist[k]

            if rt.has_valid:
                remote_band = rt.held_remote
                valid = torch.ones(b, 1, device=device, dtype=dtype)
                age = torch.full((b, 1), float(t - rt.held_emit), device=device, dtype=dtype)
                local_matched = (
                    rt.local_hist[rt.held_emit]
                    if self.local_frame == "delayed"
                    else local_band_now
                )
            else:
                remote_band = torch.zeros_like(remote_band_now)
                valid = torch.zeros(b, 1, device=device, dtype=dtype)
                age = torch.full((b, 1), float(t), device=device, dtype=dtype)
                local_matched = (
                    rt.local_hist[0] if self.local_frame == "delayed" else local_band_now
                )
        else:
            remote_band = torch.zeros_like(remote_band_now)
            valid = torch.zeros(b, 1, device=device, dtype=dtype)
            age = torch.zeros(b, 1, device=device, dtype=dtype)
            local_matched = local_band_now
            rt.local_hist.clear()

        (
            out_bin,
            _,
            rt.pair_hist,
            rt.crf_h1,
            rt.crf_h2,
            rt.mvdr_h,
        ) = self.step_fn(
            local_bin,
            local_matched,
            remote_band,
            valid,
            age,
            rt.pair_hist,
            rt.crf_h1,
            rt.crf_h2,
            rt.mvdr_h,
        )
        return out_bin

    @torch.no_grad()
    def step_spec(
        self, local_bin_l: torch.Tensor, local_bin_r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """双耳单帧频谱流式推进。"""
        if self.n_mic == 1 and local_bin_l.dim() == 4:
            local_bin_l = local_bin_l.squeeze(1)
            local_bin_r = local_bin_r.squeeze(1)
        if self.left is None or self.right is None:
            self.reset(
                batch_size=local_bin_l.shape[0],
                device=local_bin_l.device,
                dtype=local_bin_l.dtype,
            )
        comp = self.comp.to(device=local_bin_l.device, dtype=local_bin_l.dtype)
        bl = compress_band(local_bin_l, comp, self.n_linear)
        br = compress_band(local_bin_r, comp, self.n_linear)
        ref_bl = bl if self.n_mic == 1 else bl[:, 0]
        ref_br = br if self.n_mic == 1 else br[:, 0]
        t = self.frame_idx
        out_l = self._step_one_side(self.left, local_bin_l, bl, ref_br, t)
        out_r = self._step_one_side(self.right, local_bin_r, br, ref_bl, t)
        self.frame_idx += 1
        return out_l, out_r

    @torch.no_grad()
    def process_spec(
        self, xl: torch.Tensor, xr: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.reset(batch_size=xl.shape[0], device=xl.device, dtype=xl.dtype)
        outs_l, outs_r = [], []
        for t in range(xl.shape[1]):
            ol, or_ = self.step_spec(xl[:, t], xr[:, t])
            outs_l.append(ol)
            outs_r.append(or_)
        return torch.stack(outs_l, dim=1), torch.stack(outs_r, dim=1)

    @torch.no_grad()
    def process_waveform(
        self, wav_left: torch.Tensor, wav_right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if wav_left.shape != wav_right.shape:
            raise ValueError("左右波形必须同形状")
        if wav_left.dim() == 2:
            if self.n_mic != 1:
                raise ValueError(f"n_mic={self.n_mic} 时输入波形必须为 [B, n_mic, S]")
            wl, wr = wav_left.unsqueeze(1), wav_right.unsqueeze(1)
        elif wav_left.dim() == 3:
            if wav_left.shape[1] != self.n_mic:
                raise ValueError(f"输入通道数 {wav_left.shape[1]} 与 n_mic={self.n_mic} 不一致")
            wl, wr = wav_left, wav_right
        else:
            raise ValueError("波形维度必须为 [B, S] 或 [B, n_mic, S]")

        b, m, n = wl.shape
        xl_flat = strip_dc(self.stft.analyze(wl.reshape(b * m, n)))
        xr_flat = strip_dc(self.stft.analyze(wr.reshape(b * m, n)))
        t = xl_flat.shape[1]
        xl = xl_flat.reshape(b, m, t, self.n_bins, 2).transpose(1, 2)
        xr = xr_flat.reshape(b, m, t, self.n_bins, 2).transpose(1, 2)
        if self.n_mic == 1:
            xl, xr = xl.squeeze(2), xr.squeeze(2)
        yl_spec, yr_spec = self.process_spec(xl, xr)
        return (
            self.stft.synthesize(pad_dc_zero(yl_spec), length=n),
            self.stft.synthesize(pad_dc_zero(yr_spec), length=n),
        )


def make_onnxruntime_step_fn(onnx_path: str | Path):
    """把导出的单帧 ONNX 模型包装成可供 BinauralStreamingRunner 调用的 step_fn。"""
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("需要安装 onnxruntime 才能运行 ONNX 流式校验") from exc

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = list(StreamingSideModel.INPUT_NAMES)

    def _step(*tensors: torch.Tensor):
        device, dtype = tensors[0].device, tensors[0].dtype
        feed = {name: t.detach().cpu().numpy() for name, t in zip(in_names, tensors)}
        outs = sess.run(None, feed)
        return tuple(torch.from_numpy(arr).to(device=device, dtype=dtype) for arr in outs)

    return _step


@torch.no_grad()
def verify_streaming_equivalence(
    train_model,
    step_fn: Callable | None = None,
    seconds: float = 0.25,
    batch_size: int = 1,
    atol: float = 1e-5,
) -> float:
    """校验并行训练模型与单帧流式网络的整段波形最大绝对误差。"""
    was_training = train_model.training
    train_model.eval()
    try:
        device = train_model.comp.device
        dtype = train_model.comp.dtype
        n_samples = max(1, int(seconds * train_model.stft.sample_rate))
        gen = torch.Generator(device="cpu").manual_seed(42)
        shape = (
            (batch_size, n_samples)
            if train_model.n_mic == 1
            else (batch_size, train_model.n_mic, n_samples)
        )
        xl = torch.randn(*shape, generator=gen, dtype=dtype).to(device)
        xr = torch.randn(*shape, generator=gen, dtype=dtype).to(device)

        yl_par, yr_par, _ = train_model(xl, xr)
        runner = BinauralStreamingRunner(train_model, step_fn=step_fn)
        yl_str, yr_str = runner.process_waveform(xl, xr)

        err_l = (yl_par - yl_str).abs().max().item()
        err_r = (yr_par - yr_str).abs().max().item()
        max_err = max(err_l, err_r)
        if max_err > atol:
            raise AssertionError(
                f"流式单帧网络与并行模型输出偏差超限：{max_err:.3e} > {atol:.3e}"
            )
        return max_err
    finally:
        if was_training:
            train_model.train()


@torch.no_grad()
def export_torchscript(
    train_model,
    out_path: str | Path,
    batch_size: int = 1,
    verify: bool = True,
    atol: float = 1e-5,
) -> tuple[Path, float | None]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream = train_model.build_streaming_model()
    dummy = stream.dummy_inputs(batch_size=batch_size)
    traced = torch.jit.trace(stream, dummy, strict=False)
    traced.save(str(out_path))

    max_err = None
    if verify:
        loaded = torch.jit.load(str(out_path), map_location=train_model.comp.device)
        loaded.eval()
        max_err = verify_streaming_equivalence(
            train_model, step_fn=loaded, batch_size=batch_size, atol=atol
        )
    return out_path, max_err


@torch.no_grad()
def export_onnx(
    train_model,
    out_path: str | Path,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    verify: bool = True,
    atol: float = 1e-4,
) -> tuple[Path, float | None]:
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError("导出 ONNX 需要安装 onnx 包（pip install onnx）") from exc

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream = train_model.build_streaming_model().cpu().eval()
    dummy = stream.dummy_inputs(batch_size=batch_size, device=torch.device("cpu"))

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            name: {0: "batch"}
            for name in (*StreamingSideModel.INPUT_NAMES, *StreamingSideModel.OUTPUT_NAMES)
        }

    export_kwargs = dict(
        input_names=list(StreamingSideModel.INPUT_NAMES),
        output_names=list(StreamingSideModel.OUTPUT_NAMES),
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    try:
        torch.onnx.export(stream, dummy, str(out_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(stream, dummy, str(out_path), **export_kwargs)

    max_err = None
    if verify:
        try:
            ort_step = make_onnxruntime_step_fn(out_path)
        except ImportError:
            ort_step = None
        if ort_step is not None:
            cpu_model = train_model
            if train_model.comp.device.type != "cpu":
                cpu_model = copy.deepcopy(train_model).cpu().eval()
            max_err = verify_streaming_equivalence(
                cpu_model, step_fn=ort_step, batch_size=batch_size, atol=atol
            )
    return out_path, max_err
