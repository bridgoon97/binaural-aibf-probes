"""主仓合入契约与算法语义回归测试（不依赖真实数据集）。"""
import copy
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import torch
import yaml

from binaural_aibf import (
    BANDWIDTH_PRESETS,
    VERSION_PRESETS,
    BinauralAIBF,
    BitrateReport,
    CrossEarMessage,
    preset,
)
from binaural_aibf.bands import build_band_matrices, compress_band, compress_power, strip_dc
from binaural_aibf.complexops import cabs
from binaural_aibf.data import check_splits, load_example
from binaural_aibf.features import build_crf_input, compress_current_spec
from binaural_aibf.losses import (
    binaural_loss,
    coherence_penalty,
    training_loss,
    training_loss_batch,
)
from binaural_aibf.message import quantize_phase_ste
from binaural_aibf.metrics import scores
from binaural_aibf.stft import CausalSTFT
from binaural_aibf.streaming import (
    export_onnx,
    export_torchscript,
    make_onnxruntime_step_fn,
    verify_streaming_equivalence,
)

HERE = Path(__file__).resolve().parent


class MockFrozenSTFT:
    """严格仅提供 §4.2 冻结的 9 个成员，用于验证无隐藏接口依赖。"""

    def __init__(self, sample_rate=16000, hop=80, win=240, n_fft=256):
        self.sample_rate = sample_rate
        self.hop = hop
        self.win = win
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.frame_rate = sample_rate / hop
        self.bin_hz = sample_rate / n_fft
        self._inner = CausalSTFT(sample_rate, hop, win, n_fft)

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner.analyze(x)

    def synthesize(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        return self._inner.synthesize(spec, length=length)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))

    def test_package_relocatable_and_no_flat_imports(self):
        """G1：把 binaural_aibf/ 复制到孤立临时子目录后仍可导入并前向运行，且无平铺导入。"""
        pkg_src = HERE / "binaural_aibf"
        flat_re = re.compile(
            r"^\s*(?:from|import)\s+(?:bands|complexops|data|features|losses|message|metrics|model|modules|presets|stft|streaming)\b",
            re.M,
        )
        for py in pkg_src.glob("*.py"):
            txt = py.read_text(encoding="utf-8")
            self.assertIsNone(flat_re.search(txt), f"发现禁止的平铺导入：{py.name}")

        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "outer_pkg"
            nested.mkdir()
            (nested / "__init__.py").write_text("", encoding="utf-8")
            shutil.copytree(pkg_src, nested / "binaural_aibf")
            code = (
                "import torch, yaml; "
                "from outer_pkg.binaural_aibf import BinauralAIBF, preset; "
                f"cfg = yaml.safe_load(open({repr(str(HERE / 'config.yaml'))})); "
                "m = BinauralAIBF(cfg).eval(); "
                "x = torch.randn(1, 320); "
                "yl, yr, info = m(x, x); "
                "assert yl.shape == (1, 320) and info['n_bins'] == 128"
            )
            res = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp,
                capture_output=True,
                text=True,
                env={"PATH": "", "PYTHONPATH": tmp},
            )
            self.assertEqual(res.returncode, 0, f"孤立路径导入失败：{res.stderr}")

    def test_forbidden_deps_absent(self):
        """G5：包内禁止导入 hydra / omegaconf / lightning / pytorch_lightning。"""
        pkg_src = HERE / "binaural_aibf"
        forb = re.compile(r"\b(hydra|omegaconf|lightning|pytorch_lightning)\b")
        for py in pkg_src.glob("*.py"):
            txt = py.read_text(encoding="utf-8")
            self.assertIsNone(forb.search(txt), f"{py.name} 出现禁止依赖关键词")

    def test_config_7_sections_and_literal_contract(self):
        """G4：config.yaml 及 configs/*.yaml 严格只有 7 个冻结小节，无插值，含 synthesis_window_size。"""
        expected_sections = ["stft", "bands", "local", "message", "crf", "scm", "mvdr"]
        paths = [HERE / "config.yaml", *(HERE / "configs" / f"{v}.yaml" for v in VERSION_PRESETS)]
        for p in paths:
            txt = p.read_text(encoding="utf-8")
            self.assertNotIn("${", txt, f"{p.name} 不得包含插值")
            c = yaml.safe_load(txt)
            self.assertEqual(list(c.keys()), expected_sections, f"{p.name} 顶层小节不符")
            self.assertIn("synthesis_window_size", c["stft"])
            self.assertIsNone(c["stft"]["synthesis_window_size"])

        st_none = CausalSTFT(16000, 80, 240, 256, synthesis_window_size=None)
        st_same = CausalSTFT(16000, 80, 240, 256, synthesis_window_size=240)
        x = torch.randn(1, 320)
        torch.testing.assert_close(
            st_none.synthesize(st_none.analyze(x), length=320),
            st_same.synthesize(st_same.analyze(x), length=320),
        )
        with self.assertRaises(NotImplementedError):
            CausalSTFT(16000, 80, 240, 256, synthesis_window_size=128)

    def test_mock_frozen_stft_replacement(self):
        """§4.2：用仅含 9 个冻结成员的 MockSTFT 替换后，模型、批量损失与指标零报错通过。"""
        for n_mic in (1, 2):
            c = copy.deepcopy(self.cfg)
            c["local"]["n_mic"] = n_mic
            m = BinauralAIBF(c).eval()
            m.stft = MockFrozenSTFT(16000, 80, 240, 256)
            shape = (2, 400) if n_mic == 1 else (2, n_mic, 400)
            xl, xr = torch.randn(*shape), torch.randn(*shape)
            yl, yr, info = m(xl, xr)
            self.assertEqual(yl.shape, (2, 400))
            self.assertEqual(info["n_mic"], n_mic)
            est = torch.stack([yl, yr], dim=1)
            loss_b, parts = training_loss_batch(est, est * 0.5, est, m.stft)
            self.assertTrue(torch.isfinite(loss_b))
            self.assertIn("wave", parts)
            loss_s = training_loss(est[0], est[0] * 0.5, est[0], m.stft)
            self.assertTrue(torch.isfinite(loss_s))
            sc = scores(est[0], est[0], est[0] * 0.5, m.stft)
            self.assertIn("output_rms", sc)

    def test_state_dict_frozen_top_prefixes_and_legacy_load(self):
        """§4.7：顶层 state_dict 模块名严格属于冻结清单，兼容旧版 GroupedGRUCell 加载。"""
        allowed = {"stft", "comp", "expa", "band_centers", "msg", "crf", "scm", "mvdr"}
        for ver in VERSION_PRESETS:
            for n_mic in (1, 2):
                c = preset(self.cfg, ver)
                c["local"]["n_mic"] = n_mic
                m = BinauralAIBF(c)
                prefixes = {k.split(".", 1)[0] for k in m.state_dict().keys()}
                self.assertTrue(prefixes.issubset(allowed), f"{ver}/M={n_mic} 非法前缀：{prefixes - allowed}")

        m1 = BinauralAIBF(preset(self.cfg, "s0_default")).eval()
        legacy_sd = {}
        for k, v in m1.state_dict().items():
            if k.endswith("_l0") and ".grus." in k:
                head, tail = k.rsplit(".grus.", 1)
                legacy_sd[f"{head}.cells.{tail[:-3]}"] = v.clone()
            else:
                legacy_sd[k] = v.clone()
        m2 = BinauralAIBF(preset(self.cfg, "s0_default")).eval()
        m2.load_state_dict(legacy_sd, strict=True)
        xl, xr = torch.randn(1, 800), torch.randn(1, 800)
        with torch.no_grad():
            y1l, y1r, _ = m1(xl, xr)
            y2l, y2r, _ = m2(xl, xr)
        torch.testing.assert_close(y1l, y2l, atol=0, rtol=0)
        torch.testing.assert_close(y1r, y2r, atol=0, rtol=0)

    def test_full_waveform(self):
        st = CausalSTFT(16000, 80, 240, 256)
        for n in (1, 79, 80, 239, 240, 241, 1001):
            x = torch.randn(2, n)
            y = st.synthesize(st.analyze(x), length=n)
            self.assertEqual(y.shape, x.shape)
            torch.testing.assert_close(y, x, atol=2e-6, rtol=1e-5)

    def test_no_communication(self):
        m = BinauralAIBF(preset(self.cfg, "local")).eval()
        x, r = torch.randn(1, 641), torch.randn(1, 641)
        with torch.no_grad():
            a, b = m(x, r)[0], m(x, r * 0 + 10)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(m.bitrate_report().app_kbps_two_way, 0)

    def test_magnitude_phase_invariance(self):
        m = BinauralAIBF(preset(self.cfg, "magnitude")).eval()
        x = strip_dc(m.stft.analyze(torch.randn(1, 961)))
        local = compress_band(x, m.comp, m.n_linear)
        remote = torch.randn_like(local)
        rotated = torch.stack([-remote[..., 1], remote[..., 0]], -1)
        with torch.no_grad():
            a, b = m._run_side(x, local, remote), m._run_side(x, local, rotated)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_frequency_budget_and_delay(self):
        m_float = BinauralAIBF(preset(self.cfg, "float"))
        self.assertIsNone(m_float.bitrate_report().bits_per_message)
        source = torch.randn(1, 8, m_float.n_bands, 2)
        decoded = m_float.msg._decode(m_float.msg._encode(source), source.device)
        torch.testing.assert_close(source, decoded, atol=1e-5, rtol=1e-5)

        d_s0 = BinauralAIBF(preset(self.cfg, "s0_default"))
        self.assertEqual(d_s0.bitrate_report().bits_per_message, 64)
        self.assertEqual(d_s0.bitrate_report().app_kbps_two_way, 9.6)
        self.assertEqual(int(d_s0.msg.valid_band.sum()), 8)

        d_v1 = BinauralAIBF(preset(self.cfg, "v1_mixed"))
        self.assertEqual(d_v1.bitrate_report().bits_per_message, 64)
        self.assertEqual(d_v1.bitrate_report().app_kbps_two_way, 9.6)
        source = torch.randn(1, 20, d_v1.n_bands, 2)
        received, matched, valid, age, decoded = d_v1.msg.deliver(source)
        for t in range(20):
            if t < 4:
                self.assertEqual(received[:, t].abs().max().item(), 0)
            else:
                idx = ((t - 4) // 4) * 4
                torch.testing.assert_close(received[:, t], decoded[:, idx])
                self.assertEqual(int(age[t]), t - idx)

    def test_waveform_causality(self):
        for ver in ("s0_default", "v1_mixed"):
            m = BinauralAIBF(preset(self.cfg, ver)).eval()
            x, r = torch.randn(1, 1601), torch.randn(1, 1601)
            x2, r2 = x.clone(), r.clone()
            x2[:, 1000:] += 3
            r2[:, 1000:] -= 2
            with torch.no_grad():
                a, b = m(x, r)[0], m(x2, r2)[0]
            torch.testing.assert_close(a[:, :760], b[:, :760], atol=1e-6, rtol=0)
            self.assertGreater((a[:, 1000:] - b[:, 1000:]).abs().max().item(), 1e-4)

    def test_streaming_equivalence_all_presets(self):
        """G3：全部 preset 的并行与流式最大波形误差 < 1e-5。"""
        all_names = list(
            dict.fromkeys([*BANDWIDTH_PRESETS, *VERSION_PRESETS, "delay_0", "delay_16"])
        )
        for name in all_names:
            m = BinauralAIBF(preset(self.cfg, name)).eval()
            err = verify_streaming_equivalence(m, seconds=0.18, batch_size=2, atol=1e-5)
            self.assertLess(err, 1e-5, f"{name} 流式偏差过大: {err}")

        extra_cases = [
            ("scm", "local_frame", "current"),
            ("crf", "n_taps", 1),
            ("mvdr", "use_current_spec", False),
            ("mvdr", "normalize_current_spec", False),
            ("scm", "instance_norm", False),
        ]
        for section, key, value in extra_cases:
            c = copy.deepcopy(self.cfg)
            c[section][key] = value
            m = BinauralAIBF(c).eval()
            err = verify_streaming_equivalence(m, seconds=0.15, batch_size=2, atol=1e-5)
            self.assertLess(err, 1e-5, f"{section}.{key}={value} 流式偏差过大: {err}")

    def test_multimic_forward_backward_streaming_and_export(self):
        """§5 & §9：n_mic=1、2、3 前向/反向、仅第0麦通信、并行与流式等价、及 n_mic=2 ONNX/TS 导出。"""
        for n_mic in (1, 2, 3):
            c = copy.deepcopy(self.cfg)
            c["local"]["n_mic"] = n_mic
            m = BinauralAIBF(c)
            self.assertEqual(m.n_mic, n_mic)
            self.assertEqual(m.n_ch, n_mic + 1)
            # 码率与每耳麦数无关
            self.assertEqual(m.bitrate_report().bits_per_message, 64)
            self.assertAlmostEqual(m.bitrate_report().payload_kbps_one_way, 3.20, places=5)

            wav_l = torch.randn(2, n_mic, 640)
            wav_r = torch.randn(2, n_mic, 640)
            yl, yr, info = m(wav_l, wav_r)
            self.assertEqual(yl.shape, (2, 640))
            self.assertEqual(yr.shape, (2, 640))
            self.assertEqual(info["n_mic"], n_mic)

            est = torch.stack([yl, yr], dim=1)
            loss, _ = training_loss_batch(est, est * 0.5, est, m.stft)
            m.zero_grad()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

            # n_mic >= 2 时改变对侧第 1 辅助麦输入不影响本侧跨耳消息（因为只发第 0 麦）
            if n_mic >= 2:
                m.eval()
                wav_r_mod = wav_r.clone()
                wav_r_mod[:, 1:, :] += 2.0
                with torch.no_grad():
                    yl_a = m(wav_l, wav_r)[0]
                    yl_b = m(wav_l, wav_r_mod)[0]
                torch.testing.assert_close(yl_a, yl_b, atol=1e-6, rtol=1e-5)

            # n_mic=1 与 [B, S] 等价校验
            if n_mic == 1:
                m.eval()
                with torch.no_grad():
                    yl_2d = m(wav_l[:, 0], wav_r[:, 0])[0]
                    yl_3d = m(wav_l, wav_r)[0]
                torch.testing.assert_close(yl_2d, yl_3d, atol=0, rtol=0)

            err = verify_streaming_equivalence(m.eval(), seconds=0.18, batch_size=2, atol=1e-5)
            self.assertLess(err, 1e-5, f"n_mic={n_mic} 流式偏差过大: {err}")

        # n_mic=1 与 n_mic=2 均验证 B=1/B=2 动态批次 TorchScript 与 ONNX 导出
        for ver in ("s0_default", "v1_mixed"):
            for n_mic in (1, 2):
                c = preset(self.cfg, ver)
                c["local"]["n_mic"] = n_mic
                m = BinauralAIBF(c).eval()
                m_str = m.build_streaming_model()
                with tempfile.TemporaryDirectory() as tmp:
                    ts_path, ts_err = export_torchscript(
                        m, Path(tmp) / f"{ver}_m{n_mic}.pt", batch_size=1, verify=True, atol=1e-5
                    )
                    self.assertLess(ts_err, 1e-5)
                    ts_loaded = torch.jit.load(str(ts_path))
                    dummy_b2 = m_str.dummy_inputs(batch_size=2)
                    self.assertEqual(tuple(ts_loaded(*dummy_b2)[0].shape), (2, 128, 2))
                    err_b2 = verify_streaming_equivalence(
                        m, step_fn=ts_loaded, seconds=0.15, batch_size=2, atol=1e-5
                    )
                    self.assertLess(err_b2, 1e-5)

                    onnx_path, onnx_err = export_onnx(
                        m, Path(tmp) / f"{ver}_m{n_mic}.onnx", batch_size=1, verify=True, atol=1e-4
                    )
                    self.assertTrue(onnx_path.is_file())
                    self.assertLess(onnx_err, 1e-4)
                    ort_step = make_onnxruntime_step_fn(onnx_path)
                    self.assertEqual(tuple(ort_step(*dummy_b2)[0].shape), (2, 128, 2))
                    err_ort_b2 = verify_streaming_equivalence(
                        m, step_fn=ort_step, seconds=0.15, batch_size=2, atol=1e-4
                    )
                    self.assertLess(err_ort_b2, 1e-4)

    def test_parallel_vs_streaming_gradients(self):
        """验证 s0_default (n_mic=1) 与 v1_mixed (n_mic=1, n_mic=2) 的并行与流式梯度严格等价。"""
        cases = [("s0_default", 1), ("v1_mixed", 1), ("v1_mixed", 2)]
        for ver, n_mic in cases:
            c = preset(self.cfg, ver)
            c["local"]["n_mic"] = n_mic
            m_par = BinauralAIBF(c)
            m_str = m_par.build_streaming_model()
            m_str.train()
            for p in m_str.parameters():
                p.requires_grad_(True)

            shape = (2, 1600) if n_mic == 1 else (2, n_mic, 1600)
            xl_wav = torch.randn(*shape)
            xr_wav = torch.randn(*shape)
            xl_w = xl_wav.unsqueeze(1) if n_mic == 1 else xl_wav
            xr_w = xr_wav.unsqueeze(1) if n_mic == 1 else xr_wav

            xl_f = strip_dc(m_par.stft.analyze(xl_w.reshape(2 * n_mic, 1600)))
            xr_f = strip_dc(m_par.stft.analyze(xr_w.reshape(2 * n_mic, 1600)))
            t = xl_f.shape[1]
            xl = xl_f.reshape(2, n_mic, t, 128, 2).transpose(1, 2)
            xr = xr_f.reshape(2, n_mic, t, 128, 2).transpose(1, 2)
            if n_mic == 1:
                xl, xr = xl.squeeze(2), xr.squeeze(2)
            bl = compress_band(xl, m_par.comp, m_par.n_linear)
            br = compress_band(xr, m_par.comp, m_par.n_linear)
            br_ref = br if n_mic == 1 else br[:, :, 0]

            y_par = m_par._run_side(xl, bl, br_ref)
            loss_par = y_par.pow(2).mean()
            m_par.zero_grad()
            loss_par.backward()

            recv, matched_idx, valid, age, _ = m_par.msg.deliver(br_ref)
            local_matched = bl[:, matched_idx]
            pair_hist, crf_h1, crf_h2, mvdr_h = m_str.init_state(batch_size=2)
            outs = []
            for ti in range(t):
                age_t = torch.full((2, 1), float(age[ti].item()))
                out_t, _, pair_hist, crf_h1, crf_h2, mvdr_h = m_str(
                    xl[:, ti],
                    local_matched[:, ti],
                    recv[:, ti],
                    valid[:, ti],
                    age_t,
                    pair_hist,
                    crf_h1,
                    crf_h2,
                    mvdr_h,
                )
                outs.append(out_t)
            y_str = torch.stack(outs, dim=1)
            torch.testing.assert_close(y_par, y_str, atol=1e-5, rtol=1e-5)

            loss_str = y_str.pow(2).mean()
            m_str.zero_grad()
            loss_str.backward()

            str_params = dict(m_str.named_parameters())
            for name, p_par in m_par.named_parameters():
                str_name = name
                if str_name.endswith("_l0") and ".grus." in str_name:
                    head, tail = str_name.rsplit(".grus.", 1)
                    str_name = f"{head}.cells.{tail[:-3]}"
                torch.testing.assert_close(
                    p_par.grad,
                    str_params[str_name].grad,
                    atol=1e-4,
                    rtol=1e-3,
                    msg=f"{ver}/M={n_mic} 梯度不一致: {name}",
                )

    def test_training_loss_batch_contract(self):
        """§7：批量损失 [B, 2, S] 返回 (total, detached parts)，静音与非静音互不污染。"""
        stft = CausalSTFT(16000, 80, 240, 256)
        mix = torch.randn(3, 2, 480) * 0.1
        tgt = mix * 0.5
        tgt[1] = 0.0  # 批内包含 1 条空目标样本
        est = (mix * 0.4).requires_grad_(True)

        total, parts = training_loss_batch(est, tgt, mix, stft)
        self.assertEqual(total.ndim, 0)
        self.assertTrue(torch.isfinite(total))
        for k in ("wave", "spec", "ild", "ipd", "empty"):
            self.assertIn(k, parts)
            self.assertFalse(parts[k].requires_grad)
            self.assertTrue(torch.isfinite(parts[k]))

        total.backward()
        self.assertTrue(torch.isfinite(est.grad).all())

        # 验证非静音分项严格等于只取非静音子批 [0, 2] 的分项（不受样本 1 静音影响）
        with torch.no_grad():
            _, parts_nonsilent = training_loss_batch(est[[0, 2]], tgt[[0, 2]], mix[[0, 2]], stft)
            for k in ("wave", "spec", "ild", "ipd"):
                torch.testing.assert_close(parts[k], parts_nonsilent[k], atol=1e-6, rtol=1e-6)
            # 对比单样本入口
            l0_single = training_loss(est[0], tgt[0], mix[0], stft)
            l0_batch, _ = training_loss_batch(est[0:1], tgt[0:1], mix[0:1], stft)
            torch.testing.assert_close(l0_single, l0_batch, atol=1e-6, rtol=1e-6)

        # 验证默认 125-1500 Hz IPD 分频过滤
        freqs = torch.arange(1, stft.n_bins, dtype=torch.float32) * stft.bin_hz
        ref = torch.randn(1, 8, 128, 2) + 0.2
        est_hi = ref.clone()
        est_hi[:, :, freqs >= 2000.0, :] *= -1.0
        loss_cfg = {"ipd_freq_range": [125.0, 1500.0], "energy_weighted": True, "energy_power": 0.5}
        _, p0 = binaural_loss(ref, ref, ref, ref, freqs_hz=freqs, loss_cfg=loss_cfg)
        _, p1 = binaural_loss(est_hi, ref, ref, ref, freqs_hz=freqs, loss_cfg=loss_cfg)
        torch.testing.assert_close(p0["ipd"], p1["ipd"], atol=1e-6, rtol=1e-6)

    def test_skip_dc_128_bins(self):
        m = BinauralAIBF(preset(self.cfg, "s0_default")).eval()
        self.assertEqual(m.n_bins, 128)
        self.assertEqual(m.msg.selected_idx.tolist(), [1, 3, 5, 7, 9, 11, 13, 15])
        xl_wav, xr_wav = torch.randn(1, 800), torch.randn(1, 800)
        x_full, r_full = m.stft.analyze(xl_wav), m.stft.analyze(xr_wav)
        x_dc, r_dc = x_full.clone(), r_full.clone()
        x_dc[:, :, 0, :] += 1000.0
        r_dc[:, :, 0, :] -= 1000.0
        with torch.no_grad():
            x0, r0 = strip_dc(x_full), strip_dc(r_full)
            x1, r1 = strip_dc(x_dc), strip_dc(r_dc)
            y0 = m._run_side(x0, compress_band(x0, m.comp, m.n_linear), compress_band(r0, m.comp, m.n_linear))
            y1 = m._run_side(x1, compress_band(x1, m.comp, m.n_linear), compress_band(r1, m.comp, m.n_linear))
            torch.testing.assert_close(y0, y1, atol=0.0, rtol=0.0)

    def test_crf_cross_group_gradient(self):
        for ver in ("s0_default", "v1_mixed"):
            m = BinauralAIBF(preset(self.cfg, ver)).eval()
            feat = torch.randn(1, 10, m.crf.d_in, requires_grad=True)
            fir, _ = m.crf(feat)
            g_lo = torch.autograd.grad(fir[:, :, :32].sum(), feat, retain_graph=True)[0]
            g_hi = torch.autograd.grad(fir[:, :, 32:].sum(), feat)[0]
            self.assertGreater(g_lo[:, :, 129:].abs().max().item(), 1e-6)
            self.assertGreater(g_hi[:, :, :129].abs().max().item(), 1e-6)

    def test_power_pooling_no_cancellation(self):
        m = BinauralAIBF(self.cfg).eval()
        white = torch.randn(1, 16000)
        imp = torch.zeros(1, 1600)
        imp[0, 400] = 1.0
        for wav in (white, imp):
            spec_128 = strip_dc(m.stft.analyze(wav))
            band = compress_band(spec_128, m.comp, m.n_linear)
            mag_band = cabs(band)
            mag_power = compress_power(spec_128, m.comp)
            active = mag_power > 1e-4
            self.assertTrue(active.any())
            diff_db = (20.0 * torch.log10((mag_band + 1e-8) / (mag_power + 1e-8))).abs()
            self.assertLess(diff_db[active].max().item(), 1e-4)

    def test_circular_phase_quantization(self):
        phase = torch.linspace(-torch.pi, torch.pi, 1000, requires_grad=True)
        for bits in (2, 4):
            q = quantize_phase_ste(phase, bits)
            uniq = torch.unique(torch.round(q * 1e5))
            self.assertEqual(uniq.numel(), 2 ** bits)
            diff = torch.atan2(torch.sin(q - phase), torch.cos(q - phase)).abs()
            self.assertLessEqual(diff.max().item(), torch.pi / (2 ** bits) + 1e-5)
        q4 = quantize_phase_ste(phase, 4)
        q4.sum().backward()
        torch.testing.assert_close(phase.grad, torch.ones_like(phase), atol=1e-6, rtol=1e-6)

    def test_current_spec_level_invariance(self):
        x = torch.randn(2, 5, 64, 2) + 0.1
        torch.testing.assert_close(
            compress_current_spec(x * 100.0),
            compress_current_spec(x),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_mixed_payload_64bit_and_quantization(self):
        m = BinauralAIBF(preset(self.cfg, "v1_mixed")).eval()
        br = m.bitrate_report()
        self.assertIsInstance(br, BitrateReport)
        self.assertIsInstance(m.msg, CrossEarMessage)
        self.assertEqual(br.bits_per_message, 64)
        self.assertAlmostEqual(br.payload_kbps_one_way, 3.20, places=5)
        self.assertAlmostEqual(br.app_kbps_two_way, 9.60, places=5)
        p_idx, m_idx = m.msg.phase_idx.tolist(), m.msg.mag_idx.tolist()
        self.assertEqual(len(p_idx), 6)
        self.assertEqual(len(m_idx), 7)
        self.assertTrue(set(p_idx).isdisjoint(set(m_idx)))

    def test_frontal_features_and_film_gate(self):
        c = preset(self.cfg, "v1_mixed")
        m = BinauralAIBF(c).eval()
        band = torch.randn(1, 6, 64, 2)
        band[:, :, 32:, 1] = 0.0
        band[:, :, 32:, 0] = band[:, :, 32:, 0].abs() + 0.1
        feat = build_crf_input(
            c, band, band, band, torch.ones(1, 6, 1), torch.zeros(6),
            m.msg.valid_band, m.msg.valid_band_phase
        )
        self.assertEqual(feat.shape[-1], 325)
        frontal = feat[0, :, -69:-2]
        torch.testing.assert_close(frontal[:, 0], torch.ones_like(frontal[:, 0]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(frontal[:, 1], torch.zeros_like(frontal[:, 1]), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(frontal[:, 2], torch.ones_like(frontal[:, 2]), atol=1e-4, rtol=1e-4)

        crf_h = torch.randn(1, 6, c["crf"]["hidden"])
        g1_0, b1_0, g2_0, b2_0 = m.mvdr.film_bridge(crf_h, torch.zeros(6), torch.zeros(1, 6, 1))
        for t in (g1_0, b1_0, g2_0, b2_0):
            self.assertEqual(t.abs().max().item(), 0.0)

    def test_coherence(self):
        x, y = torch.randn(1, 1000, 4, 2), torch.randn(1, 1000, 4, 2)
        mask = torch.ones(1, 1000, 4)
        self.assertGreater(coherence_penalty(x, x, x, y, mask).item(), 0.8)
        self.assertEqual(coherence_penalty(x, y, x, y, mask).item(), 0)

    def test_band_edges(self):
        _, e = build_band_matrices(128, 32, 32)
        self.assertEqual(e[32, 0].item(), 0)
        self.assertEqual(e[127, 0].item(), 0)
        torch.testing.assert_close(e.sum(-1), torch.ones(128))

    def test_wav_and_npz_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            signal = np.arange(34, dtype=np.int16).reshape(17, 2)
            with wave.open(str(wav), "wb") as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(signal.tobytes())
            row = {"id": "wav", "mixture": str(wav), "target": str(wav)}
            mix, target = load_example(row, 16000)
            torch.testing.assert_close(mix, torch.tensor(signal.T.astype("float32") / 32768))
            torch.testing.assert_close(mix, target)
            with self.assertRaises(ValueError):
                load_example(row, 24000)
            path = Path(tmp) / "audio.npz"
            np.savez(path, mixture=mix.numpy(), target=target.numpy(), sample_rate=16000)
            recovered, _ = load_example({"id": "npz", "npz": str(path)}, 16000)
            torch.testing.assert_close(recovered, mix)

    def test_split_leakage(self):
        with self.assertRaises(ValueError):
            check_splits([{"id": "a", "group": "room"}], [{"id": "b", "group": "room"}])


if __name__ == "__main__":
    unittest.main()
