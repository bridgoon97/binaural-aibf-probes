"""实验语义回归测试。"""
import copy
import unittest
import tempfile
import wave
import numpy as np
from pathlib import Path
import torch
import yaml
from bands import compress_complex, build_band_matrices
from data import check_splits, load_example
from losses import coherence_penalty, training_loss
from model import BinauralAIBF
from run import preset
from stft import CausalSTFT
from streaming import (
    export_onnx,
    export_torchscript,
    make_onnxruntime_step_fn,
    verify_streaming_equivalence,
)

class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        self.cfg = yaml.safe_load((Path(__file__).parent / 'config.yaml').read_text())

    def test_full_waveform(self):
        st = CausalSTFT(16000, 80, 240, 256)
        for n in (1, 79, 80, 239, 240, 241, 1001):
            x = torch.randn(2, n)
            y = st.synthesize(st.analyze(x), length=n)
            self.assertEqual(y.shape, x.shape)
            torch.testing.assert_close(y, x, atol=2e-6, rtol=1e-5)

    def test_no_communication(self):
        m = BinauralAIBF(preset(self.cfg, 'local')).eval()
        x, r = torch.randn(1, 641), torch.randn(1, 641)
        with torch.no_grad():
            a, b = m(x, r)[0], m(x, r * 0 + 10)[0]
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertEqual(m.bitrate_report().app_kbps_two_way, 0)

    def test_magnitude_phase_invariance(self):
        m = BinauralAIBF(preset(self.cfg, 'magnitude')).eval()
        x = m.stft.analyze(torch.randn(1, 961))
        local = compress_complex(x, m.comp)
        remote = torch.randn_like(local)
        rotated = torch.stack([-remote[..., 1], remote[..., 0]], -1)
        with torch.no_grad():
            a, b = m._run_side(x, local, remote), m._run_side(x, local, rotated)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_frequency_budget_and_delay(self):
        m = BinauralAIBF(preset(self.cfg, 'float'))
        self.assertIsNone(m.bitrate_report().bits_per_message)
        source = torch.randn(1, 8, m.n_bands, 2)
        decoded = m.msg._decode(m.msg._encode(source), source.device)
        torch.testing.assert_close(source, decoded, atol=1e-5, rtol=1e-5)
        d = BinauralAIBF(self.cfg)
        self.assertEqual(d.bitrate_report().bits_per_message, 64)
        self.assertEqual(d.bitrate_report().app_kbps_two_way, 9.6)
        self.assertEqual(int(d.msg.valid_band.sum()), 8)
        source = torch.randn(1, 20, d.n_bands, 2)
        received, matched, valid, age, decoded = d.msg.deliver(source)
        for t in range(20):
            if t < 4:
                self.assertEqual(received[:, t].abs().max().item(), 0)
            else:
                idx = ((t - 4) // 4) * 4
                torch.testing.assert_close(received[:, t], decoded[:, idx])
                self.assertEqual(int(age[t]), t - idx)

    def test_waveform_causality(self):
        m = BinauralAIBF(self.cfg).eval()
        x, r = torch.randn(1, 1601), torch.randn(1, 1601)
        x2, r2 = x.clone(), r.clone()
        x2[:, 1000:] += 3
        r2[:, 1000:] -= 2
        with torch.no_grad():
            a, b = m(x, r)[0], m(x2, r2)[0]
        torch.testing.assert_close(a[:, :760], b[:, :760], atol=1e-6, rtol=0)
        self.assertGreater((a[:, 1000:] - b[:, 1000:]).abs().max().item(), 1e-4)

    def test_losses_and_backward(self):
        m = BinauralAIBF(self.cfg)
        mix = torch.randn(2, 401) * .1
        for target in (torch.zeros_like(mix), mix * .5):
            y, z, _ = m(mix[0:1], mix[1:2])
            loss = training_loss(torch.cat([y, z]), target, mix, m.stft)
            m.zero_grad()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_coherence(self):
        x, y = torch.randn(1, 1000, 4, 2), torch.randn(1, 1000, 4, 2)
        mask = torch.ones(1, 1000, 4)
        self.assertGreater(coherence_penalty(x, x, x, y, mask).item(), .8)
        self.assertEqual(coherence_penalty(x, y, x, y, mask).item(), 0)

    def test_band_edges(self):
        _, e = build_band_matrices(129, 32, 32)
        self.assertEqual(e[32, 0].item(), 0)
        self.assertEqual(e[128, 0].item(), 0)
        torch.testing.assert_close(e.sum(-1), torch.ones(129))

    def test_wav_and_npz_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / 'audio.wav'
            signal = np.arange(34, dtype=np.int16).reshape(17, 2)
            with wave.open(str(wav), 'wb') as f:
                f.setnchannels(2)
                f.setsampwidth(2)
                f.setframerate(16000)
                f.writeframes(signal.tobytes())
            row = {'id': 'wav', 'mixture': str(wav), 'target': str(wav)}
            mix, target = load_example(row, 16000)
            torch.testing.assert_close(mix, torch.tensor(signal.T.astype('float32') / 32768))
            torch.testing.assert_close(mix, target)
            with self.assertRaises(ValueError):
                load_example(row, 24000)
            path = Path(tmp) / 'audio.npz'
            np.savez(path, mixture=mix.numpy(), target=target.numpy(), sample_rate=16000)
            recovered, _ = load_example({'id': 'npz', 'npz': str(path)}, 16000)
            torch.testing.assert_close(recovered, mix)

    def test_split_leakage(self):
        with self.assertRaises(ValueError):
            check_splits([{'id': 'a', 'group': 'room'}], [{'id': 'b', 'group': 'room'}])

    def test_streaming_equivalence_presets(self):
        for name in ('default', 'local', 'float', 'magnitude', 'slow', 'delay_0', 'delay_16'):
            m = BinauralAIBF(preset(self.cfg, name)).eval()
            err = verify_streaming_equivalence(m, seconds=0.2, batch_size=2, atol=1e-5)
            self.assertLess(err, 1e-5, f'{name} 流式偏差过大: {err}')

        # 额外覆盖边界配置：current 对齐、单抽头、关闭当前谱直连与关闭 InstanceNorm
        extra_cases = [
            ('scm', 'local_frame', 'current'),
            ('crf', 'n_taps', 1),
            ('mvdr', 'use_current_spec', False),
            ('scm', 'instance_norm', False),
        ]
        for section, key, value in extra_cases:
            c = copy.deepcopy(self.cfg)
            c[section][key] = value
            m = BinauralAIBF(c).eval()
            err = verify_streaming_equivalence(m, seconds=0.15, batch_size=2, atol=1e-5)
            self.assertLess(err, 1e-5, f'{section}.{key}={value} 流式偏差过大: {err}')

    def test_parallel_vs_streaming_gradients(self):
        """验证全帧并行反向传播与单帧流式逐帧展开的梯度严格等价。"""
        m_par = BinauralAIBF(self.cfg)
        m_str = m_par.build_streaming_model()
        m_str.train()
        for p in m_str.parameters():
            p.requires_grad_(True)

        xl_wav = torch.randn(2, 1600)
        xr_wav = torch.randn(2, 1600)
        xl = m_par.stft.analyze(xl_wav)
        xr = m_par.stft.analyze(xr_wav)
        bl = compress_complex(xl, m_par.comp)
        br = compress_complex(xr, m_par.comp)

        # 并行路径（单侧）
        y_par = m_par._run_side(xl, bl, br)
        loss_par = y_par.pow(2).mean()
        m_par.zero_grad()
        loss_par.backward()

        # 单帧流式逐帧展开路径（同一份输入与延迟消息）
        recv, matched_idx, valid, age, _ = m_par.msg.deliver(br)
        local_matched = bl[:, matched_idx]
        pair_hist, crf_h1, crf_h2, mvdr_h = m_str.init_state(batch_size=xl.shape[0])
        outs = []
        for t in range(xl.shape[1]):
            age_t = torch.full((xl.shape[0], 1), float(age[t].item()))
            out_t, _, pair_hist, crf_h1, crf_h2, mvdr_h = m_str(
                xl[:, t],
                local_matched[:, t],
                recv[:, t],
                valid[:, t],
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
        checked = 0
        for name, p_par in m_par.named_parameters():
            str_name = name
            if str_name.endswith('_l0') and '.grus.' in str_name:
                head, tail = str_name.rsplit('.grus.', 1)
                str_name = f'{head}.cells.{tail[:-3]}'
            self.assertIn(str_name, str_params, f'未找到对应流式参数: {name} -> {str_name}')
            p_str = str_params[str_name]
            self.assertIsNotNone(p_par.grad)
            self.assertIsNotNone(p_str.grad)
            torch.testing.assert_close(p_par.grad, p_str.grad, atol=1e-4, rtol=1e-3)
            checked += 1
        self.assertEqual(checked, len(list(m_par.parameters())))

    def test_export_torchscript_and_onnx_dynamic_batch(self):
        m = BinauralAIBF(self.cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            ts_path, ts_err = export_torchscript(
                m, Path(tmp) / 'side_step.pt', batch_size=1, verify=True, atol=1e-5
            )
            self.assertLess(ts_err, 1e-5)
            # 验证 B=1 导出的 TorchScript 在 B=2 动态批次下依然等价
            ts_loaded = torch.jit.load(str(ts_path))
            err_b2 = verify_streaming_equivalence(
                m, step_fn=ts_loaded, seconds=0.15, batch_size=2, atol=1e-5
            )
            self.assertLess(err_b2, 1e-5)

            onnx_path, onnx_err = export_onnx(
                m, Path(tmp) / 'side_step.onnx', batch_size=1, verify=True, atol=1e-4
            )
            self.assertTrue(onnx_path.is_file())
            if onnx_err is not None:
                self.assertLess(onnx_err, 1e-4)
                ort_step = make_onnxruntime_step_fn(onnx_path)
                err_ort_b2 = verify_streaming_equivalence(
                    m, step_fn=ort_step, seconds=0.15, batch_size=2, atol=1e-4
                )
                self.assertLess(err_ort_b2, 1e-4)

    def test_legacy_cell_checkpoint_loading(self):
        m1 = BinauralAIBF(self.cfg).eval()
        legacy_sd = {}
        for k, v in m1.state_dict().items():
            if k.endswith('_l0') and '.grus.' in k:
                head, tail = k.rsplit('.grus.', 1)
                nk = f'{head}.cells.{tail[:-3]}'
                legacy_sd[nk] = v.clone()
            else:
                legacy_sd[k] = v.clone()
        m2 = BinauralAIBF(self.cfg).eval()
        m2.load_state_dict(legacy_sd, strict=True)
        xl, xr = torch.randn(1, 800), torch.randn(1, 800)
        with torch.no_grad():
            y1l, y1r, _ = m1(xl, xr)
            y2l, y2r, _ = m2(xl, xr)
        torch.testing.assert_close(y1l, y2l, atol=0, rtol=0)
        torch.testing.assert_close(y1r, y2r, atol=0, rtol=0)

if __name__ == '__main__':
    unittest.main()
