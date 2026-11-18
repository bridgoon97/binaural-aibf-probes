"""实验语义回归测试。"""
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

if __name__ == '__main__':
    unittest.main()
