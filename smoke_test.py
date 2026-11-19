"""自检脚本（基于 binaural_aibf 独立包）。"""

import copy
from pathlib import Path

import torch
import yaml

from binaural_aibf import BinauralAIBF, CrossEarMessage, VERSION_PRESETS, preset
from binaural_aibf.bands import compress_band, strip_dc
from binaural_aibf.stft import CausalSTFT
from binaural_aibf.streaming import verify_streaming_equivalence

HERE = Path(__file__).resolve().parent


def load_cfg(path=HERE / "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def estimate_core_macs_per_frame(model: BinauralAIBF) -> tuple[int, int]:
    crf = model.crf
    crf_macs = (
        (crf.inp.d_in // crf.inp.groups) * crf.inp.d_out
        + 2 * 3 * (crf.hidden // crf.groups) * crf.hidden * 2
        + (crf.out.d_in // crf.out.groups) * crf.out.d_out
    )
    mvdr = model.mvdr
    if mvdr.film_bridge is not None:
        fb = mvdr.film_bridge
        crf_macs += (
            fb.crf_hidden * fb.n_bands * fb.film_dim
            + fb.n_bands * fb.film_dim * 4 * fb.mvdr_hidden
            + 2 * 16
            + 16
        )

    per_band = (
        (mvdr.inp.d_in // mvdr.inp.groups) * mvdr.inp.d_out
        + mvdr.n_layers * 3 * (mvdr.hidden // mvdr.groups) * mvdr.hidden * 2
        + (mvdr.mid.d_in // mvdr.mid.groups) * mvdr.mid.d_out
        + mvdr.hidden * 2 * mvdr.n_mic
    )
    mvdr_macs = per_band * model.n_bands
    if mvdr.cross_band is not None:
        cb = mvdr.cross_band
        mvdr_macs += (
            model.n_bands * cb.kernel_size * cb.hidden
            + model.n_bands * cb.hidden * cb.hidden
            + model.n_bands * cb.hidden * cb.ch_mid * 2
            + cb.ch_mid * model.n_bands * cb.bottleneck * 2
        )
    return crf_macs, mvdr_macs


def test_stft(cfg):
    s = cfg["stft"]
    st = CausalSTFT(
        s["sample_rate"],
        s["hop"],
        s["win"],
        s["n_fft"],
        synthesis_window_size=s.get("synthesis_window_size"),
    )
    x = torch.randn(2, 4000)
    spec = st.analyze(x)
    y = st.synthesize(spec)
    lo, hi = s["win"], y.shape[1] - s["win"]
    err = (y[:, lo:hi] - x[:, lo:hi]).abs().max().item()
    print(
        f"[STFT] 原始频点 {st.n_bins}，模型频点 {st.n_bins - 1}（跳过 DC），"
        f"帧率 {st.frame_rate:.0f} fps，频点间隔 {st.bin_hz:.2f} Hz"
    )
    print(f"[STFT] 前端算法延迟 {1000 * s['win'] / s['sample_rate']:.1f} ms")
    print(f"[STFT] 内部区间重建最大误差 {err:.2e}")
    assert err < 1e-4, "WOLA 重建误差过大"


def test_message(cfg):
    centers = (torch.arange(64, dtype=torch.float32) + 1) * 62.5
    for ver in ("s0_default", "v1_mixed"):
        c = preset(cfg, ver)
        ch = CrossEarMessage(c, centers, 200.0)
        t = 24
        spec = torch.zeros(1, t, 64, 2)
        for i in range(t):
            spec[0, i, :, 0] = float(i + 1)
        recv, matched, valid, age, _ = ch.deliver(spec)
        u, d = ch.update_period, ch.delay_frames
        for i in range(t):
            want = max(((i - d) // u) * u, 0) if i - d >= 0 else 0
            assert matched[i].item() == want
            if i - d >= 0:
                assert valid[0, i, 0].item() == 1.0
                assert age[i].item() == i - want
                assert matched[i].item() <= i - d
            else:
                assert valid[0, i, 0].item() == 0.0
        print(
            f"[消息:{ver}] 载荷 {ch.payload}，更新周期 {u} 帧，延迟 {d} 帧，"
            f"年龄 {age[d:].min().item()}..{age[d:].max().item()} 帧"
        )
        print(f"  {ch.bitrate().describe()}")


def test_model(cfg):
    for n_mic in (1, 2):
        c = copy.deepcopy(cfg)
        c["local"]["n_mic"] = n_mic
        m = BinauralAIBF(c)
        n_param = sum(p.numel() for p in m.parameters())
        shape = (1, 3200) if n_mic == 1 else (1, n_mic, 3200)
        xl, xr = torch.randn(*shape), torch.randn(*shape)
        yl, yr, info = m(xl, xr)
        assert yl.shape == (1, 3200) and yr.shape == (1, 3200)
        loss = yl.pow(2).mean() + yr.pow(2).mean()
        loss.backward()
        n_grad = sum(1 for p in m.parameters() if p.grad is not None)
        print(
            f"[模型 n_mic={n_mic}] 参数 {n_param:,}，频带 {info['n_bands']}，"
            f"频点 {info['n_bins']}，通道 {m.n_ch}，梯度张量 {n_grad}"
        )
        assert n_grad > 0


def test_causality(cfg):
    m = BinauralAIBF(cfg).eval()
    hop = cfg["stft"]["hop"]
    s, cut = 2400, 1600
    xl, xr = torch.randn(1, s), torch.randn(1, s)

    def spec_out(wl, wr):
        xa = strip_dc(m.stft.analyze(wl))
        xb = strip_dc(m.stft.analyze(wr))
        return m._run_side(
            xa,
            compress_band(xa, m.comp, m.n_linear),
            compress_band(xb, m.comp, m.n_linear),
        )

    with torch.no_grad():
        y0 = spec_out(xl, xr)
        xl2, xr2 = xl.clone(), xr.clone()
        xl2[:, cut:] += 5.0
        xr2[:, cut:] += 5.0
        y1 = spec_out(xl2, xr2)

    safe_t = cut // hop
    d_safe = (y0[:, :safe_t] - y1[:, :safe_t]).abs().max().item()
    d_after = (y0[:, safe_t:] - y1[:, safe_t:]).abs().max().item()
    print(
        f"[因果] 安全帧 {safe_t} / 总帧 {y0.shape[1]}，"
        f"安全区间最大差 {d_safe:.2e}，扰动区间最大差 {d_after:.2e}"
    )
    assert d_safe < 1e-6
    assert d_after > 1e-4


def test_streaming_equivalence(cfg):
    for ver in VERSION_PRESETS:
        m = BinauralAIBF(preset(cfg, ver)).eval()
        max_err = verify_streaming_equivalence(m, seconds=0.2, batch_size=1, atol=1e-5)
        print(f"[流式:{ver} n_mic=1] 最大波形差 {max_err:.2e}")
    c2 = copy.deepcopy(cfg)
    c2["local"]["n_mic"] = 2
    m2 = BinauralAIBF(c2).eval()
    err2 = verify_streaming_equivalence(m2, seconds=0.2, batch_size=1, atol=1e-5)
    print(f"[流式:v1_mixed n_mic=2] 最大波形差 {err2:.2e}")


def print_versions_summary(cfg):
    print("\n[四组模型版本总览]")
    for ver in VERSION_PRESETS:
        c = preset(cfg, ver)
        m = BinauralAIBF(c)
        n_param = sum(p.numel() for p in m.parameters())
        crf_macs, mvdr_macs = estimate_core_macs_per_frame(m)
        tot = (crf_macs + mvdr_macs) * 200.0 / 1e6
        br = m.bitrate_report()
        rate_str = (
            "0.00 kbps"
            if not m.communication
            else (
                "浮点无量化"
                if br.payload_kbps_one_way is None
                else f"{br.payload_kbps_one_way:.2f} kbps/向 (双向应用层 {br.app_kbps_two_way:.2f})"
            )
        )
        print(
            f"  {ver:10s} 参数 {n_param:7,d} | 核心算力 {tot:6.2f} MMAC/s "
            f"| MVDR-H={c['mvdr']['hidden']} FiLM={c['mvdr']['use_film']} "
            f"CB={c['mvdr']['use_cross_band']} | 通信: {rate_str}"
        )


def test_sweeps(cfg):
    base = preset(cfg, "s0_default")
    print("\n[实验一] 延迟扫描，其余字段固定")
    for d in [0, 1, 2, 4, 8, 16]:
        c = copy.deepcopy(base)
        c["message"]["delay_frames"] = d
        ms = 1000.0 * d * c["stft"]["hop"] / c["stft"]["sample_rate"]
        print(f"  delay_frames={d:2d}  单向延迟 {ms:5.1f} ms")

    print("\n[实验二] 带宽扫描，延迟固定")
    rows = [
        ("complex", [125, 1000], 8, 4, 4, 4),
        ("complex", [125, 1000], 8, 4, 4, 1),
        ("complex", [62.5, 1000], 16, 4, 4, 4),
        ("complex", [125, 1000], 8, 6, 6, 4),
        ("complex", [62.5, 2000], 16, 4, 4, 4),
        ("logmag", [62.5, 8000], 16, 4, 0, 4),
        ("logmag", [62.5, 8000], 8, 4, 0, 8),
    ]
    for payload, rng, nb, bm, bp, up in rows:
        c = copy.deepcopy(base)
        c["message"].update(
            payload=payload,
            band_range=rng,
            n_msg_bands=nb,
            bits_mag=bm,
            bits_phase=bp,
            update_period=up,
        )
        mdl = BinauralAIBF(c)
        r = mdl.bitrate_report()
        print(
            f"  {payload:8s} {str(rng):12s} 带{nb:3d} 幅{bm}bit 相{bp}bit 周期{up:2d}"
            f"  净载荷 {r.payload_kbps_one_way:6.2f} kbps/向"
            f"  应用层双向 {r.app_kbps_two_way:6.2f} kbps"
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg = load_cfg()
    test_stft(cfg)
    print()
    test_message(cfg)
    print()
    test_model(cfg)
    print()
    test_causality(cfg)
    print()
    test_streaming_equivalence(cfg)
    print_versions_summary(cfg)
    test_sweeps(cfg)
    print("\n全部自检通过")
