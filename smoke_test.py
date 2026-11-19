"""自检脚本。

检查频谱流程并打印扫描表
1. STFT 与 WOLA 合成能重建波形。
2. 跨耳消息通道的延迟与保持语义正确，且不读未来帧。
3. 模型能前向、能反向，输出形状正确。
4. 码率核算与配置一致。

运行
    python3 smoke_test.py
"""

import copy

import torch
import yaml

from message import CrossEarMessage
from model import BinauralAIBF
from stft import CausalSTFT


def load_cfg(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_stft(cfg):
    s = cfg["stft"]
    st = CausalSTFT(s["sample_rate"], s["hop"], s["win"], s["n_fft"])
    x = torch.randn(2, 4000)
    spec = st.analyze(x)
    y = st.synthesize(spec)
    # 只检查完全重叠的内部区间，首尾不满 win 的部分不参与
    lo, hi = s["win"], y.shape[1] - s["win"]
    err = (y[:, lo:hi] - x[:, lo:hi]).abs().max().item()
    print(f"[STFT] 频点 {st.n_bins}，帧率 {st.frame_rate:.0f} fps，"
          f"频点间隔 {st.bin_hz:.2f} Hz")
    print(f"[STFT] 前端算法延迟 {1000 * s['win'] / s['sample_rate']:.1f} ms")
    print(f"[STFT] 内部区间重建最大误差 {err:.2e}")
    assert err < 1e-4, "WOLA 重建误差过大"


def test_message(cfg):
    centers = torch.arange(64, dtype=torch.float32) * 62.5
    ch = CrossEarMessage(cfg, centers, 200.0)
    t = 24
    spec = torch.zeros(1, t, 64, 2)
    # 每帧幅度等于帧号，便于核对交付的是哪一帧
    for i in range(t):
        spec[0, i, :, 0] = float(i + 1)
    recv, matched, valid, age, _ = ch.deliver(spec)

    u, d = ch.update_period, ch.delay_frames
    for i in range(t):
        want = max(((i - d) // u) * u, 0) if i - d >= 0 else 0
        assert matched[i].item() == want, f"帧 {i} 配对索引错: {matched[i]} != {want}"
        if i - d >= 0:
            assert valid[0, i, 0].item() == 1.0
            assert age[i].item() == i - want
            # 交付的内容必须来自配对帧，不能来自未来
            assert matched[i].item() <= i - d
        else:
            assert valid[0, i, 0].item() == 0.0
    print(f"[消息] 更新周期 {u} 帧，延迟 {d} 帧，年龄范围 "
          f"{age[d:].min().item()} 到 {age[d:].max().item()} 帧")
    print(f"[消息] {ch.bitrate().describe()}")


def test_model(cfg):
    m = BinauralAIBF(cfg)
    n_param = sum(p.numel() for p in m.parameters())
    xl, xr = torch.randn(1, 3200), torch.randn(1, 3200)
    yl, yr, info = m(xl, xr)
    assert yl.shape == xl.shape and yr.shape == xr.shape, "输出长度与输入不一致"
    loss = yl.pow(2).mean() + yr.pow(2).mean()
    loss.backward()
    n_grad = sum(1 for p in m.parameters() if p.grad is not None)
    print(f"[模型] 参数 {n_param:,}，频带 {info['n_bands']}，"
          f"频点 {info['n_bins']}")
    print(f"[模型] 前端延迟 {info['frontend_delay_ms']:.1f} ms，"
          f"链路延迟 {info['link_delay_ms']:.1f} ms")
    print(f"[模型] 输出形状 {tuple(yl.shape)}，收到梯度的参数张量 {n_grad}")
    assert n_grad > 0, "没有任何参数收到梯度"


def test_causality(cfg):
    """扰动输入尾部，检查早期输出不变。这是最关键的正确性属性。"""
    from bands import compress_complex

    m = BinauralAIBF(cfg).eval()
    hop, win = cfg["stft"]["hop"], cfg["stft"]["win"]
    s, cut = 2400, 1600
    xl, xr = torch.randn(1, s), torch.randn(1, s)

    def spec_out(wl, wr):
        xa, xb = m.stft.analyze(wl), m.stft.analyze(wr)
        return m._run_side(xa, compress_complex(xa, m.comp), compress_complex(xb, m.comp))

    with torch.no_grad():
        y0 = spec_out(xl, xr)
        xl2, xr2 = xl.clone(), xr.clone()
        xl2[:, cut:] += 5.0
        xr2[:, cut:] += 5.0
        y1 = spec_out(xl2, xr2)

    # 左补 win-hop 后，帧 t 的输入截止于 (t+1)*hop
    safe_t = cut // hop
    d_safe = (y0[:, :safe_t] - y1[:, :safe_t]).abs().max().item()
    d_after = (y0[:, safe_t:] - y1[:, safe_t:]).abs().max().item()
    print(f"[因果] 安全帧 {safe_t} / 总帧 {y0.shape[1]}，"
          f"安全区间最大差 {d_safe:.2e}，扰动区间最大差 {d_after:.2e}")
    assert d_safe < 1e-6, "因果性被破坏，早期输出受未来输入影响"
    assert d_after > 1e-4, "扰动没有传播，这个测试本身无效"


def test_streaming_equivalence(cfg):
    """验证全帧并行模型与单帧流式构造网络的整段波形等价性。"""
    from streaming import verify_streaming_equivalence

    m = BinauralAIBF(cfg).eval()
    max_err = verify_streaming_equivalence(m, seconds=0.2, batch_size=1, atol=1e-5)
    print(f"[流式] 并行训练与单帧流式波形最大差 {max_err:.2e}")


def test_sweeps(cfg):
    print("\n[实验一] 延迟扫描，其余字段固定")
    for d in [0, 1, 2, 4, 8, 16]:
        c = copy.deepcopy(cfg)
        c["message"]["delay_frames"] = d
        ms = 1000.0 * d * c["stft"]["hop"] / c["stft"]["sample_rate"]
        print(f"  delay_frames={d:2d}  单向延迟 {ms:5.1f} ms")

    print("\n[实验二] 带宽扫描，延迟固定")
    rows = [
        ("complex", [0, 1000], 8, 4, 4, 4),
        ("complex", [0, 1000], 8, 4, 4, 1),
        ("complex", [0, 1000], 16, 4, 4, 4),
        ("complex", [0, 1000], 8, 6, 6, 4),
        ("complex", [0, 2000], 16, 4, 4, 4),
        ("logmag", [0, 8000], 16, 4, 0, 4),
        ("logmag", [0, 8000], 8, 4, 0, 8),
    ]
    for payload, rng, nb, bm, bp, up in rows:
        c = copy.deepcopy(cfg)
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
    test_sweeps(cfg)
    print("\n全部自检通过")
