"""逐样本指标。信噪比采用目标能量与重建误差，保留绝对增益。"""
import math
import numpy as np
import torch

BANDS = [(125, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def scores(est, mixture, target, stft):
    result = {}
    for ear, name in enumerate(("left", "right")):
        ref = target[ear]
        if ref.square().mean() <= 1e-10:
            result[f"snr_gain_{name}"] = None
        else:
            before = (mixture[ear] - ref).square().sum()
            after = (est[ear] - ref).square().sum()
            result[f"snr_gain_{name}"] = (10 * torch.log10((before + 1e-10) / (after + 1e-10))).item()
        r = stft.analyze(ref[None])[0]
        e = stft.analyze(est[ear:ear + 1])[0]
        x = stft.analyze(mixture[ear:ear + 1])[0]
        freq = stft.bin_freqs(ref.device)
        for lo, hi in BANDS:
            mask = (freq >= lo) & (freq < hi if hi < 8000 else freq <= hi)
            power = r[:, mask].square().sum()
            key = f"gain_{lo}_{hi}_{name}"
            result[key] = None if power <= 1e-10 else (10 * torch.log10(
                ((x[:, mask] - r[:, mask]).square().sum() + 1e-10) /
                ((e[:, mask] - r[:, mask]).square().sum() + 1e-10))).item()
    e, r = stft.analyze(est), stft.analyze(target)
    em, rm = e.square().sum(-1).sqrt(), r.square().sum(-1).sqrt()
    active = (rm[0] > rm[0].max() * .0031623) & (rm[1] > rm[1].max() * .0031623)
    if active.any():
        ild_e = 20 * torch.log10((em[0] + 1e-8) / (em[1] + 1e-8))
        ild_r = 20 * torch.log10((rm[0] + 1e-8) / (rm[1] + 1e-8))
        result['ild_mae_db'] = (ild_e - ild_r).abs()[active].mean().item()
        phase_e = torch.atan2(e[..., 1], e[..., 0])
        phase_r = torch.atan2(r[..., 1], r[..., 0])
        d = phase_e[0] - phase_e[1] - phase_r[0] + phase_r[1]
        error = torch.atan2(d.sin(), d.cos()).abs()
        freq = stft.bin_freqs(est.device)
        for lo, hi in BANDS:
            mask = active & ((freq >= lo) & (freq < hi if hi < 8000 else freq <= hi))[None]
            result[f'ipd_mae_rad_{lo}_{hi}'] = error[mask].mean().item() if mask.any() else None
    result['output_rms'] = est.square().mean().sqrt().item()
    return result


def summarize(rows):
    # 场景组内先平均，再对场景组等权汇总。固定种子的组级自举。
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (float, int))})
    rng = np.random.default_rng(0)
    out = {}
    for key in keys:
        groups = {}
        for row in rows:
            value = row.get(key)
            if value is not None and math.isfinite(value):
                groups.setdefault(row['group'], []).append(value)
        values = np.array([np.mean(v) for v in groups.values()])
        if not len(values):
            continue
        ci = None
        if len(values) >= 2:
            boot = rng.choice(values, (1000, len(values)), replace=True).mean(1)
            ci = np.quantile(boot, [.025, .975]).tolist()
        out[key] = {"mean": float(values.mean()), "groups": len(values), "ci95": ci}
    return out
