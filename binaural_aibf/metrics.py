"""逐样本指标与场景组级汇总。信噪比保留绝对增益，跳过直流 bin 0。"""
import math

import numpy as np
import torch

from .bands import model_bin_freqs, remove_dc_waveform, strip_dc

BANDS = [(125, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def scores(est, mixture, target, stft):
    result = {}
    est_ac = remove_dc_waveform(est, stft)
    mix_ac = remove_dc_waveform(mixture, stft)
    target_ac = remove_dc_waveform(target, stft)
    e = strip_dc(stft.analyze(est))
    x = strip_dc(stft.analyze(mixture))
    r = strip_dc(stft.analyze(target))
    freq = model_bin_freqs(stft, device=est.device)
    for ear, name in enumerate(("left", "right")):
        ref_w = target_ac[ear]
        if ref_w.square().mean() <= 1e-10:
            result[f"snr_gain_{name}"] = None
        else:
            before = (mix_ac[ear] - ref_w).square().sum()
            after = (est_ac[ear] - ref_w).square().sum()
            result[f"snr_gain_{name}"] = (
                10 * torch.log10((before + 1e-10) / (after + 1e-10))
            ).item()
        re, ee, xe = r[ear], e[ear], x[ear]
        for lo, hi in BANDS:
            mask = (freq >= lo) & (freq < hi if hi < 8000 else freq <= hi)
            power = re[:, mask].square().sum()
            key = f"gain_{lo}_{hi}_{name}"
            result[key] = (
                None
                if power <= 1e-10
                else (
                    10
                    * torch.log10(
                        ((xe[:, mask] - re[:, mask]).square().sum() + 1e-10)
                        / ((ee[:, mask] - re[:, mask]).square().sum() + 1e-10)
                    )
                ).item()
            )
    em, rm = e.square().sum(-1).sqrt(), r.square().sum(-1).sqrt()
    active = (rm[0] > rm[0].max() * 0.0031623) & (rm[1] > rm[1].max() * 0.0031623)
    if active.any():
        ild_e = 20 * torch.log10((em[0] + 1e-8) / (em[1] + 1e-8))
        ild_r = 20 * torch.log10((rm[0] + 1e-8) / (rm[1] + 1e-8))
        result["ild_mae_db"] = (ild_e - ild_r).abs()[active].mean().item()
        phase_e = torch.atan2(e[..., 1], e[..., 0])
        phase_r = torch.atan2(r[..., 1], r[..., 0])
        d = phase_e[0] - phase_e[1] - phase_r[0] + phase_r[1]
        error = torch.atan2(d.sin(), d.cos()).abs()
        for lo, hi in BANDS:
            mask = active & ((freq >= lo) & (freq < hi if hi < 8000 else freq <= hi))[None]
            result[f"ipd_mae_rad_{lo}_{hi}"] = error[mask].mean().item() if mask.any() else None
    result["output_rms"] = est_ac.square().mean().sqrt().item()
    return result


def summarize(rows):
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (float, int))})
    rng = np.random.default_rng(0)
    out = {}
    for key in keys:
        groups = {}
        for row in rows:
            value = row.get(key)
            if value is not None and math.isfinite(value):
                groups.setdefault(row["group"], []).append(value)
        values = np.array([np.mean(v) for v in groups.values()])
        if not len(values):
            continue
        ci = None
        if len(values) >= 2:
            boot = rng.choice(values, (1000, len(values)), replace=True).mean(1)
            ci = np.quantile(boot, [0.025, 0.975]).tolist()
        out[key] = {"mean": float(values.mean()), "groups": len(values), "ci95": ci}
    return out
