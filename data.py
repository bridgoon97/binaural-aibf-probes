"""严格的双耳数据清单。支持 NPZ 和双声道 PCM WAV。"""
import json
import wave
from pathlib import Path

import numpy as np
import torch


def read_manifest(path):
    path = Path(path).resolve()
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "group"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"{path}:{line_no} 缺少字符串字段 {key}")
        keys = ("npz",) if "npz" in row else ("mixture", "target")
        for key in keys:
            if key not in row:
                raise ValueError(f"记录 {row['id']} 缺少 {key}")
            row[key] = str((path.parent / row[key]).resolve())
            if not Path(row[key]).is_file():
                raise ValueError(f"文件不存在：{row[key]}")
        rows.append(row)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("清单为空或 id 重复")
    return rows


def check_splits(*splits):
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            for key in ("id", "group"):
                overlap = {r[key] for r in left} & {r[key] for r in right}
                if overlap:
                    raise ValueError(f"集合泄漏：{key} 重复 {sorted(overlap)[:3]}")
            def paths(rows):
                return {r[k] for r in rows for k in ("npz", "mixture", "target") if k in r}
            if paths(left) & paths(right):
                raise ValueError("集合泄漏：音频路径重复")


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        sr, channels, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        if channels != 2 or width not in (2, 4) or f.getcomptype() != "NONE":
            raise ValueError("WAV 必须为双声道、16 或 32 bit 整数 PCM")
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2" if width == 2 else "<i4")
    return raw.reshape(-1, 2).T.astype(np.float32) / (2 ** (width * 8 - 1)), sr


def load_example(row, sample_rate):
    if "npz" in row:
        with np.load(row["npz"], allow_pickle=False) as z:
            mix, target = z["mixture"], z["target"]
            sr = int(z["sample_rate"].item())
    else:
        mix, sr = read_wav(row["mixture"])
        target, target_sr = read_wav(row["target"])
        if sr != target_sr:
            raise ValueError(f"{row['id']} 混合与目标采样率不同")
    if sr != sample_rate:
        raise ValueError(f"{row['id']} 采样率为 {sr}，要求 {sample_rate}，请先重采样")
    if mix.ndim != 2 or mix.shape[0] != 2 or mix.shape != target.shape or mix.shape[1] < 1:
        raise ValueError(f"{row['id']} 要求 mixture/target 均为同形状 [2, 样本数]")
    if not np.isfinite(mix).all() or not np.isfinite(target).all():
        raise ValueError(f"{row['id']} 含非有限值")
    if max(np.abs(mix).max(), np.abs(target).max()) > 1.0001:
        raise ValueError(f"{row['id']} 幅度超出 [-1, 1]，请对双耳混合与目标使用共同缩放")
    return torch.from_numpy(mix.astype(np.float32).copy()), torch.from_numpy(target.astype(np.float32).copy())


def validate_manifest(rows, sample_rate):
    samples = 0
    for row in rows:
        x, _ = load_example(row, sample_rate)
        samples += x.shape[-1]
    return {"条目": len(rows), "场景组": len({r['group'] for r in rows}), "秒数": samples / sample_rate}
