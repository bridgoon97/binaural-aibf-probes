"""双耳探针与多版本训练工作流脚本（包外薄封装，复用 binaural_aibf 包）。"""
import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from binaural_aibf import BinauralAIBF
from binaural_aibf.data import check_splits, load_example, read_manifest, validate_manifest
from binaural_aibf.losses import training_loss, training_loss_batch
from binaural_aibf.metrics import scores, summarize
from binaural_aibf.presets import BANDWIDTH_PRESETS, VERSION_PRESETS, preset
from binaural_aibf.streaming import (
    StreamingSideModel,
    export_onnx,
    export_torchscript,
    verify_streaming_equivalence,
)

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(model, mix, shuffle=False):
    if mix.dim() == 2:
        left, right, _ = model(mix[0:1], mix[1:2], shuffle_remote=shuffle)
        return torch.cat([left, right], dim=0)
    left, right, _ = model(mix[:, 0], mix[:, 1], shuffle_remote=shuffle)
    return torch.stack([left, right], dim=1)


def evaluate(model, rows, device, shuffle=False):
    results = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            mix, target = load_example(row, model.cfg["stft"]["sample_rate"])
            mix, target = mix.to(device), target.to(device)
            est = predict(model, mix, shuffle)
            loss = training_loss(est, target, mix, model.stft).item()
            if not np.isfinite(loss):
                raise RuntimeError(f"评测损失非有限：{row['id']}")
            results.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "loss": loss,
                    **scores(est, mix, target, model.stft),
                }
            )
    return results


def save_evaluation(out, model, rows, device):
    normal = evaluate(model, rows, device)
    shuffled = evaluate(model, rows, device, shuffle=True)
    delta = paired_delta(normal, shuffled)
    write_json(out / "per_sample.json", normal)
    write_json(out / "shuffled_per_sample.json", shuffled)
    write_json(
        out / "metrics.json",
        {
            "normal": summarize(normal),
            "shuffled": summarize(shuffled),
            "normal_minus_shuffled": summarize(delta),
            "bitrate": asdict(model.bitrate_report()),
            "note": "消息诊断将远端帧循环平移半段。正的 gain 差表示正常消息更好。",
        },
    )
    return normal


def paired_delta(left, right):
    by_id = {r["id"]: r for r in right}
    result = []
    for a in left:
        b = by_id[a["id"]]
        d = {"id": a["id"], "group": a["group"]}
        for k, v in a.items():
            if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
                d[k] = v - b[k]
        result.append(d)
    return result


def _collate_batch(examples):
    max_len = max(m.shape[-1] for m, _ in examples)
    mix_list, tgt_list = [], []
    for mix, target in examples:
        pad = max_len - mix.shape[-1]
        if pad > 0:
            mix = torch.nn.functional.pad(mix, (0, pad))
            target = torch.nn.functional.pad(target, (0, pad))
        mix_list.append(mix)
        tgt_list.append(target)
    return torch.stack(mix_list, dim=0), torch.stack(tgt_list, dim=0)


def train_model(cfg, train_rows, val_rows, out, args):
    out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model = BinauralAIBF(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=1e-5
        )
        if args.lr_schedule == "cosine"
        else None
    )

    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    pkg_dir = HERE / "binaural_aibf"
    provenance = {
        "train": train_rows,
        "validation": val_rows,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "segment_seconds": args.segment_seconds,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "torch": str(torch.__version__),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(pkg_dir.glob("*.py"))
        },
    }
    write_json(out / "provenance.json", provenance)
    best = float("inf")
    history = []
    sr = cfg["stft"]["sample_rate"]
    count = int(args.segment_seconds * sr)

    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_rows)))
        random.shuffle(indices)
        losses = []
        for start_i in range(0, len(indices), args.batch_size):
            batch_ids = indices[start_i : start_i + args.batch_size]
            examples = []
            for idx in batch_ids:
                mix, target = load_example(train_rows[idx], sr)
                if count > 0 and mix.shape[-1] > count:
                    start = random.randrange(mix.shape[-1] - count + 1)
                    mix = mix[:, start : start + count]
                    target = target[:, start : start + count]
                examples.append((mix, target))
            mix_b, target_b = _collate_batch(examples)
            mix_b, target_b = mix_b.to(args.device), target_b.to(args.device)

            optimizer.zero_grad(set_to_none=True)
            est_b = predict(model, mix_b)
            loss, _ = training_loss_batch(est_b, target_b, mix_b, model.stft)
            if not torch.isfinite(loss):
                raise RuntimeError(f"训练损失非有限：批次 {batch_ids}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(loss.item())

        cur_lr = float(optimizer.param_groups[0]["lr"])
        if scheduler is not None:
            scheduler.step()

        validation = evaluate(model, val_rows, args.device)
        value = summarize(validation)["loss"]["mean"]
        record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_loss": value,
            "lr": cur_lr,
        }
        history.append(record)
        payload = {
            "model": model.state_dict(),
            "config": cfg,
            "epoch": epoch + 1,
            "provenance": provenance,
            "validation_loss": value,
        }
        torch.save(payload, out / "last.pt")
        if value < best:
            best = value
            torch.save(payload, out / "best.pt")
        write_json(out / "history.json", history)
        print(
            f"{out.name} 轮次 {epoch + 1}/{args.epochs} "
            f"训练 {record['train_loss']:.5f} 验证 {value:.5f} lr {cur_lr:.2e}",
            flush=True,
        )
    return load_checkpoint(out / "best.pt", args.device)[0]


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BinauralAIBF(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="双耳探针与多版本训练")
    sub = parser.add_subparsers(dest="command", required=True)
    all_presets = list(dict.fromkeys([*BANDWIDTH_PRESETS, *VERSION_PRESETS]))
    for command in ("validate", "train", "probe", "evaluate", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", default=str(HERE / "config.yaml"))
        p.add_argument("--device", default="cpu")
        p.add_argument("--threads", type=int, default=1)
        if command in ("validate", "train", "probe"):
            p.add_argument("--train", required=True)
            p.add_argument("--val", required=True)
        if command in ("validate", "probe", "evaluate"):
            p.add_argument("--test", required=True)
        if command not in ("validate", "export"):
            p.add_argument("--out", required=True)
        if command in ("train", "probe"):
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--segment-seconds", type=float, default=3.0)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--lr", type=float, default=0.001)
            p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
        if command == "train":
            p.add_argument(
                "--preset",
                default=None,
                choices=all_presets,
                help="可选：显式指定预设档位名；未传时直接使用 --config 内容",
            )
        if command == "probe":
            p.add_argument(
                "--kind", choices=["versions", "delay", "bandwidth"], required=True
            )
            p.add_argument(
                "--versions",
                nargs="+",
                default=list(VERSION_PRESETS),
                help="--kind versions 时对比的预设名列表",
            )
            p.add_argument(
                "--delays", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16]
            )
        if command == "evaluate":
            p.add_argument("--checkpoint", required=True)
            p.add_argument(
                "--delay-frames", type=int, help="仅改推理延迟，属于鲁棒性扫描，不是重训结果"
            )
        if command == "export":
            p.add_argument(
                "--checkpoint", help="训练检查点路径；省略时按 --config 初始化导出结构验证"
            )
            p.add_argument("--onnx", help="导出单帧流式 ONNX 文件路径")
            p.add_argument("--torchscript", help="导出单帧流式 TorchScript 文件路径")
            p.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
            p.add_argument(
                "--no-verify", action="store_true", help="跳过导出后并行与流式等价性校验"
            )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads 必须大于零")
    torch.set_num_threads(args.threads)

    if args.command == "export":
        if not args.onnx and not args.torchscript:
            parser.error("export 至少需要指定 --onnx 或 --torchscript 之一")
        if args.checkpoint:
            model, _ = load_checkpoint(args.checkpoint, args.device)
        else:
            model = BinauralAIBF(config(args.config)).to(args.device)
        model.eval()
        verify = not args.no_verify
        py_err = verify_streaming_equivalence(model) if verify else None
        if py_err is not None:
            print(f"PyTorch 流式单帧校验通过，最大波形误差 {py_err:.2e}", flush=True)
        if args.torchscript:
            ts_path, ts_err = export_torchscript(model, args.torchscript, verify=verify)
            msg = f"TorchScript 已导出：{ts_path}"
            if ts_err is not None:
                msg += f"，校验最大误差 {ts_err:.2e}"
            print(msg, flush=True)
        if args.onnx:
            onnx_path, onnx_err = export_onnx(
                model, args.onnx, opset_version=args.opset, verify=verify
            )
            msg = f"ONNX 已导出：{onnx_path}"
            if onnx_err is not None:
                msg += f"，onnxruntime 校验最大误差 {onnx_err:.2e}"
            print(msg, flush=True)
        print(
            f"单帧输入：{list(StreamingSideModel.INPUT_NAMES)}\n"
            f"单帧输出：{list(StreamingSideModel.OUTPUT_NAMES)}",
            flush=True,
        )
        return

    cfg = config(args.config)
    if args.command in ("train", "probe") and (
        args.epochs < 1 or args.batch_size < 1 or args.segment_seconds < 0 or args.lr <= 0
    ):
        parser.error("epochs、batch-size 与 lr 必须大于零，segment-seconds 不得为负")

    splits = {
        key: read_manifest(getattr(args, key))
        for key in ("train", "val", "test")
        if hasattr(args, key)
    }
    check_splits(*splits.values())
    if args.command == "evaluate":
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        cfg = model.cfg
        check_splits(
            checkpoint["provenance"]["train"],
            checkpoint["provenance"]["validation"],
            splits["test"],
        )
    for key, rows in splits.items():
        print(key, validate_manifest(rows, cfg["stft"]["sample_rate"]), flush=True)
    if args.command == "validate":
        return

    out = Path(args.out)
    if args.command == "train":
        run_cfg = preset(cfg, args.preset) if args.preset else cfg
        train_model(run_cfg, splits["train"], splits["val"], out, args)
    elif args.command == "evaluate":
        if args.delay_frames is not None:
            if args.delay_frames < 0:
                parser.error("delay-frames 不得为负")
            updated = copy.deepcopy(model.cfg)
            updated["message"]["delay_frames"] = args.delay_frames
            changed = BinauralAIBF(updated).to(args.device)
            changed.load_state_dict(model.state_dict())
            model = changed
        out.mkdir(parents=True, exist_ok=False)
        write_json(
            out / "evaluation.json",
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "config": model.cfg,
                "test": splits["test"],
                "mode": "推理延迟改动" if args.delay_frames is not None else "原训练配置",
            },
        )
        save_evaluation(out, model, splits["test"], args.device)
    else:
        if any(d < 0 for d in args.delays) or len(set(args.delays)) != len(args.delays):
            parser.error("延迟必须非负且不重复")
        out.mkdir(parents=True, exist_ok=False)
        if args.kind == "versions":
            names = list(args.versions)
        elif args.kind == "delay":
            names = ["local"] + [f"delay_{d}" for d in args.delays]
        else:
            names = list(BANDWIDTH_PRESETS)

        all_results = {}
        for name in names:
            model = train_model(
                preset(cfg, name), splits["train"], splits["val"], out / name, args
            )
            all_results[name] = save_evaluation(
                out / name, model, splits["test"], args.device
            )
        ref_name = "local" if "local" in all_results else names[0]
        comparison = {
            name: summarize(paired_delta(rows, all_results[ref_name]))
            for name, rows in all_results.items()
            if name != ref_name
        }
        write_json(out / "comparison_to_local.json", comparison)
        write_json(
            out / "experiment.json",
            {
                "kind": args.kind,
                "reference": ref_name,
                "variants": names,
                "mode": "每档独立重训，同一随机种子与训练轮数",
                "test": splits["test"],
                "note": "置信区间按场景组重采样，不能覆盖训练随机性。正式结论请重复多个 seed。",
            },
        )
        print(
            f"完成。相对 {ref_name} 基线的配对结果：{out / 'comparison_to_local.json'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
