"""数据检查、训练、评测及探针矩阵入口。"""
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

from data import read_manifest, check_splits, load_example, validate_manifest
from losses import training_loss
from metrics import scores, summarize
from model import BinauralAIBF
from streaming import (
    StreamingSideModel,
    export_onnx,
    export_torchscript,
    verify_streaming_equivalence,
)

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config(path):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def preset(cfg, name):
    c = copy.deepcopy(cfg)
    m = c['message']
    m['enabled'] = name != 'local'
    if name == 'float':
        m.update(payload='complex', band_range=[0, 8000], n_msg_bands=c['bands']['n_linear'] + c['bands']['n_tri'],
                 bits_mag=0, bits_phase=0, update_period=1, delay_frames=0)
    elif name in ('complex_2bit', 'complex_6bit'):
        bits = 2 if name == 'complex_2bit' else 6
        m.update(payload='complex', bits_mag=bits, bits_phase=bits)
    elif name == 'slow':
        m['update_period'] = 8
    elif name == 'magnitude':
        m.update(payload='logmag', bits_phase=0)
    elif name == 'matched_rate_magnitude':
        m.update(payload='logmag', bits_phase=0, bits_mag=m['bits_mag'] + m['bits_phase'])
    elif name.startswith('delay_'):
        m['delay_frames'] = int(name.split('_')[1])
    elif name not in ('local', 'default'):
        raise ValueError(f'未知配置 {name}')
    return c


def predict(model, mix, shuffle=False):
    left, right, _ = model(mix[0:1], mix[1:2], shuffle_remote=shuffle)
    return torch.cat([left, right], 0)


def evaluate(model, rows, device, shuffle=False):
    results = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            mix, target = load_example(row, model.cfg['stft']['sample_rate'])
            mix, target = mix.to(device), target.to(device)
            est = predict(model, mix, shuffle)
            loss = training_loss(est, target, mix, model.stft).item()
            if not np.isfinite(loss):
                raise RuntimeError(f"评测损失非有限：{row['id']}")
            results.append({'id': row['id'], 'group': row['group'], 'loss': loss,
                            **scores(est, mix, target, model.stft)})
    return results


def save_evaluation(out, model, rows, device):
    normal = evaluate(model, rows, device)
    shuffled = evaluate(model, rows, device, shuffle=True)
    delta = paired_delta(normal, shuffled)
    write_json(out / 'per_sample.json', normal)
    write_json(out / 'shuffled_per_sample.json', shuffled)
    write_json(out / 'metrics.json', {'normal': summarize(normal), 'shuffled': summarize(shuffled),
               'normal_minus_shuffled': summarize(delta), 'bitrate': asdict(model.bitrate_report()),
               'note': '消息诊断将远端帧循环平移半段。正的 gain 差表示正常消息更好。'})
    return normal


def paired_delta(left, right):
    by_id = {r['id']: r for r in right}
    result = []
    for a in left:
        b = by_id[a['id']]
        d = {'id': a['id'], 'group': a['group']}
        for k, v in a.items():
            if isinstance(v, (int, float)) and isinstance(b.get(k), (int, float)):
                d[k] = v - b[k]
        result.append(d)
    return result


def train_model(cfg, train_rows, val_rows, out, args):
    out.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    model = BinauralAIBF(cfg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    provenance = {'train': train_rows, 'validation': val_rows, 'seed': args.seed,
                  'epochs': args.epochs, 'segment_seconds': args.segment_seconds,
                  'lr': args.lr, 'torch': str(torch.__version__),
                  'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    write_json(out / 'provenance.json', provenance)
    best = float('inf')
    history = []
    for epoch in range(args.epochs):
        model.train()
        indices = list(range(len(train_rows)))
        random.shuffle(indices)
        losses = []
        for idx in indices:
            mix, target = load_example(train_rows[idx], cfg['stft']['sample_rate'])
            count = int(args.segment_seconds * cfg['stft']['sample_rate'])
            if count > 0 and mix.shape[-1] > count:
                start = random.randrange(mix.shape[-1] - count + 1)
                mix, target = mix[:, start:start + count], target[:, start:start + count]
            mix, target = mix.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            est = predict(model, mix)
            loss = training_loss(est, target, mix, model.stft)
            if not torch.isfinite(loss):
                raise RuntimeError(f"训练损失非有限：{train_rows[idx]['id']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, val_rows, args.device)
        value = summarize(validation)['loss']['mean']
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_loss': value}
        history.append(record)
        payload = {'model': model.state_dict(), 'config': cfg, 'epoch': epoch + 1,
                   'provenance': provenance, 'validation_loss': value}
        torch.save(payload, out / 'last.pt')
        if value < best:
            best = value
            torch.save(payload, out / 'best.pt')
        write_json(out / 'history.json', history)
        print(f"{out.name} 轮次 {epoch + 1}/{args.epochs} 训练 {record['train_loss']:.5f} 验证 {value:.5f}", flush=True)
    return load_checkpoint(out / 'best.pt', args.device)[0]


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = BinauralAIBF(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description='双耳探针实验')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('validate', 'train', 'probe', 'evaluate', 'export'):
        p = sub.add_parser(command)
        p.add_argument('--config', default=str(HERE / 'config.yaml'))
        p.add_argument('--device', default='cpu')
        p.add_argument('--threads', type=int, default=1)
        if command in ('validate', 'train', 'probe'):
            p.add_argument('--train', required=True)
            p.add_argument('--val', required=True)
        if command in ('validate', 'probe', 'evaluate'):
            p.add_argument('--test', required=True)
        if command not in ('validate', 'export'):
            p.add_argument('--out', required=True)
        if command in ('train', 'probe'):
            p.add_argument('--epochs', type=int, default=20)
            p.add_argument('--segment-seconds', type=float, default=1.0)
            p.add_argument('--seed', type=int, default=0)
            p.add_argument('--lr', type=float, default=.001)
        if command == 'train':
            p.add_argument('--preset', default='default', choices=['local', 'float', 'default', 'complex_2bit', 'complex_6bit', 'slow', 'magnitude', 'matched_rate_magnitude'])
        if command == 'probe':
            p.add_argument('--kind', choices=['delay', 'bandwidth'], required=True)
            p.add_argument('--delays', type=int, nargs='+', default=[0, 1, 2, 4, 8, 16])
        if command == 'evaluate':
            p.add_argument('--checkpoint', required=True)
            p.add_argument('--delay-frames', type=int, help='仅改推理延迟，属于鲁棒性扫描，不是重训结果')
        if command == 'export':
            p.add_argument('--checkpoint', help='训练检查点路径；省略时按 --config 初始化导出结构验证')
            p.add_argument('--onnx', help='导出单帧流式 ONNX 文件路径')
            p.add_argument('--torchscript', help='导出单帧流式 TorchScript 文件路径')
            p.add_argument('--opset', type=int, default=17, help='ONNX opset 版本')
            p.add_argument('--no-verify', action='store_true', help='跳过导出后并行与流式等价性校验')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads 必须大于零')
    torch.set_num_threads(args.threads)
    if args.command == 'export':
        if not args.onnx and not args.torchscript:
            parser.error('export 至少需要指定 --onnx 或 --torchscript 之一')
        if args.checkpoint:
            model, _ = load_checkpoint(args.checkpoint, args.device)
        else:
            model = BinauralAIBF(config(args.config)).to(args.device)
        model.eval()
        verify = not args.no_verify
        py_err = verify_streaming_equivalence(model) if verify else None
        if py_err is not None:
            print(f'PyTorch 流式单帧校验通过，最大波形误差 {py_err:.2e}', flush=True)
        if args.torchscript:
            ts_path, ts_err = export_torchscript(model, args.torchscript, verify=verify)
            msg = f'TorchScript 已导出：{ts_path}'
            if ts_err is not None:
                msg += f'，校验最大误差 {ts_err:.2e}'
            print(msg, flush=True)
        if args.onnx:
            onnx_path, onnx_err = export_onnx(
                model, args.onnx, opset_version=args.opset, verify=verify
            )
            msg = f'ONNX 已导出：{onnx_path}'
            if onnx_err is not None:
                msg += f'，onnxruntime 校验最大误差 {onnx_err:.2e}'
            print(msg, flush=True)
        print(
            f'单帧输入：{list(StreamingSideModel.INPUT_NAMES)}\n'
            f'单帧输出：{list(StreamingSideModel.OUTPUT_NAMES)}',
            flush=True,
        )
        return
    cfg = config(args.config)
    if args.command in ('train', 'probe') and (args.epochs < 1 or args.segment_seconds < 0 or args.lr <= 0):
        parser.error('epochs 与 lr 必须大于零，segment-seconds 不得为负')
    splits = {key: read_manifest(getattr(args, key)) for key in ('train', 'val', 'test') if hasattr(args, key)}
    check_splits(*splits.values())
    if args.command == 'evaluate':
        model, checkpoint = load_checkpoint(args.checkpoint, args.device)
        cfg = model.cfg
        check_splits(checkpoint['provenance']['train'], checkpoint['provenance']['validation'], splits['test'])
    for key, rows in splits.items():
        print(key, validate_manifest(rows, cfg['stft']['sample_rate']), flush=True)
    if args.command == 'validate':
        return
    out = Path(args.out)
    if args.command == 'train':
        train_model(preset(cfg, args.preset), splits['train'], splits['val'], out, args)
    elif args.command == 'evaluate':
        if args.delay_frames is not None:
            if args.delay_frames < 0:
                parser.error('delay-frames 不得为负')
            updated = copy.deepcopy(model.cfg)
            updated['message']['delay_frames'] = args.delay_frames
            changed = BinauralAIBF(updated).to(args.device)
            changed.load_state_dict(model.state_dict())
            model = changed
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / 'evaluation.json', {'checkpoint': str(Path(args.checkpoint).resolve()),
                   'config': model.cfg, 'test': splits['test'], 'mode': '推理延迟改动' if args.delay_frames is not None else '原训练配置'})
        save_evaluation(out, model, splits['test'], args.device)
    else:
        if any(d < 0 for d in args.delays) or len(set(args.delays)) != len(args.delays):
            parser.error('延迟必须非负且不重复')
        out.mkdir(parents=True, exist_ok=False)
        names = ['local'] + ([f'delay_{d}' for d in args.delays] if args.kind == 'delay' else
                             ['float', 'complex_2bit', 'default', 'complex_6bit', 'slow', 'magnitude', 'matched_rate_magnitude'])
        all_results = {}
        for name in names:
            model = train_model(preset(cfg, name), splits['train'], splits['val'], out / name, args)
            all_results[name] = save_evaluation(out / name, model, splits['test'], args.device)
        comparison = {name: summarize(paired_delta(rows, all_results['local'])) for name, rows in all_results.items() if name != 'local'}
        write_json(out / 'comparison_to_local.json', comparison)
        write_json(out / 'experiment.json', {'kind': args.kind, 'variants': names,
                   'mode': '每档独立重训，同一随机种子与训练轮数', 'test': splits['test'],
                   'note': '置信区间按场景组重采样，不能覆盖训练随机性。正式结论请重复多个 seed。'})
        print(f'完成。相对本地基线的配对结果：{out / "comparison_to_local.json"}', flush=True)


if __name__ == '__main__':
    main()
