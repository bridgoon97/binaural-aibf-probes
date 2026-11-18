"""生成仅用于链路验收的合成数据，不用于论证双耳算法收益。"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=.2)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('时长必须大于零')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(7)
    sr = 16000
    t = np.arange(max(1, int(sr * args.seconds))) / sr
    for split, count in [('train', 4), ('val', 2), ('test', 2)]:
        rows = []
        for i in range(count):
            voice = .1 * np.sin(2 * np.pi * rng.uniform(180, 450) * t)
            target = np.stack([voice, voice * .9])
            if split == 'train' and i == count - 1:
                target[:] = 0
            noise = rng.normal(0, .03, target.shape)
            mixture = target + noise
            name = f'{split}_{i}'
            np.savez(out / f'{name}.npz', mixture=mixture.astype('float32'), target=target.astype('float32'), sample_rate=sr)
            rows.append({'id': name, 'group': f'{split}_scene_{i}', 'npz': f'{name}.npz'})
        (out / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    print(f'合成验收数据已写入 {out}')


if __name__ == '__main__':
    main()
