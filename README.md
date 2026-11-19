# 双耳 AIBF 数据探针与可合入模型包

本仓库包含两层内容：
- **位置无关模型包 `binaural_aibf/`**：可直接复制到主仓作为子包挂载，支持 `local.n_mic >= 1`（每耳单麦或多麦）、128 非直流频点（跳过 DC bin 0）、批量损失 `training_loss_batch`、纯函数 `preset` 及单帧 TorchScript / ONNX 导出。
- **包外工作流与配置**：`config.yaml`（严格 7 小节纯字面量）、`configs/`（4 组对照版本 YAML）、`run.py`、`make_demo.py`、`smoke_test.py`、`test_probe.py`。

首轮任务为固定正前方定向拾音，左右目标保留各自声像。
真实无线打包、丢包与硬件调度仍未包含。

## 仓库分层与主仓合入

```text
binaural_aibf/          # 可直接复制进主仓的模型包（全相对导入，无 hydra/lightning 依赖）
├── __init__.py         # 导出 BinauralAIBF、CrossEarMessage、BitrateReport、preset
├── bands.py            # 低频恒等 + 高频功率池化压缩、去直流/补零助手
├── complexops.py       # 纯实数张量复数运算
├── data.py             # JSONL + NPZ/WAV 读取与防泄漏检验
├── features.py         # 跨耳 IPD/ILD、0° 正前方先验、当前帧压缩归一化谱
├── losses.py           # training_loss_batch([B,2,S])、training_loss([2,S])、binaural_loss
├── message.py          # complex / logmag / mixed(64-bit 混合载荷) 编码与延迟保持
├── metrics.py          # 分带增益与 ILD/IPD 误差
├── model.py            # BinauralAIBF（支持 n_mic >= 1，默认 V1 架构）
├── modules.py          # CRFEstimator(含通道混洗)、NeuralSCM、MVDRBandEstimator(内置 FiLM/CrossBand)
├── presets.py          # 纯函数 preset(cfg, name)
├── stft.py             # CausalSTFT（含可选 synthesis_window_size，仅暴露 9 个冻结接口）
└── streaming.py        # StreamingSideModel、BinauralStreamingRunner、TorchScript/ONNX 导出
config.yaml             # 7 小节纯字面量 schema（stft/bands/local/message/crf/scm/mvdr）
configs/                # local.yaml、float.yaml、s0_default.yaml、v1_mixed.yaml
run.py                  # 训练、评测与探针矩阵工作流
test_probe.py           # 包契约、多麦、因果与动态批次导出测试
```

## 安装与自检

在本目录执行以下命令。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -v test_probe
python smoke_test.py
```

依赖为 PyTorch、NumPy 和 PyYAML。
本轮验收使用 Python 3.12、PyTorch 2.15.0，设备为 CPU。
训练默认使用 CPU，可加 `--device cuda`。
GPU 路径尚未实测。

## 数据契约

准备三个 JSONL 清单：`train.jsonl`、`val.jsonl`、`test.jsonl`。
每行一个样本，路径相对清单所在目录解析。
每段音频必须包含同步的左右两个声道。

推荐 NPZ 格式：

```json
{"id":"train_001","group":"room01_person01","npz":"audio/train_001.npz"}
```

NPZ 包含以下三个数组。

| 键 | 形状与类型 | 含义 |
|---|---|---|
| `mixture` | `[2, S]`，float32 | 含目标、干扰与噪声的左右耳波形 |
| `target` | `[2, S]`，float32 | 正前方区域内目标的左右耳声像 |
| `sample_rate` | 标量整数，16000 | 采样率 |

生成一个 NPZ 的示例：

```python
import numpy as np
np.savez("sample.npz", mixture=mixture.astype(np.float32),
         target=target.astype(np.float32), sample_rate=np.int64(16000))
```

`mixture` 和 `target` 由你的仿真或标注流程提供。
第 0 行为左耳，第 1 行为右耳。
同一条记录的两个数组必须等长且有限。
数值范围为 −1 到 1。
如果需要缩放，对双耳混合与目标使用同一个系数。
不得独立归一化左右耳。

也支持双声道整数 PCM WAV：

```json
{"id":"train_002","group":"room02_person02","mixture":"mix/002.wav","target":"target/002.wav"}
```

WAV 必须为 16 或 32 bit 整数 PCM，采样率为 16 kHz。
浮点 WAV 请先转换为 NPZ。
工具不隐式重采样，也不裁剪不等长声道。
无目标样本的 `target` 全为零。

`id` 在整个实验中唯一。
`group` 定义独立场景，三个集合不得复用。
同一房间响应、说话人或佩戴者的泄漏需由数据划分额外排查。
工具只能检查声明的分组、样本编号和文件路径。
组级置信区间也使用 `group`，不要把每一帧设成独立组。

检查数据：

```bash
python run.py validate --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl
```

检查读取全部文件，报告条目数、场景组数和时长。

## 先跑一个模型与四版本对比

```bash
python run.py train --config configs/v1_mixed.yaml --train data/train.jsonl --val data/val.jsonl --out runs/v1_mixed
python run.py evaluate --checkpoint runs/v1_mixed/best.pt --test data/test.jsonl --out runs/v1_mixed_test
```

可用 4 个纯 7 小节配置文件直接切换版本（也可用 `--preset` 或 `run.py probe --kind versions` 一键跑完）：

```bash
python run.py probe --kind versions --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out runs/versions
```

| 版本 | 配置文件 | 跨耳载荷与码率 | 正前方先验 / FiLM / CrossBand | MVDR hidden |
|---|---|---|---|---:|
| `local` | `configs/local.yaml` | 关闭（0 kbps） | 开 / 开 / 开 | 48 |
| `float` | `configs/float.yaml` | 64 带浮点零延迟（不报码率） | 开 / 开 / 开 | 48 |
| `s0_default` | `configs/s0_default.yaml` | 125–1000 Hz 8 点 `4+4 bit`（`3.20 kbps/向`） | 关 / 关 / 关 | 64 |
| `v1_mixed` | `configs/v1_mixed.yaml` | 6 相位(`5b`) + 7 幅度(`4b`) + `6b` 参考（`3.20 kbps/向`） | 开 / 开 / 开 | 48 |

默认训练使用 `batch_size=8`、`segment_seconds=3.0`、`lr_schedule=cosine`、`epochs=20`、`lr=0.001`。
批内短样本自动右补零对齐；`--segment-seconds 0` 表示使用整段。
验证和测试始终使用整段音频。
训练会保存 `last.pt` 和 `best.pt`。

## 探针一：延迟

```bash
python run.py probe --kind delay --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out runs/delay --epochs 20 --segment-seconds 1
```

自动训练本地基线与 6 个延迟档。
默认延迟为 0、1、2、4、8、16 帧。
一帧为 5 ms，对应 0、5、10、20、40、80 ms。
每档独立重训，使用相同随机种子与轮数。

缩小矩阵可用 `--delays 0 4 16`。
各档保持 `crf.tap_stride` 与年龄归一化尺度不变。

如果只想测固定模型对延迟变化的适应性：

```bash
python run.py evaluate --checkpoint runs/delay/delay_4/best.pt --test data/test.jsonl --out runs/inference_delay8 --delay-frames 8
```

这属于推理鲁棒性测试，不能混入独立重训曲线。

## 探针二：带宽

```bash
python run.py probe --kind bandwidth --train data/train.jsonl --val data/val.jsonl --test data/test.jsonl --out runs/bandwidth --epochs 20 --segment-seconds 1
```

默认配置下自动训练以下 8 档。
码率按 1000 bit/s 计算。

| 档名 | 内容 | 单向净载荷 | 双向应用层 |
|---|---|---:|---:|
| `local` | 严格无通信 | 0 | 0 |
| `float` | 全部模型频带，浮点，逐帧，无延迟 | 不报告 | 不报告 |
| `complex_2bit` | 幅度 2 bit，相位 2 bit | 1.60 kbps | 6.40 kbps |
| `default` | 幅度 4 bit，相位 4 bit | 3.20 kbps | 9.60 kbps |
| `complex_6bit` | 幅度 6 bit，相位 6 bit | 4.80 kbps | 12.80 kbps |
| `slow` | 默认量化，每 8 帧更新 | 1.60 kbps | 4.80 kbps |
| `magnitude` | 幅度 4 bit，不传相位 | 1.60 kbps | 6.40 kbps |
| `matched_rate_magnitude` | 幅度 8 bit，不传相位 | 3.20 kbps | 9.60 kbps |

量化档默认跳过直流 bin 0，使用 128 个非直流频点（62.5–8000 Hz）；`complex` 档默认选取 125–1000 Hz 内 8 个固定模型频带。
除 `slow` 外，每 4 帧更新。链路延迟为 4 帧。
应用层按每包 32 bit 头部，并把整包向上对齐到字节。
真实链路层开销、重传与功耗不在此表。
`float` 仍经过模型频带压缩，不是原始多麦集中式上界。

`default` 与 `magnitude` 固定幅度精度，比较去相位的整体影响。
`default` 与 `matched_rate_magnitude` 比较相同码率的两种分配。
幅度分支的空间矩阵表示也不同，结果不能只归因于相位位宽。

如需其他频率范围或频带数量，复制 `config.yaml`。
使用 `--config 新配置.yaml` 训练。
零位宽表示浮点参照，不表示零比特传输。

## 结果文件与判读

每档输出以下文件。

| 文件 | 内容 |
|---|---|
| `config.yaml` | 实际训练配置 |
| `provenance.json` | 数据清单、随机种子、训练参数、源码哈希 |
| `history.json` | 每轮训练和验证损失 |
| `best.pt`、`last.pt` | 验证最优与最后一轮检查点 |
| `per_sample.json` | 正常消息的逐样本指标 |
| `shuffled_per_sample.json` | 远端消息循环平移半段后的指标 |
| `metrics.json` | 场景组均值、95% 区间、码率与消息诊断 |

探针根目录另有 `comparison_to_local.json`。
该文件先按样本配对，再按场景组汇总相对本地基线的差值。
`experiment.json` 保存实验模式与测试清单。

`snr_gain_left/right` 为混合误差与输出误差的能量比改善量。
它保留绝对增益，不是尺度不变信噪比。
分带改善量覆盖 125–1000、1000–2000、2000–4000、4000–8000 Hz。
同时报告耳间电平差误差和分带圆周相位误差。
空目标的信噪比改善量记为 `null`，另报输出均方根。
空间误差基于含残余噪声的输出，不能解释为纯目标失真。

置信区间按场景组自举 1000 次。
少于 2 组时不报告区间。
少量场景的区间不能支撑泛化结论。
正式结论至少重复多个 `--seed`，各次使用不同输出目录。

先查看正常消息与打乱消息的差值。
如果质量差值接近零，先检查网络是否利用通信。
半段平移在平稳场景中可能过弱，需结合真实动态数据判断。
脚本不据此自动宣告实验成功。

## 网络与损失边界

消息只进入估计支路，最终滤波只作用于当前本耳谱。
左右共享权重，各自保留段内状态。
本地基线保留相同参数尺寸，并切断全部远端声学输入。

消息频带采用固定索引，发送端与接收端使用同一配置。
不再选择峰值相位，也不向其他频点广播相位。
幅度模式屏蔽 IPD，并使用幅度形式构造空间矩阵。

抽头间隔固定为 4 帧，读取历史时刻持有的消息。
同一消息可以进入多个抽头。
改变更新周期会改变消息年龄，但不改变抽头的墙钟间隔。

滤波器模长上限为 2。
短时傅里叶变换补齐首尾，保留任意长度音频。
前端保守延迟预算为 15 ms，尚无硬件测量。

训练损失为增益保真的波形误差，加 0.5 倍谱与双耳损失。
谱与双耳组合采用谱项 1、ILD 项 0.1、IPD 项 0.1。
空目标单独使用按输入能量归一化的输出能量。
相干度函数已改为先沿时间平均互谱与自谱。
总训练损失禁用相干度项，避免混淆目标与残余噪声监督。

## 并行训练与单帧流式导出

训练期在 `model.py` 中并行拼接多路延迟抽头与当前帧数据流。
循环层在 `modules.py` 中改用基于 `nn.GRU` 的 `GroupedGRU`。
`CausalSTFT.synthesize` 改用 `F.fold` 向量化，左右耳合并进批次维一次前向。
新版模型可直接加载旧版 `GroupedGRUCell` 检查点。

导出单帧模型时，调用 `model.build_streaming_model()` 或命令行子命令 `export`。
流式网络 `StreamingSideModel` 位于 `streaming.py`，使用 `GroupedGRUCell` 与显式状态张量。

```bash
python run.py export --checkpoint runs/default/best.pt --onnx export/side_step.onnx --torchscript export/side_step.pt
```

导出脚本默认运行并行模型与单帧流式模型的整段波形比对。
如果省略 `--checkpoint`，脚本按 `--config` 初始化模型并执行结构导出检查。

## 无真实数据时的流程验收

```bash
python make_demo.py --out demo_data --seconds 0.2
python run.py probe --kind versions --train demo_data/train.jsonl --val demo_data/val.jsonl --test demo_data/test.jsonl --out demo_versions --epochs 1 --batch-size 2 --segment-seconds 0.2
python run.py probe --kind delay --delays 0 4 16 --train demo_data/train.jsonl --val demo_data/val.jsonl --test demo_data/test.jsonl --out demo_delay --epochs 1 --batch-size 2 --segment-seconds 0.2
python run.py probe --kind bandwidth --train demo_data/train.jsonl --val demo_data/val.jsonl --test demo_data/test.jsonl --out demo_bandwidth --epochs 1 --batch-size 2 --segment-seconds 0.2
```

演示数据是正弦信号加噪声，只用于检查完整流程。
不得用演示数据的数值论证定向拾音或跨耳收益。

验收范围见 [验收记录](验收记录.md)。
