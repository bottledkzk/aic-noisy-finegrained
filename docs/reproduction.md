# 运行与复现

`demo` 使用运行时生成的合成分数，`demo-images` 生成小规模图像并跑通完整流程，`train` 与 `infer` 读取用户准备的原图，`scores` 单独回放既有分数。赛题数据仅限比赛授权范围，演示和后续研究可使用自己的数据。 [赛题数据约束](https://www.aicomp.cn/tracks/tracks-1/3714.html)

## 软件与输入

分数示例需要 Python 3.10+、NumPy 与 SciPy。图像入口还依赖 PyTorch、Torchvision、Transformers 与 Pillow。正式训练用 CUDA，默认 `batch=256`、视觉尺寸 `544`，显存不足时可用 `--batch` 降低批量。正式提交配置使用 RTX 4090。按系统和驱动选择 PyTorch 与 Torchvision，再安装视觉依赖。 [PyTorch 安装指引](https://pytorch.org/get-started/locally/)

```bash
python -m pip install -e '.[vision]'
```

### 用合成图像验证完整流程

Linux 和 Windows 如果只跑 CPU，可以在安装视觉依赖前选用 CPU 版 PyTorch。下面是仓库自动测试使用的版本搭配。 [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v280)

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[vision]'
python -m aic_finegrained demo-images --out runs/image-demo
```

命令需要一个尚未创建的输出目录。它生成 12 张训练图和 6 张查询图，给两张训练图设置错标，并用已知的合成近重复组测试组外监督。每个阶段训练 9 轮，最后保留 `training/student.pt`、`predictions.csv` 和 6×3 的 `canonical.npy`。可把该矩阵交给下文的 `scores` 命令做配对对比。这个小数据集用于验证流程，规模与难度均小于赛题。

### 使用自己的图像和权重

准备一个用 Transformers 格式保存的本地 `openai/clip-vit-base-patch32` 目录，包含配置和权重。加载使用本地文件。比赛历史环境包含 PyTorch `2.8.0+cu128`、Torchvision `0.23.0+cu128`、Transformers `5.17.0`、NumPy `2.3.2` 和 Pillow `11.3.0`。 [OpenAI CLIP](https://github.com/openai/CLIP)

数据目录下可按任意子目录放图。训练 CSV 使用 UTF-8，`path` 为相对于 `--data-root` 的路径，`label` 为原始类别字符串。程序按标签字符串排序编码，预测时还原回同一字符串。清单的行序决定近重复组和检索索引的解释。

```csv
path,label
train/one.jpg,0000
train/two.jpg,0001
```

查询 CSV 只有 `path` 一列，输出按查询行序写出。

```csv
path
query/three.jpg
query/four.jpg
```

## 训练与单模型推理

在仓库根目录运行。`--work` 存放本机实验产物，每个阶段从本目录读取前驱，输出 `ce.pt`、`groupcache.pt`、`sp2.pt`、`teacher.pt` 和 `student.pt`。两条 SCE2 分支独立校准。首次建立训练 JPEG 缓存，后续沿用同一目录的缓存。预先得到的训练组 ID 可以用 `--group-ids` 提供，格式为与训练清单逐行对应的一维整数 NumPy 数组。省略时自动从初始视觉特征生成。

```bash
python -m aic_finegrained train --data-root data --train-csv data/train.csv --base-model models/clip --work runs/experiment --stage all
python -m aic_finegrained infer --data-root data --train-csv data/train.csv --query-csv data/query.csv --base-model models/clip --checkpoint runs/experiment/student.pt --out runs/predictions.csv --canonical-out runs/canonical.npy
```

训练中断后可用 `--stage` 启动首个未完整输出的阶段，并使用同一份训练输入、分辨率及缓存参数。正式设置为每阶段 80 轮。SCE2 的公开入口最少需要 9 轮，以执行第 8 轮校准和首次反向项。小规模功能检查可以同时降低 `--size`、`--cache-size`、`--batch`、`--feature-batch`，例如设为 `224`、`256`、`8`、`4`，并设 `--workers 0`。图像尺寸须为 CLIP patch 尺寸的倍数。相同权重能单独重算后处理。

```bash
python -m aic_finegrained scores --input runs/canonical.npy --method none --out runs/probs-uncalibrated.npy
python -m aic_finegrained scores --input runs/canonical.npy --method balanced --out runs/probs-balanced.npy
python -m aic_finegrained scores --input runs/canonical.npy --method sot --rho 9 --temperature 1 --out runs/probs-sot.npy
```

矩阵形状为查询数 × 类别数，进入 `scores` 时应已完成视觉、检索和可选弱图融合。只在整个查询集上得到同一种 canonical 分数后才比较各处理器。使用 `--bank-indices` 可指定训练检索行，格式为相对于训练 CSV 的零基、一维整数 `.npy` 数组。省略时用全部训练图。推理 `--batch` 只控制图像特征前向批量，程序先收齐特征再做图与传输。

## 和参赛复现的关系

本实现保留已采用方案的算法和主要参数，使用通用 CSV 输入。复现实验时应使用有权访问的图像、类别顺序与检查点，并为每个对照固定训练检索库的索引。重新分组、替换检索库或更换运行环境都可能改变结果。

该版本用合成图像验证了五阶段训练、单模型推理和同一份 canonical 分数的回放。历史上从原图跨 GPU 全量复算时，部分预测与参赛输出不同。合成示例用于验证代码执行和数学行为，赛题规模下的表现需要另行评估。

运行 `python -m unittest discover -s tests` 可重跑数值测试。安装视觉依赖后，同一个命令还会运行合成图像示例，检查五个阶段的权重、查询顺序和后处理回放。GitHub Actions 从安装后的包运行这些测试。
