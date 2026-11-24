# AIC 2026｜噪声标签下的细粒度图像识别

[English](README.en.md)

这是 AIC 2026 半决赛细粒度图像识别方案的可运行参考实现。它以 CLIP ViT-B/32 为骨干，结合组外近邻软标签、独立教师重训、单模型推理和半松弛最优传输。

赛题要求从带噪声标签的训练图中学习细粒度类别。近重复图可能在检索中反复为同一个标签投票，相似异类又可能只在局部细节上有区别。训练监督来自赛题当前阶段的训练图，最终由一个学生模型提取特征。 [AIC 官方赛题](https://www.aicomp.cn/tracks/tracks-1/3714.html)

## 实际采用的路线

| 步骤 | 带来的变化 |
| --- | --- |
| 复刻基线 | 512 分辨率及严格均衡 Sinkhorn |
| GroupCache | 用教师的组外近邻构造软目标 |
| 544 | 提高输入分辨率，训练使用 608 短边缓存 |
| SP2 | 用在线 EMA 教师和训练池 OT 更新监督 |
| SCE2 | 重新训练教师和学生，各自校准完整 CE 加期望反向 CE |
| **SOT1** | **在同一个 SCE2 学生上将硬类别均衡改为软惩罚** |

SOT1 固定了模型、五视图、训练检索与弱图，只替换最后的分数校准。这个对照比同时调整教师与损失的训练实验更容易归因。 [实验复盘](docs/results.md)

## 快速开始

在仓库根目录使用 Python 3.10 或更新版本。分数示例只需 NumPy 与 SciPy，在同一组合成分数上比较无约束、严格均衡和半松弛 OT。

```bash
python -m pip install -e .
python -m aic_finegrained demo
python -m unittest discover -s tests
```

图像示例在本地生成带两处错标的小数据集和轻量 CLIP 模型，在 CPU 上跑完五阶段训练、权重重载、五视图推理和 SOT。无需下载比赛数据或模型权重。

```bash
python -m pip install -e '.[vision]'
python -m aic_finegrained demo-images --out runs/image-demo
```

运行后可查看 `runs/image-demo/predictions.csv`。使用自己的数据时需准备训练清单、查询清单和本地 CLIP 权重目录。命令与输出说明见[运行与复现](docs/reproduction.md)。赛事数据、原始输出和模型权重由各自授权渠道管理。

## 阅读顺序

| 文档 | 内容 |
| --- | --- |
| [赛题与方法](docs/method.md) | 五个训练阶段、组排除、EMA、SCE2、五视图、弱图与 SOT 数学公式 |
| [实验复盘](docs/results.md) | 匹配对照、采用决策、失效方案及归因边界 |
| [运行与复现](docs/reproduction.md) | 数据接口、训练与推理示例、原环境和实际验证范围 |
| [后续改进](docs/next-steps.md) | 置信度加权 SOT、检索证据门控、SCE2 拆解和验证顺序 |
| [许可与依赖](THIRD_PARTY_NOTICES.md) | 仓库许可、上游依赖及数据使用边界 |

代码位于 `src/aic_finegrained/`。`groupcache.py` 和 `losses.py` 对应核心监督方法，`train.py` 和 `model.py` 串起视觉训练，`infer.py`、`graph.py`、`scores.py` 和 `transport.py` 组成原图推理与后处理。
