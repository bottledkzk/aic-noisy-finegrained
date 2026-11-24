# 第三方项目与许可

本仓库原创代码及文档采用 [MIT](LICENSE)。运行依赖及使用者自行取得的模型、数据适用各自的授权条件。

- 视觉骨干是 OpenAI CLIP ViT-B/32。通过 Transformers 加载使用者准备的本地权重目录。 [OpenAI CLIP 源码与许可](https://github.com/openai/CLIP/blob/main/LICENSE)
- 模型与图像处理依赖 Transformers、PyTorch、Torchvision 和 Pillow。数值求解依赖 NumPy 与 SciPy。安装包由各上游项目分发。
- 赛事数据的使用由组委会规则约定。公开示例运行时生成合成分数，图像入口使用者自行准备具备相应使用权限的输入。 [AIC 赛题规则](https://www.aicomp.cn/tracks/tracks-1/3714.html)
