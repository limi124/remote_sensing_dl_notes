# 遥感深度学习转码路线

[![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing-059669)](#)
[![Topic](https://img.shields.io/badge/Topic-Deep%20Learning-2563eb)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20Chinese-111827)](#)

**语言:** 中文 | [English](README.en.md)

这是一个面向地信遥感方向的深度学习转码学习仓库，用来记录从 Python 地信数据分析转向 **遥感深度学习科研** 的完整路线。

项目会长期整理：神经网络底层原理、PyTorch 训练闭环、CNN、U-Net、DeepLabV3+、Transformer 视觉模型、遥感语义分割、变化检测、遥感基础模型和论文复现。

目标不是只会调用模型，而是逐步理解训练机制、模型结构、实验设计、常见失败原因和创新来源。

## 学习路线

```text
Karpathy 底层原理
-> PyTorch 训练框架
-> CNN / U-Net / DeepLabV3+
-> Transformer / ViT / SegFormer
-> 遥感语义分割 / 变化检测 / 多时相建模
-> 遥感基础模型论文复现
-> 自己的创新实验
```

## 课程目录

完整路线见：[COURSE_OUTLINE.md](COURSE_OUTLINE.md)

| Part | 标题 | Markdown | Notebook | 状态 |
| --- | --- | --- | --- | --- |
| 0 | 深度学习底层原理与遥感 CV 路线总览 | [文档](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Notebook](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) | 已完成 |
| 1 | PyTorch Training Fundamentals | [文档](docs/foundations/01-pytorch-training-fundamentals.md) | [Notebook](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) | 已完成 |
| 2 | CNN Image Classification | [理论文档](docs/algorithms/02-cnn-image-classification.md) / [实操文档](docs/algorithms/02-cnn-eurosat-local-practice.md) | [理论 Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) / [本地实操](notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb) | 已完成 |
| 3 | Remote Sensing Data Pipeline | [文档](docs/practices/03-remote-sensing-data-pipeline.md) | [Notebook](notebooks/practices/03-remote-sensing-data-pipeline.ipynb) | 已完成 |
| 4 | U-Net Semantic Segmentation | [文档](docs/algorithms/04-unet-semantic-segmentation.md) | [Notebook](notebooks/algorithms/04-unet-semantic-segmentation.ipynb) | 已完成 |
| 5 | DeepLabV3+ Land Cover Segmentation | [文档](docs/algorithms/05-deeplabv3plus-land-cover.md) | [Notebook](notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb) | 已完成 |
| 6 | Transformer Vision Models | [文档](docs/algorithms/06-transformer-vision-models.md) | [Notebook](notebooks/algorithms/06-transformer-vision-models.ipynb) | 已完成 |
| 7 | Optical Change Detection | [文档](docs/practices/07-optical-change-detection.md) | [Notebook](notebooks/practices/07-optical-change-detection.ipynb) | 已完成 |
| 8 | Remote Sensing Foundation Models | [文档](docs/papers/08-remote-sensing-foundation-models.md) | [Notebook](notebooks/papers/08-remote-sensing-foundation-models.ipynb) | 已完成 |
| 9 | Paper Reproduction Workflow | [文档](docs/practices/09-paper-reproduction-workflow.md) | [Notebook](notebooks/practices/09-paper-reproduction-workflow.ipynb) | 已完成 |

## 当前内容

| 模块 | 重点 | Markdown | Notebook |
| --- | --- | --- | --- |
| Part 0 | 从 Karpathy 底层原理到 CNN、U-Net、DeepLabV3+、Transformer 的全局路线 | [阅读](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [运行](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) |
| Part 1 | PyTorch 训练闭环：Dataset、DataLoader、模型、loss、optimizer、验证、checkpoint、实验日志 | [阅读](docs/foundations/01-pytorch-training-fundamentals.md) | [运行](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) |
| Part 2 | CNN 图像分类：Conv2d、shape、TinyCNN、EuroSAT 遥感场景分类 | [理论](docs/algorithms/02-cnn-image-classification.md) / [实操](docs/algorithms/02-cnn-eurosat-local-practice.md) | [理论 Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) / [本地实操](notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb) |
| Part 3 | 遥感数据管线：GeoTIFF、CRS、矢量栅格化、切片、同步增强、Dataset | [阅读](docs/practices/03-remote-sensing-data-pipeline.md) | [运行](notebooks/practices/03-remote-sensing-data-pipeline.ipynb) |
| Part 4–6 | U-Net、DeepLabV3+、ViT/Swin/SegFormer 分割模型路线 | [算法文档](docs/algorithms/) | [算法 Notebook](notebooks/algorithms/) |
| Part 7–9 | 变化检测、遥感基础模型与论文复现工作流 | [实践/论文文档](docs/) | [实践/论文 Notebook](notebooks/) |

## Part 3：从遥感数据到深度学习样本

Part 3 用一个自包含的合成遥感场景，把常见的空间数据处理步骤串成可理解、可复现的工作流：

- [教学 Notebook](notebooks/practices/03-remote-sensing-data-pipeline.ipynb)
- [脚本与运行说明](scripts/part3_data_pipeline/README.md)
- 学习主线：4 波段 GeoTIFF + GeoJSON → mask → tiles → augmentation → Dataset

这一章重点不是“记住 API”，而是理解影像、矢量标注、栅格 mask、切片和 PyTorch Dataset 之间如何衔接；掌握这条链路后，读者可以替换为自己的遥感数据。

## Part 2：EuroSAT 图像分类实操

Part 2 在理论 Notebook 之外提供 EuroSAT RGB 的轻量实操，用于体验从数据集、训练循环到评估和保存模型的完整闭环：

- [本地实操 Notebook](notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb)
- [训练脚本与说明](scripts/part2_cnn/README.md)
- 使用 TinyCNN 作为清晰、易修改的 baseline
- 数据与训练输出不随仓库上传；按说明下载后即可在本地复现

## Part 4–9：从模型理解到可信实验

Part 4–9 不止介绍 U-Net、DeepLabV3+、SegFormer、变化检测和基础模型，也把它们放进真实训练实验中：LandCover.ai 语义分割、LEVIR-CD 变化检测、官方 TorchGeo Scale-MAE 线性探针，以及三随机种子复现。

训练脚本、环境定义和结果解读均公开在仓库中：

- [真实训练与复现说明](scripts/real_training/README.md)
- [实验协议与本机实测结果](docs/REAL_TRAINING_RESULTS.md)

为了让仓库保持轻量、便于学习和长期维护，原始数据、模型权重、训练日志与预测图不提交到 GitHub；读者可根据文档在本地独立复现。

## 关于这个仓库

这份开源笔记大约花了我三个月持续整理、编写和实验。它记录的是我从地信/遥感数据分析走向深度学习研究的一条学习路线，也希望能为正在转向遥感 AI 的同学提供一份可以动手跟练的教材。

受时间、经验和设备条件限制，仓库中一定还有不严谨、不完整或表达不够清楚的地方。非常欢迎大家通过 Issue、Discussion 或 Pull Request 提出问题、补充案例、修正文档与共建新章节；希望它能在开源协作中慢慢变得更好。

## 目录结构

```text
remote_sensing_dl_notes/
├─ README.md
├─ README.en.md
├─ README.zh-CN.md
├─ COURSE_OUTLINE.md
├─ docs/
│  ├─ foundations/
│  ├─ algorithms/
│  ├─ practices/
│  └─ papers/
├─ notebooks/
│  ├─ foundations/
│  ├─ algorithms/
│  ├─ practices/
│  └─ papers/
└─ projects/
```

## 每章固定结构

每一章尽量采用同一套结构，方便长期维护：

```text
1. 这一章要解决什么问题
2. 理论核心
3. 最小代码实现
4. Shape / 流程图 / 可视化
5. 小实验
6. 常见坑
7. 遥感迁移
8. 作业与扩展
```

## 适合谁

- 会 Python 数据分析，想转深度学习的地信/遥感学生
- 想从 PyTorch、CV 进入遥感深度学习的人
- 想复现遥感论文、做高水平论文选题的人
- 想把计算机视觉方法迁移到遥感任务的人

## 学习建议

- 不要只收藏笔记，每章至少跑通一个 Notebook。
- 每个模型先 overfit 一小批数据，再上完整数据集。
- 每次实验都记录数据、模型、loss、metric、运行命令、随机种子和结论。
- 进入遥感任务后，重点关注投影、切片、类别不平衡、空间自相关和跨区域泛化。

