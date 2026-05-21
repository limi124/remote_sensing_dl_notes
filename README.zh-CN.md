# 遥感深度学习转码路线

[![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing-059669)](#)
[![Topic](https://img.shields.io/badge/Topic-Deep%20Learning-2563eb)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20Chinese-111827)](#)

**语言:** [English](README.md) | 中文

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
| 2 | CNN Image Classification | [文档](docs/algorithms/02-cnn-image-classification.md) | [Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) | 已完成 |
| 3 | Remote Sensing Data Pipeline | docs/practices/03-remote-sensing-data-pipeline.md | notebooks/practices/03-remote-sensing-data-pipeline.ipynb | 计划中 |
| 4 | U-Net Semantic Segmentation | docs/algorithms/04-unet-semantic-segmentation.md | notebooks/algorithms/04-unet-semantic-segmentation.ipynb | 计划中 |
| 5 | DeepLabV3+ Land Cover Segmentation | docs/algorithms/05-deeplabv3plus-land-cover.md | notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb | 计划中 |
| 6 | Transformer Vision Models | docs/algorithms/06-transformer-vision-models.md | notebooks/algorithms/06-transformer-vision-models.ipynb | 计划中 |
| 7 | Optical Change Detection | docs/practices/07-optical-change-detection.md | notebooks/practices/07-optical-change-detection.ipynb | 计划中 |
| 8 | Remote Sensing Foundation Models | docs/papers/08-remote-sensing-foundation-models.md | notebooks/papers/08-remote-sensing-foundation-models.ipynb | 计划中 |
| 9 | Paper Reproduction Workflow | docs/practices/09-paper-reproduction-workflow.md | notebooks/practices/09-paper-reproduction-workflow.ipynb | 计划中 |

## 当前内容

| 模块 | 重点 | Markdown | Notebook |
| --- | --- | --- | --- |
| Part 0 | 从 Karpathy 底层原理到 CNN、U-Net、DeepLabV3+、Transformer 的全局路线 | [阅读](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [运行](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) |
| Part 1 | PyTorch 训练闭环：Dataset、DataLoader、模型、loss、optimizer、验证、checkpoint、实验日志 | [阅读](docs/foundations/01-pytorch-training-fundamentals.md) | [运行](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) |
| Part 2 | CNN 图像分类：Conv2d、shape、TinyCNN、EuroSAT 遥感场景分类 | [阅读](docs/algorithms/02-cnn-image-classification.md) | [运行](notebooks/algorithms/02-cnn-image-classification.ipynb) |

## 目录结构

```text
remote_sensing_dl_notes/
├─ README.md
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

