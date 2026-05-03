# 遥感深度学习转码笔记

[![Remote Sensing](https://img.shields.io/badge/Domain-Remote%20Sensing-059669)](#)
[![Deep Learning](https://img.shields.io/badge/Topic-Deep%20Learning-2563eb)](#)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20Chinese-111827)](#)

**语言:** [English](README.md) | 中文

面向地信遥感方向的深度学习转码学习笔记与实战仓库。

这个项目用于系统整理从深度学习底层原理到计算机视觉、遥感语义分割、变化检测、遥感基础模型和论文复现的学习材料。目标不是只会调用模型，而是逐步理解训练机制、模型结构、实验设计和创新来源。

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

## 当前内容

| 模块 | Markdown | Notebook | 说明 |
| --- | --- | --- | --- |
| 深度学习基础 + CV / 遥感分割入门 | [文档](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Notebook](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) | 基于 Karpathy 路线，插入 CNN、U-Net、DeepLabV3+，面向遥感深度学习 |

## 目录结构

```text
remote-sensing-dl-notes/
├─ README.md
├─ README.zh-CN.md
├─ docs/
│  ├─ foundations/      # 基础知识文档
│  ├─ algorithms/       # 后续放算法介绍，如 U-Net、DeepLab、SegFormer、Swin 等
│  ├─ papers/           # 后续放论文精读
│  └─ practices/        # 后续放实战流程
├─ notebooks/
│  ├─ foundations/      # 可运行基础 notebook
│  ├─ algorithms/       # 算法 notebook
│  └─ practices/        # 实战 notebook
└─ projects/            # 后续放完整小项目
```

## 后续计划

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| 1 | PyTorch 基础训练框架 | 会写训练、验证、保存模型、画 loss 曲线 |
| 2 | CNN 图像分类 | 跑通 CIFAR10 / EuroSAT 分类 |
| 3 | U-Net 遥感分割 | 跑通建筑物、水体或土地覆盖分割 |
| 4 | DeepLabV3+ | 理解 ASPP、多尺度上下文和 output stride |
| 5 | SegFormer / ViT | 从 CNN 过渡到 Transformer 视觉模型 |
| 6 | 变化检测 | 学习 LEVIR-CD 等光学变化检测任务 |
| 7 | 遥感基础模型 | 阅读和复现 MAE、DINO、遥感 VLM 等方向 |
| 8 | 论文复现 | 形成可复现实验日志和改进思路 |

## 适合谁

- 会 Python 数据分析，想转深度学习的地信/遥感学生
- 想从 PyTorch、CV 进入遥感深度学习的人
- 想复现遥感论文、做高水平论文选题的人
- 想把计算机视觉方法迁移到遥感任务的人

## 使用方式

阅读 Markdown：

```text
docs/foundations/karpathy-cv-remote-sensing-foundations.md
```

运行 Notebook：

```text
notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb
```

建议边读边运行代码，并把自己的实验结果、报错和理解补充到笔记里。

## Notes

- 文档主要使用中文，关键技术术语保留英文，方便检索论文和代码。
- Notebook 中的代码以教学演示为主，后续会逐步扩展为完整训练项目。
- 本仓库会持续加入遥感实战、算法详解和论文复现内容。
