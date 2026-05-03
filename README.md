# Remote Sensing Deep Learning Notes

[![Remote Sensing](https://img.shields.io/badge/Domain-Remote%20Sensing-059669)](#)
[![Deep Learning](https://img.shields.io/badge/Topic-Deep%20Learning-2563eb)](#)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20Chinese-111827)](#)

**Language:** English | [中文](README.zh-CN.md)

A learning and practice repository for transitioning into **deep learning for geospatial AI and remote sensing**.

This project organizes notes, notebooks, algorithm explanations, and hands-on projects from deep learning fundamentals to computer vision, remote sensing semantic segmentation, change detection, foundation models, and paper reproduction. The goal is not only to call existing models, but to understand training mechanics, model structures, experiment design, and where research innovation can come from.

## Learning Roadmap

```text
Karpathy-style neural network fundamentals
-> PyTorch training framework
-> CNN / U-Net / DeepLabV3+
-> Transformer / ViT / SegFormer
-> Remote sensing segmentation / change detection / multi-temporal modeling
-> Remote sensing foundation model reproduction
-> Your own research experiments
```

## Current Contents

| Module | Markdown | Notebook | Description |
| --- | --- | --- | --- |
| Deep learning foundations + CV / remote sensing segmentation | [Docs](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Notebook](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) | A Karpathy-inspired foundation route with CNN, U-Net, and DeepLabV3+ inserted for remote sensing research |

## Repository Structure

```text
remote-sensing-dl-notes/
├─ README.md
├─ README.zh-CN.md
├─ docs/
│  ├─ foundations/      # Foundation notes
│  ├─ algorithms/       # Algorithm notes: U-Net, DeepLab, SegFormer, Swin, etc.
│  ├─ papers/           # Paper reading notes
│  └─ practices/        # Practical remote sensing workflows
├─ notebooks/
│  ├─ foundations/      # Runnable foundation notebooks
│  ├─ algorithms/       # Algorithm notebooks
│  └─ practices/        # Practice notebooks
└─ projects/            # End-to-end mini projects
```

## Planned Topics

| Stage | Topic | Goal |
| --- | --- | --- |
| 1 | PyTorch training framework | Write training, validation, checkpointing, and loss-curve visualization code |
| 2 | CNN image classification | Run CIFAR10 / EuroSAT classification experiments |
| 3 | U-Net remote sensing segmentation | Run building, water, or land-cover segmentation |
| 4 | DeepLabV3+ | Understand ASPP, multi-scale context, and output stride |
| 5 | SegFormer / ViT | Move from CNNs to Transformer-based vision models |
| 6 | Change detection | Study optical change detection tasks such as LEVIR-CD |
| 7 | Remote sensing foundation models | Read and reproduce MAE, DINO, remote sensing VLMs, and related models |
| 8 | Paper reproduction | Build reproducible experiment logs and research improvement ideas |

## Who This Is For

- Geospatial or remote sensing students who know Python data analysis and want to learn deep learning
- Learners moving from PyTorch and computer vision into remote sensing deep learning
- Researchers who want to reproduce remote sensing papers and develop high-quality research ideas
- Anyone interested in transferring computer vision methods to remote sensing tasks

## How to Use

Read the Markdown notes:

```text
docs/foundations/karpathy-cv-remote-sensing-foundations.md
```

Run the notebook:

```text
notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb
```

Recommended workflow: read the notes, run the code cells, modify the examples, and record your own experiment results, errors, and insights.

## Notes

- The main study notes are currently written in Chinese, while key technical terms are preserved in English for easier paper and code search.
- Notebook code is currently educational and will gradually evolve into complete training projects.
- This repository will continue to add remote sensing practices, algorithm explanations, and paper reproduction notes.
