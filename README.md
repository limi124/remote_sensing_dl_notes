# Remote Sensing Deep Learning Roadmap

[![Domain](https://img.shields.io/badge/Domain-Remote%20Sensing-059669)](#)
[![Topic](https://img.shields.io/badge/Topic-Deep%20Learning-2563eb)](#)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)](#)
[![Language](https://img.shields.io/badge/Language-English%20%7C%20Chinese-111827)](#)

**Language:** English | [中文](README.zh-CN.md)

A structured learning repository for transitioning from Python geospatial data analysis to **remote sensing deep learning research**.

This project is organized as a long-term study path: from neural network fundamentals and PyTorch training loops to CNNs, U-Net, DeepLabV3+, Transformer vision models, semantic segmentation, change detection, remote sensing foundation models, and paper reproduction.

The goal is not just to call existing models. The goal is to understand training mechanics, model structures, experiment design, failure modes, and where research ideas can come from.

## Learning Path

```text
Karpathy-style neural network fundamentals
-> PyTorch training framework
-> CNN / U-Net / DeepLabV3+
-> Transformer / ViT / SegFormer
-> Remote sensing semantic segmentation / change detection / multi-temporal modeling
-> Remote sensing foundation model paper reproduction
-> Your own research experiments
```

## Course Outline

See the full roadmap here: [COURSE_OUTLINE.md](COURSE_OUTLINE.md)

| Part | Title | Markdown | Notebook | Status |
| --- | --- | --- | --- | --- |
| 0 | Deep Learning Foundations + CV / Remote Sensing Roadmap | [Docs](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Notebook](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) | Done |
| 1 | PyTorch Training Fundamentals | [Docs](docs/foundations/01-pytorch-training-fundamentals.md) | [Notebook](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) | Done |
| 2 | CNN Image Classification | [Docs](docs/algorithms/02-cnn-image-classification.md) | [Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) | Done |
| 3 | Remote Sensing Data Pipeline | docs/practices/03-remote-sensing-data-pipeline.md | notebooks/practices/03-remote-sensing-data-pipeline.ipynb | Planned |
| 4 | U-Net Semantic Segmentation | docs/algorithms/04-unet-semantic-segmentation.md | notebooks/algorithms/04-unet-semantic-segmentation.ipynb | Planned |
| 5 | DeepLabV3+ Land Cover Segmentation | docs/algorithms/05-deeplabv3plus-land-cover.md | notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb | Planned |
| 6 | Transformer Vision Models | docs/algorithms/06-transformer-vision-models.md | notebooks/algorithms/06-transformer-vision-models.ipynb | Planned |
| 7 | Optical Change Detection | docs/practices/07-optical-change-detection.md | notebooks/practices/07-optical-change-detection.ipynb | Planned |
| 8 | Remote Sensing Foundation Models | docs/papers/08-remote-sensing-foundation-models.md | notebooks/papers/08-remote-sensing-foundation-models.ipynb | Planned |
| 9 | Paper Reproduction Workflow | docs/practices/09-paper-reproduction-workflow.md | notebooks/practices/09-paper-reproduction-workflow.ipynb | Planned |

## Current Contents

| Module | Focus | Markdown | Notebook |
| --- | --- | --- | --- |
| Part 0 | Big-picture roadmap from Karpathy fundamentals to CNN, U-Net, DeepLabV3+, and Transformer models | [Read](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Run](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) |
| Part 1 | PyTorch training loop: Dataset, DataLoader, model, loss, optimizer, evaluation, checkpoint, experiment log | [Read](docs/foundations/01-pytorch-training-fundamentals.md) | [Run](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) |
| Part 2 | CNN image classification: Conv2d, shapes, TinyCNN, and EuroSAT scene classification | [Read](docs/algorithms/02-cnn-image-classification.md) | [Theory Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) / [Local Practice](notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb) |

## Part 2 Local Practice

Part 2 also includes a runnable EuroSAT RGB local practice workflow:

- [Local practice notebook](notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb)
- [Training scripts and notes](scripts/part2_cnn/README.md)
- Scope: TinyCNN baseline on EuroSAT RGB subset
- Dataset files are not committed; place them under `data/eurosat/EuroSAT_RGB`
- Training outputs and promo figures are generated under `outputs/`, ignored by git by default

## Repository Structure

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

## Chapter Template

Each chapter follows the same pattern:

```text
1. Problem and goals
2. Core theory
3. Minimal implementation
4. Shape / flow diagrams / visualization
5. Small experiments
6. Common pitfalls
7. Remote sensing transfer
8. Exercises and extensions
```

## Who This Is For

- Remote sensing or GIS learners who already know some Python and want to move into deep learning
- Students preparing for remote sensing paper reproduction or research projects
- Learners who want to connect computer vision methods with geospatial applications
- Anyone building a long-term study portfolio around PyTorch and remote sensing deep learning

## Study Advice

- Run the notebooks instead of only reading the Markdown.
- For every model, first overfit a tiny dataset before using full data.
- Record every experiment: data, model, loss, metric, command, seed, and conclusion.
- When moving into remote sensing tasks, pay attention to CRS, tiling, class imbalance, spatial autocorrelation, and cross-region generalization.

