# 遥感深度学习转码课程目录

这个目录用于规划 `remote_sensing_dl_notes` 仓库的长期学习路线。每一章建议同时维护 Markdown 文档和可运行 Notebook。

## 总体路线

```text
Part 0  深度学习底层原理与遥感 CV 路线总览
Part 1  PyTorch Training Fundamentals：从训练原理到可复现实验
Part 2  CNN Image Classification：从卷积到遥感场景分类
Part 3  Remote Sensing Data Pipeline：遥感影像数据读取、切片与增强
Part 4  U-Net Semantic Segmentation：遥感建筑物/水体/道路提取
Part 5  DeepLabV3+ Land Cover Segmentation：多尺度上下文与土地覆盖
Part 6  Transformer Vision Models：ViT、SegFormer 与遥感分割
Part 7  Optical Change Detection：双时相光学变化检测
Part 8  Remote Sensing Foundation Models：MAE、DINO、CLIP、遥感 VLM
Part 9  Paper Reproduction Workflow：论文复现、消融实验与创新点设计
```

## 章节清单

| Part | 标题 | Markdown | Notebook | 核心目标 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 0 | 深度学习底层原理与遥感 CV 路线总览 | [MD](docs/foundations/karpathy-cv-remote-sensing-foundations.md) | [Notebook](notebooks/foundations/karpathy-cv-remote-sensing-foundations.ipynb) | 建立从 micrograd 到 U-Net、DeepLabV3+、Transformer 的全局地图 | 已完成 |
| 1 | PyTorch Training Fundamentals | [MD](docs/foundations/01-pytorch-training-fundamentals.md) | [Notebook](notebooks/foundations/01-pytorch-training-fundamentals.ipynb) | 跑通 Dataset、DataLoader、train/eval、loss、optimizer、checkpoint、实验记录 | 已完成 |
| 2 | CNN Image Classification | [MD](docs/algorithms/02-cnn-image-classification.md) | [Notebook](notebooks/algorithms/02-cnn-image-classification.ipynb) | 从 Conv2d 到 EuroSAT 遥感场景分类 | 已完成 |
| 3 | Remote Sensing Data Pipeline | docs/practices/03-remote-sensing-data-pipeline.md | notebooks/practices/03-remote-sensing-data-pipeline.ipynb | GeoTIFF、shapefile、CRS、切片、mask、增强 | 计划中 |
| 4 | U-Net Semantic Segmentation | docs/algorithms/04-unet-semantic-segmentation.md | notebooks/algorithms/04-unet-semantic-segmentation.ipynb | 建筑物、水体、道路提取，掌握 IoU、Dice、可视化 | 计划中 |
| 5 | DeepLabV3+ Land Cover Segmentation | docs/algorithms/05-deeplabv3plus-land-cover.md | notebooks/algorithms/05-deeplabv3plus-land-cover.ipynb | 空洞卷积、ASPP、多尺度上下文 | 计划中 |
| 6 | Transformer Vision Models | docs/algorithms/06-transformer-vision-models.md | notebooks/algorithms/06-transformer-vision-models.ipynb | ViT、Swin、SegFormer、遥感大图中的 attention | 计划中 |
| 7 | Optical Change Detection | docs/practices/07-optical-change-detection.md | notebooks/practices/07-optical-change-detection.ipynb | 双时相输入、Siamese、差异特征、LEVIR-CD | 计划中 |
| 8 | Remote Sensing Foundation Models | docs/papers/08-remote-sensing-foundation-models.md | notebooks/papers/08-remote-sensing-foundation-models.ipynb | MAE、DINO、CLIP、遥感 VLM、多光谱适配 | 计划中 |
| 9 | Paper Reproduction Workflow | docs/practices/09-paper-reproduction-workflow.md | notebooks/practices/09-paper-reproduction-workflow.ipynb | 论文复现、实验日志、消融实验、创新点设计 | 计划中 |

## 每章固定结构

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

