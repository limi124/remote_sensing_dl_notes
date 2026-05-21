# Part 2 CNN 实操

本目录用于本地跑通 EuroSAT RGB 遥感场景分类。

数据已经放在：

```text
data/eurosat/EuroSAT_RGB/
```

目录结构应该是：

```text
EuroSAT_RGB/
  AnnualCrop/
  Forest/
  HerbaceousVegetation/
  Highway/
  Industrial/
  Pasture/
  PermanentCrop/
  Residential/
  River/
  SeaLake/
```

## 1. 安装依赖

如果当前环境还没有 PyTorch，先按你的 CUDA/CPU 环境安装 PyTorch。CPU 版可以用：

```bash
pip install torch pillow matplotlib
```

如果有 NVIDIA GPU，建议按 PyTorch 官网选择 CUDA 版本：

```text
https://pytorch.org/get-started/locally/
```

## 2. 检查数据

```bash
python scripts/part2_cnn/check_eurosat_data.py
```

它会打印每个类别的图片数量，并输出一张样例网格：

```text
outputs/part2_cnn/eurosat_samples.png
```

## 3. 先跑一个快速训练

这个实操的定位是：

```text
TinyCNN baseline on EuroSAT RGB subset
```

默认 notebook 使用 3000 张均衡子集、训练 30 轮。不同电脑速度不同，核心参数都可以改：

- `--max-samples 3000`：默认均衡子集，适合展示和学习。
- `--max-samples 0`：使用完整 27000 张。
- `--epochs 30`：默认展示曲线更完整。
- `--epochs 3` 或 `5`：只想快速确认代码能跑。
- `--batch-size 64`：显存不够可改成 32 或 16。

第一次只想快速试跑，可以用 3000 张图、3 个 epoch：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 3000 --epochs 3 --batch-size 64
```

想复现 notebook 默认宣传图曲线，可以用：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 3000 --epochs 30 --batch-size 64
```

训练输出会保存到：

```text
outputs/part2_cnn/
```

包括：

- `best_tiny_cnn.pt`
- `history.csv`
- `training_curve.png`
- `confusion_matrix.png`
- `class_to_idx.json`

## 4. 跑完整数据

确认快速训练没问题后，再跑完整 27000 张：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 0 --epochs 10 --batch-size 64
```

`--max-samples 0` 表示使用全部数据。
