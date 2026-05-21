# Part 2 实操：TinyCNN 训练 EuroSAT RGB

这是 Part 2 的本地实操补充，对应 notebook：

- [02-cnn-eurosat-local-practice.ipynb](../../notebooks/algorithms/02-cnn-eurosat-local-practice.ipynb)

定位：

```text
TinyCNN baseline on EuroSAT RGB subset
```

它不是完整 EuroSAT SOTA 实验，而是一个教学 baseline，用来把前面讲过的 `Conv2d`、CNN shape、训练闭环、验证指标和混淆矩阵真正跑起来。

---

## 1. 本章解决什么问题

理论版 Part 2 讲的是：

```text
Conv2d -> CNN feature extractor -> image classification -> EuroSAT scene classification
```

这一份实操补充负责把它落到本地代码：

- 下载并检查 EuroSAT RGB 数据
- 用本地文件夹构造 `Dataset`
- 训练一个小型 `TinyCNNClassifier`
- 做分层训练/验证划分，避免只取到单一类别
- 保存 checkpoint、训练日志、训练曲线和混淆矩阵
- 生成适合仓库展示的宣传图

---

## 2. 数据位置

数据不随 GitHub 仓库上传。请放在：

```text
data/eurosat/EuroSAT_RGB/
```

目录结构应为：

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

完整数据共 27000 张 RGB 图像。

---

## 3. 推荐运行方式

先检查数据：

```bash
python scripts/part2_cnn/check_eurosat_data.py
```

快速试跑：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 3000 --epochs 3 --batch-size 64
```

默认展示曲线：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 3000 --epochs 30 --batch-size 64
```

完整数据实验：

```bash
python scripts/part2_cnn/train_tiny_cnn_eurosat.py --max-samples 0 --epochs 10 --batch-size 64
```

---

## 4. 可调参数

不同电脑训练速度不同，建议直接改这些参数：

| 参数 | 推荐值 | 说明 |
| --- | --- | --- |
| `MAX_SAMPLES` / `--max-samples` | `3000` | 默认均衡子集，适合学习和展示 |
| `MAX_SAMPLES` / `--max-samples` | `0` | 使用完整 27000 张 |
| `EPOCHS` / `--epochs` | `3-5` | 快速确认代码能跑 |
| `EPOCHS` / `--epochs` | `30` | 生成更完整的训练曲线 |
| `BATCH_SIZE` / `--batch-size` | `64` | 默认值 |
| `BATCH_SIZE` / `--batch-size` | `32` 或 `16` | 显存不足时使用 |

---

## 5. 为什么要分层抽样

EuroSAT 文件夹按类别排序。如果直接取前 3000 张：

```python
indices = list(range(len(dataset)))
indices = indices[:3000]
```

会只取到 `AnnualCrop` 一个类别，得到虚假的 `val acc = 1.0`。

因此实操代码使用：

```text
按类别均衡抽样 -> 每个类别内部切分 train / val
```

默认 `MAX_SAMPLES = 3000` 时：

```text
每类 300 张
train 每类 240 张
val 每类 60 张
```

---

## 6. 输出文件

训练输出默认在：

```text
outputs/part2_cnn_notebook/
```

包括：

- `best_tiny_cnn.pt`
- `history.csv`
- `training_curve.png`
- `confusion_matrix.png`
- `class_to_idx.json`

宣传图输出在：

```text
outputs/part2_cnn_promo/
```

包括数据分布、样例网格、CNN shape flow 和训练看板。

这些输出目录默认被 `.gitignore` 忽略，不会上传到 GitHub。

---

## 7. 本章小结

这一章实操的重点不是追求最高精度，而是建立一个可信的遥感 CNN baseline：

```text
本地数据 -> Dataset -> DataLoader -> TinyCNN -> train/eval -> checkpoint -> curve/confusion matrix
```

后续可以在这个 baseline 上继续做：

- 更深的 CNN
- ResNet 迁移学习
- Sentinel-2 13 波段输入
- 更严格的空间划分验证
- U-Net 语义分割

