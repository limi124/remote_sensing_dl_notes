# Part 4–9 本机真实训练结果

本页记录一次可审计的教学基线运行，而不是论文 SOTA 声明。所有训练均使用 PyTorch 2.5.1 + CUDA、RTX 4060 Laptop GPU 8GB；数据、checkpoint、日志和图片只保留在本机的 `data/` 与 `outputs/`，不随 Git 提交。

## 实验协议

- Part 4–6：LandCover.ai，固定 train/val 空间划分；1200 个训练样本、300 个验证样本、256×256、8 epoch、seed 42。
- Part 7：LEVIR-CD，445/64 个 train/val 三元组、256×256、12 epoch、seed 42；交叉熵类别权重为 `[1, 5]`，以避免“全不变”预测。
- Part 8：EuroSAT RGB 的 1500 个分层样本（80/20），224×224、5 epoch、seed 42；冻结 Scale-MAE 编码器，仅训练线性头，并与随机冻结编码器在同一划分对比。
- Part 9：LandCover.ai 固定划分，900/240 个 train/val 样本、256×256、5 epoch，seeds 13/42/87。速度为 batch=1 的 CUDA 推理时间。

## 单次基线

| 章节 | 模型 | 主指标 | 规模 | 峰值显存 | 推理速度 |
| --- | --- | ---: | ---: | ---: | ---: |
| Part 4 | Tiny U-Net | mIoU 0.45900；mean Dice 0.51699 | 1.085M | 2.91GB | 5.07 ms/图 |
| Part 5 | Tiny DeepLabV3+ | mIoU 0.45966；mean Dice 0.51915 | 0.981M | 1.49GB | 5.42 ms/图 |
| Part 6 | SegFormer-B0 | mIoU 0.65911；mean Dice 0.76296 | 3.715M | 0.76GB | 11.91 ms/图 |
| Part 7 | Siamese U-Net | change IoU 0.48218；F1 0.65063 | 0.242M | 1.43GB | — |
| Part 8 | Scale-MAE frozen probe | accuracy 0.83333（随机冻结：0.49000） | 303.312M 总参数；10,250 可训练 | 1.61GB | — |

Part 7 的 change precision/recall 分别为 0.60230/0.70745；其二分类混淆矩阵为 `[[3936081, 82218], [51490, 124515]]`（行是真值，列是预测）。

## Part 9：三随机种子复现

| 模型 | mIoU（均值±标准差） | mean Dice | 参数量 | 最大显存 | 平均推理速度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tiny U-Net | 0.46490 ± 0.00481 | 0.52158 | 1.085M | 2.91GB | 5.01 ms/图 |
| Tiny DeepLabV3+ | 0.48398 ± 0.00487 | 0.54073 | 0.981M | 1.75GB | 5.37 ms/图 |
| SegFormer-B0 | 0.48007 ± 0.07075 | 0.54704 | 3.715M | 0.78GB | 8.61 ms/图 |

SegFormer-B0 在该短预算下方差明显更大，因此不应仅用单种子最高分得出结论。较长的 Part 6 单次训练显示出更高上限，但 Part 9 的结论应以均值和标准差为准。

## 复现

环境在仓库根目录的 [`environment-real-training.yml`](../environment-real-training.yml)，训练命令见 [`scripts/real_training/README.md`](../scripts/real_training/README.md)。先执行 smoke test，再按章节运行脚本。提交前须确认：

```powershell
git check-ignore data outputs
git status --short
```

其中 `git status` 不得包含数据集、`.pt`、CSV 日志或输出图片。
