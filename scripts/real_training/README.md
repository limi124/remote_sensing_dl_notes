# Parts 4–9 真实训练

环境：`remote-sensing-dl`（见仓库根目录 `environment-real-training.yml`）。

```powershell
conda run -n remote-sensing-dl python scripts/real_training/smoke_test.py
conda run -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model unet --output outputs/part4_unet
conda run -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model deeplabv3plus --output outputs/part5_deeplabv3plus
conda run -n remote-sensing-dl python scripts/real_training/train_segmentation.py --model segformer_b0 --output outputs/part6_segformer
conda run -n remote-sensing-dl python scripts/real_training/train_change.py --data-root data/levir_cd --output outputs/part7_change_weighted
conda run -n remote-sensing-dl python scripts/real_training/train_foundation.py --data-root data/eurosat/EuroSAT_RGB --output outputs/part8_scalemae
powershell -ExecutionPolicy Bypass -File scripts/real_training/run_part9.ps1
```

Part 4–6 保存 `best.pt`、`last.pt`、`config.json`、`history.csv`、`metrics.json`、`training_curves.png` 和 `predictions.png`；Part 7 另有 `confusion_matrix.png`；Part 8 分别保存随机/预训练 probe 的权重和历史；Part 9 保存每个 seed 的完整结果及汇总 CSV/对比图。

在提交前验证本地结果不会被上传：

```powershell
git check-ignore data outputs
git status --short
```
