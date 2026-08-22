from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def confusion_update(matrix: torch.Tensor, pred: torch.Tensor, target: torch.Tensor, classes: int):
    valid = (target >= 0) & (target < classes)
    idx = classes * target[valid].to(torch.int64) + pred[valid]
    matrix += torch.bincount(idx, minlength=classes * classes).reshape(classes, classes).cpu()


def metrics_from_confusion(cm: torch.Tensor):
    cm = cm.double(); tp = cm.diag(); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    iou = tp / (tp + fp + fn).clamp_min(1)
    dice = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    acc = tp.sum() / cm.sum().clamp_min(1)
    return {"accuracy": float(acc), "miou": float(iou.mean()), "mean_dice": float(dice.mean()),
            "class_iou": [float(x) for x in iou], "class_dice": [float(x) for x in dice]}


def save_history(rows: list[dict], output: Path):
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax[0].plot(epochs, [r["val_loss"] for r in rows], label="val"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(epochs, [r["val_miou"] for r in rows]); ax[1].set_title("Validation mIoU")
    fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=160); plt.close(fig)


def atomic_checkpoint(path: Path, payload: dict):
    tmp = path.with_suffix(".tmp"); torch.save(payload, tmp); tmp.replace(path)


def benchmark_inference(model, sample, device, repeats=20):
    model.eval()
    with torch.no_grad():
        for _ in range(3): model(*sample) if isinstance(sample, tuple) else model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats): model(*sample) if isinstance(sample, tuple) else model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - start) / repeats


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
