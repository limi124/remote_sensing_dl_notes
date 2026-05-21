import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split


CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

EUROSAT_MEAN = torch.tensor([0.3444, 0.3809, 0.4082]).view(3, 1, 1)
EUROSAT_STD = torch.tensor([0.2037, 0.1366, 0.1148]).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    raw = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    tensor = raw.view(image.size[1], image.size[0], 3).permute(2, 0, 1).float()
    return tensor / 255.0


class EuroSATRGBDataset(Dataset):
    def __init__(self, root: Path, train: bool = True, image_size: int = 64):
        self.root = Path(root)
        self.train = train
        self.image_size = image_size
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")
            for path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No jpg images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, image: Image.Image) -> Image.Image:
        if self.train:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        return image

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self._augment(image)
        tensor = pil_to_tensor(image)
        tensor = (tensor - EUROSAT_MEAN) / EUROSAT_STD
        return tensor, torch.tensor(label, dtype=torch.long)


class TinyCNNClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return {"loss": total_loss / total_count, "acc": total_correct / total_count}


@torch.no_grad()
def confusion_matrix(model, dataloader, device, num_classes: int):
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    model.eval()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        for target, pred in zip(labels.cpu(), preds.cpu()):
            matrix[target, pred] += 1
    return matrix


def save_history(history, out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_outputs(history, cm, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip plots.")
        return

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a TinyCNN baseline on a local EuroSAT RGB subset. "
            "Tune --epochs, --max-samples, and --batch-size for your computer."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/eurosat/EuroSAT_RGB"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/part2_cnn"))
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Default is 30 for a smoother demo curve. Use 3-5 for a quick test.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Balanced subset size. Default 3000 is quick; 0 means use all 27000 images.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def make_balanced_stratified_split(dataset, max_samples=0, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    indices_by_class = {class_idx: [] for class_idx in range(len(CLASS_NAMES))}

    for idx, (_, label) in enumerate(dataset.samples):
        indices_by_class[label].append(idx)

    selected_indices = []
    if max_samples and max_samples > 0:
        per_class = max_samples // len(CLASS_NAMES)
        remainder = max_samples % len(CLASS_NAMES)
        for class_idx in range(len(CLASS_NAMES)):
            class_indices = indices_by_class[class_idx][:]
            rng.shuffle(class_indices)
            take = per_class + (1 if class_idx < remainder else 0)
            selected_indices.extend(class_indices[: min(take, len(class_indices))])
    else:
        for class_indices in indices_by_class.values():
            selected_indices.extend(class_indices)

    selected_by_class = {class_idx: [] for class_idx in range(len(CLASS_NAMES))}
    for idx in selected_indices:
        label = dataset.samples[idx][1]
        selected_by_class[label].append(idx)

    train_indices = []
    val_indices = []
    for class_indices in selected_by_class.values():
        rng.shuffle(class_indices)
        val_count = max(1, int(round(len(class_indices) * val_ratio))) if class_indices else 0
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    train_base = EuroSATRGBDataset(args.data_root, train=True)
    val_base = EuroSATRGBDataset(args.data_root, train=False)
    train_indices, val_indices = make_balanced_stratified_split(
        train_base,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = Subset(train_base, list(train_indices))
    val_dataset = Subset(val_base, list(val_indices))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("train images:", len(train_dataset))
    print("val images:", len(val_dataset))

    model = TinyCNNClassifier(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val_acc = -1.0
    best_path = args.out_dir / "best_tiny_cnn.pt"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)

        if row["val_acc"] > best_val_acc:
            best_val_acc = row["val_acc"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "args": vars(args),
                "best_val_acc": best_val_acc,
            }
            # PyTorch 2.0 on Windows may mis-handle non-ASCII Path strings.
            # Saving through an opened file object works reliably.
            with best_path.open("wb") as f:
                torch.save(checkpoint, f)

        print(
            f"epoch {epoch:02d} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f}"
        )

    cm = confusion_matrix(model, val_loader, device, len(CLASS_NAMES))
    save_history(history, args.out_dir / "history.csv")
    plot_outputs(history, cm, args.out_dir)
    (args.out_dir / "class_to_idx.json").write_text(
        json.dumps({name: idx for idx, name in enumerate(CLASS_NAMES)}, indent=2),
        encoding="utf-8",
    )

    print("\nbest val acc:", f"{best_val_acc:.4f}")
    print("saved:", best_path)
    print("outputs:", args.out_dir)


if __name__ == "__main__":
    main()
