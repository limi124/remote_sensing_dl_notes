import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import numpy as np


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

PALETTE = [
    "#2f6f4e",
    "#5a8f3c",
    "#8bb174",
    "#7b6d8d",
    "#c65d3b",
    "#d9a441",
    "#3f88c5",
    "#a23e48",
    "#2a9d8f",
    "#546a7b",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    root = repo_root() / "data" / "eurosat" / "EuroSAT_RGB"
    if not root.exists():
        raise FileNotFoundError(f"Missing EuroSAT RGB data: {root}")
    return root


def out_dir() -> Path:
    path = repo_root() / "outputs" / "part2_cnn_promo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_images(root: Path):
    return {name: len(list((root / name).glob("*.jpg"))) for name in CLASS_NAMES}


def sample_image(path: Path, size=128) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def save_class_distribution(counts, output: Path):
    names = list(counts.keys())
    values = [counts[name] for name in names]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(names, values, color=PALETTE, edgecolor="#1f2933", linewidth=0.8)
    ax.set_title("EuroSAT RGB: 27,000 Sentinel-2 Scene Images", fontsize=16, weight="bold")
    ax.set_ylabel("images")
    ax.set_ylim(0, max(values) * 1.18)
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 55,
            f"{value}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_large_sample_grid(root: Path, output: Path):
    rng = random.Random(42)
    rows, cols = 4, 5
    tile = 160
    label_h = 34
    pad = 12
    canvas_w = cols * tile + (cols + 1) * pad
    canvas_h = rows * (tile + label_h) + (rows + 1) * pad + 70
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#f5f7f2")
    draw = ImageDraw.Draw(canvas)

    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        title_font = label_font = small_font = ImageFont.load_default()

    draw.text((pad, 18), "EuroSAT RGB Samples", fill="#16213e", font=title_font)
    draw.text(
        (pad, 50),
        "Remote sensing scene classification: 10 land-use / land-cover classes",
        fill="#53616f",
        font=small_font,
    )

    sample_paths = []
    for class_name in CLASS_NAMES:
        files = sorted((root / class_name).glob("*.jpg"))
        sample_paths.extend((class_name, p) for p in rng.sample(files, 2))

    for idx, (class_name, path) in enumerate(sample_paths[: rows * cols]):
        r = idx // cols
        c = idx % cols
        x = pad + c * (tile + pad)
        y = 82 + pad + r * (tile + label_h + pad)
        image = sample_image(path, size=tile)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y, x + tile, y + tile), outline="#ffffff", width=3)
        draw.text((x + 4, y + tile + 8), class_name, fill="#16213e", font=label_font)

    canvas.save(output)


def save_class_strips(root: Path, output: Path):
    rng = random.Random(7)
    examples_per_class = 6
    tile = 92
    label_w = 180
    pad = 8
    canvas_w = label_w + examples_per_class * (tile + pad) + pad
    canvas_h = len(CLASS_NAMES) * (tile + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    try:
        label_font = ImageFont.truetype("arial.ttf", 16)
        count_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        label_font = count_font = ImageFont.load_default()

    for row, class_name in enumerate(CLASS_NAMES):
        files = sorted((root / class_name).glob("*.jpg"))
        samples = rng.sample(files, examples_per_class)
        y = pad + row * (tile + pad)
        draw.rectangle((0, y, label_w - 10, y + tile), fill=PALETTE[row])
        draw.text((14, y + 25), class_name, fill="white", font=label_font)
        draw.text((14, y + 51), f"{len(files)} images", fill="#eef2f4", font=count_font)
        for col, path in enumerate(samples):
            x = label_w + col * (tile + pad)
            image = sample_image(path, size=tile)
            canvas.paste(image, (x, y))

    canvas.save(output)


def save_cnn_shape_flow(output: Path):
    stages = [
        ("Input", "[B, 3, 64, 64]", "#3f88c5"),
        ("Conv Block 1", "[B, 16, 32, 32]", "#2a9d8f"),
        ("Conv Block 2", "[B, 32, 16, 16]", "#d9a441"),
        ("Conv Block 3", "[B, 64, 16, 16]", "#c65d3b"),
        ("Global Pool", "[B, 64]", "#7b6d8d"),
        ("Classifier", "[B, 10]", "#a23e48"),
    ]

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.text(0.2, 3.55, "TinyCNN Shape Flow", fontsize=18, weight="bold", color="#16213e")
    ax.text(0.2, 3.25, "From remote-sensing image patches to land-cover logits", fontsize=11, color="#53616f")

    x_positions = np.linspace(0.5, 11.2, len(stages))
    widths = [1.35, 1.75, 1.75, 1.75, 1.55, 1.4]
    heights = [1.4, 1.25, 1.05, 0.95, 0.75, 0.65]

    for i, ((name, shape, color), x, w, h) in enumerate(zip(stages, x_positions, widths, heights)):
        y = 1.25 + (1.4 - h) / 2
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#1f2933", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + 0.13, name, ha="center", va="center", fontsize=10, color="white", weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.22, shape, ha="center", va="center", fontsize=9, color="white")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.12, 1.95),
                xytext=(x + w + 0.1, 1.95),
                arrowprops=dict(arrowstyle="->", color="#53616f", lw=1.8),
            )

    ax.text(0.2, 0.45, "Conv2d + BatchNorm + ReLU + Pooling -> compact feature maps -> scene class", fontsize=11, color="#16213e")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def read_history(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_training_dashboard(history_path: Path, output: Path):
    history = read_history(history_path)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axis("off")
    ax.text(0.04, 0.88, "TinyCNN Baseline on EuroSAT RGB Subset", fontsize=18, weight="bold", color="#16213e")

    if not history:
        ax.text(0.04, 0.72, "No history.csv found yet. Run the notebook training cells first.", fontsize=12, color="#53616f")
    else:
        epochs = [int(row["epoch"]) for row in history]
        train_acc = [float(row["train_acc"]) for row in history]
        val_acc = [float(row["val_acc"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]

        left = fig.add_axes([0.08, 0.18, 0.38, 0.48])
        right = fig.add_axes([0.56, 0.18, 0.34, 0.48])

        left.plot(epochs, train_acc, marker="o", label="train", color="#2a9d8f")
        left.plot(epochs, val_acc, marker="o", label="val", color="#c65d3b")
        left.set_title("Accuracy")
        left.set_xlabel("epoch")
        left.set_ylim(0, 1.05)
        left.grid(alpha=0.22)
        left.legend()

        right.plot(epochs, train_loss, marker="o", label="train", color="#3f88c5")
        right.plot(epochs, val_loss, marker="o", label="val", color="#d9a441")
        right.set_title("Loss")
        right.set_xlabel("epoch")
        right.grid(alpha=0.22)
        right.legend()

        ax.text(0.04, 0.75, f"Epochs: {len(epochs)}", fontsize=12, color="#53616f")
        ax.text(0.26, 0.75, f"Best val acc: {max(val_acc):.3f}", fontsize=12, color="#53616f")
        ax.text(0.54, 0.75, "Model: TinyCNNClassifier", fontsize=12, color="#53616f")
        ax.text(0.04, 0.08, "Teaching baseline, not a full EuroSAT SOTA result", fontsize=10, color="#6b7280")

    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    root = data_root()
    output_root = out_dir()
    counts = count_images(root)

    save_class_distribution(counts, output_root / "01_class_distribution.png")
    save_large_sample_grid(root, output_root / "02_eurosat_sample_grid.png")
    save_class_strips(root, output_root / "03_class_example_strips.png")
    save_cnn_shape_flow(output_root / "04_tinycnn_shape_flow.png")
    save_training_dashboard(
        repo_root() / "outputs" / "part2_cnn_notebook" / "history.csv",
        output_root / "05_training_dashboard.png",
    )

    print("promo figures saved to:", output_root)
    for path in sorted(output_root.glob("*.png")):
        print(path.name, path.stat().st_size)


if __name__ == "__main__":
    main()
