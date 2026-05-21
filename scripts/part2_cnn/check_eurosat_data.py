from pathlib import Path

from PIL import Image


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


def find_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "eurosat" / "EuroSAT_RGB"
    if not data_root.exists():
        raise FileNotFoundError(f"EuroSAT_RGB not found: {data_root}")
    return data_root


def main() -> None:
    data_root = find_data_root()
    print("data root:", data_root)
    print()

    samples = []
    total = 0
    for class_name in CLASS_NAMES:
        class_dir = data_root / class_name
        files = sorted(class_dir.glob("*.jpg"))
        total += len(files)
        print(f"{class_name:22s} {len(files):5d}")
        if files:
            samples.append((class_name, files[0]))

    print(f"\ntotal images: {total}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skip sample grid.")
        return

    out_dir = data_root.parents[2] / "outputs" / "part2_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eurosat_samples.png"

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, (class_name, image_path) in zip(axes.ravel(), samples):
        image = Image.open(image_path).convert("RGB")
        ax.imshow(image)
        ax.set_title(class_name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("sample grid saved:", out_path)


if __name__ == "__main__":
    main()

