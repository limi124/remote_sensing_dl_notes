"""Build the Part 3 tutorial notebook from reproducible cell sources."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "practices" / "03-remote-sensing-data-pipeline.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.x"}
nb["cells"] = [
    nbf.v4.new_markdown_cell("""# Part 3: Remote Sensing Data Pipeline

从带地理参考的多波段影像出发，跑通：

`GeoTIFF → CRS/transform → vector → mask → tiles → augmentation → Dataset`

**适合读者**：已完成 Part 1–2、了解 NumPy 和 PyTorch 基本 shape。  
**完成标准**：所有单元格可顺序运行，最终得到 `[C,H,W]` 影像 tile 与 `[H,W]` mask。"""),
    nbf.v4.new_markdown_cell("""## 学习路线

1. 创建无需下载的合成遥感场景
2. 读取 GeoTIFF 元数据与波段
3. 像素坐标和地图坐标互转
4. 矢量标注栅格化
5. 滑窗切片并处理边界
6. 影像/mask 同步增强
7. 构造 PyTorch Dataset
8. 完成练习与质量检查"""),
    nbf.v4.new_code_cell("""from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# 兼容从仓库根目录或 notebooks/practices 启动。
cwd = Path.cwd().resolve()
root_override = os.environ.get("REMOTE_SENSING_DL_ROOT")
ROOT = Path(root_override).resolve() if root_override else (
    cwd if (cwd / "scripts" / "part3_data_pipeline").exists() else cwd.parents[1]
)
SCRIPT_DIR = ROOT / "scripts" / "part3_data_pipeline"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (SegmentationTileDataset, create_demo_scene, rasterize_geojson,
                      read_scene, synchronized_augment, write_tiles)

output_override = os.environ.get("PART3_OUTPUT_DIR")
OUTPUT_DIR = Path(output_override).resolve() if output_override else ROOT / "outputs" / "part3_data_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("repo root:", ROOT)
print("rasterio:", rasterio.__version__)"""),
    nbf.v4.new_markdown_cell("""## 1. 创建一个可复现的多波段场景

示例生成 4 波段 `uint16` GeoTIFF、UTM 50N 地理参考，以及两个 polygon 类别。真实项目中只需把路径替换成自己的数据。"""),
    nbf.v4.new_code_cell("""raster_path, vector_path = create_demo_scene(OUTPUT_DIR / "scene")
image, profile = read_scene(raster_path)
print("image:", image.shape, image.dtype)
print("CRS:", profile["crs"])
print("transform:", profile["transform"])
print("range:", int(image.min()), int(image.max()))"""),
    nbf.v4.new_markdown_cell("""## 2. 坐标转换

`row/col` 是数组位置，`x/y` 是地图坐标。不要把 `(row, col)` 和 `(x, y)` 的顺序混用。"""),
    nbf.v4.new_code_cell("""with rasterio.open(raster_path) as src:
    row, col = 100, 120
    x, y = src.xy(row, col)
    recovered_row, recovered_col = src.index(x, y)
    print(f"pixel ({row}, {col}) -> map ({x:.1f}, {y:.1f})")
    print("round trip:", (recovered_row, recovered_col))
    assert (row, col) == (recovered_row, recovered_col)"""),
    nbf.v4.new_markdown_cell("""## 3. 矢量标注栅格化并叠加检查

`mask=0` 是背景，`1` 是 water，`2` 是 built-up。栅格化时以 GeoTIFF 的 shape 和 transform 为唯一参考。"""),
    nbf.v4.new_code_cell("""mask = rasterize_geojson(vector_path, raster_path)
print("mask:", mask.shape, mask.dtype, "classes:", np.unique(mask))

rgb = np.moveaxis(image[:3].astype("float32"), 0, -1)
lo, hi = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(rgb); axes[0].set_title("RGB composite")
axes[1].imshow(rgb); axes[1].imshow(mask, alpha=0.45, vmin=0, vmax=2, cmap="viridis")
axes[1].set_title("Aligned labels")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()"""),
    nbf.v4.new_markdown_cell("""## 4. 滑窗切片

这里用 `128×128` 且 stride=128，256×256 场景得到 4 对严格对齐的 image/mask tile。`write_tiles` 使用 window 读取影像，适合扩展到大图。"""),
    nbf.v4.new_code_cell("""pairs = write_tiles(raster_path, mask, OUTPUT_DIR / "tiles", tile_size=128, stride=128)
tile_image = np.load(pairs[0][0])
tile_mask = np.load(pairs[0][1])
print("tile pairs:", len(pairs))
print("image tile:", tile_image.shape, tile_image.dtype)
print("mask tile:", tile_mask.shape, tile_mask.dtype, np.unique(tile_mask))"""),
    nbf.v4.new_markdown_cell("""## 5. 同步增强

空间变换必须同时作用于影像与 mask。mask 不做颜色增强，缩放时必须使用 nearest-neighbor。"""),
    nbf.v4.new_code_cell("""aug_image, aug_mask = synchronized_augment(tile_image, tile_mask, horizontal=True, quarter_turns=1)
assert aug_image.shape[1:] == aug_mask.shape
assert set(np.unique(aug_mask)).issubset(set(np.unique(mask)))
print("augmentation alignment check passed:", aug_image.shape, aug_mask.shape)"""),
    nbf.v4.new_markdown_cell("""## 6. Dataset 与 batch shape

安装 PyTorch 时返回 tensor；如果当前环境未安装，则返回同 shape 的 NumPy 数组，地理数据流水线仍可完整验证。"""),
    nbf.v4.new_code_cell("""dataset = SegmentationTileDataset(OUTPUT_DIR / "tiles", augment=True)
sample_image, sample_mask = dataset[0]
print("dataset length:", len(dataset))
print("sample image:", type(sample_image).__name__, sample_image.shape, sample_image.dtype)
print("sample mask:", type(sample_mask).__name__, sample_mask.shape, sample_mask.dtype)
assert tuple(sample_image.shape) == (4, 128, 128)
assert tuple(sample_mask.shape) == (128, 128)"""),
    nbf.v4.new_markdown_cell("""## 练习

把 `tile_size` 改为 64、`stride` 改为 32：

1. 预测会生成多少个 window；
2. 运行并验证 image/mask 数量相等；
3. 思考为什么不能把高度重叠的 tile 随机分到 train 和 val。"""),
    nbf.v4.new_code_cell("""# 答案脚手架：取消注释后运行
# exercise_pairs = write_tiles(
#     raster_path, mask, OUTPUT_DIR / "exercise_tiles", tile_size=64, stride=32
# )
# print("exercise tile pairs:", len(exercise_pairs))

# 256x256 场景按 range(0, 256, 32) 在两个方向各取 8 个起点，因此是 64 对。
expected = 8 * 8
print("expected tile pairs:", expected)"""),
    nbf.v4.new_markdown_cell("""## 常见坑与下一步

- **CRS 相同不代表网格相同**：还要核对 transform、resolution、width/height。
- **mask 不能双线性插值**：会制造非法类别。
- **先切后随机划分会泄漏**：应按 scene、区域或空间块划分。
- **不要默认除以 255**：先确认数据 dtype、scale 和 NoData。

下一章将把这里生成的 `[C,H,W]` image 和 `[H,W]` mask 送入 U-Net。"""),
]

nbf.write(nb, OUT)
print(f"Wrote {OUT}")
