"""Reusable building blocks for Part 3's remote-sensing data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window


def create_demo_scene(out_dir: str | Path, size: int = 256) -> tuple[Path, Path]:
    """Create a deterministic 4-band GeoTIFF and GeoJSON labels."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raster_path = out_dir / "demo_multispectral.tif"
    vector_path = out_dir / "demo_labels.geojson"

    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[:size, :size]
    base = (xx + yy) / (2 * size)
    image = np.stack(
        [base, xx / size, yy / size, 0.6 * base + 0.4 * rng.random((size, size))]
    )
    image = np.clip(image * 10000, 0, 10000).astype("uint16")
    transform = from_origin(500000, 3500000, 10, 10)
    profile = {
        "driver": "GTiff", "height": size, "width": size, "count": 4,
        "dtype": "uint16", "crs": "EPSG:32650", "transform": transform,
        "nodata": 0, "compress": "deflate",
    }
    with rasterio.open(raster_path, "w", **profile) as dst:
        dst.write(image)
        dst.set_band_description(1, "red")
        dst.set_band_description(2, "green")
        dst.set_band_description(3, "blue")
        dst.set_band_description(4, "nir")

    def world_polygon(c0: int, r0: int, c1: int, r1: int) -> list[list[float]]:
        corners = [(c0, r0), (c1, r0), (c1, r1), (c0, r1), (c0, r0)]
        return [[transform * (col, row) for col, row in corners]]

    features = []
    for class_id, name, box in [
        (1, "water", (25, 35, 115, 135)),
        (2, "built_up", (145, 95, 235, 220)),
    ]:
        features.append({
            "type": "Feature",
            "properties": {"class_id": class_id, "name": name},
            "geometry": {"type": "Polygon", "coordinates": world_polygon(*box)},
        })
    vector_path.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "demo_labels",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32650"}},
        "features": features,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return raster_path, vector_path


def read_scene(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read all bands as [C, H, W] and return a copy of the raster profile."""
    with rasterio.open(path) as src:
        return src.read(), src.profile.copy()


def rasterize_geojson(vector_path: str | Path, reference_raster: str | Path) -> np.ndarray:
    """Burn GeoJSON class_id values into a mask aligned to a reference raster."""
    collection = json.loads(Path(vector_path).read_text(encoding="utf-8"))
    shapes = [(f["geometry"], int(f["properties"]["class_id"])) for f in collection["features"]]
    with rasterio.open(reference_raster) as src:
        return rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform,
                         fill=0, dtype="uint8")


def iter_windows(width: int, height: int, tile_size: int, stride: int) -> Iterable[Window]:
    """Yield fixed-size windows; border windows may extend beyond the scene."""
    for row in range(0, height, stride):
        for col in range(0, width, stride):
            yield Window(col, row, tile_size, tile_size)


def write_tiles(raster_path: str | Path, mask: np.ndarray, out_dir: str | Path,
                tile_size: int = 128, stride: int = 128) -> list[tuple[Path, Path]]:
    """Write aligned image/mask tiles, padding scene borders with zeros."""
    out_dir = Path(out_dir)
    image_dir, mask_dir = out_dir / "images", out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    with rasterio.open(raster_path) as src:
        for index, window in enumerate(iter_windows(src.width, src.height, tile_size, stride)):
            image = src.read(window=window, boundless=True, fill_value=0)
            r0, c0 = int(window.row_off), int(window.col_off)
            mask_tile = np.zeros((tile_size, tile_size), dtype="uint8")
            valid_h = max(0, min(tile_size, mask.shape[0] - r0))
            valid_w = max(0, min(tile_size, mask.shape[1] - c0))
            mask_tile[:valid_h, :valid_w] = mask[r0:r0 + valid_h, c0:c0 + valid_w]
            image_path, mask_path = image_dir / f"tile_{index:03d}.npy", mask_dir / f"tile_{index:03d}.npy"
            np.save(image_path, image)
            np.save(mask_path, mask_tile)
            pairs.append((image_path, mask_path))
    return pairs


def synchronized_augment(image: np.ndarray, mask: np.ndarray, *, horizontal: bool = False,
                         vertical: bool = False, quarter_turns: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical spatial transforms to [C,H,W] imagery and [H,W] masks."""
    if horizontal:
        image, mask = image[:, :, ::-1], mask[:, ::-1]
    if vertical:
        image, mask = image[:, ::-1, :], mask[::-1, :]
    k = quarter_turns % 4
    if k:
        image, mask = np.rot90(image, k, axes=(1, 2)), np.rot90(mask, k)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


class SegmentationTileDataset:
    """Minimal paired-tile dataset; returns torch tensors when torch is installed."""

    def __init__(self, root: str | Path, augment: bool = False):
        root = Path(root)
        self.images = sorted((root / "images").glob("*.npy"))
        self.masks = sorted((root / "masks").glob("*.npy"))
        if not self.images or len(self.images) != len(self.masks):
            raise ValueError("Expected equal, non-empty images/ and masks/ tile folders")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = np.load(self.images[index]).astype("float32") / 10000.0
        mask = np.load(self.masks[index]).astype("int64")
        if self.augment:
            image, mask = synchronized_augment(image, mask, horizontal=index % 2 == 0,
                                                quarter_turns=index % 4)
        try:
            import torch
            return torch.from_numpy(image), torch.from_numpy(mask)
        except ImportError:
            return image, mask
