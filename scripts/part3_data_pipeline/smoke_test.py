"""Run Part 3 end-to-end without external data."""

from pathlib import Path
import shutil
import tempfile

import numpy as np

from pipeline import (SegmentationTileDataset, create_demo_scene, rasterize_geojson,
                      read_scene, write_tiles)


def main() -> None:
    temp_dir = Path(tempfile.mkdtemp(prefix="part3_pipeline_"))
    try:
        raster_path, vector_path = create_demo_scene(temp_dir / "scene")
        image, profile = read_scene(raster_path)
        mask = rasterize_geojson(vector_path, raster_path)
        pairs = write_tiles(raster_path, mask, temp_dir / "tiles", tile_size=128, stride=128)
        dataset = SegmentationTileDataset(temp_dir / "tiles", augment=True)
        sample_image, sample_mask = dataset[0]
        assert image.shape == (4, 256, 256)
        assert profile["crs"].to_epsg() == 32650
        assert set(np.unique(mask)) == {0, 1, 2}
        assert len(pairs) == len(dataset) == 4
        assert tuple(sample_image.shape) == (4, 128, 128)
        assert tuple(sample_mask.shape) == (128, 128)
        print("Part 3 smoke test passed: GeoTIFF -> mask -> tiles -> dataset")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
