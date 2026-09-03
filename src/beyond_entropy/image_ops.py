from __future__ import annotations

from typing import Any

from .schema import BBox


def normalized_crop_resized_to_source(image: Any, bbox: BBox) -> Any:
    """Crop a normalized box and resize it to the source raster dimensions."""

    from PIL import Image

    width, height = image.size
    pixel_box = (
        round(bbox.x1 * width),
        round(bbox.y1 * height),
        round(bbox.x2 * width),
        round(bbox.y2 * height),
    )
    crop = image.crop(pixel_box)
    return crop.resize((width, height), Image.Resampling.LANCZOS)
