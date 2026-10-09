from __future__ import annotations

import cv2
import numpy as np


def extract_regions(
    mask: np.ndarray,
    *,
    min_area: int = 100,
    min_area_fraction: float = 0.0005,
    max_regions: int = 100,
) -> list[dict]:
    """Extract connected candidate regions, sorted by pixel area."""
    if mask.ndim != 2:
        raise ValueError("Expected a single-channel binary mask.")
    if min_area < 1 or max_regions < 1:
        raise ValueError("min_area and max_regions must be positive.")
    if not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be between 0 and 1.")

    height, width = mask.shape
    minimum = max(min_area, int(height * width * min_area_fraction))

    binary = (mask > 0).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    regions = []

    for label_id in range(1, count):
        x, y, box_width, box_height, area = stats[label_id]

        if area < minimum:
            continue

        regions.append({
            "bbox_xywh": [
                int(x), int(y), int(box_width), int(box_height)
            ],
            "area_pixels": int(area),
            "area_fraction": round(float(area / (height * width)), 6),
            "centroid_xy": [
                round(float(centroids[label_id][0]), 2),
                round(float(centroids[label_id][1]), 2),
            ],
        })

    regions.sort(key=lambda region: region["area_pixels"], reverse=True)
    regions = regions[:max_regions]

    for index, region in enumerate(regions, start=1):
        region["id"] = index

    return regions
