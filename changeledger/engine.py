from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .alignment import align_images, decode_image
from .detection import detect_changes
from .regions import extract_regions


ENGINE_VERSION = "0.1.0"
METHOD_NAME = "ORB alignment + pixel-difference baseline"


def compare_images(
    before_image: bytes | bytearray | np.ndarray,
    after_image: bytes | bytearray | np.ndarray,
    *,
    threshold: int = 25,
    min_area: int = 100,
    min_area_fraction: float = 0.0005,
    max_regions: int = 100,
) -> dict[str, Any]:
    """
    Compare a before/after image pair.

    Inputs may be encoded image bytes or OpenCV/NumPy image arrays.

    Returns:
        report: JSON-serializable metadata and candidate regions
        images: RGB visualizations and the single-channel change mask

    Coordinates use the BEFORE image's pixel coordinate system.
    """
    before = decode_image(before_image)
    after = decode_image(after_image)

    aligned = align_images(before, after)

    detection = detect_changes(
        aligned.before,
        aligned.after_aligned,
        aligned.valid_mask,
        threshold=threshold,
    )

    regions = extract_regions(
        detection["mask"],
        min_area=min_area,
        min_area_fraction=min_area_fraction,
        max_regions=max_regions,
    )

    height, width = before.shape[:2]

    annotated = aligned.after_aligned.copy()
    for region in regions:
        x, y, box_width, box_height = region["bbox_xywh"]
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            annotated,
            str(region["id"]),
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    overlay = before.copy()
    overlay[detection["mask"] > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(before, 0.6, overlay, 0.4, 0)

    changed_pixels = int(np.count_nonzero(detection["mask"]))
    valid_pixels = int(np.count_nonzero(aligned.valid_mask))
    valid_changed_pixels = int(
        np.count_nonzero(
            (detection["mask"] > 0) & (aligned.valid_mask > 0)
        )
    )

    report = {
        "engine_version": ENGINE_VERSION,
        "method": METHOD_NAME,
        "image_size": {
            "width": int(width),
            "height": int(height),
        },
        "parameters": {
            "threshold": int(threshold),
            "min_area": int(min_area),
            "min_area_fraction": float(min_area_fraction),
            "max_regions": int(max_regions),
        },
        "alignment": aligned.diagnostics,
        "valid_pixel_fraction": round(
            float(valid_pixels / (width * height)), 6
        ),
        "changed_pixel_count": changed_pixels,
        "changed_pixel_fraction": round(
            float(valid_changed_pixels / max(valid_pixels, 1)), 6
        ),
        "candidate_region_count": len(regions),
        "regions": regions,
        "limitations": [
            "This is an ORB and pixel-difference baseline, not the learned WACV model.",
            "Regions indicate candidate image differences, not verified real-world changes.",
            "Lighting, shadows, perspective, occlusion, and moving objects may affect results.",
            "Review the alignment and candidate regions before relying on the report.",
        ],
    }

    def to_rgb(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return {
        "report": report,
        "images": {
            "before": to_rgb(before),
            "after_aligned": to_rgb(aligned.after_aligned),
            "annotated": to_rgb(annotated),
            "overlay": to_rgb(overlay),
            "change_mask": detection["mask"].copy(),
            "difference_heatmap": to_rgb(detection["heatmap"]),
        },
    }
