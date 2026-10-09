from __future__ import annotations

import cv2
import numpy as np


def detect_changes(
    before: np.ndarray,
    after_aligned: np.ndarray,
    valid_mask: np.ndarray,
    *,
    threshold: int = 25,
    blur_kernel: int = 5,
    open_kernel: int = 3,
    close_kernel: int = 7,
) -> dict[str, np.ndarray]:
    """Compute candidate pixel differences between aligned images."""
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255.")
    for name, size in [
        ("blur_kernel", blur_kernel),
        ("open_kernel", open_kernel),
        ("close_kernel", close_kernel),
    ]:
        if size < 1 or size % 2 == 0:
            raise ValueError(f"{name} must be a positive odd number.")

    if before.shape != after_aligned.shape:
        raise ValueError("Images must have the same dimensions.")
    if valid_mask.shape != before.shape[:2]:
        raise ValueError("valid_mask dimensions must match the images.")

    diff = cv2.absdiff(before, after_aligned)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(gray_diff, (blur_kernel, blur_kernel), 0)

    _, mask = cv2.threshold(
        smoothed, threshold, 255, cv2.THRESH_BINARY
    )

    mask[valid_mask == 0] = 0

    open_element = np.ones((open_kernel, open_kernel), np.uint8)
    close_element = np.ones((close_kernel, close_kernel), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_element)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_element)

    # Morphological operations can expand into invalid warp borders.
    mask[valid_mask == 0] = 0

    heatmap = cv2.applyColorMap(smoothed, cv2.COLORMAP_INFERNO)
    heatmap[valid_mask == 0] = 0

    return {
        "mask": mask,
        "difference_gray": gray_diff,
        "heatmap": heatmap,
    }
