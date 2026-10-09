from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize report metadata as readable JSON."""
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report_to_json(report), encoding="utf-8")
    return destination


def save_image(
    image: np.ndarray,
    path: str | Path,
    *,
    image_format: str = "PNG",
) -> Path:
    """Save an RGB or grayscale NumPy image."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 3 and image.shape[2] == 3:
        output = Image.fromarray(image.astype(np.uint8), mode="RGB")
    elif image.ndim == 2:
        output = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        raise ValueError("Expected an RGB or grayscale image.")

    output.save(destination, format=image_format)
    return destination


def encode_png(image: np.ndarray) -> bytes:
    """Encode an RGB or grayscale image to PNG bytes."""
    if image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 2:
        bgr = image
    else:
        raise ValueError("Expected an RGB or grayscale image.")

    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Could not encode image as PNG.")
    return encoded.tobytes()
