import io
import json

import numpy as np
import pytest
from PIL import Image

from changeledger import (
    ImageInputError,
    compare_images,
    encode_png,
    report_to_json,
)


def make_png(seed=1, changed=False):
    """Create a reproducible synthetic test image."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)

    # Add stable shapes to provide repeatable visual features.
    import cv2
    cv2.rectangle(image, (30, 30), (100, 100), (20, 180, 240), -1)
    cv2.circle(image, (180, 170), 30, (220, 30, 60), -1)

    if changed:
        cv2.rectangle(image, (120, 60), (170, 120), (0, 255, 0), -1)

    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def test_rejects_empty_image():
    with pytest.raises(ImageInputError):
        compare_images(b"", make_png())


def test_rejects_invalid_image():
    with pytest.raises(ImageInputError):
        compare_images(b"not an image", make_png())


def test_compare_returns_expected_schema():
    result = compare_images(
        make_png(),
        make_png(changed=True),
        threshold=25,
        min_area=20,
    )

    assert set(result) == {"report", "images"}

    report = result["report"]
    assert report["engine_version"]
    assert report["method"]
    assert report["image_size"]["width"] == 256
    assert report["image_size"]["height"] == 256
    assert "alignment" in report
    assert "regions" in report
    assert isinstance(report["regions"], list)

    images = result["images"]
    assert images["before"].shape == (256, 256, 3)
    assert images["after_aligned"].shape == (256, 256, 3)
    assert images["annotated"].shape == (256, 256, 3)
    assert images["overlay"].shape == (256, 256, 3)
    assert images["change_mask"].shape == (256, 256)


def test_report_is_valid_json():
    result = compare_images(make_png(), make_png(changed=True))
    parsed = json.loads(report_to_json(result["report"]))

    assert parsed["candidate_region_count"] == len(parsed["regions"])


def test_png_export_has_valid_signature():
    result = compare_images(make_png(), make_png(changed=True))
    encoded = encode_png(result["images"]["annotated"])

    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")


def test_numpy_image_input():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    result = compare_images(image, image)

    assert result["report"]["image_size"] == {
        "width": 64,
        "height": 64,
    }
