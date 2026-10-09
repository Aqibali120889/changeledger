from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


class ImageInputError(ValueError):
    """Raised when an input image cannot be decoded or is invalid."""


@dataclass
class AlignmentResult:
    before: np.ndarray
    after_aligned: np.ndarray
    valid_mask: np.ndarray
    diagnostics: dict[str, Any]


def decode_image(image_data: bytes | bytearray | np.ndarray) -> np.ndarray:
    """Decode encoded image bytes into a BGR OpenCV image."""
    if isinstance(image_data, np.ndarray):
        image = image_data.copy()
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.ndim != 3 or image.shape[2] != 3:
            raise ImageInputError("Expected a grayscale, BGR, or BGRA image.")
    else:
        if not isinstance(image_data, (bytes, bytearray)):
            raise ImageInputError("Image input must be bytes or a NumPy array.")
        if not image_data:
            raise ImageInputError("The uploaded image is empty.")

        buffer = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        if image is None:
            raise ImageInputError("Could not decode the image file.")

    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ImageInputError("Image dimensions must be at least 8 × 8 pixels.")

    return np.ascontiguousarray(image)


def align_images(
    before: np.ndarray,
    after: np.ndarray,
    *,
    max_features: int = 3000,
    ratio_test: float = 0.75,
    ransac_reprojection_threshold: float = 5.0,
    min_matches: int = 8,
    min_inliers: int = 8,
    min_inlier_ratio: float = 0.15,
) -> AlignmentResult:
    """
    Align the after image to the before image using ORB and homography.

    If there are insufficient reliable matches, use a resized after image
    without geometric alignment. Diagnostics make this fallback explicit.
    """
    if max_features < 1 or min_matches < 4 or min_inliers < 4:
        raise ValueError("Feature and match limits are too small.")
    if not 0 < ratio_test < 1:
        raise ValueError("ratio_test must be between 0 and 1.")
    if ransac_reprojection_threshold <= 0:
        raise ValueError("RANSAC threshold must be positive.")

    h, w = before.shape[:2]
    resized_after = cv2.resize(after, (w, h))

    diagnostics: dict[str, Any] = {
        "method": "resize_only",
        "keypoints_before": 0,
        "keypoints_after": 0,
        "good_matches": 0,
        "homography_inliers": 0,
        "inlier_ratio": 0.0,
        "fallback_reason": None,
    }

    identity_mask = np.full((h, w), 255, dtype=np.uint8)

    def fallback(reason: str) -> AlignmentResult:
        diagnostics["fallback_reason"] = reason
        return AlignmentResult(
            before=before,
            after_aligned=resized_after,
            valid_mask=identity_mask,
            diagnostics=diagnostics.copy(),
        )

    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(resized_after, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kp1, des1 = orb.detectAndCompute(gray_before, None)
    kp2, des2 = orb.detectAndCompute(gray_after, None)

    diagnostics["keypoints_before"] = len(kp1)
    diagnostics["keypoints_after"] = len(kp2)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return fallback("Insufficient image features.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des1, des2, k=2)
    good = [
        m
        for pair in pairs
        if len(pair) == 2
        for m, n in [pair]
        if m.distance < ratio_test * n.distance
    ]
    diagnostics["good_matches"] = len(good)

    if len(good) < min_matches:
        return fallback("Insufficient feature matches.")

    # Homography maps points in AFTER to corresponding points in BEFORE.
    points_before = np.float32([kp1[m.queryIdx].pt for m in good])
    points_after = np.float32([kp2[m.trainIdx].pt for m in good])

    H, inlier_mask = cv2.findHomography(
        points_after,
        points_before,
        cv2.RANSAC,
        ransac_reprojection_threshold,
    )

    if H is None or inlier_mask is None or not np.isfinite(H).all():
        return fallback("Homography estimation failed.")

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / len(good)
    diagnostics["homography_inliers"] = inliers
    diagnostics["inlier_ratio"] = round(inlier_ratio, 4)

    if inliers < min_inliers or inlier_ratio < min_inlier_ratio:
        return fallback("Homography did not pass reliability checks.")

    # Reject numerically unstable or implausibly large projective warps.
    if abs(H[2, 2]) < 1e-8:
        return fallback("Homography is numerically unstable.")

    H = H / H[2, 2]
    if not np.isfinite(H).all() or np.linalg.cond(H) > 1e10:
        return fallback("Homography is ill-conditioned.")

    corners = np.float32([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H)
    if projected is None or not np.isfinite(projected).all():
        return fallback("Projected image geometry is invalid.")

    aligned = cv2.warpPerspective(
        resized_after, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    source_mask = np.full((h, w), 255, dtype=np.uint8)
    valid_mask = cv2.warpPerspective(
        source_mask, H, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    diagnostics["method"] = "orb_homography"
    diagnostics["fallback_reason"] = None

    return AlignmentResult(
        before=before,
        after_aligned=aligned,
        valid_mask=valid_mask,
        diagnostics=diagnostics,
    )
