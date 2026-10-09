# ChangeLedger

A research-informed computer vision engine for comparing before-and-after images and identifying candidate regions of change.

## Current Status

- Version: 0.1.0
- Milestone 1: Baseline Engine
- Method: ORB feature matching, homography alignment, pixel-difference detection, morphology, and connected components.

## Features

- Accepts image bytes and NumPy arrays.
- Aligns before-and-after images.
- Detects candidate changed regions.
- Generates bounding boxes and region statistics.
- Produces annotated images, overlays, heatmaps, and masks.
- Exports JSON reports.
- Includes automated tests.

## Project Structure

    changeledger/
    ├── changeledger/
    │   ├── __init__.py
    │   ├── alignment.py
    │   ├── detection.py
    │   ├── regions.py
    │   ├── engine.py
    │   └── reporting.py
    ├── tests/
    ├── examples/
    ├── requirements.txt
    ├── pyproject.toml
    ├── README.md
    └── .gitignore

## Installation

Python 3.10 or newer is recommended.

    python -m pip install -e ".[dev]"

## Usage

    from changeledger import compare_images

    result = compare_images(
        before_image,
        after_image
    )

    report = result["report"]
    images = result["images"]

    print(report["candidate_region_count"])
    print(report["regions"])

Pass image bytes or NumPy arrays as supported by the engine.

## Outputs

The engine provides a JSON-serializable report and image arrays for the original image, aligned image, annotated results, overlay, change mask, and difference heatmap.

## Tests

Run:

    python -m pytest -v

## Research Context

This project is informed by research into visual correspondence and object-level change detection, including the WACV 2025 paper "Improving Zero-Shot Object-Level Change Detection by Incorporating Visual Correspondence."

This implementation is a classical computer-vision baseline, not a reproduction of the paper's learned model.

## Limitations

- Lighting, shadows, reflections, and camera viewpoint differences may cause false detections.
- Alignment can fail when images have insufficient matching features.
- Detected regions are candidates, not semantically verified objects.
- Results should be reviewed before operational use.

## Roadmap

1. Package and test the baseline engine.
2. Build a user interface.
3. Improve change-region filtering.
4. Add downloadable inspection reports.
5. Evaluate against labeled image pairs.

## License

No license has been specified. Do not assume this project is open-source licensed until a license is added.
