"""Shared contract for every checker (DIP, CNN, VLM), the app and the evaluation.

Do not change this file from a unit agent; it is owned by the orchestrator.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFECTS = ["blur", "noise", "dark", "non_white_bg", "bad_framing", "cropped"]

DEFECT_DESCRIPTIONS = {
    "blur": "Image is out of focus or motion-blurred",
    "noise": "Visible grain / sensor noise",
    "dark": "Under-exposed or low contrast",
    "non_white_bg": "Background is not plain white",
    "bad_framing": "Product too small in the frame or off-centre",
    "cropped": "Product is cut off at the image edge",
}

IMG_SIZE = 512  # all generated dataset images are IMG_SIZE x IMG_SIZE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


def empty_result(method: str) -> dict:
    """Result skeleton every checker fills in. All defects start as pass."""
    return {
        "method": method,
        "defects": {d: {"fail": False, "score": None, "reason": ""} for d in DEFECTS},
        "passed": True,
        "intermediates": {},
        "time_s": 0.0,
    }


def finalize(result: dict) -> dict:
    """Set result['passed'] from the per-defect flags."""
    result["passed"] = not any(v["fail"] for v in result["defects"].values())
    return result


def load_image(path: str | Path) -> np.ndarray:
    """Load an image as BGR uint8. Raises if unreadable."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR uint8 image."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def save_image(path: str | Path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
