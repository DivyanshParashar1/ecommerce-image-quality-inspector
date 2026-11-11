"""CNN defect checker: EfficientNet-B0, 6 sigmoid outputs (one per common.DEFECTS)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np

from common import DEFECTS, MODELS_DIR, empty_result, finalize

INPUT_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_cache: dict = {}


def _model_dir() -> Path:
    return Path(os.environ.get("CNN_MODEL_DIR") or MODELS_DIR)


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load():
    d = _model_dir()
    key = str(d)
    if key in _cache:
        return _cache[key]
    import torch
    import torchvision

    wpath = d / "cnn.pt"
    if not wpath.exists():
        raise FileNotFoundError(
            f"CNN weights not found at {wpath}. Train on Kaggle and place cnn.pt in {d}.")
    model = torchvision.models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(1280, len(DEFECTS))
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    dev = _device()
    model.to(dev).eval()
    th = {}
    tpath = d / "thresholds.json"
    if tpath.exists():
        th = json.loads(tpath.read_text())
    thresholds = [float(th.get(n, 0.5)) for n in DEFECTS]
    _cache[key] = (model, dev, thresholds)
    return _cache[key]


def is_available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _probs(imgs: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    import torch
    model, dev, thresholds = _load()
    x = torch.from_numpy(np.stack([_preprocess(i) for i in imgs])).to(dev)
    with torch.no_grad():
        p = torch.sigmoid(model(x)).float().cpu().numpy()
    return p, thresholds


def _build(p: np.ndarray, thresholds: list[float], t: float) -> dict:
    res = empty_result("cnn")
    for i, name in enumerate(DEFECTS):
        pi, th = float(p[i]), thresholds[i]
        fail = pi >= th
        res["defects"][name] = {
            "fail": bool(fail),
            "score": pi,
            "reason": f"p={pi:.2f} {'≥' if fail else '<'} {th:.2f}",
        }
    res["time_s"] = t
    return finalize(res)


def check(img_bgr: np.ndarray) -> dict:
    t0 = time.perf_counter()
    p, th = _probs([img_bgr])
    return _build(p[0], th, time.perf_counter() - t0)


def check_batch(imgs: list[np.ndarray], batch_size: int = 32) -> list[dict]:
    out = []
    for s in range(0, len(imgs), batch_size):
        chunk = imgs[s:s + batch_size]
        t0 = time.perf_counter()
        p, th = _probs(chunk)
        dt = (time.perf_counter() - t0) / len(chunk)
        out += [_build(p[i], th, dt) for i in range(len(chunk))]
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python src/cnn_checker.py <image>")
    from common import load_image
    r = check(load_image(sys.argv[1]))
    r.pop("intermediates", None)
    print(json.dumps(r, indent=2, ensure_ascii=False))
