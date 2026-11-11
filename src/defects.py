"""Synthetic defect generator. Self-contained (numpy + cv2 only) so it can be
inlined into the Kaggle training kernel.

Pipeline for each source photo:
  normalize_product()  -> tight crop of the product, centred on a white
                          IMG_SIZE canvas, longest side = 80% of the canvas
  apply_defects()      -> applies the chosen defects with random strength
"""
from __future__ import annotations

import cv2
import numpy as np

DEFECTS = ["blur", "noise", "dark", "non_white_bg", "bad_framing", "cropped"]
IMG_SIZE = 512
WHITE_T = 235  # a pixel is "white" if every channel > WHITE_T


def product_mask(img: np.ndarray) -> np.ndarray:
    """uint8 mask (255 = product) for a photo on a white background."""
    nonwhite = (img.min(axis=2) <= WHITE_T).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    nonwhite = cv2.morphologyEx(nonwhite, cv2.MORPH_CLOSE, k)
    # fill holes (e.g. white parts of the product) from the border inwards
    h, w = nonwhite.shape
    flood = nonwhite.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    return nonwhite | cv2.bitwise_not(flood)


def is_clean_source(img: np.ndarray) -> bool:
    """Accept only photos with a white border and a reasonably sized product."""
    b = 10
    border = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                             img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    if (border.min(axis=1) > WHITE_T).mean() < 0.97:
        return False
    frac = (product_mask(img) > 0).mean()
    return 0.03 < frac < 0.9


def normalize_product(img: np.ndarray, fill: float = 0.8) -> np.ndarray:
    """Tight-crop the product and centre it on a white IMG_SIZE canvas."""
    s = 1024 / max(img.shape[:2])
    if s < 1:  # speed: sources are 1800x2400, output is only 512x512
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = product_mask(img)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return place_on_canvas(crop, fill, (0.5, 0.5))


def place_on_canvas(crop: np.ndarray, fill: float, center: tuple[float, float],
                    bg: np.ndarray | None = None) -> np.ndarray:
    """Resize crop so its longest side = fill*IMG_SIZE and paste its centre at
    `center` (fractions of the canvas). Parts outside the canvas are clipped."""
    h, w = crop.shape[:2]
    s = fill * IMG_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8) if bg is None else bg.copy()
    cx, cy = int(center[0] * IMG_SIZE), int(center[1] * IMG_SIZE)
    x0, y0 = cx - nw // 2, cy - nh // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(IMG_SIZE, x0 + nw), min(IMG_SIZE, y0 + nh)
    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = crop[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return canvas


# ---------------------------------------------------------------- defects
def _bad_framing(img, rng):
    m = product_mask(img)
    ys, xs = np.nonzero(m)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    fill = rng.uniform(0.25, 0.45)
    if rng.random() < 0.5:  # small and centred
        c = (0.5, 0.5)
    else:  # off-centre, still fully visible
        half = fill / 2
        c = tuple(rng.choice([rng.uniform(half + 0.02, 0.32), rng.uniform(0.68, 1 - half - 0.02)])
                  for _ in range(2))
    return place_on_canvas(crop, fill, c)


def _cropped(img, rng):
    m = product_mask(img)
    ys, xs = np.nonzero(m)
    crop = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    fill = rng.uniform(0.75, 0.95)
    shift = rng.uniform(0.18, 0.35)  # 18-35% of canvas past the centre -> product cut off
    side = rng.integers(4)
    c = [0.5, 0.5]
    c[side % 2] += shift if side < 2 else -shift
    return place_on_canvas(crop, fill, tuple(c))


def _non_white_bg(img, rng):
    m = product_mask(img) > 0
    kind = rng.integers(3)
    h, w = img.shape[:2]
    if kind == 0:  # solid colour
        col = rng.integers(40, 220, 3)
        bg = np.empty_like(img); bg[:] = col
    elif kind == 1:  # gradient
        a, b = rng.integers(30, 230, 3), rng.integers(30, 230, 3)
        t = np.linspace(0, 1, h)[:, None, None]
        bg = np.broadcast_to(a * (1 - t) + b * t, img.shape).astype(np.uint8)
    else:  # cluttered texture (smoothed random blobs)
        small = rng.integers(0, 255, (h // 16, w // 16, 3)).astype(np.uint8)
        bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        bg = cv2.GaussianBlur(bg, (0, 0), 3)
    out = bg.copy()
    out[m] = img[m]
    return out


def _blur(img, rng):
    if rng.random() < 0.6:
        return cv2.GaussianBlur(img, (0, 0), rng.uniform(2.5, 6.0))
    k = int(rng.integers(13, 31))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _noise(img, rng):
    sigma = rng.uniform(15, 35)
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _dark(img, rng):
    f = img.astype(np.float32) / 255.0
    if rng.random() < 0.5:
        f = (f ** rng.uniform(1.5, 2.5)) * rng.uniform(0.4, 0.65)  # under-exposed
    else:
        c = rng.uniform(0.3, 0.5)                        # low contrast, greyish
        f = 0.5 * (1 - c) + f * c - rng.uniform(0.05, 0.2)
    return np.clip(f * 255, 0, 255).astype(np.uint8)


# Order matters: geometry first, then background, then photometric defects.
_ORDER = ["bad_framing", "cropped", "non_white_bg", "dark", "blur", "noise"]
_FNS = {"blur": _blur, "noise": _noise, "dark": _dark, "non_white_bg": _non_white_bg,
        "bad_framing": _bad_framing, "cropped": _cropped}


def sample_defects(rng: np.random.Generator, p_clean: float = 0.35) -> list[str]:
    """35% clean, otherwise 1-2 defects; bad_framing and cropped never together."""
    if rng.random() < p_clean:
        return []
    k = 1 if rng.random() < 0.6 else 2
    while True:
        chosen = list(rng.choice(DEFECTS, size=k, replace=False))
        if not {"bad_framing", "cropped"} <= set(chosen):
            return chosen


def apply_defects(img: np.ndarray, defects: list[str], rng: np.random.Generator) -> np.ndarray:
    """img must already be normalize_product() output."""
    out = img.copy()
    for d in _ORDER:
        if d in defects:
            out = _FNS[d](out, rng)
    return out


def labels_row(defects: list[str]) -> dict:
    return {d: int(d in defects) for d in DEFECTS}
