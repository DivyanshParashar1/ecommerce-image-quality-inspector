"""Classical DIP quality checker (OpenCV + numpy only).

Public API
    features(img_bgr) -> dict[str, float]   raw scores (used by scripts/tune_dip.py)
    check(img_bgr)    -> dict               contract result (see PLAN.md), method="dip"
Thresholds come from models/dip_thresholds.json when present, else DEFAULT_THRESHOLDS.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cv2
import numpy as np

from common import DEFECTS, MODELS_DIR, empty_result, finalize

WORK_SIZE = 512
THRESHOLDS_PATH = MODELS_DIR / "dip_thresholds.json"

# defect -> (feature name used as score, direction, threshold, label)
#   direction "gt": fail when score > t ; "lt": fail when score < t
DEFAULT_THRESHOLDS = {
    "blur":         {"feature": "blur_score",    "dir": "lt", "t": 110.0},
    "noise":        {"feature": "noise_score",   "dir": "gt", "t": 4.0},
    "dark":         {"feature": "dark_score",    "dir": "lt", "t": 200.0},
    "non_white_bg": {"feature": "bg_nonwhite",   "dir": "gt", "t": 0.5},
    "bad_framing":  {"feature": "framing_score", "dir": "gt", "t": 0.3},
    "cropped":      {"feature": "crop_score",    "dir": "gt", "t": 0.04},
}
_MESSAGES = {
    "blur": ("Laplacian variance (near edges)", "blurry"),
    "noise": ("flat-region noise std", "noisy"),
    "dark": ("95th-percentile brightness", "under-exposed / low contrast"),
    "non_white_bg": ("non-white background fraction", "background not white"),
    "bad_framing": ("framing badness (small/off-centre)", "badly framed"),
    "cropped": ("border contact fraction", "product cropped"),
}


def load_thresholds() -> dict:
    th = {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()}
    try:
        if THRESHOLDS_PATH.exists():
            saved = json.loads(THRESHOLDS_PATH.read_text())
            for k, v in saved.items():
                if k in th:
                    th[k].update({kk: v[kk] for kk in ("feature", "dir", "t") if kk in v})
    except Exception:
        pass
    return th


# ------------------------------------------------------------------ helpers
def _resize(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    s = WORK_SIZE / max(h, w)
    if s != 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return img


def _border_ring(h, w, b=8):
    m = np.zeros((h, w), bool)
    m[:b] = m[-b:] = True
    m[:, :b] = m[:, -b:] = True
    return m


def _segment(img: np.ndarray):
    """Return (raw_mask, clean_mask, dist) as uint8 masks (255=product).

    Background model: k-means (K=3) on the border ring in LAB; clusters holding
    >=15% of border pixels are "background"; dist = min LAB distance to them.
    Works for white, coloured and gradient backgrounds."""
    h, w = img.shape[:2]
    smooth = cv2.GaussianBlur(img, (0, 0), 2.0)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = _border_ring(h, w)
    pts = lab[ring]
    k = 3
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(pts, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(labels.ravel(), minlength=k) / len(labels)
    keep = centers[share >= 0.15]
    if len(keep) == 0:
        keep = centers[[int(np.argmax(share))]]
    dist = np.min([np.linalg.norm(lab - c, axis=2) for c in keep], axis=0)
    d8 = np.clip(dist * 4, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 4.0, 14.0)  # floor so flat backgrounds are not split by noise
    raw = (dist > thr).astype(np.uint8) * 255

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k5)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k9)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (cc == big).astype(np.uint8) * 255
    else:
        m = np.zeros_like(m)
    # fill holes: flood the background from a padded corner
    p = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = np.zeros((p.shape[0] + 2, p.shape[1] + 2), np.uint8)
    flood = p.copy()
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    m = m | holes
    return raw, m, dist


def _hist_image(gray: np.ndarray) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = np.sqrt(hist)  # compress the dominant white-background peak
    H, W = 200, 512
    out = np.full((H + 20, W, 3), 255, np.uint8)
    hmax = max(hist.max(), 1e-6)
    for i, v in enumerate(hist):
        hh = int(v / hmax * (H - 5))
        cv2.rectangle(out, (2 * i, H - hh), (2 * i + 1, H), (90, 90, 90), -1)
    cv2.line(out, (2 * 30, 0), (2 * 30, H), (0, 0, 255), 1)
    cv2.putText(out, "0", (2, H + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(out, "255 (sqrt counts; red = 30)", (W - 190, H + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return out


# ------------------------------------------------------------------ analysis
def _analyse(img_in: np.ndarray):
    img = _resize(img_in)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    f: dict[str, float] = {}

    # ---- segmentation
    raw, mask, dist = _segment(img)
    mb = mask > 0
    area = float(mb.sum())
    f["mask_frac"] = area / (h * w)

    # ---- dark / exposure
    f["gray_mean"] = float(gray.mean())
    f["gray_std"] = float(gray.std())
    f["pct_below_30"] = float((gray < 30).mean())
    f["dark_score"] = float(np.percentile(gray, 95))  # brightness of the "white" background

    # ---- blur: Laplacian variance of a lightly denoised image near product edges
    den = cv2.medianBlur(gray, 3)
    lap = cv2.Laplacian(den.astype(np.float32), cv2.CV_32F, ksize=3)
    if area > 0:
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))) > 0
        region = near
    else:
        region = np.ones_like(mb)
    f["lap_var_global"] = float(lap.var())
    f["blur_score"] = float(lap[region].var())
    gx = cv2.Sobel(den, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(den, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.hypot(gx, gy)
    f["grad_p99"] = float(np.percentile(gm[region], 99))

    # ---- noise: residual vs median 3x3 inside flat regions
    resid = gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)
    g2 = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    gg = np.hypot(cv2.Sobel(g2, cv2.CV_32F, 1, 0), cv2.Sobel(g2, cv2.CV_32F, 0, 1))
    flat = (gg < 6.0).astype(np.uint8)
    flat = cv2.erode(flat, np.ones((5, 5), np.uint8)) > 0
    if flat.sum() < 500:  # fall back to the flattest 20% of pixels
        flat = gg <= np.percentile(gg, 20)
    f["noise_score"] = float(1.4826 * np.median(np.abs(resid[flat] - np.median(resid[flat]))))
    f["noise_std"] = float(resid[flat].std())
    f["flat_frac"] = float(flat.mean())

    # ---- background
    bgm = ~mb
    if bgm.sum() < 50:
        bgm = _border_ring(h, w, 4)
    L = lab[..., 0].astype(np.float32) * 100 / 255
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    Lref = max(float(np.percentile(gray, 99)) * 100 / 255, 1.0)
    bg_nonwhite_pix = (chroma > 12) | (L < 0.9 * Lref)
    bg_nonwhite_pix = bg_nonwhite_pix & bgm
    f["bg_nonwhite"] = float(bg_nonwhite_pix.sum() / max(bgm.sum(), 1))
    f["bg_chroma"] = float(np.median(chroma[bgm]))
    f["bg_L"] = float(np.median(L[bgm]))
    f["bg_L_std"] = float(L[bgm].std())
    f["bg_L_rel"] = float(np.median(L[bgm]) / Lref)

    # ---- framing & cropping
    ys, xs = np.nonzero(mb)
    if len(xs):
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        bbox = (int(x0), int(y0), int(bw), int(bh))
        cx, cy = (x0 + x1 + 1) / 2, (y0 + y1 + 1) / 2
        mcx, mcy = float(xs.mean()), float(ys.mean())
    else:
        bbox, cx, cy, mcx, mcy, bw, bh = (0, 0, w, h), w / 2, h / 2, w / 2, h / 2, w, h
    long_ratio = max(bw, bh) / max(h, w)
    f["bbox_area_ratio"] = float(bw * bh / (h * w))
    f["bbox_long_ratio"] = float(long_ratio)
    off = float(np.hypot(cx - w / 2, cy - h / 2) / max(h, w))
    f["centroid_offset"] = off
    t = 3
    top, bot = mb[:t].any(axis=0).mean() if len(xs) else 0, mb[-t:].any(axis=0).mean() if len(xs) else 0
    lef, rig = mb[:, :t].any(axis=1).mean() if len(xs) else 0, mb[:, -t:].any(axis=1).mean() if len(xs) else 0
    f["crop_score"] = float(max(top, bot, lef, rig))
    small_term = max(0.0, 1.0 - long_ratio / 0.8)
    off_term = 2.0 * off if f["crop_score"] < 0.04 else 0.0  # offset is meaningless once cropped
    f["framing_score"] = float(max(small_term, off_term))

    # ---- intermediates
    inter = {}
    inter["Grayscale"] = gray
    lap_vis = np.clip(np.abs(lap) * 4, 0, 255).astype(np.uint8)
    inter["Laplacian (edges)"] = lap_vis
    inter["Noise residual"] = np.clip(np.abs(resid) * 12, 0, 255).astype(np.uint8)
    inter["Histogram"] = _hist_image(gray)
    inter["Threshold mask"] = raw
    inter["Mask after morphology"] = mask
    det = img.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(det, cnts, -1, (0, 200, 0), 1)
    cv2.rectangle(det, (bbox[0], bbox[1]), (bbox[0] + bbox[2] - 1, bbox[1] + bbox[3] - 1), (0, 0, 255), 2)
    cv2.circle(det, (int(mcx), int(mcy)), 6, (255, 0, 0), -1)
    cv2.drawMarker(det, (w // 2, h // 2), (0, 165, 255), cv2.MARKER_CROSS, 24, 2)
    inter["Detected product"] = det
    return f, inter


def features(img_bgr: np.ndarray) -> dict[str, float]:
    """Raw scores for every defect (plus helper measures)."""
    return _analyse(img_bgr)[0]


def check(img_bgr: np.ndarray) -> dict:
    t0 = time.perf_counter()
    f, inter = _analyse(img_bgr)
    th = load_thresholds()
    res = empty_result("dip")
    for d in DEFECTS:
        cfg = th[d]
        s = float(f[cfg["feature"]])
        t = float(cfg["t"])
        fail = s > t if cfg["dir"] == "gt" else s < t
        what, verdict = _MESSAGES[d]
        if cfg["feature"] != DEFAULT_THRESHOLDS[d]["feature"]:
            what = cfg["feature"]
        op = ">" if cfg["dir"] == "gt" else "<"
        if fail:
            reason = f"{what} {s:.3g} {op} {t:.3g} → {verdict}"
        else:
            reason = f"{what} {s:.3g} {'<=' if cfg['dir'] == 'gt' else '>='} {t:.3g} → OK"
        res["defects"][d] = {"fail": bool(fail), "score": s, "reason": reason}
    res["intermediates"] = inter
    res["time_s"] = time.perf_counter() - t0
    return finalize(res)


if __name__ == "__main__":
    from common import load_image
    for p in sys.argv[1:]:
        r = check(load_image(p))
        print(p, "PASS" if r["passed"] else "FAIL", f"{r['time_s']*1000:.0f} ms")
        for d, v in r["defects"].items():
            print(f"  {d:13s} {'FAIL' if v['fail'] else 'ok  '} {v['reason']}")
