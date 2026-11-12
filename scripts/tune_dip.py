"""Tune DIP thresholds (max F1 per defect) on a labelled split.

  python scripts/tune_dip.py --data data/dataset --split train,val --out models/dip_thresholds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from common import DEFECTS, load_image
import dip_checks

# candidate features per defect: (feature, direction). Best F1 over candidates wins.
CANDIDATES = {
    "blur": [("blur_score", "lt"), ("grad_p99", "lt")],
    "noise": [("noise_score", "gt"), ("noise_std", "gt")],
    "dark": [("dark_score", "lt"), ("gray_mean", "lt")],
    "non_white_bg": [("bg_nonwhite", "gt"), ("bg_chroma", "gt")],
    "bad_framing": [("framing_score", "gt")],
    "cropped": [("crop_score", "gt")],
}


def f1(y, pred):
    tp = int((y & pred).sum()); fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def best_threshold(x, y, direction):
    vals = np.unique(np.concatenate([np.unique(x), np.percentile(x, np.linspace(1, 99, 99))]))
    cands = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
    scores = np.array([f1(y, x > t if direction == "gt" else x < t) for t in cands])
    top = cands[scores >= scores.max() - 1e-9]
    return float(scores.max()), float(np.median(top))  # middle of the best plateau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default=str(ROOT / "models" / "dip_thresholds.json"))
    a = ap.parse_args()
    data = Path(a.data)
    df = pd.read_csv(data / "labels.csv")
    df = df[df["split"].isin(a.split.split(","))].reset_index(drop=True)
    rows = []
    for i, fn in enumerate(df["filename"]):
        rows.append(dip_checks.features(load_image(data / "images" / fn)))
        if (i + 1) % 100 == 0:
            print(f"  features {i + 1}/{len(df)}", flush=True)
    F = pd.DataFrame(rows)

    out = {}
    print(f"{'defect':14s} {'feature':14s} dir  {'F1 default':>10s} {'F1 tuned':>9s}  threshold")
    for d in DEFECTS:
        y = df[d].values.astype(bool)
        dflt = dip_checks.DEFAULT_THRESHOLDS[d]
        x0 = F[dflt["feature"]].values
        p0 = x0 > dflt["t"] if dflt["dir"] == "gt" else x0 < dflt["t"]
        before = f1(y, p0)
        best = None
        for feat, direction in CANDIDATES[d]:
            s, t = best_threshold(F[feat].values, y, direction)
            if best is None or s > best[0]:
                best = (s, t, feat, direction)
        s, t, feat, direction = best
        out[d] = {"feature": feat, "dir": direction, "t": t, "f1": s, "n_pos": int(y.sum())}
        print(f"{d:14s} {feat:14s} {direction:3s}  {before:10.3f} {s:9.3f}  {t:.4g}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
