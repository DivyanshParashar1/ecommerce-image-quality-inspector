"""Build the synthetic defect dataset from a folder of clean product photos.

Usage:
  python scripts/make_dataset.py --src data/clean_samples --out data/dataset \
      --n-train 4000 --n-val 600 --n-test 400 --seed 42

Each source photo is used for one split only (no leakage). Output:
  <out>/images/<split>_<idx>.jpg   and   <out>/labels.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from defects import (DEFECTS, apply_defects, is_clean_source, labels_row,  # noqa: E402
                     normalize_product, sample_defects)


def build(src: Path, out: Path, n: dict, seed: int, max_sources: int | None = None) -> None:
    rng = np.random.default_rng(seed)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(files)
    if max_sources:
        files = files[:max_sources]
    # split source photos by proportion of requested images
    total = sum(n.values())
    cuts, start, parts = {}, 0, list(n.items())
    for i, (split, cnt) in enumerate(parts):
        end = len(files) if i == len(parts) - 1 else start + int(len(files) * cnt / total)
        cuts[split] = files[start:end]
        start = end

    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for split, cnt in n.items():
        srcs, made, i = cuts[split], 0, 0
        while made < cnt and srcs:
            p = srcs[i % len(srcs)]
            i += 1
            if i > 20 * cnt:
                break
            img = cv2.imread(str(p))
            if img is None or not is_clean_source(img):
                continue
            base = normalize_product(img)
            ds = sample_defects(rng)
            name = f"{split}_{made:05d}.jpg"
            cv2.imwrite(str(out / "images" / name), apply_defects(base, ds, rng),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append({"filename": name, "split": split, "source": p.name, **labels_row(ds)})
            made += 1
        print(f"{split}: {made} images from {len(srcs)} sources")

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "split", "source", *DEFECTS])
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-sources", type=int, default=None)
    a = ap.parse_args()
    build(Path(a.src), Path(a.out), {"train": a.n_train, "val": a.n_val, "test": a.n_test},
          a.seed, a.max_sources)
