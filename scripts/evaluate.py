#!/usr/bin/env python
"""Evaluate DIP / CNN / VLM checkers on the synthetic test set.

Usage:
  python scripts/evaluate.py [--data data/dataset] [--split test]
         [--methods dip,cnn,vlm] [--vlm-limit 200] [--out results/]
         [--workers 4] [--fresh]

Predictions are cached in <out>/preds_<method>.csv and resumed on re-run, so
paid VLM calls are never repeated. Set CHECKERS_PATH to prepend a directory to
sys.path (used for testing with stub checkers).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if os.environ.get("CHECKERS_PATH"):
    sys.path.insert(0, os.environ["CHECKERS_PATH"])

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from common import DEFECTS, load_image

log = logging.getLogger("evaluate")
MODULES = {"dip": "dip_checks", "cnn": "cnn_checker", "vlm": "vlm_checker"}
METHOD_COLORS = {"dip": "#4C78A8", "cnn": "#F58518", "vlm": "#54A24B"}


# ----------------------------------------------------------------- loading
def load_method(name: str):
    try:
        mod = importlib.import_module(MODULES[name])
    except Exception as e:  # noqa: BLE001
        log.warning("Method %s skipped: cannot import %s (%s)", name, MODULES[name], e)
        return None
    try:
        if hasattr(mod, "is_available") and not mod.is_available():
            log.warning("Method %s skipped: is_available() is False", name)
            return None
    except Exception as e:  # noqa: BLE001
        log.warning("Method %s skipped: is_available() raised %s", name, e)
        return None
    return mod


def pred_columns():
    cols = ["filename"]
    cols += [f"{d}_pred" for d in DEFECTS]
    cols += [f"{d}_score" for d in DEFECTS]
    return cols + ["time_s", "cost"]


def result_to_row(fn: str, res: dict) -> dict:
    row = {"filename": fn}
    for d in DEFECTS:
        info = res["defects"][d]
        row[f"{d}_pred"] = int(bool(info.get("fail")))
        s = info.get("score")
        row[f"{d}_score"] = float("nan") if s is None else float(s)
    row["time_s"] = float(res.get("time_s", 0.0) or 0.0)
    usage = res.get("usage") or {}
    try:
        row["cost"] = float(usage.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        row["cost"] = 0.0
    return row


class Cache:
    def __init__(self, path: Path, fresh: bool):
        self.path = path
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if path.exists() and not fresh:
            try:
                df = pd.read_csv(path)
                for r in df.to_dict("records"):
                    self.rows[r["filename"]] = r
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read cache %s (%s); starting empty", path, e)
        self._since = 0
        if fresh or not path.exists():
            self._flush()

    def _flush(self):
        pd.DataFrame(list(self.rows.values()), columns=pred_columns()).to_csv(self.path, index=False)

    def add(self, row: dict, every: int = 10):
        with self.lock:
            self.rows[row["filename"]] = row
            self._since += 1
            if self._since >= every:
                self._flush()
                self._since = 0

    def flush(self):
        with self.lock:
            self._flush()
            self._since = 0


# ----------------------------------------------------------------- running
def run_method(name, mod, files, img_dir: Path, cache: Cache, workers: int):
    todo = [f for f in files if f not in cache.rows]
    log.info("[%s] %d images requested, %d cached, %d to run", name, len(files), len(files) - len(todo), len(todo))
    if not todo:
        return
    t0 = time.time()
    done = 0
    try:
        if name == "vlm":
            def work(fn):
                return fn, mod.check(load_image(img_dir / fn))

            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                futs = {ex.submit(work, fn): fn for fn in todo}
                for fut in as_completed(futs):
                    fn = futs[fut]
                    try:
                        _, res = fut.result()
                        cache.add(result_to_row(fn, res), every=1)
                    except Exception as e:  # noqa: BLE001
                        log.error("[vlm] %s failed, skipped: %s", fn, e)
                    done += 1
                    if done % 10 == 0:
                        log.info("[vlm] %d/%d", done, len(todo))
        elif name == "cnn" and hasattr(mod, "check_batch"):
            bs = 32
            for i in range(0, len(todo), bs):
                chunk = todo[i:i + bs]
                imgs, ok = [], []
                for fn in chunk:
                    try:
                        imgs.append(load_image(img_dir / fn))
                        ok.append(fn)
                    except Exception as e:  # noqa: BLE001
                        log.error("[cnn] %s unreadable: %s", fn, e)
                if not imgs:
                    continue
                t = time.perf_counter()
                try:
                    results = mod.check_batch(imgs)
                except Exception as e:  # noqa: BLE001
                    log.error("[cnn] batch failed (%s); falling back to single", e)
                    results = [mod.check(im) for im in imgs]
                per = (time.perf_counter() - t) / len(imgs)
                for fn, res in zip(ok, results):
                    if not res.get("time_s"):
                        res["time_s"] = per
                    cache.add(result_to_row(fn, res))
                log.info("[cnn] %d/%d", min(i + bs, len(todo)), len(todo))
        else:
            for fn in todo:
                try:
                    cache.add(result_to_row(fn, mod.check(load_image(img_dir / fn))))
                except Exception as e:  # noqa: BLE001
                    log.error("[%s] %s failed, skipped: %s", name, fn, e)
                done += 1
                if done % 50 == 0:
                    log.info("[%s] %d/%d", name, done, len(todo))
    finally:
        cache.flush()
    log.info("[%s] finished in %.1fs", name, time.time() - t0)


# ----------------------------------------------------------------- metrics
def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, int(y.sum())


def merged(labels: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    return labels.merge(preds, on="filename", how="inner")


def per_defect_metrics(m: pd.DataFrame, method: str) -> list[dict]:
    out = []
    for d in DEFECTS:
        p, r, f, s = prf(m[d].values.astype(int), m[f"{d}_pred"].values.astype(int))
        out.append(dict(method=method, defect=d, precision=p, recall=r, f1=f, support=s, n_images=len(m)))
    return out


def summary_metrics(m: pd.DataFrame, method: str) -> dict:
    Y = m[DEFECTS].values.astype(int)
    P = m[[f"{d}_pred" for d in DEFECTS]].values.astype(int)
    f1s = [prf(Y[:, i], P[:, i])[2] for i in range(len(DEFECTS))]
    has_def = Y.any(axis=1)
    pred_fail = P.any(axis=1)
    return dict(
        method=method,
        macro_f1=float(np.mean(f1s)),
        defect_recall=float(pred_fail[has_def].mean()) if has_def.any() else float("nan"),
        false_reject_rate=float(pred_fail[~has_def].mean()) if (~has_def).any() else float("nan"),
        image_accuracy=float((pred_fail == has_def).mean()),
        exact_match=float((Y == P).all(axis=1).mean()),
        mean_time_s=float(m["time_s"].mean()),
        total_cost=float(m["cost"].fillna(0).sum()),
        n_images=len(m),
    )


# ----------------------------------------------------------------- figures
def fig_f1(pd_df: pd.DataFrame, path: Path, title: str):
    methods = list(pd_df["method"].unique())
    x = np.arange(len(DEFECTS))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, mth in enumerate(methods):
        sub = pd_df[pd_df.method == mth].set_index("defect").reindex(DEFECTS)
        ax.bar(x + i * w - 0.4 + w / 2, sub["f1"], w, label=mth.upper(), color=METHOD_COLORS.get(mth))
    ax.set_xticks(x)
    ax.set_xticklabels(DEFECTS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary(sm: pd.DataFrame, path: Path, title: str):
    cols = ["macro_f1", "defect_recall", "false_reject_rate", "image_accuracy", "exact_match"]
    methods = list(sm["method"])
    x = np.arange(len(cols))
    w = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, mth in enumerate(methods):
        vals = sm[sm.method == mth][cols].iloc[0].astype(float).values
        ax.bar(x + i * w - 0.4 + w / 2, np.nan_to_num(vals), w, label=mth.upper(), color=METHOD_COLORS.get(mth))
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_confusion(m: pd.DataFrame, method: str, path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, d in zip(axes.ravel(), DEFECTS):
        y = m[d].values.astype(int)
        p = m[f"{d}_pred"].values.astype(int)
        cm = np.array([[((y == 0) & (p == 0)).sum(), ((y == 0) & (p == 1)).sum()],
                       [((y == 1) & (p == 0)).sum(), ((y == 1) & (p == 1)).sum()]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pass", "fail"]); ax.set_yticklabels(["ok", "defect"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(d)
    fig.suptitle(f"Confusion matrices - {method.upper()} (n={len(m)})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- gallery
def labels_str(row, suffix="") -> str:
    on = [d for d in DEFECTS if int(row[d + suffix]) == 1]
    return ",".join(on) if on else "none"


def captioned(img, row, kind: str, defect: str, width=512):
    img = cv2.resize(img, (width, width))
    lines = [f"{row['filename']}  [{kind} {defect}]",
             "true: " + labels_str(row),
             "pred: " + labels_str(row, "_pred")]
    strip = np.full((22 * len(lines) + 8, width, 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(strip, t[:70], (6, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def failure_gallery(m: pd.DataFrame, method: str, img_dir: Path, out: Path, k=4):
    root = out / "failures" / method
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("*.jpg"):
        old.unlink()
    saved = []
    for d in DEFECTS:
        for kind, cond in (("FN", (m[d] == 1) & (m[f"{d}_pred"] == 0)),
                           ("FP", (m[d] == 0) & (m[f"{d}_pred"] == 1))):
            for n, (_, row) in enumerate(m[cond].head(k).iterrows()):
                try:
                    img = load_image(img_dir / row["filename"])
                except Exception:  # noqa: BLE001
                    continue
                cap = captioned(img, row, kind, d)
                cv2.imwrite(str(root / f"{d}_{kind}_{n}_{Path(row['filename']).stem}.jpg"), cap)
                saved.append(cap)
    if saved:
        tile = 256
        cols = 6
        tiles = [cv2.resize(c, (tile, int(c.shape[0] * tile / c.shape[1]))) for c in saved]
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(root / "_contact_sheet.jpg"), np.vstack(rows))
    log.info("[%s] saved %d failure examples", method, len(saved))


# ----------------------------------------------------------------- reporting
def to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="dip,cnn,vlm")
    ap.add_argument("--vlm-limit", type=int, default=200)
    ap.add_argument("--out", default="results/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="ignore cached predictions")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    data = Path(a.data)
    if not data.is_absolute() and not data.exists():
        data = ROOT / data
    out = Path(a.out)
    if not out.is_absolute() and not out.exists():
        out = Path.cwd() / out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    img_dir = data / "images"

    labels = pd.read_csv(data / "labels.csv")
    labels = labels[labels["split"] == a.split].sort_values("filename").reset_index(drop=True)
    if labels.empty:
        sys.exit(f"No rows with split={a.split!r} in {data/'labels.csv'}")
    all_files = list(labels["filename"])
    log.info("%d images in split %s", len(all_files), a.split)

    caches, ran = {}, []
    for name in [s.strip() for s in a.methods.split(",") if s.strip()]:
        if name not in MODULES:
            log.warning("Unknown method %s ignored", name)
            continue
        mod = load_method(name)
        if mod is None:
            continue
        files = all_files[:a.vlm_limit] if name == "vlm" else all_files
        cache = Cache(out / f"preds_{name}.csv", a.fresh)
        run_method(name, mod, files, img_dir, cache, a.workers)
        caches[name] = cache
        ran.append(name)

    # Also pick up cached predictions for methods not run now (e.g. VLM unavailable offline).
    for name in MODULES:
        if name not in caches and (out / f"preds_{name}.csv").exists() and not a.fresh \
                and name in [s.strip() for s in a.methods.split(",")]:
            caches[name] = Cache(out / f"preds_{name}.csv", False)
            log.info("[%s] using cached predictions only", name)

    merged_by = {}
    for name, c in caches.items():
        pdf = pd.DataFrame(list(c.rows.values()), columns=pred_columns())
        m = merged(labels, pdf)
        if len(m):
            merged_by[name] = m
    if not merged_by:
        sys.exit("No predictions available for any method.")

    def compute(mb):
        pds, sms = [], []
        for name, m in mb.items():
            pds += per_defect_metrics(m, name)
            sms.append(summary_metrics(m, name))
        return pd.DataFrame(pds), pd.DataFrame(sms)

    pd_df, sm_df = compute(merged_by)
    pd_df.to_csv(out / "metrics_per_defect.csv", index=False)
    sm_df.to_csv(out / "summary.csv", index=False)

    common = set.intersection(*[set(m["filename"]) for m in merged_by.values()])
    mb_c = {n: m[m["filename"].isin(common)].reset_index(drop=True) for n, m in merged_by.items()}
    pd_c, sm_c = compute(mb_c)
    pd_c.to_csv(out / "metrics_per_defect_common_subset.csv", index=False)
    sm_c.to_csv(out / "summary_common_subset.csv", index=False)

    fig_f1(pd_df, out / "figures" / "f1_per_defect.png", "Per-defect F1 (all available images)")
    fig_summary(sm_df, out / "figures" / "summary_metrics.png", "Summary metrics (all available images)")
    fig_f1(pd_c, out / "figures" / "f1_per_defect_common_subset.png", f"Per-defect F1 (common subset, n={len(common)})")
    fig_summary(sm_c, out / "figures" / "summary_metrics_common_subset.png", f"Summary metrics (common subset, n={len(common)})")
    for name, m in merged_by.items():
        fig_confusion(m, name, out / "figures" / f"confusion_{name}.png")
        failure_gallery(m, name, img_dir, out)

    # stdout + markdown
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Summary (all images each method ran on) ===")
    print(sm_df.round(3).to_string(index=False))
    print(f"\n=== Summary (common subset, n={len(common)}) ===")
    print(sm_c.round(3).to_string(index=False))
    print("\n=== Per-defect (common subset) ===")
    print(pd_c.round(3).to_string(index=False))

    md = [f"# Evaluation results (split: {a.split})", "",
          "## Summary, all images each method ran on", to_md(sm_df), "",
          f"## Summary, common subset (n={len(common)})", to_md(sm_c), "",
          "## Per-defect metrics, all images", to_md(pd_df), "",
          f"## Per-defect metrics, common subset (n={len(common)})", to_md(pd_c), ""]
    (out / "results.md").write_text("\n".join(md))
    log.info("Wrote results to %s", out)


if __name__ == "__main__":
    main()
