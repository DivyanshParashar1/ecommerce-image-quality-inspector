"""Copy a small, fixed set of reference images into data/demo/ for the app's demo gallery.

Picks from the held-out test split (never seen in training): clean, one image per single
defect, a few two-defect combos, plus two untouched original catalogue photos.

Usage: python scripts/make_demo.py
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "dataset"
OUT = ROOT / "data" / "demo"
DEFECTS = ["blur", "noise", "dark", "non_white_bg", "bad_framing", "cropped"]
COMBOS = [("non_white_bg", "bad_framing"), ("dark", "blur"), ("noise", "cropped")]
ORIGINALS = ["10000.jpg", "10006.jpg"]  # real catalogue photos from data/clean_samples


def main():
    df = pd.read_csv(SRC / "labels.csv")
    df = df[df.split == "test"].sort_values("filename")
    # prefer images whose DIP prediction matches the label, so each demo clearly shows its defect
    dip = pd.read_csv(ROOT / "results" / "preds_dip.csv").set_index("filename")
    agree = df.filename.map(lambda f: f in dip.index and all(
        int(dip.at[f, f"{d}_pred"]) == int(df.loc[df.filename == f, d].iloc[0]) for d in DEFECTS))
    df = df[agree.values]
    n = df[DEFECTS].sum(axis=1)
    picks = [("Clean", df[n == 0].iloc[1])]
    for d in DEFECTS:
        picks.append((d.replace("_", " ").capitalize(), df[(n == 1) & (df[d] == 1)].iloc[0]))
    for a, b in COMBOS:
        sel = df[(n == 2) & (df[a] == 1) & (df[b] == 1)]
        if len(sel):
            picks.append((f"{a} + {b}".replace("_", " ").capitalize(), sel.iloc[0]))

    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "images").mkdir(parents=True)
    rows = []
    for i, (title, r) in enumerate(picks):
        name = f"{i:02d}_{title.lower().replace(' + ', '_').replace(' ', '_')}.jpg"
        shutil.copy(SRC / "images" / r["filename"], OUT / "images" / name)
        rows.append({"filename": name, "title": title, "origin": f"test/{r['filename']}",
                     **{d: int(r[d]) for d in DEFECTS}})
    for j, f in enumerate(ORIGINALS):
        name = f"{len(rows):02d}_original_catalogue_photo_{j + 1}.jpg"
        shutil.copy(ROOT / "data" / "clean_samples" / f, OUT / "images" / name)
        # unlabelled: real photo, no synthetic defects added
        rows.append({"filename": name, "title": f"Original catalogue photo {j + 1}",
                     "origin": f"kaggle/{f}", **{d: "" for d in DEFECTS}})
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)
    print(f"wrote {len(rows)} demo images to {OUT}")


if __name__ == "__main__":
    main()
