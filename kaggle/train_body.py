# ---- Training body. build_kernel.py prepends src/defects.py to this file. ----
# Runs on a Kaggle GPU kernel with dataset paramaggarwal/fashion-product-images-dataset attached.
import csv
import glob
import json
import os
import random
import shutil
import time
import zipfile
from multiprocessing import Pool

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

SEED = 42
N = {"train": 4000, "val": 600, "test": 400}
INPUT = 384
EPOCHS = 10
BATCH = 32
WORK = "/kaggle/working"
DS = "/kaggle/tmp_dataset"
IMG_GLOB = "/kaggle/input/**/images/*.jpg"
if os.environ.get("SMOKE_DIR"):  # local CPU smoke test
    N = {"train": 12, "val": 6, "test": 6}
    EPOCHS, BATCH = 1, 8
    WORK = os.environ["SMOKE_DIR"]
    DS = os.path.join(WORK, "ds")
    IMG_GLOB = os.environ["SMOKE_GLOB"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def find_images():
    files = sorted(glob.glob(IMG_GLOB, recursive=True))
    if not files:
        raise SystemExit(f"no images found for {IMG_GLOB}")
    return files


def make_one(job):
    split, idx, path, seed = job
    rng = np.random.default_rng(seed)
    img = cv2.imread(path)
    if img is None:
        return None
    s = 1024 / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if not is_clean_source(img):
        return None
    base = normalize_product(img)
    ds = sample_defects(rng)
    name = f"{split}_{idx:05d}.jpg"
    cv2.imwrite(os.path.join(DS, "images", name), apply_defects(base, ds, rng),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"filename": name, "split": split, "source": os.path.basename(path), **labels_row(ds)}


def build_dataset():
    files = find_images()
    print("source images:", len(files))
    rnd = random.Random(SEED)
    rnd.shuffle(files)
    os.makedirs(os.path.join(DS, "images"), exist_ok=True)
    rows, start = [], 0
    for split, cnt in N.items():
        pool_files = files[start:start + int(cnt * 1.6)]  # some sources get rejected
        start += len(pool_files)
        jobs = [(split, i, p, SEED * 100000 + start + i) for i, p in enumerate(pool_files)]
        with Pool(os.cpu_count()) as pool:
            got = [r for r in pool.map(make_one, jobs, chunksize=16) if r]
        # keep exactly cnt, rename sequentially
        got = got[:cnt]
        for k, r in enumerate(got):
            new = f"{split}_{k:05d}.jpg"
            if r["filename"] != new:
                os.replace(os.path.join(DS, "images", r["filename"]), os.path.join(DS, "images", new))
                r["filename"] = new
        rows += got
        print(f"{split}: {len(got)} images")
    keep = {r["filename"] for r in rows}
    for f in os.listdir(os.path.join(DS, "images")):
        if f not in keep:
            os.remove(os.path.join(DS, "images", f))
    with open(os.path.join(DS, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "split", "source", *DEFECTS])
        w.writeheader()
        w.writerows(rows)
    return rows


def preprocess(img_bgr):
    x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class DS_(Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(os.path.join(DS, "images", r["filename"]))
        if self.train and random.random() < 0.5:
            img = img[:, ::-1]  # horizontal flip is label-preserving for every defect
        y = torch.tensor([float(r[d]) for d in DEFECTS])
        return preprocess(np.ascontiguousarray(img)), y


def predict(model, rows, dev):
    model.eval()
    ps, ys = [], []
    with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
        for x, y in DataLoader(DS_(rows, False), batch_size=64, num_workers=4):
            ps.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def f1(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


def main():
    t0 = time.time()
    rows = build_dataset()
    print(f"dataset built in {time.time() - t0:.0f}s")
    by = {s: [r for r in rows if r["split"] == s] for s in N}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    torch.manual_seed(SEED)
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, len(DEFECTS))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    dl = DataLoader(DS_(by["train"], True), batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * len(dl), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    best, hist = -1, []
    for ep in range(EPOCHS):
        model.train()
        tot = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast("cuda", enabled=dev == "cuda"):
                loss = lossf(model(x), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        p, y = predict(model, by["val"], dev)
        mf1 = float(np.mean([f1(y[:, j], (p[:, j] >= 0.5).astype(int)) for j in range(len(DEFECTS))]))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(dl), "val_macro_f1@0.5": mf1})
        print(hist[-1], f"{time.time() - t0:.0f}s", flush=True)
        if mf1 > best:
            best = mf1
            torch.save(model.state_dict(), os.path.join(WORK, "cnn.pt"))

    model.load_state_dict(torch.load(os.path.join(WORK, "cnn.pt"), map_location=dev))
    p, y = predict(model, by["val"], dev)
    th = {}
    for j, d in enumerate(DEFECTS):
        cands = np.linspace(0.05, 0.95, 91)
        th[d] = float(cands[int(np.argmax([f1(y[:, j], (p[:, j] >= c).astype(int)) for c in cands]))])
    json.dump(th, open(os.path.join(WORK, "thresholds.json"), "w"), indent=2)

    p, y = predict(model, by["test"], dev)
    pred = np.stack([(p[:, j] >= th[d]).astype(int) for j, d in enumerate(DEFECTS)], 1)
    per = {d: float(f1(y[:, j], pred[:, j])) for j, d in enumerate(DEFECTS)}
    clean = y.sum(1) == 0
    metrics = {"test_f1": per, "test_macro_f1": float(np.mean(list(per.values()))),
               "test_false_reject_rate": float(pred[clean].any(1).mean()),
               "test_defect_recall": float(pred[~clean].any(1).mean()),
               "thresholds": th, "history": hist, "minutes": (time.time() - t0) / 60}
    json.dump(metrics, open(os.path.join(WORK, "metrics.json"), "w"), indent=2)
    print(json.dumps(metrics, indent=2))

    # ship val+test images (+ all labels) back for local DIP tuning and evaluation
    with zipfile.ZipFile(os.path.join(WORK, "dataset_valtest.zip"), "w", zipfile.ZIP_STORED) as z:
        z.write(os.path.join(DS, "labels.csv"), "dataset/labels.csv")
        for r in rows:
            if r["split"] != "train":
                z.write(os.path.join(DS, "images", r["filename"]), f"dataset/images/{r['filename']}")
    shutil.rmtree(DS, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
