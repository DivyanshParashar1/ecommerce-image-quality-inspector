# E-Commerce Image Quality Inspector — Project Plan

Topic: HSR-AIML-056 (Automated E-commerce Image Quality and Compliance Inspector)

## Goal
Upload a product photo and get PASS/FAIL for each quality rule, with reasons and the intermediate DIP images.
Three methods are compared on the same synthetic test set:

| # | Method | Role |
|---|--------|------|
| 1 | Classical DIP rules (OpenCV, hand-set thresholds) | Baseline |
| 2 | CNN (EfficientNet-B0, multi-label, trained on Kaggle GPU) | Learned method |
| 3 | Vision LLM through OpenRouter (returns JSON) | Zero-shot VLM |

Metrics: per-defect F1, Macro-F1, defect recall, false-reject rate (clean images wrongly failed), time per image, API cost per image (VLM only).

## Defects (fixed label set, used by everything)
```
DEFECTS = ["blur", "noise", "dark", "non_white_bg", "bad_framing", "cropped"]
```
- blur: Gaussian or motion blur
- noise: Gaussian noise, sigma 15–35
- dark: underexposed / low contrast (gamma > 1.8 or contrast scaled by 0.3–0.5)
- non_white_bg: white background replaced with a solid colour, gradient or texture
- bad_framing: product shrunk to 20–40% of the frame and/or placed off-centre
- cropped: product partly outside the frame

## Dataset
- Source: a public Kaggle product-image dataset with white backgrounds (picked once the Kaggle key is available; Fashion Product Images is the first candidate).
- Keep only "clean" source images: border pixels are at least 95% near-white and the product doesn't touch the border.
- Resize to 512×512 on a white canvas.
- Synthetic generation (`src/defects.py`, fixed seed): 35% clean, otherwise 1–2 random defects (`bad_framing` and `cropped` are never combined).
- Split: train 4000 / val 600 / test 400. Labels go in `labels.csv` with columns `filename,split,blur,noise,dark,non_white_bg,bad_framing,cropped`.
- The VLM is evaluated on a 200-image subset of the test set to limit cost.

## Shared interface (contract every module follows)
Every checker exposes:
```python
def check(img_bgr: np.ndarray) -> dict
```
and returns:
```python
{
  "method": "dip" | "cnn" | "vlm",
  "defects": {name: {"fail": bool, "score": float | None, "reason": str} for name in DEFECTS},
  "passed": bool,                        # True if no defect failed
  "intermediates": {title: np.ndarray},  # DIP only; {} for the others
  "time_s": float,
}
```
`src/common.py` holds `DEFECTS`, `empty_result(method)`, and the image load/save helpers.

## Repository layout
```
dip_project/
  PLAN.md  README.md  requirements.txt  .env (OPENROUTER_API_KEY, OPENROUTER_MODEL — not committed)
  src/common.py        # contract (orchestrator)
  src/defects.py       # defect generator (orchestrator)
  src/dip_checks.py    # Unit A
  src/cnn_checker.py   # Unit B
  src/vlm_checker.py   # Unit C
  app.py               # Unit D (Streamlit)
  scripts/make_dataset.py   # orchestrator
  scripts/evaluate.py       # Unit E
  kaggle/train.py           # orchestrator (Kaggle kernel)
  models/cnn.pt             # downloaded from Kaggle output
  data/                     # generated dataset
  results/                  # metrics.csv, confusion plots, failure gallery
```

## DIP checks (Unit A)
| Defect | Technique |
|--------|-----------|
| blur | Variance of Laplacian on grayscale |
| noise | std of (gray − median3×3(gray)) in flat regions |
| dark | Histogram mean, std (contrast), % of pixels below 30 |
| non_white_bg | Product mask via Otsu threshold on LAB lightness/saturation + morphological open/close + largest connected component. Then % of background pixels that are near-white |
| bad_framing | Bounding box of the largest component: area ratio < 0.45 or centroid offset > 15% |
| cropped | Mask touches the image border |

Intermediates to return: grayscale, Laplacian edge map, histogram plot, raw threshold mask, mask after morphology, annotated bounding box.

## Execution plan (orchestrator view)
**Phase 0 — orchestrator, sequential (~short):** write `common.py`, `defects.py`, `make_dataset.py`, `requirements.txt`; test on a few images.

**Phase 1 — all in parallel:**
- Orchestrator: find the dataset on Kaggle, push `kaggle/train.py` as a GPU kernel (it builds the dataset on Kaggle using an inlined copy of `defects.py`, trains EfficientNet-B0 for about 10 epochs with BCE loss, tunes per-defect thresholds on val, saves `cnn.pt`, `thresholds.json` and the test split). Monitor it until done.
- Unit A agent: `src/dip_checks.py` and its threshold tuning on generated samples.
- Unit B agent: `src/cnn_checker.py` (loads `models/cnn.pt` and `models/thresholds.json`; architecture spec fixed in this plan).
- Unit C agent: `src/vlm_checker.py` (OpenRouter, JSON-mode prompt, retry/parse fallback).
- Unit D agent: `app.py` (Streamlit: upload, method selector, pass/fail table, intermediates grid).
- Unit E agent: `scripts/evaluate.py` (runs the methods on the test set, metrics table, confusion matrices, failure-case gallery).

Agents only write their own files, code against the contract above, and use stubs where another unit isn't ready.

**Phase 2 — orchestrator:** download the trained weights, run the integration check, run evaluation, fix mismatches.

**Phase 3:** README (setup, how to run), results figures, failure-case write-up for the report.

## CNN spec (Units B and kaggle/train.py must match)
- `torchvision.models.efficientnet_b0`, ImageNet weights, classifier replaced by `Linear(1280, 6)`.
- Input: RGB, resized to 384×384, ImageNet mean/std normalisation.
- Output: sigmoid per defect; `thresholds.json` maps defect name to a threshold (default 0.5).
- Saved as `torch.save(model.state_dict(), "cnn.pt")`.

## Expected failure cases (for the report)
- White product on a white background: segmentation fails, so `non_white_bg`, `bad_framing` and `cropped` are wrong.
- Textured products (knits, prints) read as noise; intentional soft focus reads as blur.
- The VLM is lenient or inconsistent on subtle blur/noise; the CNN can over-fit to synthetic defects.
