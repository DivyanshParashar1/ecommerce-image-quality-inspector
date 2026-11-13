# E-Commerce Image Quality Inspector

DIP course project, topic **HSR-AIML-056** (Automated E-commerce Image Quality and Compliance Inspector).

Upload a product photo and get PASS/FAIL on six marketplace rules, with a reason for each and the intermediate DIP images.
Three methods are compared on the same synthetic test set:

| Method | What it is |
|---|---|
| **DIP rules** (baseline) | Classical OpenCV: Laplacian, median-filter residual, histogram, k-means/Otsu segmentation, morphology, connected components. Thresholds tuned on the validation split. |
| **CNN** | EfficientNet-B0 (ImageNet-pretrained), multi-label, fine-tuned on a Kaggle T4 GPU (12.8 min) |
| **Vision LLM** | `google/gemma-4-26b-a4b-it` through OpenRouter, zero-shot, JSON output |

Defects: `blur`, `noise`, `dark`, `non_white_bg`, `bad_framing` (too small or off-centre), `cropped` (cut off at the edge).

## Pipeline
```
Product photo → resize (longest side 512)
   ├─ DIP: grayscale → Laplacian variance (blur)
   │        → median-filter residual in flat regions (noise)
   │        → histogram / 95th-percentile brightness (dark)
   │        → LAB border k-means + Otsu → open/close → largest component → hole fill
   │              → background chroma (non_white_bg), bbox size/offset (bad_framing), border contact (cropped)
   ├─ CNN: RGB 384×384 → EfficientNet-B0 → 6 sigmoids → per-defect thresholds
   └─ VLM: JPEG 768px → OpenRouter → JSON per defect
→ per-defect PASS/FAIL + reasons → evaluation (F1, defect recall, false-reject rate, time, cost)
```

## Data
- Source: Kaggle `paramaggarwal/fashion-product-images-dataset`, real catalogue photos on white backgrounds.
- `src/defects.py` keeps clean photos only, tight-crops each product, centres it on a 512×512 white canvas, then adds 0–2 synthetic defects with random strength. 35% of images are left clean. Labels are known by construction.
- Split by source photo: train 3893 / val 522 / test 352.
- The Kaggle dataset ships every image twice under two folders, so 59 val/test images came from photos also used in training. They are marked `*_leak` in `labels.csv` and excluded from evaluation.

## Results (test set, 352 images)

| Method | Macro-F1 | Defect recall | False-reject rate | Time / image | Cost (352 images) |
|---|---|---|---|---|---|
| DIP rules | 0.853 | 0.961 | 2.4% | 27 ms | $0 |
| CNN | **0.995** | **0.996** | **0.0%** | 33 ms | $0 |
| Vision LLM | 0.662 | 1.000 | 29.0% | 2.3 s | $0.036 |

Per-defect F1:

| Defect | DIP | CNN | VLM |
|---|---|---|---|
| blur | 0.83 | 1.00 | 0.74 |
| noise | 0.99 | 1.00 | 0.79 |
| dark | 0.82 | 0.99 | 0.69 |
| non_white_bg | 0.98 | 1.00 | 0.68 |
| bad_framing | 0.82 | 1.00 | 0.58 |
| cropped | 0.67 | 0.98 | 0.49 |

Full tables are in `results/results.md`, figures in `results/figures/`, and failure examples per method in `results/failures/<method>/`.

## Failure analysis
- **DIP, cluttered backgrounds:** the colour-noise texture background breaks the border-based segmentation, so the mask leaks to the edges and gives false `cropped` and `bad_framing` flags.
- **DIP, low-contrast "dark" images** have a low Laplacian variance and get flagged as `blur`. Solid dark-coloured backgrounds lower the 95th-percentile brightness and get flagged as `dark`. In both cases one feature is responding to two defects.
- **DIP, small cut-offs** (a few pixels of a sandal or watch strap past the edge) are missed, because the border-contact threshold has to stay high enough to tolerate segmentation noise.
- **CNN:** almost perfect, but it was trained and tested on the *same synthetic generator*. It has learnt this generator's defects, and real-world performance will be lower. The DIP rules are interpretable and need no training data.
- **VLM:** it never misses a defect (recall 1.0) but over-flags heavily, rejecting 29% of clean images. It calls darkened white backgrounds "non-white" and often gets framing and cropping wrong. It is also roughly 70× slower than the other two and the only paid method.
- **Label noise:** the `cropped` generator shifts the product towards an edge by a fixed fraction. A narrow product (e.g. earrings) can end up shifted but still fully visible while labelled `cropped`. This penalises the DIP rules unfairly on some images.

## Setup and running
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sk-or-..." > .env        # only needed for the VLM method

.venv/bin/streamlit run app.py                    # web app (click a demo image to run every method on it)
.venv/bin/python src/dip_checks.py some.jpg       # a single checker on the command line
.venv/bin/python scripts/evaluate.py --data data/dataset --split test --methods dip,cnn,vlm --vlm-limit 400
.venv/bin/python scripts/tune_dip.py --data data/dataset --split val --out models/dip_thresholds.json
```
Note: scipy and scikit-learn crash on this macOS version, so the project uses numpy/pandas only.

Re-training the CNN (needs a Kaggle account; the kernel attaches the dataset itself):
```bash
.venv/bin/python kaggle/build_kernel.py && kaggle kernels push -p kaggle/kernel
kaggle kernels output <user>/dip-ecom-image-quality-cnn -p <dir>   # cnn.pt, thresholds.json, metrics.json, dataset_valtest.zip
```

## Files
| Path | Purpose |
|---|---|
| `src/common.py` | Shared contract: defect list and result format |
| `src/defects.py` | Synthetic defect generator |
| `src/dip_checks.py` | Classical DIP checker, with intermediates |
| `src/cnn_checker.py` | CNN inference |
| `src/vlm_checker.py` | OpenRouter vision-LLM checker |
| `app.py` | Streamlit app |
| `scripts/make_dataset.py` | Builds the dataset locally |
| `scripts/make_demo.py`, `data/demo/` | 12 reference images for the app's demo gallery (held-out test images plus 2 untouched catalogue photos) |
| `scripts/tune_dip.py` | Tunes the DIP thresholds |
| `scripts/evaluate.py` | Metrics, figures, failure gallery |
| `kaggle/` | GPU training kernel |
| `models/` | `cnn.pt`, `thresholds.json`, `dip_thresholds.json`, `cnn_train_metrics.json` |
