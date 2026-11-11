"""Assemble the single-file Kaggle kernel: src/defects.py + kaggle/train_body.py.

Usage:  python kaggle/build_kernel.py && kaggle kernels push -p kaggle/kernel
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "kaggle" / "kernel"
out.mkdir(parents=True, exist_ok=True)
code = (ROOT / "src" / "defects.py").read_text() + "\n\n" + (ROOT / "kaggle" / "train_body.py").read_text()
(out / "train_kernel.py").write_text(code)
(out / "kernel-metadata.json").write_text(json.dumps({
    "id": "divyanshparashar1/dip-ecom-image-quality-cnn",
    "title": "dip-ecom-image-quality-cnn",
    "code_file": "train_kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["paramaggarwal/fashion-product-images-dataset"],
    "competition_sources": [],
    "kernel_sources": [],
}, indent=2))
print("built", out / "train_kernel.py")
