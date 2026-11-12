"""E-Commerce Image Quality Inspector - Streamlit app.

Run:  .venv/bin/streamlit run app.py
"""
import hashlib
import importlib
import os
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if os.environ.get("CHECKERS_PATH"):  # testing only: stub checkers
    sys.path.insert(0, os.environ["CHECKERS_PATH"])

from common import DEFECTS, DEFECT_DESCRIPTIONS, decode_image  # noqa: E402

METHODS = {
    "DIP rules": ("dip", "dip_checks", "DIP module src/dip_checks.py not found"),
    "CNN": ("cnn", "cnn_checker", "CNN weights not found in models/"),
    "Vision LLM": ("vlm", "vlm_checker", "OPENROUTER_API_KEY not set"),
}

st.set_page_config(page_title="Image Quality Inspector", layout="wide")


def load_module(modname):
    """Return (module, None) or (None, reason)."""
    try:
        return importlib.import_module(modname), None
    except Exception as e:  # missing file, missing deps, ...
        return None, f"{modname} could not be loaded ({type(e).__name__}: {e})"


@st.cache_resource
def get_checker(modname):
    """Import once per session; heavy models are loaded inside the module on first use."""
    mod, err = load_module(modname)
    if mod is None:
        return None, err
    return mod, None


def availability(label):
    _, modname, unavailable_msg = METHODS[label]
    mod, err = get_checker(modname)
    if mod is None:
        return False, err
    fn = getattr(mod, "is_available", None)
    try:
        if fn is not None and not fn():
            return False, unavailable_msg
    except Exception as e:
        return False, f"is_available() failed: {e}"
    return True, ""


@st.cache_data(show_spinner=False, max_entries=64)
def run_check(img_hash, method_label, img_bytes):
    """Cached per (image hash, method). Exceptions are not cached by Streamlit."""
    mod, err = get_checker(METHODS[method_label][1])
    if mod is None:
        raise RuntimeError(err)
    return mod.check(decode_image(img_bytes))


def dataset_dir():
    for name in ("dataset", "dev_dataset"):
        d = ROOT / "data" / name
        if (d / "images").is_dir():
            return d
    return None


@st.cache_data
def load_labels(d):
    f = Path(d) / "labels.csv"
    return pd.read_csv(f).set_index("filename") if f.exists() else None


def to_rgb(a):
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return a


# ---------------- sidebar ----------------
st.sidebar.title("Settings")
avail = {m: availability(m) for m in METHODS}
chosen = st.sidebar.multiselect("Methods", list(METHODS),
                                default=[m for m in METHODS if avail[m][0]] or ["DIP rules"])
for m in chosen:
    if not avail[m][0]:
        st.sidebar.warning(f"{m} unavailable: {avail[m][1]}")

ds = dataset_dir()
sample_name = None
if ds is not None:
    files = sorted(p.name for p in (ds / "images").iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    sample_name = st.sidebar.selectbox("Try a sample", [None] + files,
                                       format_func=lambda x: "(none)" if x is None else x)
    st.sidebar.caption(f"Samples from {ds.relative_to(ROOT)}/images")
else:
    st.sidebar.caption("No sample dataset found in data/.")

# ---------------- main ----------------
st.title("E-Commerce Image Quality Inspector")

DEMO = ROOT / "data" / "demo"
demo = None
if (DEMO / "labels.csv").exists():
    demo = pd.read_csv(DEMO / "labels.csv", dtype=str).fillna("")
    with st.expander("Demo gallery: click an image to run the whole system on it",
                     expanded="demo" not in st.session_state):
        cols = st.columns(6)
        for i, r in demo.iterrows():
            with cols[i % 6]:
                st.image(str(DEMO / "images" / r["filename"]), width="stretch")
                if st.button(r["title"], key=f"demo_{i}", width="stretch"):
                    st.session_state["demo"] = r["filename"]
                    st.rerun()

up = st.file_uploader("Upload a product photo", type=["jpg", "jpeg", "png", "webp"])

img_bytes, truth = None, None
if up is not None:
    img_bytes = up.getvalue()
elif st.session_state.get("demo") and demo is not None:
    r = demo[demo.filename == st.session_state["demo"]].iloc[0]
    img_bytes = (DEMO / "images" / r["filename"]).read_bytes()
    st.markdown(f"**Demo image:** {r['title']}")
    if r["blur"] != "":  # original catalogue photos are unlabelled
        truth = {d: r[d] == "1" for d in DEFECTS}
elif sample_name:
    img_bytes = (ds / "images" / sample_name).read_bytes()
    labels = load_labels(str(ds))
    if labels is not None and sample_name in labels.index:
        row = labels.loc[sample_name]
        truth = {d: bool(row[d]) for d in DEFECTS if d in row}

if img_bytes is None:
    st.info("Click a demo image above, upload a photo, or pick a sample in the sidebar.")
    st.stop()

try:
    img = decode_image(img_bytes)
except Exception as e:
    st.error(f"Could not read image: {e}")
    st.stop()

img_hash = hashlib.sha256(img_bytes).hexdigest()
c1, c2 = st.columns([1, 2])
with c1:
    st.image(to_rgb(img), caption=f"Input ({img.shape[1]}x{img.shape[0]})", width="stretch")
    if truth is not None:
        bad = [d for d, v in truth.items() if v]
        st.caption("Ground truth: " + (", ".join(bad) if bad else "clean"))

results = {}
with c2:
    if not chosen:
        st.info("Select at least one method in the sidebar.")
    for m in chosen:
        ok, why = avail[m]
        if not ok:
            st.warning(f"**{m}** is unavailable: {why}")
            continue
        try:
            with st.spinner(f"Running {m}..."):
                results[m] = run_check(img_hash, m, img_bytes)
        except Exception as e:
            st.error(f"{m} failed: {type(e).__name__}: {e}")

    for m, r in results.items():
        meta = f"time {r.get('time_s', 0):.2f}s"
        u = r.get("usage")
        if isinstance(u, dict):
            cost = u.get("cost") or u.get("cost_usd")
            if cost is not None:
                meta += f" | cost ${float(cost):.5f}"
            toks = u.get("total_tokens")
            if toks:
                meta += f" | {toks} tokens"
        if r.get("model"):
            meta += f" | {r['model']}"
        (st.success if r.get("passed") else st.error)(
            f"{m}: {'PASS' if r.get('passed') else 'FAIL'}  ({meta})")

if results:
    st.subheader("Defect report")
    rows = []
    for d in DEFECTS:
        row = {"Defect": DEFECT_DESCRIPTIONS.get(d, d)}
        if truth is not None:
            row["Ground truth"] = "❌ defect" if truth.get(d) else "✅ ok"
        for m, r in results.items():
            v = r["defects"].get(d, {})
            s = v.get("score")
            row[m] = ("❌" if v.get("fail") else "✅") + (f" {s:.3g}" if isinstance(s, (int, float)) else "")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Defect"), width="stretch")

    for m, r in results.items():
        with st.expander(f"Reasons - {m}"):
            for d in DEFECTS:
                v = r["defects"].get(d, {})
                st.markdown(f"**{d}** {'❌' if v.get('fail') else '✅'} - {v.get('reason') or ''}")

    for m, r in results.items():
        inter = r.get("intermediates") or {}
        if inter:
            st.subheader(f"{m}: intermediate images")
            cols = st.columns(3)
            for i, (title, arr) in enumerate(inter.items()):
                with cols[i % 3]:
                    st.image(to_rgb(arr), caption=title, width="stretch")
