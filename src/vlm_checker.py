"""Vision-LLM quality checker via OpenRouter (OpenAI-compatible chat completions).

Config (project-root .env or environment):
    OPENROUTER_API_KEY   required for real calls (never printed/logged)
    OPENROUTER_MODEL     optional override of DEFAULT_MODEL
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEFECTS, DEFECT_DESCRIPTIONS, ROOT, empty_result, finalize  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:  # dotenv missing: fall back to plain environment
    pass

API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Chosen from the OpenRouter /models list (image input, non-reasoning, supports
# response_format): Google Gemma 4 26B-A4B, ~$0.077/M input and $0.255/M output
# tokens -> roughly $0.0002 per image. Current-generation and very cheap.
DEFAULT_MODEL = "google/gemma-4-26b-a4b-it"
MAX_SIDE = 768
TIMEOUT_S = 60
RETRIES = 3

CRITERIA = {
    "blur": "fail if the product is visibly out of focus or motion-blurred; edges, text and texture are not crisp.",
    "noise": "fail if visible grain / speckle / sensor noise is present, especially on flat areas.",
    "dark": "fail if the image is clearly under-exposed or low contrast (overall dim, muddy, shadows crushed).",
    "non_white_bg": "fail if the background is not near-pure white (any visible colour, gray tint, gradient, texture or scene). A plain white studio background passes.",
    "bad_framing": "fail if the product occupies less than ~45% of the frame (too small) or is clearly off-centre.",
    "cropped": "fail if ANY part of the product is cut off by the image edge.",
}


def is_available() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL


def build_prompt() -> str:
    rules = "\n".join(
        f'- "{d}" ({DEFECT_DESCRIPTIONS[d]}): {CRITERIA[d]}' for d in DEFECTS
    )
    schema = ", ".join(f'"{d}": {{"fail": false, "confidence": 0.0, "reason": ""}}' for d in DEFECTS)
    return (
        "You are a strict quality inspector for an e-commerce marketplace. Product photos must be "
        "sharp, noise-free, well exposed, on a plain white background, with the whole product "
        "centred and filling most of the frame. Inspect the attached product image for these "
        "six defects:\n"
        f"{rules}\n\n"
        "Judge each defect independently. Only flag a defect when it is clearly present.\n"
        "Respond with STRICT JSON only (no markdown, no prose) with exactly these keys:\n"
        f"{{{schema}}}\n"
        '"fail" is a boolean, "confidence" is a number from 0 to 1 (your confidence that the '
        'image HAS that defect when fail is true, otherwise your confidence it is fine), and '
        '"reason" is a short phrase (max 15 words).'
    )


def encode_image(img_bgr: np.ndarray) -> str:
    h, w = img_bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img_bgr = cv2.resize(img_bgr, (max(1, round(w * s)), max(1, round(h * s))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def parse_response(text: str) -> dict:
    """Parse model text into {defect: {"fail", "confidence", "reason"}} (raises ValueError)."""
    t = (text or "").strip()
    t = re.sub(r"```(?:json|JSON)?", "", t)
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    # balanced-brace scan for the first complete {...} block
    depth, in_str, esc, end = 0, False, False, -1
    for i in range(start, len(t)):
        c = t[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("unterminated JSON object in model output")
    obj = json.loads(t[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model JSON is not an object")
    out = {}
    for d in DEFECTS:
        v = obj.get(d)
        if v is None:
            out[d] = {"fail": False, "confidence": None, "reason": "not reported"}
            continue
        if not isinstance(v, dict):
            v = {"fail": v}
        f = v.get("fail", False)
        if isinstance(f, str):
            f = f.strip().lower() in ("true", "yes", "1")
        try:
            conf = float(v.get("confidence"))
            conf = min(1.0, max(0.0, conf))
        except (TypeError, ValueError):
            conf = None
        out[d] = {"fail": bool(f), "confidence": conf, "reason": str(v.get("reason", ""))[:200]}
    return out


def _call_api(payload: dict, key: str) -> dict:
    """POST with retries/backoff. Returns response JSON. Raises RuntimeError."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "DIP product image QC"}
    last = "unknown error"
    use_rf = True
    attempt = 0
    while attempt < RETRIES:
        body = dict(payload)
        if not use_rf:
            body.pop("response_format", None)
        try:
            r = requests.post(API_URL, headers=headers, json=body, timeout=TIMEOUT_S)
        except (requests.Timeout, requests.ConnectionError) as e:
            last = f"network error: {type(e).__name__}"
        else:
            if r.status_code == 200:
                try:
                    data = r.json()
                except ValueError:
                    data = {}
                if data.get("choices"):
                    return data
                last = f"empty/invalid response: {str(data.get('error', ''))[:200]}"
            elif r.status_code == 400 and use_rf and "response_format" in r.text.lower():
                use_rf = False  # model rejects response_format: retry without it (no attempt used)
                continue
            elif r.status_code in (401, 403):
                raise RuntimeError(f"OpenRouter rejected the API key (HTTP {r.status_code}).")
            elif r.status_code == 429 or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
            else:
                raise RuntimeError(f"OpenRouter error HTTP {r.status_code}: {r.text[:300]}")
        attempt += 1
        if attempt < RETRIES:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"VLM request failed after {RETRIES} attempts: {last}")


def check(img_bgr: np.ndarray) -> dict:
    t0 = time.time()
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (add it to .env in the project root).")
    model = get_model()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": build_prompt()},
            {"type": "image_url", "image_url": {"url": encode_image(img_bgr)}},
        ]}],
    }
    parsed, data = None, None
    err = "unknown"
    for _ in range(2):  # one extra request if the output cannot be parsed
        data = _call_api(payload, key)
        content = data["choices"][0].get("message", {}).get("content") or ""
        try:
            parsed = parse_response(content)
            break
        except (ValueError, json.JSONDecodeError) as e:
            err = f"{e}; output starts: {content[:120]!r}"
    if parsed is None:
        raise RuntimeError(f"Could not parse VLM output: {err}")

    res = empty_result("vlm")
    for d in DEFECTS:
        p = parsed[d]
        res["defects"][d] = {"fail": p["fail"], "score": p["confidence"], "reason": p["reason"]}
    u = data.get("usage") or {}
    res["usage"] = {"prompt_tokens": u.get("prompt_tokens"),
                    "completion_tokens": u.get("completion_tokens"),
                    "cost": u.get("cost")}
    res["model"] = data.get("model") or model
    res["time_s"] = time.time() - t0
    return finalize(res)


if __name__ == "__main__":
    from common import load_image
    if len(sys.argv) < 2:
        sys.exit("usage: python src/vlm_checker.py <image>")
    out = check(load_image(sys.argv[1]))
    print(json.dumps(out, indent=2, default=str))
