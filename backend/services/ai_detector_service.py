import os
import re
import json
import logging
import httpx
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Preferred model order — first one that responds successfully is used
HF_MODELS = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
]

# Base URLs — try modern router endpoint first, legacy endpoint second
HF_BASE_URLS = [
    "https://router.huggingface.co/hf-inference/v1/models",
    "https://api-inference.huggingface.co/models",
]

# ── Substring pattern sets ───────────────────────────────────────────────────
# Substring patterns indicating AI-generated content
AI_POSITIVE_PATTERNS = [
    "fake", "generated", "synthetic", "artificial", "deepfake", "sdxl",
    "midjourney", "ai_generated", "ai-generated", "machine"
]

# Substring patterns indicating human / authentic content
HUMAN_POSITIVE_PATTERNS = [
    "human", "authentic", "natural", "photograph", "photo"
]


def _is_ai_positive_label(label: str) -> bool:
    """Returns True if the label matches AI-positive substring patterns or word boundary 'ai'."""
    lbl = label.lower().strip()
    if any(p in lbl for p in AI_POSITIVE_PATTERNS):
        return True
    # Word boundary check for 'ai' (to prevent false matches in words like 'portrait' or 'mountain')
    if re.search(r'\bai\b', lbl):
        return True
    return False


def _is_human_positive_label(label: str) -> bool:
    """Returns True if the label matches human/authentic substring patterns or word boundary 'real'."""
    lbl = label.lower().strip()
    if any(p in lbl for p in HUMAN_POSITIVE_PATTERNS):
        return True
    # Word boundary check for 'real'
    if re.search(r'\breal\b', lbl):
        return True
    return False


def _parse_confidence(response_json: Any, *, log_prefix: str = "") -> Optional[float]:
    """
    Parse raw HF classifier response into a 0-100 AI-generation confidence.

    Strategy:
    1. Extract items list (unwrapping outer batch list if present).
    2. Log raw summary of all returned labels and scores.
    3. Take the highest-scoring label:
       - If AI-positive: confidence = score * 100
       - If human/authentic: confidence = (1 - score) * 100
    4. Fallback: check secondary labels if top label did not match.
    5. If no label matches either pattern set: log warning with unmatched label and return None.
    """
    if not isinstance(response_json, list):
        msg = f"[AI-detector{log_prefix}] Unexpected response format (not a list): {type(response_json).__name__} - raw: {response_json!r}"
        logger.warning(msg)
        print(msg)
        return None

    # Unwrap batch outer list [[{...}]] -> [{...}]
    items = response_json[0] if response_json and isinstance(response_json[0], list) else response_json

    if not items or not isinstance(items, list):
        msg = f"[AI-detector{log_prefix}] Empty or invalid items in response: {response_json!r}"
        logger.warning(msg)
        print(msg)
        return None

    # Log labels and scores summary
    labels_summary = ", ".join(
        f"{str(i.get('label'))!r}:{float(i.get('score', 0)):.4f}"
        for i in items if isinstance(i, dict)
    )
    logger.info(f"[AI-detector{log_prefix}] Parsed response labels: [{labels_summary}]")
    print(f"[AI-detector{log_prefix}] Parsed response labels: [{labels_summary}]")

    # Take the highest-scoring label
    try:
        top_item = max(items, key=lambda x: float(x.get("score", 0.0)) if isinstance(x, dict) else -1.0)
    except Exception as e:
        msg = f"[AI-detector{log_prefix}] Error finding top-scoring label item: {e}"
        logger.warning(msg)
        print(msg)
        return None

    if not isinstance(top_item, dict):
        msg = f"[AI-detector{log_prefix}] Top item is not a dict: {top_item!r}"
        logger.warning(msg)
        print(msg)
        return None

    top_label = str(top_item.get("label", "")).strip()
    top_score = float(top_item.get("score", 0.0))

    # Match top label against AI-positive patterns
    if _is_ai_positive_label(top_label):
        conf = round(top_score * 100, 2)
        msg = f"[AI-detector{log_prefix}] Top label '{top_label}' matched AI-positive pattern -> confidence = {conf:.2f}%"
        logger.info(msg)
        print(msg)
        return conf

    # Match top label against human/authentic patterns
    if _is_human_positive_label(top_label):
        conf = round((1.0 - top_score) * 100, 2)
        msg = f"[AI-detector{log_prefix}] Top label '{top_label}' matched human/authentic pattern (score={top_score:.4f}) -> inverted AI conf = {conf:.2f}%"
        logger.info(msg)
        print(msg)
        return conf

    # Secondary check: if top label didn't match, check other returned items
    for item in items:
        if not isinstance(item, dict):
            continue
        lbl = str(item.get("label", "")).strip()
        sc = float(item.get("score", 0.0))
        if _is_ai_positive_label(lbl):
            conf = round(sc * 100, 2)
            msg = f"[AI-detector{log_prefix}] Secondary label '{lbl}' matched AI-positive pattern -> confidence = {conf:.2f}%"
            logger.info(msg)
            print(msg)
            return conf
        if _is_human_positive_label(lbl):
            conf = round((1.0 - sc) * 100, 2)
            msg = f"[AI-detector{log_prefix}] Secondary label '{lbl}' matched human/authentic pattern (score={sc:.4f}) -> inverted AI conf = {conf:.2f}%"
            logger.info(msg)
            print(msg)
            return conf

    # Unmatched label fallback: log warning with unmatched label and return None
    unmatched_labels = [str(i.get("label")) for i in items if isinstance(i, dict)]
    msg = f"[AI-detector{log_prefix}] WARNING: Unmatched label string(s) returned by HF model: {unmatched_labels} (top label was '{top_label}')"
    logger.warning(msg)
    print(msg)
    return None


def run_ai_detector(image_bytes: bytes, log_prefix: str = "") -> Dict[str, Any]:
    """
    Calls Hugging Face Inference API image-classification models to estimate
    AI-generation confidence. Tries HF_MODELS and HF_BASE_URLS in order, returning on first success.

    log_prefix: optional tag appended to diagnostic prints (e.g. " [frame 2]")

    Returns:
      {"ai_generation_confidence": float (0-100), "model_used": str, "raw_label": str}
    or on failure:
      {"ai_generation_confidence": None, "note": "<reason>"}

    Never crashes — all exceptions are caught and degrade gracefully.
    """
    token = os.environ.get("HF_API_TOKEN", "").strip()

    headers = {
        "Content-Type": "application/octet-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for model_id in HF_MODELS:
        for base_url in HF_BASE_URLS:
            url = f"{base_url}/{model_id}"
            msg = f"[AI-detector{log_prefix}] POST {url} ({len(image_bytes)} bytes)"
            logger.info(msg)
            print(msg)

            try:
                response = httpx.post(
                    url,
                    content=image_bytes,
                    headers=headers,
                    timeout=10.0,
                )
                msg = f"[AI-detector{log_prefix}] HTTP {response.status_code} from {model_id}"
                logger.info(msg)
                print(msg)

                if response.status_code == 200:
                    data = response.json()
                    
                    # REQUIREMENT 1: Log full raw JSON response from HF API call at INFO level
                    raw_json_str = json.dumps(data)
                    raw_msg = f"[AI-detector{log_prefix}] Full raw JSON response from HF API ({model_id}): {raw_json_str}"
                    logger.info(raw_msg)
                    print(raw_msg)

                    confidence = _parse_confidence(data, log_prefix=f"{log_prefix} [{model_id}]")
                    if confidence is not None:
                        items = data[0] if data and isinstance(data[0], list) else data
                        raw_label = str(items[0].get("label", "")) if items and isinstance(items[0], dict) else ""
                        return {
                            "ai_generation_confidence": confidence,
                            "model_used": model_id,
                            "raw_label": raw_label,
                        }
                    # confidence is None — log warning and try next model
                    warn_msg = f"[AI-detector{log_prefix}] {model_id} returned unparseable/unmatched labels, trying next model"
                    logger.warning(warn_msg)
                    print(warn_msg)

                elif response.status_code == 503:
                    # Model still loading
                    body_preview = response.text[:200]
                    msg = f"[AI-detector{log_prefix}] {model_id} loading (503): {body_preview}"
                    logger.warning(msg)
                    print(msg)
                    break  # Try next model

                elif response.status_code == 401:
                    msg = f"[AI-detector{log_prefix}] HF_API_TOKEN rejected or missing (401) for {model_id}"
                    logger.warning(msg)
                    print(msg)
                    break  # Try next model/URL

                else:
                    msg = f"[AI-detector{log_prefix}] {model_id} returned HTTP {response.status_code}: {response.text[:300]}"
                    logger.warning(msg)
                    print(msg)

            except httpx.TimeoutException:
                msg = f"[AI-detector{log_prefix}] {model_id} timed out (10s)"
                logger.warning(msg)
                print(msg)
            except Exception as e:
                msg = f"[AI-detector{log_prefix}] {model_id} exception: {type(e).__name__}: {e}"
                logger.error(msg)
                print(msg)

    msg = f"[AI-detector{log_prefix}] All models/endpoints exhausted or returned unavailable — returning None confidence"
    logger.info(msg)
    print(msg)
    return {"ai_generation_confidence": None, "note": "AI detector unavailable"}
