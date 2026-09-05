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
AI_POSITIVE_PATTERNS = [
    "fake", "generated", "synthetic", "artificial", "deepfake", "sdxl",
    "midjourney", "ai_generated", "ai-generated", "machine"
]

HUMAN_POSITIVE_PATTERNS = [
    "human", "authentic", "natural", "photograph", "photo"
]


def _is_ai_positive_label(label: str) -> bool:
    """Returns True if the label matches AI-positive substring patterns or word boundary 'ai'."""
    lbl = label.lower().strip()
    if any(p in lbl for p in AI_POSITIVE_PATTERNS):
        return True
    if re.search(r'\bai\b', lbl):
        return True
    return False


def _is_human_positive_label(label: str) -> bool:
    """Returns True if the label matches human/authentic substring patterns or word boundary 'real'."""
    lbl = label.lower().strip()
    if any(p in lbl for p in HUMAN_POSITIVE_PATTERNS):
        return True
    if re.search(r'\breal\b', lbl):
        return True
    return False


def _parse_confidence(response_json: Any, *, log_prefix: str = "") -> Optional[float]:
    """
    Parse raw HF classifier response into a 0-100 AI-generation confidence.
    """
    if not isinstance(response_json, list):
        msg = f"[AI-detector{log_prefix}] Unexpected response format (not a list): {type(response_json).__name__} - raw: {response_json!r}"
        logger.warning(msg)
        print(msg)
        return None

    items = response_json[0] if response_json and isinstance(response_json[0], list) else response_json

    if not items or not isinstance(items, list):
        msg = f"[AI-detector{log_prefix}] Empty or invalid items in response: {response_json!r}"
        logger.warning(msg)
        print(msg)
        return None

    labels_summary = ", ".join(
        f"{str(i.get('label'))!r}:{float(i.get('score', 0)):.4f}"
        for i in items if isinstance(i, dict)
    )
    logger.info(f"[AI-detector{log_prefix}] Parsed response labels: [{labels_summary}]")
    print(f"[AI-detector{log_prefix}] Parsed response labels: [{labels_summary}]")

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

    if _is_ai_positive_label(top_label):
        conf = round(top_score * 100, 2)
        msg = f"[AI-detector{log_prefix}] Top label '{top_label}' matched AI-positive pattern -> confidence = {conf:.2f}%"
        logger.info(msg)
        print(msg)
        return conf

    if _is_human_positive_label(top_label):
        conf = round((1.0 - top_score) * 100, 2)
        msg = f"[AI-detector{log_prefix}] Top label '{top_label}' matched human/authentic pattern (score={top_score:.4f}) -> inverted AI conf = {conf:.2f}%"
        logger.info(msg)
        print(msg)
        return conf

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

    unmatched_labels = [str(i.get("label")) for i in items if isinstance(i, dict)]
    msg = f"[AI-detector{log_prefix}] WARNING: Unmatched label string(s) returned by HF model: {unmatched_labels} (top label was '{top_label}')"
    logger.warning(msg)
    print(msg)
    return None


async def run_ai_detector_async(
    image_bytes: bytes,
    log_prefix: str = "",
    client: Optional[httpx.AsyncClient] = None
) -> Dict[str, Any]:
    """
    Non-blocking async call to Hugging Face Inference API image-classification models.
    Supports reusing an existing httpx.AsyncClient for connection pooling.
    """
    token = os.environ.get("HF_API_TOKEN", "").strip()

    headers = {
        "Content-Type": "application/octet-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=10.0)
        close_client = True

    try:
        for model_id in HF_MODELS:
            for base_url in HF_BASE_URLS:
                url = f"{base_url}/{model_id}"
                msg = f"[AI-detector-async{log_prefix}] POST {url} ({len(image_bytes)} bytes)"
                logger.info(msg)
                print(msg)

                try:
                    response = await client.post(
                        url,
                        content=image_bytes,
                        headers=headers,
                    )
                    msg = f"[AI-detector-async{log_prefix}] HTTP {response.status_code} from {model_id}"
                    logger.info(msg)
                    print(msg)

                    if response.status_code == 200:
                        data = response.json()
                        raw_json_str = json.dumps(data)
                        raw_msg = f"[AI-detector-async{log_prefix}] Full raw JSON response from HF API ({model_id}): {raw_json_str}"
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
                        warn_msg = f"[AI-detector-async{log_prefix}] {model_id} returned unparseable/unmatched labels, trying next model"
                        logger.warning(warn_msg)
                        print(warn_msg)

                    elif response.status_code == 503:
                        body_preview = response.text[:200]
                        msg = f"[AI-detector-async{log_prefix}] {model_id} loading (503): {body_preview}"
                        logger.warning(msg)
                        print(msg)
                        break

                    elif response.status_code == 401:
                        msg = f"[AI-detector-async{log_prefix}] HF_API_TOKEN rejected or missing (401) for {model_id}"
                        logger.warning(msg)
                        print(msg)
                        break

                    else:
                        msg = f"[AI-detector-async{log_prefix}] {model_id} returned HTTP {response.status_code}: {response.text[:300]}"
                        logger.warning(msg)
                        print(msg)

                except httpx.TimeoutException:
                    msg = f"[AI-detector-async{log_prefix}] {model_id} timed out (10s)"
                    logger.warning(msg)
                    print(msg)
                except Exception as e:
                    msg = f"[AI-detector-async{log_prefix}] {model_id} exception: {type(e).__name__}: {e}"
                    logger.error(msg)
                    print(msg)

        msg = f"[AI-detector-async{log_prefix}] All models/endpoints exhausted or returned unavailable — returning None confidence"
        logger.info(msg)
        print(msg)
        return {"ai_generation_confidence": None, "note": "AI detector unavailable"}
    finally:
        if close_client:
            await client.aclose()


def run_ai_detector(image_bytes: bytes, log_prefix: str = "") -> Dict[str, Any]:
    """
    Synchronous wrapper for run_ai_detector_async for backward compatibility.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(run_ai_detector_async(image_bytes, log_prefix=log_prefix))
        else:
            return loop.run_until_complete(run_ai_detector_async(image_bytes, log_prefix=log_prefix))
    except Exception:
        return asyncio.run(run_ai_detector_async(image_bytes, log_prefix=log_prefix))
