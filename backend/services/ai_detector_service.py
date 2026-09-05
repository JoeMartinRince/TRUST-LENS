import os
import re
import json
import logging
import httpx
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ── Service Constants ────────────────────────────────────────────────────────
HF_HTTP_TIMEOUT_SECONDS: float = 10.0

HF_MODELS: List[str] = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
]

HF_BASE_URLS: List[str] = [
    "https://router.huggingface.co/hf-inference/v1/models",
    "https://api-inference.huggingface.co/models",
]

# ── Substring pattern sets ───────────────────────────────────────────────────
AI_POSITIVE_PATTERNS: List[str] = [
    "fake", "generated", "synthetic", "artificial", "deepfake", "sdxl",
    "midjourney", "ai_generated", "ai-generated", "machine"
]

HUMAN_POSITIVE_PATTERNS: List[str] = [
    "human", "authentic", "natural", "photograph", "photo"
]


def _is_ai_positive_label(label: str) -> bool:
    """
    Returns True if the label string matches AI-positive substring patterns or word boundary 'ai'.

    Args:
        label (str): Label string returned by Hugging Face classifier.

    Returns:
        bool: True if label indicates AI-generated content.
    """
    lbl = label.lower().strip()
    if any(p in lbl for p in AI_POSITIVE_PATTERNS):
        return True
    if re.search(r'\bai\b', lbl):
        return True
    return False


def _is_human_positive_label(label: str) -> bool:
    """
    Returns True if the label string matches human/authentic substring patterns or word boundary 'real'.

    Args:
        label (str): Label string returned by Hugging Face classifier.

    Returns:
        bool: True if label indicates human/authentic content.
    """
    lbl = label.lower().strip()
    if any(p in lbl for p in HUMAN_POSITIVE_PATTERNS):
        return True
    if re.search(r'\breal\b', lbl):
        return True
    return False


def _parse_confidence(response_json: Any, *, log_prefix: str = "") -> Optional[float]:
    """
    Parses raw Hugging Face classifier response payload into a 0-100 AI-generation confidence score.

    Args:
        response_json (Any): Parsed JSON response from Hugging Face API.
        log_prefix (str): Prefix tag for diagnostic logging.

    Returns:
        Optional[float]: 0.0 - 100.0 confidence score if parsed successfully, otherwise None.
    """
    if not isinstance(response_json, list):
        logger.warning(f"[AI-detector{log_prefix}] Unexpected response format (not a list): {type(response_json).__name__}")
        return None

    items = response_json[0] if response_json and isinstance(response_json[0], list) else response_json

    if not items or not isinstance(items, list):
        logger.warning(f"[AI-detector{log_prefix}] Empty or invalid items list in response: {response_json!r}")
        return None

    labels_summary = ", ".join(
        f"{str(i.get('label'))!r}:{float(i.get('score', 0)):.4f}"
        for i in items if isinstance(i, dict)
    )
    logger.info(f"[AI-detector{log_prefix}] Parsed response labels: [{labels_summary}]")

    try:
        top_item = max(items, key=lambda x: float(x.get("score", 0.0)) if isinstance(x, dict) else -1.0)
    except Exception as e:
        logger.warning(f"[AI-detector{log_prefix}] Error finding top-scoring label item: {e}")
        return None

    if not isinstance(top_item, dict):
        logger.warning(f"[AI-detector{log_prefix}] Top item is not a dictionary: {top_item!r}")
        return None

    top_label = str(top_item.get("label", "")).strip()
    top_score = float(top_item.get("score", 0.0))

    if _is_ai_positive_label(top_label):
        conf = round(top_score * 100, 2)
        logger.info(f"[AI-detector{log_prefix}] Top label '{top_label}' matched AI-positive pattern -> confidence = {conf:.2f}%")
        return conf

    if _is_human_positive_label(top_label):
        conf = round((1.0 - top_score) * 100, 2)
        logger.info(f"[AI-detector{log_prefix}] Top label '{top_label}' matched human/authentic pattern (score={top_score:.4f}) -> inverted AI conf = {conf:.2f}%")
        return conf

    for item in items:
        if not isinstance(item, dict):
            continue
        lbl = str(item.get("label", "")).strip()
        sc = float(item.get("score", 0.0))
        if _is_ai_positive_label(lbl):
            conf = round(sc * 100, 2)
            logger.info(f"[AI-detector{log_prefix}] Secondary label '{lbl}' matched AI-positive pattern -> confidence = {conf:.2f}%")
            return conf
        if _is_human_positive_label(lbl):
            conf = round((1.0 - sc) * 100, 2)
            logger.info(f"[AI-detector{log_prefix}] Secondary label '{lbl}' matched human/authentic pattern (score={sc:.4f}) -> inverted AI conf = {conf:.2f}%")
            return conf

    unmatched_labels = [str(i.get("label")) for i in items if isinstance(i, dict)]
    logger.warning(f"[AI-detector{log_prefix}] WARNING: Unmatched label string(s) returned by HF model: {unmatched_labels} (top label was '{top_label}')")
    return None


async def run_ai_detector_async(
    image_bytes: bytes,
    log_prefix: str = "",
    client: Optional[httpx.AsyncClient] = None
) -> Dict[str, Any]:
    """
    Non-blocking async call to Hugging Face Inference API image-classification models.
    Supports reusing an existing httpx.AsyncClient for connection pooling.

    Args:
        image_bytes (bytes): Raw binary bytes of image file.
        log_prefix (str): Optional diagnostic logging prefix (e.g. " [frame 2]").
        client (Optional[httpx.AsyncClient]): Shared HTTP client instance for connection reuse.

    Returns:
        Dict[str, Any]: {"ai_generation_confidence": float|None, "model_used": str, "raw_label": str}
    """
    token = os.environ.get("HF_API_TOKEN", "").strip()

    headers = {
        "Content-Type": "application/octet-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    close_client = False
    if client is None:
        client = httpx.AsyncClient(timeout=HF_HTTP_TIMEOUT_SECONDS)
        close_client = True

    try:
        for model_id in HF_MODELS:
            for base_url in HF_BASE_URLS:
                url = f"{base_url}/{model_id}"
                logger.info(f"[AI-detector-async{log_prefix}] POST {url} ({len(image_bytes)} bytes)")

                try:
                    response = await client.post(
                        url,
                        content=image_bytes,
                        headers=headers,
                    )
                    logger.info(f"[AI-detector-async{log_prefix}] HTTP {response.status_code} from {model_id}")

                    if response.status_code == 200:
                        data = response.json()
                        raw_json_str = json.dumps(data)
                        logger.info(f"[AI-detector-async{log_prefix}] Full raw JSON response from HF API ({model_id}): {raw_json_str}")

                        confidence = _parse_confidence(data, log_prefix=f"{log_prefix} [{model_id}]")
                        if confidence is not None:
                            items = data[0] if data and isinstance(data[0], list) else data
                            raw_label = str(items[0].get("label", "")) if items and isinstance(items[0], dict) else ""
                            return {
                                "ai_generation_confidence": confidence,
                                "model_used": model_id,
                                "raw_label": raw_label,
                            }
                        logger.warning(f"[AI-detector-async{log_prefix}] {model_id} returned unparseable/unmatched labels, trying next model")

                    elif response.status_code == 503:
                        logger.warning(f"[AI-detector-async{log_prefix}] {model_id} loading (503): {response.text[:200]}")
                        break

                    elif response.status_code == 401:
                        logger.warning(f"[AI-detector-async{log_prefix}] HF_API_TOKEN rejected or missing (401) for {model_id}")
                        break

                    else:
                        logger.warning(f"[AI-detector-async{log_prefix}] {model_id} returned HTTP {response.status_code}: {response.text[:300]}")

                except httpx.TimeoutException:
                    logger.warning(f"[AI-detector-async{log_prefix}] {model_id} timed out ({HF_HTTP_TIMEOUT_SECONDS}s)")
                except Exception as e:
                    logger.error(f"[AI-detector-async{log_prefix}] {model_id} exception: {type(e).__name__}: {e}")

        logger.info(f"[AI-detector-async{log_prefix}] All models/endpoints exhausted or returned unavailable — returning None confidence")
        return {"ai_generation_confidence": None, "note": "AI detector unavailable"}
    finally:
        if close_client:
            await client.aclose()


def run_ai_detector(image_bytes: bytes, log_prefix: str = "") -> Dict[str, Any]:
    """
    Synchronous wrapper for run_ai_detector_async for backward compatibility.

    Args:
        image_bytes (bytes): Raw binary bytes of image file.
        log_prefix (str): Optional diagnostic logging prefix.

    Returns:
        Dict[str, Any]: AI generation detector results.
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
