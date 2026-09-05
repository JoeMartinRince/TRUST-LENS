import os
import logging
import httpx
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)

# ── Service Constants ────────────────────────────────────────────────────────
HF_AUDIO_TIMEOUT_SECONDS: float = 10.0

HF_AUDIO_MODELS: List[str] = [
    "MelodyMachine/Deepfake-audio-detection-V2",
    "bartowski/audio-deepfake-detection",
]

HF_BASE_URL: str = "https://api-inference.huggingface.co/models"

FAKE_LABELS: Set[str] = {
    "fake", "spoof", "deepfake", "synthetic", "ai", "generated",
    "ai-generated", "artificial", "tts", "vc"
}


def _parse_deepfake_confidence(response_json: Any) -> Optional[float]:
    """
    Parses Hugging Face classifier response into deepfake confidence score (0-100).

    Args:
        response_json (Any): Raw JSON payload from Hugging Face API.

    Returns:
        Optional[float]: Deepfake confidence score if matched, otherwise None.
    """
    try:
        if isinstance(response_json, list):
            items = (
                response_json[0]
                if response_json and isinstance(response_json[0], list)
                else response_json
            )
            for item in items:
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0))
                if any(fl in label for fl in FAKE_LABELS):
                    return round(score * 100, 2)
    except Exception as err:
        logger.warning(f"Failed to parse deepfake confidence from HF response: {err}")
    return None


def run_deepfake_audio_detector(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Calls Hugging Face Inference API to detect deepfake / AI-generated audio.
    Tries HF_AUDIO_MODELS in order, returns on first success.

    Args:
        audio_bytes (bytes): Raw binary bytes of audio file.

    Returns:
        Dict[str, Any]: {
            "deepfake_audio_confidence": float|None,
            "model_used": str,
            "raw_label": str,
            "note": str (optional)
        }
    """
    if not audio_bytes:
        return {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable (no audio bytes)"}

    token = os.environ.get("HF_API_TOKEN", "").strip()
    if not token:
        return {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable (no HF_API_TOKEN)"}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
    }

    for model_id in HF_AUDIO_MODELS:
        url = f"{HF_BASE_URL}/{model_id}"
        try:
            response = httpx.post(url, content=audio_bytes, headers=headers, timeout=HF_AUDIO_TIMEOUT_SECONDS)
            if response.status_code == 200:
                data = response.json()
                confidence = _parse_deepfake_confidence(data)
                if confidence is not None:
                    items = data[0] if data and isinstance(data[0], list) else data
                    raw_label = str(items[0].get("label", "")) if items else ""
                    return {
                        "deepfake_audio_confidence": confidence,
                        "model_used": model_id,
                        "raw_label": raw_label,
                    }
            elif response.status_code == 503:
                logger.warning(f"HF audio model {model_id} is loading (503), trying next.")
                continue
            else:
                logger.warning(f"HF deepfake audio detector returned HTTP {response.status_code} for {model_id}")
        except httpx.TimeoutException:
            logger.warning(f"HF deepfake audio detector timed out ({HF_AUDIO_TIMEOUT_SECONDS}s) for {model_id}")
        except Exception as e:
            logger.error(f"HF deepfake audio detector error for {model_id}: {e}")

    return {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable"}
