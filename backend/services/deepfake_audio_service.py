import os
import httpx
from typing import Dict, Any

# Primary model and fallback chain
HF_AUDIO_MODELS = [
    "MelodyMachine/Deepfake-audio-detection-V2",
    "bartowski/audio-deepfake-detection",
]

HF_BASE_URL = "https://api-inference.huggingface.co/models"

# Label strings that indicate the "deepfake / AI-generated" class
FAKE_LABELS = {
    "fake", "spoof", "deepfake", "synthetic", "ai", "generated",
    "ai-generated", "artificial", "tts", "vc"
}


def _parse_deepfake_confidence(response_json: Any) -> float | None:
    """
    Parse HF classifier response → deepfake confidence 0-100.
    Handles both list-of-dicts and nested batch formats.
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
    except Exception:
        pass
    return None


def run_deepfake_audio_detector(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Calls Hugging Face Inference API to detect deepfake / AI-generated audio.
    Tries HF_AUDIO_MODELS in order, returns on first success.

    Returns:
      {
        "deepfake_audio_confidence": float (0-100),
        "model_used": str,
        "raw_label": str
      }
    or on failure / missing token:
      {
        "deepfake_audio_confidence": None,
        "note": "Deepfake audio detector unavailable"
      }

    Never crashes — all exceptions are caught.
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
            response = httpx.post(url, content=audio_bytes, headers=headers, timeout=10.0)
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
                print(f"HF audio model {model_id} is loading (503), trying next.")
                continue
            else:
                print(f"HF deepfake audio detector returned {response.status_code} for {model_id}")
        except httpx.TimeoutException:
            print(f"HF deepfake audio detector timed out for {model_id}")
        except Exception as e:
            print(f"HF deepfake audio detector error for {model_id}: {e}")

    return {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable"}
