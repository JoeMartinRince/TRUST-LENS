import os
import re
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ── Service Constants & Regex Patterns ───────────────────────────────────────
AI_WORDS_RE = re.compile(
    r"(\bai\b|generated|sora|runway|pika|kling|veo|synthesia|heygen|deepfake|midjourney|dalle|dall-e|stable[_-]?diffusion|sdxl|gemini|leonardo|output|flux|chatgpt|ideogram|playground)",
    re.IGNORECASE
)

AI_SEQUENTIAL_RE = re.compile(
    r"^(image|video|generated[_-]?(image|video)?|ai[_-]?(image|video)?|output)[_\s\-\(\)]*\(\d+\)\.(jpg|jpeg|png|webp|gif|mp4|mov|wav|mp3|mkv|avi|webm)$",
    re.IGNORECASE
)

CAMERA_NATIVE_RE = re.compile(
    r"^(IMG|DSC|PXL|DCIM|GOPR|SAM|MAH|CIMG|VID|MVI)[_-]?\d+([_-]\d+)*(\.\w+)?$|^(PXL[_-]|\d{8}[_-])?\d{8}[_-]\d{6,}(\.\w+)?$",
    re.IGNORECASE
)

GENERIC_DOWNLOAD_RE = re.compile(
    r"^(download|download\s*\(\d+\)|image|video|untitled|media|file|document|photo|picture|screenshot)(\.(jpg|jpeg|png|webp|gif|mp4|mov|wav|mp3|mkv|avi|webm))?$",
    re.IGNORECASE
)


def analyze_filename(filename: str) -> Dict[str, Any]:
    """
    Analyzes an uploaded file's original filename for AI generator, camera native, or generic download patterns.

    Args:
        filename (str): Original filename of uploaded media.

    Returns:
        Dict[str, Any]: {
            "filename": str,
            "pattern_type": "ai_generator" | "camera_native" | "generic_download" | "unrecognized",
            "note": str
        }
    """
    try:
        clean_name = os.path.basename(str(filename or "")).strip()
        if not clean_name:
            return {
                "filename": "unknown",
                "pattern_type": "unrecognized",
                "note": "Filename unavailable"
            }

        # 1. Check AI-generator naming patterns
        if AI_WORDS_RE.search(clean_name) or AI_SEQUENTIAL_RE.match(clean_name):
            return {
                "filename": clean_name,
                "pattern_type": "ai_generator",
                "note": "Filename contains AI generator or synthetic model naming signature"
            }

        # 2. Check Camera-native patterns (authentic capture device signal)
        name_no_ext = os.path.splitext(clean_name)[0]
        if CAMERA_NATIVE_RE.match(clean_name) or CAMERA_NATIVE_RE.match(name_no_ext):
            return {
                "filename": clean_name,
                "pattern_type": "camera_native",
                "note": "Matches standard camera or smartphone hardware capture naming convention"
            }

        # 3. Check Generic/downloaded patterns
        if GENERIC_DOWNLOAD_RE.match(clean_name):
            return {
                "filename": clean_name,
                "pattern_type": "generic_download",
                "note": "Filename gives no provenance info, likely re-saved or downloaded"
            }

        # 4. Fallback default
        return {
            "filename": clean_name,
            "pattern_type": "unrecognized",
            "note": "Custom or unclassified filename pattern"
        }
    except Exception as err:
        logger.error(f"Filename analysis error: {err}")
        return {
            "filename": str(filename or "unknown"),
            "pattern_type": "unrecognized",
            "note": "Filename pattern unrecognized"
        }
