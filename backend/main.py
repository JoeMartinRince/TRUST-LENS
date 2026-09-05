import os
import io
import hashlib
import asyncio
import logging
import httpx
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.services.metadata_service import extract_metadata
from backend.services.ela_service import run_ela_analysis
from backend.services.reverse_search_service import trace_source
from backend.services.audio_service import analyze_audio_file
from backend.services.filename_service import analyze_filename
from backend.services.ai_detector_service import run_ai_detector_async, run_ai_detector
from backend.services.deepfake_audio_service import run_deepfake_audio_detector
from backend.services.video_keyframe_service import run_video_ai_detector_async, run_video_ai_detector
from backend.services.gemini_synthesis import generate_synthesis, generate_video_synthesis, generate_audio_synthesis

logger = logging.getLogger(__name__)

# ── Pipeline Constants ───────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024  # 100MB maximum upload payload limit
CHUNK_SIZE_BYTES: int = 64 * 1024               # 64KB chunk buffer size for streaming
MAX_CACHE_ENTRIES: int = 100                    # Maximum entries for SHA-256 response cache
HTTP_FETCH_TIMEOUT_SECONDS: float = 15.0        # Timeout for URL payload fetching

DEFAULT_IMAGE_FILENAME: str = "upload.jpg"
DEFAULT_AUDIO_FILENAME: str = "audio.wav"

VIDEO_EXTENSIONS: set = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

# ── In-Memory SHA-256 Response Caching ───────────────────────────────────────
_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_cache_key(data_bytes: bytes, target_url: Optional[str] = None) -> str:
    """
    Computes a deterministic SHA-256 hex digest cache key for media payload bytes or URL.

    Args:
        data_bytes (bytes): Raw binary bytes of uploaded media file.
        target_url (Optional[str]): Source URL if provided via JSON request.

    Returns:
        str: 64-character SHA-256 hex digest string.
    """
    hasher = hashlib.sha256()
    if target_url:
        hasher.update(f"url:{target_url}".encode("utf-8"))
    if data_bytes:
        hasher.update(data_bytes)
    return hasher.hexdigest()


def _cache_set(key: str, val: Dict[str, Any]) -> None:
    """
    Stores analysis response in LRU-style in-memory response cache.

    Args:
        key (str): SHA-256 hex digest cache key.
        val (Dict[str, Any]): Complete response payload dictionary to cache.
    """
    if len(_RESPONSE_CACHE) >= MAX_CACHE_ENTRIES:
        first_key = next(iter(_RESPONSE_CACHE))
        _RESPONSE_CACHE.pop(first_key, None)
    _RESPONSE_CACHE[key] = val


app = FastAPI(
    title="TrustLens API",
    description="Backend verification pipeline for image, video, and audio authenticity.",
    version="1.1.0"
)

# Enable CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving ELA heatmap images
static_path = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.on_event("startup")
async def startup_check_environment() -> None:
    """
    Checks environment variables on app startup and logs warnings for any missing credentials.
    """
    hf_token = os.environ.get("HF_API_TOKEN", "").strip()
    if not hf_token:
        logger.warning(
            "⚠️  [STARTUP WARNING] HF_API_TOKEN is missing or empty in environment / .env! "
            "Hugging Face AI detector calls will fail or operate unauthenticated."
        )
    else:
        logger.info("✅ [STARTUP CHECK] HF_API_TOKEN successfully loaded from environment.")

    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        logger.warning(
            "⚠️  [STARTUP WARNING] GEMINI_API_KEY is missing or empty in environment / .env! "
            "Gemini synthesis will fall back to local heuristics."
        )
    else:
        logger.info("✅ [STARTUP CHECK] GEMINI_API_KEY successfully loaded from environment.")



async def _extract_payload_bytes(
    request: Request,
    file: Optional[UploadFile] = None,
    default_filename: str = DEFAULT_IMAGE_FILENAME
) -> Tuple[bytes, str, Optional[str]]:
    """
    Explicitly checks request Content-Type header to route payload parsing:
    1. Content-Type 'application/json': parses {"url": "..."}, downloads media bytes server-side using httpx.
    2. Content-Type 'multipart/form-data': streams uploaded file bytes in 64KB chunks up to MAX_UPLOAD_SIZE_BYTES.

    Args:
        request (Request): Incoming FastAPI HTTP request object.
        file (Optional[UploadFile]): Uploaded file from multipart form data.
        default_filename (str): Default filename fallback if unprovided.

    Returns:
        Tuple[bytes, str, Optional[str]]: (raw_bytes, filename, target_url)

    Raises:
        HTTPException: HTTP 400 for bad payloads or HTTP 413 if upload exceeds size limit.
    """
    content_type = request.headers.get("content-type", "").lower()

    # Handler 1: JSON body with URL payload
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception as err:
            logger.warning(f"Failed to parse JSON request body: {err}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {err}")

        if not isinstance(body, dict) or "url" not in body or not str(body.get("url", "")).strip():
            raise HTTPException(status_code=400, detail="JSON request payload must contain a non-empty 'url' string field")

        target_url = str(body.get("url")).strip()
        filename = default_filename
        path_basename = os.path.basename(target_url.split("?")[0])
        if path_basename and "." in path_basename:
            filename = path_basename

        try:
            async with httpx.AsyncClient(timeout=HTTP_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
                resp = await client.get(target_url)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch media from URL (HTTP {resp.status_code})")
                if not resp.content:
                    raise HTTPException(status_code=400, detail="Downloaded media content from URL is empty")
                if len(resp.content) > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Downloaded media exceeds max upload size limit ({MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB)"
                    )
                return resp.content, filename, target_url
        except HTTPException:
            raise
        except Exception as fetch_err:
            logger.error(f"Failed to download media from URL '{target_url}': {fetch_err}")
            raise HTTPException(status_code=400, detail=f"Failed to download media from URL: {fetch_err}")

    # Handler 2: Multipart form-data file upload (Streamed in 64KB chunks)
    elif "multipart/form-data" in content_type or file is not None:
        if file:
            filename = file.filename or default_filename
            byte_buf = bytearray()
            total_read = 0

            while chunk := await file.read(CHUNK_SIZE_BYTES):
                total_read += len(chunk)
                if total_read > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Uploaded file exceeds maximum allowed size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)}MB"
                    )
                byte_buf.extend(chunk)

            if not byte_buf:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            return bytes(byte_buf), filename, None

        # Manual form parsing fallback
        try:
            form = await request.form()
            file_obj = form.get("file")
            if file_obj and hasattr(file_obj, "read"):
                fn = getattr(file_obj, "filename", None) or default_filename
                content = await file_obj.read()
                if not content:
                    raise HTTPException(status_code=400, detail="Uploaded file is empty")
                if len(content) > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file exceeds max upload size limit")
                return content, fn, None
        except HTTPException:
            raise
        except Exception as form_err:
            logger.error(f"Failed to parse multipart form data: {form_err}")
            raise HTTPException(status_code=400, detail=f"Failed to parse multipart form data: {form_err}")

        raise HTTPException(status_code=400, detail="No file found in multipart/form-data upload")

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Content-Type '{content_type}'. Must be 'application/json' or 'multipart/form-data'"
        )


@app.get("/")
def root() -> Dict[str, Any]:
    """Health check root endpoint returning API status and active cache count."""
    return {"status": "ok", "service": "TrustLens API Server", "cached_entries": len(_RESPONSE_CACHE)}


@app.post("/analyze")
async def analyze(
    request: Request,
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Main verification pipeline endpoint for image and video analysis.
    Accepts multipart/form-data upload or JSON {"url": "..."}.
    Checks SHA-256 payload cache first for 0ms instant response on repeat files.
    Decodes PIL Image instance once in RAM and offloads CPU tasks to worker thread pool.
    """
    raw_bytes, filename, target_url = await _extract_payload_bytes(request, file=file, default_filename=DEFAULT_IMAGE_FILENAME)

    # Check Response Cache (0ms Instant Hit)
    cache_key = _get_cache_key(raw_bytes, target_url=target_url)
    if cache_key in _RESPONSE_CACHE:
        logger.info(f"[TrustLens Cache] Cache HIT for SHA-256 key {cache_key[:12]}... (0ms instant return)")
        return _RESPONSE_CACHE[cache_key]

    file_ext = os.path.splitext(filename)[1].lower()
    is_video = file_ext in VIDEO_EXTENSIONS or "video" in request.url.path

    # In-memory Image Pre-decoding (Decode once into RAM, share across metadata & ELA)
    pre_decoded_pil: Optional[Image.Image] = None
    if not is_video:
        try:
            pre_decoded_pil = Image.open(io.BytesIO(raw_bytes))
        except Exception:
            pre_decoded_pil = None

    # Signal 0: Filename Pattern Analysis (CPU offloaded)
    filename_analysis = await asyncio.to_thread(analyze_filename, filename)

    # Signal 1: Metadata Analysis (CPU offloaded, pre-decoded image passed)
    metadata_analysis = {
        "has_exif": False,
        "camera_make": None,
        "software_tag": None,
        "creation_date": None,
        "suspicious_flags": ["EXIF metadata is completely missing"]
    }
    try:
        extracted = await asyncio.to_thread(extract_metadata, raw_bytes, filename=filename, pre_decoded_image=pre_decoded_pil)
        if isinstance(extracted, dict):
            metadata_analysis = extracted
    except Exception as meta_err:
        logger.error(f"Metadata signal computation error: {meta_err}")

    # Signal 2: Error Level Analysis (ELA) (CPU offloaded, pre-decoded image passed)
    base_url = str(request.base_url).rstrip("/")
    ela_analysis = {
        "score": 0,
        "heatmap_url": None,
        "original_image_url": None,
        "note": "ELA scan unavailable"
    }
    try:
        ela_input = pre_decoded_pil if pre_decoded_pil is not None else raw_bytes
        ela_res = await asyncio.to_thread(run_ela_analysis, ela_input, 90, 15.0, base_url)
        if isinstance(ela_res, dict):
            ela_analysis = {
                "score": ela_res.get("score", 0),
                "heatmap_url": ela_res.get("heatmap_url", None),
                "original_image_url": ela_res.get("original_image_url", None)
            }
    except Exception as ela_err:
        logger.error(f"ELA signal computation error: {ela_err}")

    # Signal 3: Source Trace Analysis
    source_trace = {"found_matches": False, "note": "reverse search unavailable"}
    try:
        source_res = await asyncio.to_thread(trace_source, url=target_url, image_bytes=raw_bytes, metadata=metadata_analysis)
        if isinstance(source_res, dict):
            source_trace = source_res
    except Exception as source_err:
        logger.error(f"Source trace signal computation error: {source_err}")

    # Signal 4: HuggingFace AI Detector (IO Async / Concurrent Keyframes for Video)
    ai_detector = {"ai_generation_confidence": None, "note": "AI detector unavailable"}
    try:
        if is_video:
            ai_res = await run_video_ai_detector_async(raw_bytes)
        else:
            ai_res = await run_ai_detector_async(raw_bytes)
        if isinstance(ai_res, dict):
            ai_detector = ai_res
    except Exception as ai_err:
        logger.error(f"AI detector signal computation error: {ai_err}")

    # Signal 5: Gemini LLM Synthesis
    if is_video:
        synthesis = {
            "malicious_percentage": 30,
            "explanation": "Video verification completed based on keyframe classification and technical forensic signals.",
            "red_flags": metadata_analysis.get("suspicious_flags", []),
            "limitations": "Some verification signals were unavailable or incomplete."
        }
        try:
            syn_res = await asyncio.to_thread(
                generate_video_synthesis, metadata_analysis, ela_analysis, source_trace, filename_analysis, ai_detector
            )
            if isinstance(syn_res, dict) and "malicious_percentage" in syn_res:
                synthesis = syn_res
        except Exception as gemini_err:
            logger.error(f"Gemini video synthesis computation error: {gemini_err}")

        response_payload = {
            "malicious_percentage": synthesis.get("malicious_percentage", 30),
            "explanation": synthesis.get("explanation", ""),
            "red_flags": synthesis.get("red_flags", []),
            "limitations": synthesis.get("limitations", ""),
            "metadata_analysis": metadata_analysis,
            "filename_analysis": filename_analysis,
            "ela_analysis": ela_analysis,
            "source_trace": source_trace,
            "ai_detector": ai_detector,
            "is_video": True,
            "is_audio": False
        }
    else:
        synthesis = {
            "trust_score": 70,
            "verdict": "Suspicious",
            "explanation": "Media verification completed based on available technical forensic signals.",
            "red_flags": metadata_analysis.get("suspicious_flags", []),
            "limitations": "Some verification signals were unavailable or incomplete."
        }
        try:
            syn_res = await asyncio.to_thread(
                generate_synthesis, metadata_analysis, ela_analysis, source_trace, filename_analysis, ai_detector
            )
            if isinstance(syn_res, dict) and "trust_score" in syn_res:
                synthesis = syn_res
        except Exception as gemini_err:
            logger.error(f"Gemini synthesis computation error: {gemini_err}")

        response_payload = {
            "trust_score": synthesis.get("trust_score", 70),
            "verdict": synthesis.get("verdict", "Suspicious"),
            "explanation": synthesis.get("explanation", ""),
            "red_flags": synthesis.get("red_flags", []),
            "limitations": synthesis.get("limitations", ""),
            "metadata_analysis": metadata_analysis,
            "filename_analysis": filename_analysis,
            "ela_analysis": ela_analysis,
            "source_trace": source_trace,
            "ai_detector": ai_detector,
            "is_video": False,
            "is_audio": False
        }

    # Store in SHA-256 Response Cache
    _cache_set(cache_key, response_payload)
    return response_payload


@app.post("/analyze-video")
async def analyze_video(
    request: Request,
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Dedicated endpoint for video analysis — delegates to main analyze logic with video forced.
    """
    return await analyze(request, file=file)


@app.post("/analyze-audio")
async def analyze_audio(
    request: Request,
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    """
    Accepts audio uploads or URLs.
    Checks SHA-256 cache first for 0ms instant return.
    Streams upload bytes and offloads CPU prosody calculation to thread pool.
    """
    audio_bytes, filename, target_url = await _extract_payload_bytes(request, file=file, default_filename=DEFAULT_AUDIO_FILENAME)

    # Check Response Cache (0ms Instant Hit)
    cache_key = _get_cache_key(audio_bytes, target_url=target_url)
    if cache_key in _RESPONSE_CACHE:
        logger.info(f"[TrustLens Cache] Cache HIT for SHA-256 audio key {cache_key[:12]}...")
        return _RESPONSE_CACHE[cache_key]

    # Signal 0: Filename Pattern Analysis
    filename_analysis = await asyncio.to_thread(analyze_filename, filename)

    # Signal 1: Librosa Audio Spectral Analysis (CPU offloaded to thread pool)
    audio_analysis = {
        "spectral_flatness": 0.0,
        "pitch_variance": 0.0,
        "harmonic_to_noise_ratio": 0.0,
        "suspicious_flags": ["Audio spectral analysis signal unavailable"]
    }
    try:
        extracted_audio = await asyncio.to_thread(analyze_audio_file, audio_bytes, filename=filename)
        if isinstance(extracted_audio, dict):
            audio_analysis = extracted_audio
    except Exception as audio_err:
        logger.error(f"Audio analysis signal computation error: {audio_err}")

    # Signal 2: HuggingFace Deepfake Audio Detector
    deepfake_detector = {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable"}
    try:
        df_res = await asyncio.to_thread(run_deepfake_audio_detector, audio_bytes)
        if isinstance(df_res, dict):
            deepfake_detector = df_res
    except Exception as df_err:
        logger.error(f"Deepfake audio detector signal computation error: {df_err}")

    # Signal 3: Gemini Audio Synthesis
    synthesis = {
        "trust_score": 70,
        "verdict": "Suspicious",
        "explanation": "Audio prosody verification completed based on available acoustic spectral features.",
        "red_flags": audio_analysis.get("suspicious_flags", []),
        "limitations": "Voice biometrics were unavailable; assessment is based on prosody metrics."
    }
    try:
        syn_res = await asyncio.to_thread(
            generate_audio_synthesis, audio_analysis, filename_analysis, deepfake_detector
        )
        if isinstance(syn_res, dict) and "trust_score" in syn_res:
            synthesis = syn_res
    except Exception as gemini_audio_err:
        logger.error(f"Gemini audio synthesis error: {gemini_audio_err}")

    response_payload = {
        "trust_score": synthesis.get("trust_score", 70),
        "verdict": synthesis.get("verdict", "Suspicious"),
        "explanation": synthesis.get("explanation", ""),
        "red_flags": synthesis.get("red_flags", []),
        "limitations": synthesis.get("limitations", ""),
        "audio_analysis": audio_analysis,
        "deepfake_detector": deepfake_detector,
        "filename_analysis": filename_analysis,
        "is_audio": True,
        "metadata_analysis": {
            "has_exif": False,
            "camera_make": None,
            "software_tag": "Audio Prosody Filter",
            "creation_date": None,
            "suspicious_flags": audio_analysis.get("suspicious_flags", [])
        },
        "ela_analysis": {
            "score": 0,
            "heatmap_url": None,
            "original_image_url": None
        },
        "source_trace": {
            "found_matches": False,
            "note": "audio reverse search unavailable"
        }
    }

    _cache_set(cache_key, response_payload)
    return response_payload
