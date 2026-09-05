import os
import uuid
import logging
import tempfile
import asyncio
import httpx
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

_CV2_AVAILABLE = False
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass

from backend.services.ai_detector_service import run_ai_detector_async, run_ai_detector

TARGET_KEYFRAMES = 5
AI_FLAG_THRESHOLD = 70.0


def _extract_keyframes(video_bytes: bytes) -> Tuple[List[bytes], List[str]]:
    """
    Decode video bytes with OpenCV, sample exactly TARGET_KEYFRAMES evenly
    spaced frames, save each keyframe to a unique file path on disk, and return
    both the frame bytes list and keyframe file paths list.
    """
    if not _CV2_AVAILABLE:
        logger.warning("Video keyframe extraction unavailable: cv2 module not installed")
        return [], []

    tmp_path = None
    cap = None
    try:
        suffix = ".mp4"
        if video_bytes[:4] == b"RIFF":
            suffix = ".avi"
        elif video_bytes[:4] in (b"\x1a\x45\xdf\xa3",):
            suffix = ".mkv"

        unique_video_id = uuid.uuid4().hex[:8]
        tmp_dir = tempfile.mkdtemp(prefix=f"trustlens_vid_{unique_video_id}_")
        tmp_path = os.path.join(tmp_dir, f"source_video{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            logger.warning(f"VideoCapture failed to open temporary video file at {tmp_path}")
            return [], []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("VideoCapture reported invalid total frame count <= 0")
            return [], []

        duration_sec = total_frames / fps
        logger.info(f"Video loaded from {tmp_path}: total_frames={total_frames}, fps={fps:.2f}, duration={duration_sec:.2f}s")
        print(f"[Video Extractor] Video source file path: {tmp_path}")

        if duration_sec <= 0.5 or TARGET_KEYFRAMES == 1:
            timestamps = [0.0]
        else:
            segment = duration_sec / TARGET_KEYFRAMES
            timestamps = [segment * i + segment / 2 for i in range(TARGET_KEYFRAMES)]

        frames: List[bytes] = []
        frame_paths: List[str] = []

        for idx, ts in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(f"Keyframe {idx+1} read failed at timestamp {ts:.2f}s")
                continue
            success, buf_out = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if success:
                frame_b = bytes(buf_out)
                kf_filename = f"keyframe_{idx+1}_ts{ts:.2f}s_{uuid.uuid4().hex[:6]}.jpg"
                kf_path = os.path.join(tmp_dir, kf_filename)
                with open(kf_path, "wb") as kf_file:
                    kf_file.write(frame_b)

                frames.append(frame_b)
                frame_paths.append(kf_path)
                logger.info(f"Saved keyframe {idx+1}/{len(timestamps)} to disk: {kf_path} ({len(frame_b)} bytes)")
                print(f"[Video Extractor] Keyframe {idx+1} file path on disk: {kf_path}")

        return frames, frame_paths

    except Exception as e:
        msg = f"Video keyframe extraction error: {e}"
        logger.error(msg)
        print(msg)
        return [], []
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


async def run_video_ai_detector_async(video_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts keyframes from a video and runs Hugging Face AI detector calls CONCURRENTLY
    across all keyframes using asyncio.gather and httpx connection pooling.
    This eliminates sequential latency (e.g. 5x sequential delays -> 1x parallel delay).
    """
    if not _CV2_AVAILABLE or not video_bytes:
        return {
            "ai_generation_confidence": None,
            "per_frame_confidence": [],
            "keyframe_paths": [],
            "frames_analyzed": 0,
            "frames_flagged_ai": 0,
            "note": "Video AI detector unavailable"
        }

    # Offload CPU OpenCV keyframe extraction to worker thread
    frames, keyframe_paths = await asyncio.to_thread(_extract_keyframes, video_bytes)
    if not frames:
        return {
            "ai_generation_confidence": None,
            "per_frame_confidence": [],
            "keyframe_paths": [],
            "frames_analyzed": 0,
            "frames_flagged_ai": 0,
            "note": "Video AI detector unavailable (keyframe extraction failed)"
        }

    print(f"\n================ VIDEO CONCURRENT PER-FRAME DETECTOR RUN ================")
    print(f"Total keyframes extracted: {len(frames)}. Executing parallel asyncio requests...")

    async with httpx.AsyncClient(timeout=12.0) as async_client:
        tasks = [
            run_ai_detector_async(frame_bytes, log_prefix=f" [frame {i+1}/{len(frames)}]", client=async_client)
            for i, frame_bytes in enumerate(frames)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    confidences: List[float] = []
    per_frame_results: List[Dict[str, Any]] = []
    model_used: str = ""

    for idx, (result, path) in enumerate(zip(results, keyframe_paths)):
        frame_status = {"frame": idx + 1, "path": path, "status": "failed", "confidence": None, "error": None}

        if isinstance(result, Exception):
            exc_msg = f"{type(result).__name__}: {result}"
            frame_status["error"] = exc_msg
            print(f"  Frame {idx+1} Result: EXCEPTION | error = {exc_msg}")
        elif isinstance(result, dict):
            conf = result.get("ai_generation_confidence")
            if conf is not None:
                conf_val = round(float(conf), 2)
                confidences.append(conf_val)
                if not model_used:
                    model_used = result.get("model_used", "")
                frame_status["status"] = "success"
                frame_status["confidence"] = conf_val
                frame_status["model_used"] = result.get("model_used")
                print(f"  Frame {idx+1} Result: SUCCESS | raw ai_generation_confidence = {conf_val}%")
            else:
                note_reason = result.get("note", "unparseable/unmatched labels")
                frame_status["error"] = note_reason
                print(f"  Frame {idx+1} Result: FAILED/NULL | note = {note_reason}")

        per_frame_results.append(frame_status)

    print("=========================================================================\n")

    if not confidences:
        return {
            "ai_generation_confidence": None,
            "per_frame_confidence": [],
            "keyframe_paths": keyframe_paths,
            "per_frame_details": per_frame_results,
            "frames_analyzed": len(frames),
            "frames_flagged_ai": 0,
            "note": "Video AI detector unavailable (all frame detections failed or returned null)"
        }

    max_conf = round(max(confidences), 2)
    frames_flagged = sum(1 for c in confidences if c > AI_FLAG_THRESHOLD)
    successful = len(confidences)
    total = len(frames)

    note_str = f"Analyzed {successful}/{total} keyframes concurrently; max confidence: {max_conf:.1f}%; {frames_flagged} frame(s) above {AI_FLAG_THRESHOLD:.0f}% threshold"

    return {
        "ai_generation_confidence": max_conf,
        "per_frame_confidence": confidences,
        "keyframe_paths": keyframe_paths,
        "per_frame_details": per_frame_results,
        "frames_analyzed": total,
        "frames_flagged_ai": frames_flagged,
        "model_used": model_used,
        "note": note_str
    }


def run_video_ai_detector(video_bytes: bytes) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_video_ai_detector_async.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(run_video_ai_detector_async(video_bytes))
        else:
            return loop.run_until_complete(run_video_ai_detector_async(video_bytes))
    except Exception:
        return asyncio.run(run_video_ai_detector_async(video_bytes))
