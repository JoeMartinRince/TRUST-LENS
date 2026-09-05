import os
import io
import uuid
import logging
import traceback
import numpy as np
from typing import Union, Dict, Any
from PIL import Image, ImageChops, ImageEnhance
import cv2

logger = logging.getLogger(__name__)

# ── ELA Service Constants ───────────────────────────────────────────────────
DEFAULT_JPEG_QUALITY: int = 90
DEFAULT_AMPLIFY_FACTOR: float = 15.0
MAX_IMAGE_DIMENSION: int = 1024

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "heatmaps")
os.makedirs(STATIC_DIR, exist_ok=True)


def run_ela_analysis(
    image_input: Union[bytes, Image.Image], 
    quality: int = DEFAULT_JPEG_QUALITY, 
    amplify: float = DEFAULT_AMPLIFY_FACTOR,
    base_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Performs Error Level Analysis (ELA) on an image:
    1. Accepts either raw bytes OR a pre-decoded in-memory PIL Image object to avoid redundant image decoding.
    2. Re-saves uploaded image as JPEG at quality=90 in memory.
    3. Computes pixel-wise difference against original with PIL & numpy.
    4. Amplifies difference (~15-20x) into a visible colorized JET heatmap.
    5. Saves heatmap PNG to static/heatmaps/ folder.
    6. Computes ela_score (0-100) based on error variance and intensity.

    Args:
        image_input (Union[bytes, Image.Image]): Binary bytes or PIL Image instance.
        quality (int): JPEG resave compression quality level (default 90).
        amplify (float): Error brightness amplification factor (default 15.0).
        base_url (str): Base server URL for constructing static image links.

    Returns:
        Dict[str, Any]: {
            "score": int (0-100),
            "heatmap_url": str|None,
            "original_image_url": str|None,
            "explanation": str
        }
    """
    img_format = "Unknown"
    img_mode = "Unknown"
    img_dimensions = "Unknown"

    try:
        if isinstance(image_input, Image.Image):
            original = image_input.convert("RGB")
            img_format = str(getattr(image_input, "format", "PIL.Image"))
        elif isinstance(image_input, bytes):
            if not image_input:
                return {
                    "score": 0,
                    "heatmap_url": None,
                    "original_image_url": None,
                    "explanation": "ELA computation failed: empty image bytes"
                }
            with Image.open(io.BytesIO(image_input)) as raw_img:
                img_format = str(raw_img.format)
                img_mode = str(raw_img.mode)
                img_dimensions = f"{raw_img.size[0]}x{raw_img.size[1]}"
                original = raw_img.convert("RGB")
        else:
            raise ValueError(f"Invalid image_input type: {type(image_input)}")

        img_mode = str(original.mode)
        img_dimensions = f"{original.size[0]}x{original.size[1]}"
        
        # Max dimension constraint for fast processing
        if max(original.size) > MAX_IMAGE_DIMENSION:
            original.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)

        # 1. Save compressed version to in-memory buffer at quality=90
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        # 2. Compute pixel-wise absolute difference between original and quality=90 JPEG
        diff_im = ImageChops.difference(original, compressed)

        # 3. Amplify (~15-20x) into a visible difference map
        extrema = diff_im.getextrema()
        max_diff = max([ex[1] for ex in extrema]) or 1
        scale_factor = 255.0 / max_diff if max_diff < (255 / amplify) else amplify
        
        amplified_diff = ImageEnhance.Brightness(diff_im).enhance(scale_factor)

        # 4. Generate visible colorized heatmap PNG using OpenCV JET colormap
        diff_np = np.array(amplified_diff)
        gray = cv2.cvtColor(diff_np, cv2.COLOR_RGB2GRAY)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap_pil = Image.fromarray(heatmap_rgb)

        # 5. Save heatmap PNG to static folder
        filename = f"ela_{uuid.uuid4().hex[:10]}.png"
        filepath = os.path.join(STATIC_DIR, filename)
        heatmap_pil.save(filepath, "PNG")

        heatmap_url = f"{base_url}/static/heatmaps/{filename}"

        # Save original preview image to static as well
        orig_filename = f"orig_{uuid.uuid4().hex[:10]}.jpg"
        orig_filepath = os.path.join(STATIC_DIR, orig_filename)
        original.save(orig_filepath, "JPEG", quality=85)
        orig_url = f"{base_url}/static/heatmaps/{orig_filename}"

        # 6. Compute ela_score (0-100) based on error intensity & variance
        mean_err = float(np.mean(gray))
        std_err = float(np.std(gray))
        peak_err = float(np.percentile(gray, 95))

        raw_score = int(min(100, max(0, (mean_err * 0.7 + std_err * 1.6 + peak_err * 0.3))))

        return {
            "score": raw_score,
            "heatmap_url": heatmap_url,
            "original_image_url": orig_url,
            "explanation": f"Error Level Analysis computed with JPEG quality={quality}. Error noise variance: {std_err:.1f}."
        }

    except Exception as e:
        err_msg = (
            f"ELA processing exception traceback:\n"
            f"Image Format: {img_format}\n"
            f"Image Mode: {img_mode}\n"
            f"Image Dimensions: {img_dimensions}\n"
            f"Exception Type: {type(e).__name__}: {e}\n"
            f"Full Traceback:\n{traceback.format_exc()}"
        )
        logger.error(err_msg)

        return {
            "score": 0,
            "heatmap_url": None,
            "original_image_url": None,
            "explanation": f"ELA computation failed: {str(e)}"
        }
