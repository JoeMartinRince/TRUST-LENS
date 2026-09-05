import io
import piexif
from PIL import Image, ExifTags
from typing import Dict, Any, List, Optional

AI_GENERATOR_KEYWORDS = [
    "midjourney", "dalle", "dall-e", "stable diffusion", "firefly", 
    "gemini", "ideogram", "leonardo", "playground", "photoshop", 
    "gimp", "automatic1111", "comfyui"
]

def extract_metadata(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Extracts real EXIF metadata from raw image bytes using Pillow and piexif.
    Returns:
      {
        "has_exif": bool,
        "camera_make": str or None,
        "software_tag": str or None,
        "creation_date": str or None,
        "suspicious_flags": List[str]
      }
    Never crashes on images without EXIF — returns has_exif: False safely.
    """
    has_exif = False
    camera_make: Optional[str] = None
    software_tag: Optional[str] = None
    creation_date: Optional[str] = None
    suspicious_flags: List[str] = []

    if not image_bytes:
        return {
            "has_exif": False,
            "camera_make": None,
            "software_tag": None,
            "creation_date": None,
            "suspicious_flags": ["EXIF metadata is completely missing"]
        }

    # 1. Attempt piexif extraction
    exif_dict = {}
    try:
        exif_dict = piexif.load(image_bytes)
        # Check if 0th, Exif, or GPS IFD contains any non-empty entries
        if exif_dict and any(exif_dict.get(k) for k in ("0th", "Exif", "GPS")):
            has_exif = True
    except Exception:
        exif_dict = {}

    # 2. Extract specific EXIF tags from piexif if present
    if has_exif:
        try:
            # Camera Make & Model
            make_bytes = exif_dict.get("0th", {}).get(piexif.ImageIFD.Make, b"")
            model_bytes = exif_dict.get("0th", {}).get(piexif.ImageIFD.Model, b"")
            
            make_str = make_bytes.decode("utf-8", errors="ignore").strip("\x00 ").strip() if isinstance(make_bytes, bytes) else str(make_bytes).strip()
            model_str = model_bytes.decode("utf-8", errors="ignore").strip("\x00 ").strip() if isinstance(model_bytes, bytes) else str(model_bytes).strip()

            if make_str and model_str:
                if make_str.lower() in model_str.lower():
                    camera_make = model_str
                else:
                    camera_make = f"{make_str} {model_str}"
            elif model_str:
                camera_make = model_str
            elif make_str:
                camera_make = make_str

            # Software Tag
            soft_bytes = exif_dict.get("0th", {}).get(piexif.ImageIFD.Software, b"")
            if soft_bytes:
                software_tag = soft_bytes.decode("utf-8", errors="ignore").strip("\x00 ").strip() if isinstance(soft_bytes, bytes) else str(soft_bytes).strip()

            # Creation Date
            date_bytes = (
                exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"") or
                exif_dict.get("0th", {}).get(piexif.ImageIFD.DateTime, b"") or
                exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeDigitized, b"")
            )
            if date_bytes:
                creation_date = date_bytes.decode("utf-8", errors="ignore").strip("\x00 ").strip() if isinstance(date_bytes, bytes) else str(date_bytes).strip()
        except Exception:
            pass

    # 3. Fallback to Pillow EXIF if piexif didn't populate tags
    if not camera_make or not creation_date or not software_tag:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            raw_exif = img._getexif()
            if raw_exif:
                has_exif = True
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    if tag_name == "Make" and not camera_make:
                        make_val = str(value).strip()
                        camera_make = make_val
                    elif tag_name == "Model" and not camera_make:
                        camera_make = str(value).strip()
                    elif tag_name == "Software" and not software_tag:
                        software_tag = str(value).strip()
                    elif tag_name in ("DateTimeOriginal", "DateTime") and not creation_date:
                        creation_date = str(value).strip()
        except Exception:
            pass

    # Clean empty string placeholders
    camera_make = camera_make if camera_make and camera_make.strip() else None
    software_tag = software_tag if software_tag and software_tag.strip() else None
    creation_date = creation_date if creation_date and creation_date.strip() else None

    # 4. Check for AI generator / editing software keywords
    detected_software_ai = None
    search_text = (software_tag or "").lower()
    
    # Also scan first 4KB of raw bytes for AI model strings if software_tag is missing
    header_sample = image_bytes[:4096].decode("latin-1", errors="ignore").lower()
    
    for kw in AI_GENERATOR_KEYWORDS:
        if kw in search_text or kw in header_sample:
            detected_software_ai = kw
            if not software_tag:
                software_tag = kw.title()
            break

    # 5. Populate suspicious_flags rules
    if not has_exif:
        suspicious_flags.append("EXIF metadata is completely missing")

    if detected_software_ai:
        suspicious_flags.append(f"Software tag indicates AI generation or digital editing ({detected_software_ai})")

    if has_exif and camera_make and not creation_date:
        suspicious_flags.append("Camera profile present but timestamp stripped")

    if has_exif and not camera_make and not software_tag and not creation_date:
        suspicious_flags.append("Metadata structure appears artificially stripped or uniform")

    return {
        "has_exif": has_exif,
        "camera_make": camera_make,
        "software_tag": software_tag,
        "creation_date": creation_date,
        "suspicious_flags": suspicious_flags
    }
