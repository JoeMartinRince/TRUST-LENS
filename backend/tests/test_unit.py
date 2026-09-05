import io
import piexif
from PIL import Image
from backend.services.metadata_service import extract_metadata
from backend.services.filename_service import analyze_filename
from backend.services.ela_service import run_ela_analysis


def test_metadata_extraction_with_and_without_exif():
    """
    Test 3a: Metadata extraction.
    - Image with EXIF returns has_exif: True and extracted fields.
    - Image without EXIF returns has_exif: False and flag for missing EXIF.
    """
    # 1. Plain image without EXIF metadata
    plain_buf = io.BytesIO()
    plain_img = Image.new("RGB", (50, 50), color="blue")
    plain_img.save(plain_buf, format="JPEG")
    plain_bytes = plain_buf.getvalue()

    meta_no_exif = extract_metadata(plain_bytes, filename="plain.jpg")
    assert meta_no_exif["has_exif"] is False
    assert any("missing" in flag.lower() for flag in meta_no_exif["suspicious_flags"])

    # 2. Image WITH EXIF metadata using piexif dump
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: b"Canon",
            piexif.ImageIFD.Model: b"Canon EOS R5",
            piexif.ImageIFD.Software: b"Adobe Photoshop 2024",
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: b"2026:09:05 12:00:00",
        },
        "GPS": {}
    }
    exif_bytes = piexif.dump(exif_dict)

    exif_buf = io.BytesIO()
    exif_img = Image.new("RGB", (50, 50), color="green")
    exif_img.save(exif_buf, format="JPEG", exif=exif_bytes)
    with_exif_bytes = exif_buf.getvalue()

    meta_with_exif = extract_metadata(with_exif_bytes, filename="photo.jpg")
    assert meta_with_exif["has_exif"] is True
    assert meta_with_exif["camera_make"] is not None
    assert "Canon" in meta_with_exif["camera_make"]
    assert meta_with_exif["creation_date"] == "2026:09:05 12:00:00"


def test_filename_pattern_classifier():
    """
    Test 3b: Filename pattern classifier for AI-generator, camera-native, generic-download, and unrecognized patterns.
    """
    # AI Generator patterns
    ai_names = ["sora_generated_video.mp4", "midjourney_output_123.jpg", "generated-image(1).png", "sdxl_render.webp"]
    for name in ai_names:
        result = analyze_filename(name)
        assert result["pattern_type"] == "ai_generator", f"Expected 'ai_generator' for '{name}', got '{result['pattern_type']}'"

    # Camera Native patterns
    camera_names = ["IMG_9842.JPG", "DSC_0001.NEF", "VID_20260905_120000.mp4", "MVI_4821.mov"]
    for name in camera_names:
        result = analyze_filename(name)
        assert result["pattern_type"] == "camera_native", f"Expected 'camera_native' for '{name}', got '{result['pattern_type']}'"

    # Generic Download patterns
    generic_names = ["download.mp4", "image.png", "video.mov", "untitled.jpg"]
    for name in generic_names:
        result = analyze_filename(name)
        assert result["pattern_type"] == "generic_download", f"Expected 'generic_download' for '{name}', got '{result['pattern_type']}'"

    # Unrecognized pattern
    custom_name = "financial_report_q3_final_v2.pdf"
    result_custom = analyze_filename(custom_name)
    assert result_custom["pattern_type"] == "unrecognized"


def test_ela_analysis_rgba_png_does_not_crash():
    """
    Test 3c: ELA function handles RGBA PNG input gracefully without crashing.
    """
    rgba_buf = io.BytesIO()
    rgba_img = Image.new("RGBA", (60, 60), color=(255, 0, 0, 128))
    rgba_img.save(rgba_buf, format="PNG")
    rgba_bytes = rgba_buf.getvalue()

    ela_result = run_ela_analysis(rgba_bytes)

    assert isinstance(ela_result, dict)
    assert "score" in ela_result
    assert isinstance(ela_result["score"], (int, float))
    assert ela_result["heatmap_url"] is not None
