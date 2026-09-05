import io
from PIL import Image

def create_dummy_image_bytes(format: str = "JPEG", size: tuple = (20, 20), color: str = "red") -> bytes:
    """Helper utility to generate dummy in-memory image bytes for API testing."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format)
    return buf.getvalue()


def test_analyze_image_endpoint_success(client):
    """
    Test 1: /analyze endpoint with valid image upload returns 200 and expected schema keys:
    trust_score, verdict, explanation, red_flags, limitations, metadata_analysis, ela_analysis, source_trace.
    """
    img_bytes = create_dummy_image_bytes(format="JPEG")
    files = {"file": ("test_sample.jpg", img_bytes, "image/jpeg")}

    response = client.post("/analyze", files=files)

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()

    expected_keys = [
        "trust_score",
        "verdict",
        "explanation",
        "red_flags",
        "limitations",
        "metadata_analysis",
        "ela_analysis",
        "source_trace",
    ]

    for key in expected_keys:
        assert key in data, f"Expected key '{key}' missing from /analyze response"

    assert isinstance(data["trust_score"], (int, float))
    assert isinstance(data["verdict"], str)
    assert isinstance(data["red_flags"], list)
    assert isinstance(data["metadata_analysis"], dict)
    assert isinstance(data["ela_analysis"], dict)
    assert isinstance(data["source_trace"], dict)


def test_analyze_video_endpoint_success(client):
    """
    Test 2: /analyze-video endpoint with valid video file returns 200 and expected video schema keys:
    malicious_percentage, red_flags, explanation, limitations.
    """
    dummy_video_bytes = b"ftypisom" + b"\x00" * 256
    files = {"file": ("test_video.mp4", dummy_video_bytes, "video/mp4")}

    response = client.post("/analyze-video", files=files)

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    data = response.json()

    expected_video_keys = [
        "malicious_percentage",
        "red_flags",
        "explanation",
        "limitations",
    ]

    for key in expected_video_keys:
        assert key in data, f"Expected key '{key}' missing from /analyze-video response"

    assert isinstance(data["malicious_percentage"], (int, float))
    assert isinstance(data["red_flags"], list)
    assert isinstance(data["explanation"], str)
    assert isinstance(data["limitations"], str)


def test_malformed_upload_returns_error(client):
    """
    Test 4: Malformed/invalid file uploads or missing payloads return a proper 400 error response instead of crashing.
    """
    headers = {"Content-Type": "text/plain"}
    response = client.post("/analyze", content=b"invalid raw data", headers=headers)
    assert response.status_code == 400
    assert "detail" in response.json()

    response_json = client.post("/analyze", json={})
    assert response_json.status_code == 400
    assert "detail" in response_json.json()

    empty_files = {"file": ("empty.jpg", b"", "image/jpeg")}
    response_empty = client.post("/analyze", files=empty_files)
    assert response_empty.status_code == 400
    assert "detail" in response_empty.json()


def test_response_caching_returns_instant_cached_data(client):
    """
    Test 5: Submitting identical file bytes returns cached response payload instantly on repeated calls.
    """
    img_bytes = create_dummy_image_bytes(format="JPEG", size=(30, 30), color="blue")
    files1 = {"file": ("cache_test.jpg", img_bytes, "image/jpeg")}
    files2 = {"file": ("cache_test.jpg", img_bytes, "image/jpeg")}

    resp1 = client.post("/analyze", files=files1)
    assert resp1.status_code == 200

    resp2 = client.post("/analyze", files=files2)
    assert resp2.status_code == 200

    assert resp1.json() == resp2.json()
