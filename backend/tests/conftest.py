import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_external_apis():
    """
    Global autouse fixture to mock external API calls (Gemini API & Hugging Face detectors)
    so all tests run fast, offline, and without relying on external API keys.
    """
    mock_img_syn = {
        "trust_score": 85,
        "verdict": "Likely Authentic",
        "explanation": "Media verification completed based on available technical forensic signals.",
        "red_flags": [],
        "limitations": "Mocked test limitations."
    }

    mock_vid_syn = {
        "malicious_percentage": 15,
        "explanation": "Video verification completed based on keyframe classification.",
        "red_flags": [],
        "limitations": "Mocked video test limitations."
    }

    mock_ai_det = {
        "ai_generation_confidence": 12.5,
        "model_used": "mock-hf-detector"
    }

    mock_vid_ai_det = {
        "ai_generation_confidence": 15.0,
        "per_frame_confidence": [10.0, 15.0],
        "frames_analyzed": 2,
        "frames_flagged_ai": 0,
        "model_used": "mock-hf-video-detector"
    }

    mock_source = {
        "found_matches": False,
        "note": "Mocked reverse search index"
    }

    with patch("backend.main.generate_synthesis", return_value=mock_img_syn), \
         patch("backend.main.generate_video_synthesis", return_value=mock_vid_syn), \
         patch("backend.main.run_ai_detector", return_value=mock_ai_det), \
         patch("backend.main.run_video_ai_detector", return_value=mock_vid_ai_det), \
         patch("backend.main.trace_source", return_value=mock_source):
        yield
