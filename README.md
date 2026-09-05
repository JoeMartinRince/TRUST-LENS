# TrustLens — Technical Media Verification & Multi-Modal Forensics Engine

**TrustLens** is a modern, high-performance media forensics platform engineered to verify the authenticity of images, videos, and audio streams. Combining technical error-level analysis (ELA), acoustic prosody spectrum scanning, Hugging Face neural classifier pipelines, filename provenance analysis, and Google Gemini AI synthesis with automatic fail-safe API key fallback, TrustLens provides a complete technical analysis of digital media.

---

## ⚡ Key Features

- **Multi-Modal Verification Engine**:
  - **Image Forensics (`/analyze`)**: Error-Level Analysis (ELA) compression heatmap generation, EXIF metadata extraction, Hugging Face AI image detector classification, reverse web source index tracing, and multi-signal Gemini synthesis.
  - **Video Forensics (`/analyze-video`)**: Keyframe extraction, per-frame neural AI detector classification, video-specific filename pattern matching (`camera_native`, `ai_generator`, `generic_download`), keyframe ELA analysis, and a specialized `malicious_percentage` risk score (0–100%).
  - **Audio Forensics (`/analyze-audio`)**: Spectral flatness, pitch variance, harmonic-to-noise ratio prosody scanning, and deepfake voice clone classification.

- **Fail-Safe API Key Architecture**:
  - Built-in multi-key failover fallback mechanism. If the primary Gemini API key hits rate limits, HTTP 429 quota limits, or network errors, the engine seamlessly retries using the fail-safe fallback key without interrupting user analysis.

- **Technical Editorial Wireframe UI**:
  - Sleek dark technical aesthetic (`#0D0D0D`) featuring fine dot-grid textures, concentric sonar/radar rings, monospace metadata tags, and a crisp monochrome silver color scheme.
  - Full-viewport layout with a multi-step sequential loading animation checklist.

---

## 🛠️ Architecture & Tech Stack

### Frontend
- **Framework**: Vite, React 18, TanStack Router / Start
- **Styling**: Tailwind CSS v4, Lucide Icons, Custom CSS animation system
- **Design System**: Technical wireframe, crisp monochrome silver accents, custom sonar ring SVG overlays

### Backend
- **Framework**: Python 3.12, FastAPI, Uvicorn
- **Image Processing & ELA**: Pillow (PIL), NumPy, OpenCV (`cv2`)
- **Neural Classifiers**: Hugging Face Inference API (`organika/sdxl-detector` / `umbrella-ai`)
- **LLM Forensic Synthesis**: Google Gemini API (`google-genai` / `google-generativeai`) with automatic multi-key failover fallback

---

## 🚀 Getting Started

### Prerequisites
- **Node.js** (v18+) or **Bun**
- **Python** (3.10+)

---

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure Environment Variables
cp .env.example .env
```

Edit `backend/.env` with your API keys:

```env
# Primary Gemini API Key
GEMINI_API_KEY=AIzaSyYourPrimaryKeyHere...

# Fail-Safe Fallback Gemini API Key (Automatically used if Primary hits rate limits/quota)
GEMINI_API_KEY_FALLBACK=AIzaSyYourFallbackKeyHere...

# Optional: Hugging Face API Token (For AI detector model classification)
HF_API_TOKEN=hf_YourHuggingFaceTokenHere...
```

Start the FastAPI server:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The API will run at `http://127.0.0.1:8000`.

---

### 2. Frontend Setup

From the project root:

```bash
# Install dependencies
bun install   # or npm install

# Start local development server
bun run dev   # or npm run dev
```

Open `http://localhost:8080` in your browser.

---

## 📡 API Reference

### `POST /analyze`
Analyzes image media via file upload (`multipart/form-data`) or URL (`application/json`).

**Response Schema**:
```json
{
  "trust_score": 85,
  "verdict": "Likely Authentic",
  "explanation": "High authenticity signals detected...",
  "red_flags": [],
  "limitations": "TrustLens uses AI-assisted signals and public metadata...",
  "metadata_analysis": { ... },
  "ela_analysis": { "score": 12, "heatmap_url": "/static/ela_xxx.jpg" },
  "source_trace": { ... },
  "ai_detector": { "ai_generation_confidence": 12.5 }
}
```

---

### `POST /analyze-video`
Analyzes video media across keyframes, returning a `malicious_percentage` risk rating.

**Response Schema**:
```json
{
  "malicious_percentage": 15,
  "red_flags": ["Camera-native filename pattern identified"],
  "explanation": "Keyframes and metadata indicate authentic device capture.",
  "limitations": "Keyframes were sampled at discrete intervals.",
  "filename_analysis": {
    "filename": "VID_20260905_120000.mp4",
    "pattern_type": "camera_native",
    "note": "Camera-native filename pattern (VID_ / IMG_ / MVI_)"
  },
  "ai_detector": {
    "ai_generation_confidence": 18.2,
    "per_frame_confidence": [14.1, 18.2, 12.0],
    "frames_analyzed": 3,
    "frames_flagged_ai": 0
  }
}
```

---

### `POST /analyze-audio`
Analyzes audio media for prosody anomalies and synthetic voice clones.

---

## 🛡️ License

Distributed under the MIT License.
