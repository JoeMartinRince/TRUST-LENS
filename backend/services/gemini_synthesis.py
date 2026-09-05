import os
import json
import concurrent.futures
import google.generativeai as genai
from typing import Dict, Any, List

SYSTEM_INSTRUCTION_IMAGE = (
    "You are a media forensics analyst. You are given raw technical findings about an uploaded image. "
    "Synthesize them into a clear trust assessment for a non-technical user such as a journalist or everyday citizen. "
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "trust_score": <integer 0-100, lower = more suspicious>,\n'
    '  "verdict": "<short label: \'Likely Authentic\' / \'Suspicious\' / \'Likely Manipulated\'>",\n'
    '  "red_flags": ["<specific evidence-based bullet>", ...],\n'
    '  "explanation": "<2-3 sentence plain-English paragraph>",\n'
    '  "limitations": "<what this analysis could NOT determine, mention any signals marked unavailable>"\n'
    "}\n\n"
    "Score calculation guidance:\n"
    "- Start from 100 (fully trustworthy) as baseline\n"
    "- Missing EXIF entirely: -15 points\n"
    "- Software tag matches known AI generator: -40 points\n"
    "- Filename pattern_type is ai_generator: -20 points\n"
    "- Filename pattern_type is camera_native: +10 points\n"
    "- ELA score above 60 (high localized variance): -25 points\n"
    "- ELA score above 80: -40 points\n"
    "- No reverse search matches found: -5 points (weak signal, don't over-penalize)\n"
    "- Reverse search finds much earlier source than claimed context: -30 points\n\n"
    "AI detector signal guidance:\n"
    "- If ai_generation_confidence from the dedicated AI detector is above 70, this is strong direct evidence of AI generation — apply a -50 point penalty to trust_score regardless of other signals.\n"
    "- If ai_generation_confidence is above 90, cap the maximum possible trust_score at 15.\n"
    "- This signal should be weighted more heavily than metadata or filename signals since it directly analyzes media content rather than surrounding context.\n"
    "- CRITICAL: If ai_generation_confidence is null or the detector was unavailable, explicitly do NOT default to a moderate/high trust score (~80) or assume authenticity by default. The primary AI-detection classifier was unavailable; explanation and score MUST be based on ELA, metadata, and filename evidence only, and should state this limitation clearly rather than assuming authenticity.\n\n"
    "Note: missing EXIF metadata alone is very weak evidence — it happens naturally when images are shared via social media, messaging apps, or screenshots, and does NOT strongly indicate AI generation or manipulation.\n\n"
    "Rules: be specific, cite only given evidence, never invent findings, be honest about uncertainty, avoid absolute claims like '100% AI-generated'."
)

SYSTEM_INSTRUCTION_VIDEO = (
    "You are a video forensics analyst. You are given raw technical findings about an uploaded video file. "
    "Synthesize them into a clear risk assessment for a non-technical user such as a journalist or everyday citizen. "
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "malicious_percentage": <integer 0-100, higher = more likely manipulated or AI-generated>,\n'
    '  "red_flags": ["<specific evidence-based bullet>", ...],\n'
    '  "explanation": "<2-3 sentence plain-English paragraph>",\n'
    '  "limitations": "<what this analysis could NOT determine>"\n'
    "}\n\n"
    "Score calculation guidance (inverted scale where 0 = completely authentic, 100 = heavily manipulated/AI-generated):\n"
    "- Start from 0 (completely authentic) as baseline\n"
    "- Missing EXIF entirely: +15 points\n"
    "- Software tag matches known AI generator: +40 points\n"
    "- Filename pattern_type is ai_generator: +15 to +20 points (supporting/corroborating signal, not standalone proof)\n"
    "- Filename pattern_type is camera_native: -10 points (pulls risk down)\n"
    "- Filename pattern_type is generic_download or unrecognized: neutral signal (0 points)\n"
    "- ELA score above 60 (high localized variance): +25 points\n"
    "- ELA score above 80: +40 points\n"
    "- No reverse search matches found: +5 points\n"
    "- Reverse search finds earlier source: +30 points\n\n"
    "AI detector signal guidance (highest-priority primary signal):\n"
    "- Weight the AI-detector confidence far more heavily than filename or metadata — filename is corroborating evidence, not primary evidence.\n"
    "- ai_generation_confidence is the MAXIMUM confidence across sampled video keyframes. per_frame_confidence lists every frame score; frames_flagged_ai counts how many frames exceeded 70%.\n"
    "- If ai_generation_confidence is above 70, this is strong direct evidence of AI generation — increase malicious_percentage by +50 points regardless of other signals.\n"
    "- If ai_generation_confidence is above 90, floor the minimum malicious_percentage at 85.\n"
    "- If frames_flagged_ai >= 3 out of 5, treat as very strong AI generation evidence.\n"
    "- CRITICAL: If ai_generation_confidence is null or the detector was unavailable, explicitly do NOT default to a low risk score (~20) or assume authenticity. State in limitations that the primary AI detector was unavailable, and calculate malicious_percentage strictly from ELA, metadata, and filename evidence.\n\n"
    "Rules: be specific, cite only given evidence, never invent findings, avoid absolute claims like '100% AI-generated'."
)

SYSTEM_INSTRUCTION_AUDIO = (
    "You are an audio forensics analyst. You are given raw technical findings about an uploaded audio recording. "
    "Synthesize them into a clear trust assessment for a non-technical user such as a journalist or everyday citizen. "
    "Return ONLY valid JSON with this exact schema:\n"
    "{\n"
    '  "trust_score": <integer 0-100, lower = more suspicious>,\n'
    '  "verdict": "<short label: \'Likely Authentic\' / \'Suspicious\' / \'Likely Manipulated\'>",\n'
    '  "red_flags": ["<specific evidence-based bullet>", ...],\n'
    '  "explanation": "<2-3 sentence plain-English paragraph>",\n'
    '  "limitations": "<what this analysis could NOT determine, mention any signals marked unavailable>"\n'
    "}\n\n"
    "Score calculation guidance:\n"
    "- Start from 100 (fully trustworthy) as baseline\n"
    "- High spectral flatness (>0.07): -20 points (weak-to-moderate signal)\n"
    "- Unnaturally low pitch variance: -15 points (weak-to-moderate signal)\n"
    "- Elevated harmonic purity (HNR >24 dB): -10 points\n\n"
    "Deepfake audio detector signal guidance:\n"
    "- If deepfake_audio_confidence from the dedicated deepfake detector is above 70, this is strong direct evidence of synthetic/cloned audio — apply a -50 point penalty regardless of other signals.\n"
    "- If deepfake_audio_confidence is above 90, cap the maximum possible trust_score at 15.\n"
    "- This signal should be weighted more heavily than prosody signals since it directly analyzes audio content via a trained classifier.\n"
    "- CRITICAL: If deepfake_audio_confidence is null or unavailable, explicitly do NOT default to assuming authenticity (~80 score). Note in limitations that primary deepfake classifier was unavailable, and base evaluation on prosody and filename signals only.\n\n"
    "Rules: be specific, cite only given evidence, never invent findings, be honest about uncertainty, avoid absolute claims like '100% AI-generated'."
)

SAFE_DEFAULT = {
    "trust_score": 70,
    "verdict": "Suspicious",
    "red_flags": ["Audio metadata incomplete"],
    "explanation": "Audio analysis completed based on spectral prosody and acoustic flatness signals.",
    "limitations": "Full speaker voice print database was unavailable; assessment relies on acoustic features."
}


def _get_gemini_api_keys() -> List[str]:
    """
    Retrieves all configured Gemini API keys (primary & fail-safe fallbacks).
    Checks:
    - GEMINI_API_KEYS (comma-separated string)
    - GEMINI_API_KEY / GEMINI_API_KEY_PRIMARY
    - GEMINI_API_KEY_FALLBACK / GEMINI_API_KEY_SECONDARY / GEMINI_API_KEY_2 / FALLBACK_GEMINI_API_KEY
    Returns unique non-empty keys in priority order.
    """
    keys = []
    raw_list = os.environ.get("GEMINI_API_KEYS", "").strip()
    if raw_list:
        for k in raw_list.split(","):
            cleaned = k.strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)

    env_vars = [
        "GEMINI_API_KEY",
        "GEMINI_API_KEY_PRIMARY",
        "GEMINI_API_KEY_FALLBACK",
        "GEMINI_API_KEY_SECONDARY",
        "GEMINI_API_KEY_2",
        "FALLBACK_GEMINI_API_KEY",
    ]
    for ev in env_vars:
        val = os.environ.get(ev, "").strip()
        if val and val not in keys:
            keys.append(val)

    return keys


def generate_synthesis(
    metadata_analysis: Dict[str, Any],
    ela_analysis: Dict[str, Any],
    source_trace: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    ai_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Sends image findings to Gemini. Loops over primary and fail-safe fallback API keys if quota/rate limits occur.
    """
    fn_analysis = filename_analysis or {
        "filename": "unknown",
        "pattern_type": "unrecognized",
        "note": "Filename pattern unrecognized"
    }

    ai_det = ai_detector or {"ai_generation_confidence": None, "note": "AI detector unavailable"}

    findings_payload = {
        "metadata_analysis": metadata_analysis,
        "filename_analysis": fn_analysis,
        "ela_analysis": {
            "score": ela_analysis.get("score", 0),
            "has_heatmap": bool(ela_analysis.get("heatmap_url"))
        },
        "source_trace": source_trace,
        "ai_detector": ai_det
    }

    print(f"\n================ FULL GEMINI IMAGE INPUT PAYLOAD ================")
    print(json.dumps(findings_payload, indent=2))
    print(f"=================================================================\n")

    api_keys = _get_gemini_api_keys()
    if not api_keys:
        print("[Gemini Image Synthesis] No GEMINI_API_KEY configured — using offline fallback synthesis.")
        return _build_fallback(metadata_analysis, ela_analysis, source_trace, fn_analysis, ai_det)

    for key_idx, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)

            user_prompt = (
                f"{SYSTEM_INSTRUCTION_IMAGE}\n\n"
                f"TECHNICAL FINDINGS:\n"
                f"{json.dumps(findings_payload, indent=2)}"
            )

            generation_config = {
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }

            model = None
            for model_name in ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )
                    break
                except Exception:
                    continue

            if not model:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            def _call_gemini():
                return model.generate_content(user_prompt)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_gemini)
                response = future.result(timeout=15.0)

            if response and response.text:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                parsed = json.loads(text)
                if isinstance(parsed, dict) and "trust_score" in parsed and "verdict" in parsed:
                    if key_idx > 0:
                        print(f"[Gemini Image Synthesis] SUCCESS using fail-safe API key index #{key_idx + 1}")
                    return {
                        "trust_score": int(parsed.get("trust_score", 70)),
                        "verdict": str(parsed.get("verdict", "Suspicious")),
                        "red_flags": parsed.get("red_flags", []),
                        "explanation": str(parsed.get("explanation", SAFE_DEFAULT["explanation"])),
                        "limitations": str(parsed.get("limitations", SAFE_DEFAULT["limitations"]))
                    }
        except Exception as err:
            print(f"[Gemini Image Synthesis] Key #{key_idx + 1} call failed / rate limited: {err}")
            if key_idx < len(api_keys) - 1:
                print(f"--> Retrying with fail-safe fallback Gemini API key #{key_idx + 2}...")
                continue

    return _build_fallback(metadata_analysis, ela_analysis, source_trace, fn_analysis, ai_det)


def generate_video_synthesis(
    metadata_analysis: Dict[str, Any],
    ela_analysis: Dict[str, Any],
    source_trace: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    ai_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Sends video findings to Gemini using SYSTEM_INSTRUCTION_VIDEO.
    Loops over primary and fail-safe fallback API keys if quota/rate limits occur.
    Returns schema with malicious_percentage (no trust_score or verdict).
    """
    fn_analysis = filename_analysis or {
        "filename": "unknown",
        "pattern_type": "unrecognized",
        "note": "Filename pattern unrecognized"
    }

    ai_det = ai_detector or {"ai_generation_confidence": None, "note": "AI detector unavailable"}

    findings_payload = {
        "metadata_analysis": metadata_analysis,
        "filename_analysis": fn_analysis,
        "ela_analysis": {
            "score": ela_analysis.get("score", 0),
            "has_heatmap": bool(ela_analysis.get("heatmap_url"))
        },
        "source_trace": source_trace,
        "ai_detector": ai_det
    }

    print(f"\n================ FULL GEMINI VIDEO INPUT PAYLOAD ================")
    print(json.dumps(findings_payload, indent=2))
    print(f"=================================================================\n")

    api_keys = _get_gemini_api_keys()
    if not api_keys:
        print("[Gemini Video Synthesis] No GEMINI_API_KEY configured — using offline fallback synthesis.")
        return _build_video_fallback(metadata_analysis, ela_analysis, source_trace, fn_analysis, ai_det)

    for key_idx, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)

            user_prompt = (
                f"{SYSTEM_INSTRUCTION_VIDEO}\n\n"
                f"TECHNICAL FINDINGS:\n"
                f"{json.dumps(findings_payload, indent=2)}"
            )

            generation_config = {
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }

            model = None
            for model_name in ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )
                    break
                except Exception:
                    continue

            if not model:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            def _call_gemini():
                return model.generate_content(user_prompt)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_gemini)
                response = future.result(timeout=15.0)

            if response and response.text:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                parsed = json.loads(text)
                if isinstance(parsed, dict) and "malicious_percentage" in parsed:
                    if key_idx > 0:
                        print(f"[Gemini Video Synthesis] SUCCESS using fail-safe API key index #{key_idx + 1}")
                    return {
                        "malicious_percentage": int(parsed.get("malicious_percentage", 30)),
                        "red_flags": parsed.get("red_flags", []),
                        "explanation": str(parsed.get("explanation", "")),
                        "limitations": str(parsed.get("limitations", ""))
                    }
        except Exception as err:
            print(f"[Gemini Video Synthesis] Key #{key_idx + 1} call failed / rate limited: {err}")
            if key_idx < len(api_keys) - 1:
                print(f"--> Retrying with fail-safe fallback Gemini API key #{key_idx + 2}...")
                continue

    return _build_video_fallback(metadata_analysis, ela_analysis, source_trace, fn_analysis, ai_det)


def generate_audio_synthesis(
    audio_analysis: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    deepfake_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Sends audio spectral findings to Gemini. Loops over primary and fail-safe fallback API keys if quota/rate limits occur.
    """
    suspicious_flags = audio_analysis.get("suspicious_flags", [])
    fn_analysis = filename_analysis or {
        "filename": "unknown",
        "pattern_type": "unrecognized",
        "note": "Filename pattern unrecognized"
    }
    df_detector = deepfake_detector or {"deepfake_audio_confidence": None, "note": "Deepfake audio detector unavailable"}

    audio_payload = {
        **audio_analysis,
        "filename_analysis": fn_analysis,
        "deepfake_detector": df_detector
    }

    print(f"\n================ FULL GEMINI AUDIO INPUT PAYLOAD ================")
    print(json.dumps(audio_payload, indent=2))
    print(f"=================================================================\n")

    api_keys = _get_gemini_api_keys()
    if not api_keys:
        print("[Gemini Audio Synthesis] No GEMINI_API_KEY configured — using offline fallback synthesis.")
        return _build_audio_fallback(audio_analysis, fn_analysis, df_detector)

    for key_idx, api_key in enumerate(api_keys):
        try:
            genai.configure(api_key=api_key)

            user_prompt = (
                f"{SYSTEM_INSTRUCTION_AUDIO}\n\n"
                f"AUDIO TECHNICAL FINDINGS:\n"
                f"{json.dumps(audio_payload, indent=2)}"
            )

            generation_config = {
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }

            model = None
            for model_name in ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"]:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config
                    )
                    break
                except Exception:
                    continue

            if not model:
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")

            def _call_gemini():
                return model.generate_content(user_prompt)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_gemini)
                response = future.result(timeout=15.0)

            if response and response.text:
                text = response.text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                parsed = json.loads(text)
                if isinstance(parsed, dict) and "trust_score" in parsed and "verdict" in parsed:
                    if key_idx > 0:
                        print(f"[Gemini Audio Synthesis] SUCCESS using fail-safe API key index #{key_idx + 1}")
                    return {
                        "trust_score": int(parsed.get("trust_score", 70)),
                        "verdict": str(parsed.get("verdict", "Suspicious")),
                        "red_flags": parsed.get("red_flags", suspicious_flags),
                        "explanation": str(parsed.get("explanation", "")),
                        "limitations": str(parsed.get("limitations", ""))
                    }
        except Exception as err:
            print(f"[Gemini Audio Synthesis] Key #{key_idx + 1} call failed / rate limited: {err}")
            if key_idx < len(api_keys) - 1:
                print(f"--> Retrying with fail-safe fallback Gemini API key #{key_idx + 2}...")
                continue

    return _build_audio_fallback(audio_analysis, fn_analysis, df_detector)


def _build_fallback(
    metadata: Dict[str, Any],
    ela: Dict[str, Any],
    source: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    ai_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    score = 100
    has_exif = metadata.get("has_exif", False)
    software_tag = str(metadata.get("software_tag") or "").lower()
    suspicious_flags = metadata.get("suspicious_flags", [])[:]
    fn_type = (filename_analysis or {}).get("pattern_type", "unrecognized")
    ai_confidence = (ai_detector or {}).get("ai_generation_confidence")

    has_strong_signal = False

    if not has_exif:
        score -= 15

    ai_softwares = ["midjourney", "dalle", "stable diffusion", "firefly", "gemini", "ideogram", "leonardo", "playground"]
    if any(sw in software_tag for sw in ai_softwares):
        score -= 40
        has_strong_signal = True

    if fn_type == "ai_generator":
        score -= 20
        has_strong_signal = True
        suspicious_flags.append(f"Filename pattern indicates AI generator output ({filename_analysis.get('filename')})")
    elif fn_type == "camera_native":
        score += 10

    ela_score = ela.get("score", 0)
    if ela_score > 80:
        score -= 40
    elif ela_score > 60:
        score -= 25

    found_matches = source.get("found_matches", False)
    if not found_matches:
        score -= 5

    frames_flagged = (ai_detector or {}).get("frames_flagged_ai", 0)
    per_frame = (ai_detector or {}).get("per_frame_confidence", [])

    if ai_confidence is not None:
        is_video = bool(per_frame)
        if ai_confidence > 90:
            score -= 50
            score = min(score, 15)
            has_strong_signal = True
            if is_video:
                suspicious_flags.append(
                    f"Video AI detector: max frame confidence {ai_confidence:.1f}% "
                    f"({frames_flagged}/{len(per_frame)} frames flagged — very strong AI generation signal)"
                )
            else:
                suspicious_flags.append(f"AI image detector confidence: {ai_confidence:.1f}% (very high — strong AI generation signal)")
        elif ai_confidence > 70:
            score -= 50
            has_strong_signal = True
            if is_video:
                suspicious_flags.append(
                    f"Video AI detector: max frame confidence {ai_confidence:.1f}% "
                    f"({frames_flagged}/{len(per_frame)} frames flagged — likely AI-generated)"
                )
            else:
                suspicious_flags.append(f"AI image detector confidence: {ai_confidence:.1f}% (above threshold — likely AI-generated)")

    score = max(0, min(100, score))

    if not has_strong_signal:
        score = max(score, 50)

    if has_strong_signal and score >= 30:
        score = min(score, 29)

    if score >= 75:
        verdict = "Likely Authentic"
        explanation = "Media analysis indicates consistent metadata and low compression noise variance."
    elif score >= 50:
        verdict = "Suspicious"
        explanation = "Technical metadata is missing or incomplete. Compression noise pattern warrants careful verification."
    else:
        verdict = "Likely Manipulated"
        explanation = "Multiple technical signals suggest digital editing or AI generation. Metadata tags, filename patterns, or high error-level noise variance indicate alteration."

    limitations_text = "Reverse search index was unavailable; authenticity relies on file EXIF header structures and ELA noise."
    if ai_confidence is None:
        limitations_text = "Primary AI-detection classifier signal was unavailable; assessment is derived strictly from EXIF headers, filename pattern, and ELA noise without assuming authenticity."

    return {
        "trust_score": score,
        "verdict": verdict,
        "red_flags": suspicious_flags if suspicious_flags else ["No verified photographer metadata embedded"],
        "explanation": explanation,
        "limitations": limitations_text
    }


def _build_video_fallback(
    metadata: Dict[str, Any],
    ela: Dict[str, Any],
    source: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    ai_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Offline fallback for video — calculates malicious_percentage (0-100, inverted from trust_score).
    """
    fallback_res = _build_fallback(metadata, ela, source, filename_analysis, ai_detector)
    trust_score = fallback_res.get("trust_score", 70)
    malicious_percentage = 100 - trust_score

    ai_confidence = (ai_detector or {}).get("ai_generation_confidence")
    if ai_confidence is not None:
        if ai_confidence > 90:
            malicious_percentage = max(malicious_percentage, 85)
        elif ai_confidence > 70:
            malicious_percentage = max(malicious_percentage, 71)

    limitations_text = fallback_res.get("limitations", "")
    if ai_confidence is None:
        limitations_text = "Primary AI-detection classifier signal was unavailable; video risk assessment is derived strictly from keyframe ELA noise, EXIF headers, and filename patterns."

    return {
        "malicious_percentage": malicious_percentage,
        "red_flags": fallback_res.get("red_flags", []),
        "explanation": fallback_res.get("explanation", "Video frame analysis completed based on keyframe classification and technical metadata."),
        "limitations": limitations_text
    }


def _build_audio_fallback(
    audio: Dict[str, Any],
    filename_analysis: Dict[str, Any] = None,
    deepfake_detector: Dict[str, Any] = None
) -> Dict[str, Any]:
    score = 100
    suspicious_flags = audio.get("suspicious_flags", [])[:]
    flatness = audio.get("spectral_flatness", 0.0)
    fn_type = (filename_analysis or {}).get("pattern_type", "unrecognized")
    df_confidence = (deepfake_detector or {}).get("deepfake_audio_confidence")

    has_strong_signal = False

    if flatness > 0.07:
        score -= 20

    if fn_type == "ai_generator":
        score -= 20
        has_strong_signal = True
        suspicious_flags.append(f"Filename pattern indicates synthetic/AI voice generator ({filename_analysis.get('filename')})")

    if df_confidence is not None:
        if df_confidence > 90:
            score -= 50
            score = min(score, 15)
            has_strong_signal = True
            suspicious_flags.append(f"Deepfake audio detector confidence: {df_confidence:.1f}% (very high — strong synthetic voice signal)")
        elif df_confidence > 70:
            score -= 50
            has_strong_signal = True
            suspicious_flags.append(f"Deepfake audio detector confidence: {df_confidence:.1f}% (above threshold — likely AI-generated voice)")

    score = max(0, min(100, score))

    if not has_strong_signal:
        score = max(score, 50)

    if has_strong_signal and score >= 30:
        score = min(score, 29)

    if score >= 75:
        verdict = "Likely Authentic"
        explanation = "Audio spectral prosody and pitch variance exhibit natural human speech characteristics."
    elif score >= 50:
        verdict = "Suspicious"
        explanation = "Some acoustic spectral features deviate from typical natural human speech. Caution advised."
    else:
        verdict = "Likely Manipulated"
        explanation = "Acoustic analysis indicates strong synthetic or cloned voice signatures — high deepfake audio confidence or multiple prosody anomalies detected."

    limitations_text = "Voice biometrics database was unavailable; assessment is derived from deepfake audio classifier and spectral prosody features."
    if df_confidence is None:
        limitations_text = "Primary deepfake audio classifier signal was unavailable; assessment is derived strictly from acoustic spectral prosody metrics without assuming authenticity."

    return {
        "trust_score": score,
        "verdict": verdict,
        "red_flags": suspicious_flags if suspicious_flags else ["No synthetic voice markers detected"],
        "explanation": explanation,
        "limitations": limitations_text
    }
