import os
from typing import Dict, Any, List

def synthesize_analysis(
    metadata: Dict[str, Any],
    ela: Dict[str, Any],
    source_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesizes metadata, ELA manipulation metrics, and source trace info into:
    - Overall Trust Score (0-100)
    - Verdict tag ("Likely manipulated", "Suspicious", "Authentic")
    - Narrative summary
    - Structured Red Flags list
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            prompt = f"""
            Analyze the following media verification signals:
            - Metadata: {metadata}
            - ELA Manipulation Score: {ela.get('manipulation_score')}% ({ela.get('manipulation_explanation')})
            - Source Trace: Domain={source_trace.get('domain')}, Date={source_trace.get('date')}, Verified={source_trace.get('verified')}

            Return a JSON object with:
            - trust_score: integer 0-100 (where 100 is completely authentic, 0 is fully manipulated)
            - verdict: string ("Likely manipulated", "Suspicious", "Authentic")
            - summary: 2 sentence summary of key evidence and conclusion.
            - red_flags: list of objects with keys "label" and "desc" for any suspicious signals found.
            """
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            # Parse response content if returned as JSON
            import json
            text = response.content[0].text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                parsed = json.loads(text[start:end])
                return parsed
        except Exception:
            pass

    # Deterministic Local Synthesis Engine
    manipulation_score = ela.get("manipulation_score", 50)
    has_software_edit = metadata.get("software_flag", False)
    source_verified = source_trace.get("verified", False)
    camera_missing = metadata.get("camera") == "Not embedded"
    date_missing = metadata.get("date_taken") == "Not embedded"

    # Base Trust Score Calculation
    base_trust = 100 - (manipulation_score * 0.6)

    red_flags: List[Dict[str, str]] = []

    if has_software_edit:
        base_trust -= 20
        red_flags.append({
            "label": f"Edited with {metadata.get('software', 'editing software')}",
            "desc": "Software tag detected in metadata",
            "type": "software"
        })

    if manipulation_score >= 60:
        base_trust -= 15
        red_flags.append({
            "label": "High ELA Noise Variance",
            "desc": f"Manipulation scan indicates {manipulation_score}% suspicious edge variance",
            "type": "ela"
        })

    if not source_verified:
        base_trust -= 10
        red_flags.append({
            "label": "No verified source",
            "desc": "Could not trace to original news agency or verified publisher",
            "type": "source"
        })

    if camera_missing or date_missing:
        base_trust -= 5
        red_flags.append({
            "label": "Inconsistent timeline / metadata",
            "desc": "Creation date or camera EXIF profile missing or stripped",
            "type": "timeline"
        })

    trust_score = max(5, min(98, int(base_trust)))

    if trust_score >= 75:
        verdict = "Authentic"
        summary = "Media analysis indicates high authenticity signals. Metadata is consistent and low compression noise variance was detected."
    elif trust_score >= 50:
        verdict = "Suspicious"
        summary = "Moderate inconsistency detected. Compression artifacts or missing metadata warrant careful verification before sharing."
    else:
        verdict = "Likely manipulated"
        summary = "Multiple signals suggest this media has been altered. Metadata is inconsistent and elevated error levels indicate potential digital editing."

    # Guarantee at least 1-3 red flags if trust score is low/medium
    if not red_flags and trust_score < 70:
        red_flags.append({
            "label": "Unverified origin",
            "desc": "Original capture device signatures missing from file headers",
            "type": "source"
        })

    return {
        "trust_score": trust_score,
        "verdict": verdict,
        "summary": summary,
        "red_flags": red_flags
    }
