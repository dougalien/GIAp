import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="GIAp", layout="wide")

CONFIG = {
    "title": "GIAp",
    "subtitle": "Geology image analysis by We are dougalien",
    "website": "www.dougalien.com",
    "image_label": "rock, mineral, fossil, sediment, thin section, or related sample image",
    "analyst_role": "careful geology image analyst and supportive geology tutor",
    "timeout": 90,
    "max_image_size": 1400,
}

PROVIDERS = ["OpenAI", "Gemini"]

SAMPLE_TYPE_OPTIONS = [
    "Auto-detect",
    "Rock hand sample",
    "Thin section",
    "Mineral hand sample",
    "Fossil",
    "Sediment / sand / soil",
    "Water / liquid sample",
    "Other / mixed geologic material",
]

SAMPLE_TYPE_GUIDANCE = {
    "Auto-detect": (
        "Infer the sample type from the image and then keep every suggestion in that same context. "
        "Do not jump to a microscope unless the image is clearly a thin section or microscope view."
    ),
    "Rock hand sample": (
        "Focus on texture, grain size, fabric, visible mineralogy, composition, vesicles, foliation, bedding, rounding, and weathering. "
        "Do not suggest microscope work."
    ),
    "Thin section": (
        "Focus on optical properties and petrographic textures: relief, interference colors, twinning, extinction, birefringence, grain boundaries, zoning, and texture. "
        "Microscope-based suggestions are appropriate here."
    ),
    "Mineral hand sample": (
        "Focus on crystal habit, luster, cleavage, fracture, transparency, color, and visible associations in hand sample. "
        "Do not suggest microscope work."
    ),
    "Fossil": (
        "Focus on morphology, symmetry, ornamentation, segmentation, chambers, curvature, and growth habit. "
        "Do not suggest microscope work unless the image is clearly microscopic."
    ),
    "Sediment / sand / soil": (
        "Focus on grain size, sorting, rounding, composition, matrix, shell fragments, lithic fragments, fossils, and sedimentary texture."
    ),
    "Water / liquid sample": (
        "Focus on visible volume, clarity, color, suspended material, layering, bubbles, container context, and obvious contents."
    ),
    "Other / mixed geologic material": (
        "Match the observational pathway to the visible sample and keep advice tied to the actual viewing context."
    ),
}

ANALYSIS_SYSTEM_PROMPT = f"""
You are a {CONFIG['analyst_role']} for a mobile-friendly educational app.

Core rules:
- Use only visible evidence from the image and user-provided sample context.
- Be accurate, cautious, and specific.
- Do not invent locality, chemistry, hardness, streak, acid reaction, or magnetism unless directly visible or explicitly stated by the user.
- If precision is not justified, return the best material group instead of overreaching.
- Distinguish visible observations from interpretation.
- Keep the wording clear and accessible.
- The field next_look must stay in the same context as the sample type. Example: do not suggest microscope work for a rock hand sample, but it is appropriate for a thin section.
- Prefer a broad but defensible material group rather than an overconfident specific name.

Return valid JSON only with this shape:
{{
  "sample_type": "best-fit sample type",
  "candidate": "best identification or most likely material group",
  "alternate": "brief alternate possibility or 'none'",
  "confidence": 1,
  "observations": ["visible feature 1", "visible feature 2", "visible feature 3"],
  "why": "short explanation grounded in visible evidence",
  "limits": "what cannot be determined from this image alone",
  "next_look": "one specific next thing to look at or think about in the same sample context"
}}

Confidence scale:
1 = weak guess
2 = plausible
3 = moderate
4 = strong
5 = very strong
""".strip()

FOLLOWUP_SYSTEM_PROMPT = """
You are continuing a geology image discussion for the same sample image.

Rules:
- Stay grounded in the same image and the existing image context.
- Be concise, helpful, and accurate.
- If the answer is uncertain, say so plainly.
- Keep suggestions matched to the sample type and viewing context.
- Do not invent tests or observations that are not visible unless you clearly label them as possible next checks.
""".strip()

GUIDED_SYSTEM_PROMPT = """
You are a supportive geology tutor and lab helper working from the same uploaded image.

Rules:
- Always begin by recognizing something the student did well, even if small.
- Be encouraging, calm, and specific.
- Guide the student toward better observations without shaming them.
- Use the hidden instructor analysis only as guidance for coaching. Do not dump the final answer immediately unless the student directly asks for it.
- Keep all advice matched to the sample type and image context.
- For rock hand samples, emphasize texture and mineralogy/composition.
- For thin sections, emphasize optical properties and textures.
- For fossils, emphasize morphology and habit.
- For sediment, sand, or soil, emphasize grain size, sorting, rounding, and composition.
- For water or liquid samples, emphasize visible volume, clarity, contents, layering, and suspended material.
- If the student's name is off, explain gently what features point elsewhere.
- End with one concrete next observation or question for the student to consider.
- Keep the response readable in short paragraphs or bullets.
""".strip()

ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sample_type": {"type": "string"},
        "candidate": {"type": "string"},
        "alternate": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "observations": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 5,
        },
        "why": {"type": "string"},
        "limits": {"type": "string"},
        "next_look": {"type": "string"},
    },
    "required": [
        "sample_type",
        "candidate",
        "alternate",
        "confidence",
        "observations",
        "why",
        "limits",
        "next_look",
    ],
    "additionalProperties": False,
}


def init_state() -> None:
    defaults = {
        "analysis_error": "",
        "source_name": "",
        "focus_zone": "Full image",
        "last_image_b64": None,
        "last_image_signature": "",
        "last_context_signature": "",
        "selected_sample_type": "Auto-detect",
        "point_results": {},
        "point_summary": None,
        "point_chat_history": {provider: [] for provider in PROVIDERS},
        "guided_chat_history": {provider: [] for provider in PROVIDERS},
        "guided_reference": {},
        "analysis_calls_used": 0,
        "provider_call_counts": {provider: 0 for provider in PROVIDERS},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()



def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default



def get_openai_model(task: str = "point") -> str:
    if task == "guided":
        return get_secret("OPENAI_GUIDED_MODEL", get_secret("OPENAI_MODEL", "gpt-4o"))
    return get_secret("OPENAI_POINT_MODEL", get_secret("OPENAI_MODEL", "gpt-4o"))



def get_gemini_model(task: str = "point") -> str:
    if task == "guided":
        return get_secret("GEMINI_GUIDED_MODEL", get_secret("GEMINI_MODEL", "gemini-2.5-flash"))
    return get_secret("GEMINI_POINT_MODEL", get_secret("GEMINI_MODEL", "gemini-2.5-flash"))



def get_session_limit() -> int:
    raw = get_secret("MAX_AI_CALLS_PER_SESSION", "0")
    try:
        limit = int(str(raw).strip())
    except Exception:
        limit = 0
    return max(0, limit)



def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        return None
    return None



def normalize_analysis_result(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        confidence = int(data.get("confidence", 1))
    except Exception:
        confidence = 1
    confidence = max(1, min(5, confidence))

    observations = data.get("observations", [])
    if not isinstance(observations, list):
        observations = []
    observations = [str(item).strip() for item in observations if str(item).strip()][:5]

    return {
        "sample_type": str(data.get("sample_type", "Auto-detect")).strip() or "Auto-detect",
        "candidate": str(data.get("candidate", "")).strip() or "Uncertain material group",
        "alternate": str(data.get("alternate", "none")).strip() or "none",
        "confidence": confidence,
        "observations": observations,
        "why": str(data.get("why", "")).strip(),
        "limits": str(data.get("limits", "")).strip(),
        "next_look": str(data.get("next_look", "")).strip(),
    }



def prepare_image(file_bytes: bytes, max_size: int = CONFIG["max_image_size"]) -> Image.Image:
    image = Image.open(io.BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((max_size, max_size))
    return image



def crop_by_zone(image: Image.Image, zone: str) -> Image.Image:
    if zone == "Full image":
        return image

    width, height = image.size
    third_w = max(1, width // 3)
    third_h = max(1, height // 3)

    boxes = {
        "Top left": (0, 0, third_w, third_h),
        "Top center": (third_w, 0, third_w * 2, third_h),
        "Top right": (third_w * 2, 0, width, third_h),
        "Middle left": (0, third_h, third_w, third_h * 2),
        "Center": (third_w, third_h, third_w * 2, third_h * 2),
        "Middle right": (third_w * 2, third_h, width, third_h * 2),
        "Bottom left": (0, third_h * 2, third_w, height),
        "Bottom center": (third_w, third_h * 2, third_w * 2, height),
        "Bottom right": (third_w * 2, third_h * 2, width, height),
    }
    return image.crop(boxes.get(zone, (0, 0, width, height)))



def image_to_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



def image_signature(file_bytes: bytes, source_name: str) -> str:
    digest = hashlib.md5(file_bytes).hexdigest()
    return f"{source_name}:{digest}"



def reset_image_dependent_state() -> None:
    st.session_state.analysis_error = ""
    st.session_state.point_results = {}
    st.session_state.point_summary = None
    st.session_state.point_chat_history = {provider: [] for provider in PROVIDERS}
    st.session_state.guided_chat_history = {provider: [] for provider in PROVIDERS}
    st.session_state.guided_reference = {}



def ensure_quota() -> None:
    limit = get_session_limit()
    used = st.session_state.analysis_calls_used
    if limit > 0 and used >= limit:
        raise RuntimeError(
            f"This session has reached its AI call limit ({used}/{limit}). "
            f"Increase MAX_AI_CALLS_PER_SESSION in secrets or set it to 0 for unlimited use."
        )



def increment_quota(provider: str) -> None:
    st.session_state.analysis_calls_used += 1
    counts = st.session_state.provider_call_counts
    counts[provider] = counts.get(provider, 0) + 1
    st.session_state.provider_call_counts = counts



def normalize_label(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\b(probably|possibly|likely|most likely|appears to be|could be|may be|suggests?)\b", "", text)
    text = re.sub(r"[^a-z0-9\s/-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def labels_overlap(a: str, b: str) -> bool:
    a_norm = normalize_label(a)
    b_norm = normalize_label(b)
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    return a_norm in b_norm or b_norm in a_norm



def observation_overlap(a_obs: List[str], b_obs: List[str]) -> int:
    a_set = {normalize_label(item) for item in a_obs if normalize_label(item)}
    b_set = {normalize_label(item) for item in b_obs if normalize_label(item)}
    if not a_set or not b_set:
        return 0
    return len(a_set.intersection(b_set))



def dedupe_text_list(items: List[str], max_items: int = 6) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        cleaned = str(item).strip()
        key = normalize_label(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            output.append(cleaned)
        if len(output) >= max_items:
            break
    return output



def build_point_prompt(sample_type: str, source_name: str, focus_zone: str) -> str:
    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    return (
        f"User-selected sample type: {sample_type}.\n"
        f"Source name: {source_name or 'upload'}.\n"
        f"Focus zone: {focus_zone}.\n"
        f"Sample-type guidance: {guidance}\n\n"
        "Analyze this image and identify the most likely geologic material or material group. "
        "Return cautious observations, a concise explanation, and one same-context next thing to look at or think about."
    )



def build_openai_multimodal_user_message(text: str, image_b64: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }



def extract_gemini_text(response_json: Dict[str, Any]) -> str:
    candidates = response_json.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")
    parts = candidates[0].get("content", {}).get("parts", [])
    texts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
    text = "\n".join(texts).strip()
    if not text:
        raise RuntimeError("Gemini returned no text output.")
    return text



def call_openai(messages: List[Dict[str, Any]], json_mode: bool = False, temperature: float = 0.2, model: Optional[str] = None) -> str:
    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    ensure_quota()

    payload: Dict[str, Any] = {
        "model": model or get_openai_model("point"),
        "temperature": temperature,
        "messages": messages,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=CONFIG["timeout"],
    )
    response.raise_for_status()
    increment_quota("OpenAI")
    return response.json()["choices"][0]["message"]["content"]



def call_gemini(
    user_text: str,
    image_b64: str,
    system_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.2,
    model: Optional[str] = None,
) -> str:
    api_key = get_secret("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.")

    ensure_quota()

    payload: Dict[str, Any] = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"text": user_text},
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    if json_mode:
        payload["generationConfig"]["responseMimeType"] = "application/json"
        payload["generationConfig"]["responseJsonSchema"] = ANALYSIS_SCHEMA

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model or get_gemini_model('point')}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=CONFIG["timeout"],
    )
    response.raise_for_status()
    increment_quota("Gemini")
    return extract_gemini_text(response.json())



def run_provider_point_analysis(
    provider: str,
    image_b64: str,
    sample_type: str,
    source_name: str,
    focus_zone: str,
) -> Dict[str, Any]:
    prompt = build_point_prompt(sample_type, source_name, focus_zone)
    if provider == "OpenAI":
        content = call_openai(
            [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                build_openai_multimodal_user_message(prompt, image_b64),
            ],
            json_mode=True,
            temperature=0.1,
            model=get_openai_model("point"),
        )
    else:
        content = call_gemini(
            user_text=prompt,
            image_b64=image_b64,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            json_mode=True,
            temperature=0.1,
            model=get_gemini_model("point"),
        )

    parsed = safe_json_loads(content)
    if not parsed:
        raise RuntimeError(f"{provider} returned unreadable JSON.")
    result = normalize_analysis_result(parsed)
    result["provider"] = provider
    result["model"] = get_openai_model("point") if provider == "OpenAI" else get_gemini_model("point")
    return result



def build_consensus_summary(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    available = {name: value for name, value in results.items() if value and not value.get("error")}
    if not available:
        raise RuntimeError("No provider result was available.")

    if len(available) == 1:
        provider, result = next(iter(available.items()))
        return {
            "status": "single-provider",
            "headline": result.get("candidate", "Uncertain material group"),
            "agreement": "Only one model available",
            "recommended_provider": provider,
            "recommended_result": result,
            "notes": f"Only {provider} returned a usable result, so treat this as a working interpretation rather than a cross-model consensus.",
            "observations": dedupe_text_list(result.get("observations", []), max_items=5),
        }

    openai_result = available.get("OpenAI")
    gemini_result = available.get("Gemini")
    if not openai_result or not gemini_result:
        provider, result = next(iter(available.items()))
        return {
            "status": "single-provider",
            "headline": result.get("candidate", "Uncertain material group"),
            "agreement": "Only one model available",
            "recommended_provider": provider,
            "recommended_result": result,
            "notes": f"Only {provider} returned a usable result, so treat this as a working interpretation rather than a cross-model consensus.",
            "observations": dedupe_text_list(result.get("observations", []), max_items=5),
        }

    candidate_match = labels_overlap(openai_result.get("candidate", ""), gemini_result.get("candidate", ""))
    cross_match = (
        labels_overlap(openai_result.get("candidate", ""), gemini_result.get("alternate", ""))
        or labels_overlap(gemini_result.get("candidate", ""), openai_result.get("alternate", ""))
    )
    sample_type_match = labels_overlap(openai_result.get("sample_type", ""), gemini_result.get("sample_type", ""))
    obs_overlap = observation_overlap(openai_result.get("observations", []), gemini_result.get("observations", []))

    openai_score = int(openai_result.get("confidence", 1))
    gemini_score = int(gemini_result.get("confidence", 1))

    if candidate_match:
        openai_score += 2
        gemini_score += 2
    elif cross_match:
        openai_score += 1
        gemini_score += 1

    if sample_type_match:
        openai_score += 1
        gemini_score += 1

    if obs_overlap > 0:
        openai_score += 1
        gemini_score += 1

    if openai_score > gemini_score:
        recommended_provider = "OpenAI"
        recommended_result = openai_result
    elif gemini_score > openai_score:
        recommended_provider = "Gemini"
        recommended_result = gemini_result
    else:
        if int(openai_result.get("confidence", 1)) >= int(gemini_result.get("confidence", 1)):
            recommended_provider = "OpenAI"
            recommended_result = openai_result
        else:
            recommended_provider = "Gemini"
            recommended_result = gemini_result

    if candidate_match:
        agreement = "High agreement"
        headline = recommended_result.get("candidate", "Uncertain material group")
        notes = "Both models landed on essentially the same working interpretation."
    elif cross_match or (sample_type_match and obs_overlap >= 2):
        agreement = "Partial agreement"
        headline = recommended_result.get("candidate", "Uncertain material group")
        notes = (
            "The models overlap in important ways, but not perfectly. Use the recommended result as the better-supported working interpretation."
        )
    else:
        agreement = "Low agreement"
        headline = f"Unresolved: {recommended_result.get('candidate', 'uncertain material group')}"
        notes = (
            "The models do not strongly agree. Treat the recommendation as provisional and lean on the visible observations before naming the sample too specifically."
        )

    combined_observations = dedupe_text_list(
        openai_result.get("observations", []) + gemini_result.get("observations", []),
        max_items=6,
    )

    return {
        "status": "consensus",
        "headline": headline,
        "agreement": agreement,
        "recommended_provider": recommended_provider,
        "recommended_result": recommended_result,
        "notes": notes,
        "observations": combined_observations,
    }



def get_or_create_guided_reference(provider: str, sample_type: str, source_name: str, focus_zone: str) -> Dict[str, Any]:
    key = f"{provider}|{sample_type}|{source_name}|{focus_zone}|{st.session_state.last_image_signature}"
    cache = st.session_state.guided_reference
    if key in cache:
        return cache[key]

    if provider in st.session_state.point_results and not st.session_state.point_results[provider].get("error"):
        cache[key] = st.session_state.point_results[provider]
        st.session_state.guided_reference = cache
        return cache[key]

    result = run_provider_point_analysis(
        provider=provider,
        image_b64=st.session_state.last_image_b64,
        sample_type=sample_type,
        source_name=source_name,
        focus_zone=focus_zone,
    )
    cache[key] = result
    st.session_state.guided_reference = cache
    return result



def build_point_followup_prompt(question: str, provider: str) -> str:
    summary = st.session_state.point_summary or {}
    provider_result = st.session_state.point_results.get(provider, {})
    recommended = summary.get("recommended_result", provider_result)
    return (
        f"Consensus headline: {summary.get('headline', '')}\n"
        f"Agreement level: {summary.get('agreement', '')}\n"
        f"Recommended provider: {summary.get('recommended_provider', '')}\n"
        f"Recommended interpretation: {recommended.get('candidate', '')}\n"
        f"Recommended observations: {', '.join(recommended.get('observations', []))}\n"
        f"Recommended why: {recommended.get('why', '')}\n"
        f"Recommended limits: {recommended.get('limits', '')}\n"
        f"Recommended next look: {recommended.get('next_look', '')}\n\n"
        f"{provider} result for this same image:\n"
        f"Sample type: {provider_result.get('sample_type', '')}\n"
        f"Likely identification: {provider_result.get('candidate', '')}\n"
        f"Alternate: {provider_result.get('alternate', '')}\n"
        f"Confidence: {provider_result.get('confidence', '')}/5\n"
        f"Observations: {', '.join(provider_result.get('observations', []))}\n"
        f"Why: {provider_result.get('why', '')}\n"
        f"Limits: {provider_result.get('limits', '')}\n"
        f"Next look: {provider_result.get('next_look', '')}\n\n"
        f"Answer this follow-up question about the same image: {question}"
    )



def point_followup(provider: str, question: str) -> str:
    if provider == "OpenAI":
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
            build_openai_multimodal_user_message(build_point_followup_prompt(question, provider), st.session_state.last_image_b64),
        ]
        for item in st.session_state.point_chat_history.get(provider, []):
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": question})
        return call_openai(messages, json_mode=False, temperature=0.2, model=get_openai_model("guided")).strip()

    prompt = build_point_followup_prompt(question, provider)
    history_lines = []
    for item in st.session_state.point_chat_history.get(provider, []):
        speaker = "User" if item.get("role") == "user" else "Assistant"
        history_lines.append(f"{speaker}: {item.get('content', '')}")
    history_text = "\n".join(history_lines)
    if history_text:
        prompt = f"{prompt}\n\nPrior conversation:\n{history_text}\n\nNewest user question: {question}"

    return call_gemini(
        user_text=prompt,
        image_b64=st.session_state.last_image_b64,
        system_prompt=FOLLOWUP_SYSTEM_PROMPT,
        json_mode=False,
        temperature=0.2,
        model=get_gemini_model("guided"),
    ).strip()



def build_guided_start_prompt(
    sample_type: str,
    observations: str,
    attempted_name: str,
    source_name: str,
    focus_zone: str,
    reference: Dict[str, Any],
) -> str:
    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    return (
        f"User-selected sample type: {sample_type}.\n"
        f"Source name: {source_name or 'upload'}.\n"
        f"Focus zone: {focus_zone}.\n"
        f"Sample-type guidance: {guidance}\n\n"
        f"Hidden instructor reference sample type: {reference.get('sample_type', '')}\n"
        f"Hidden instructor reference likely identification: {reference.get('candidate', '')}\n"
        f"Hidden instructor alternate: {reference.get('alternate', '')}\n"
        f"Hidden instructor observations: {', '.join(reference.get('observations', []))}\n"
        f"Hidden instructor explanation: {reference.get('why', '')}\n"
        f"Hidden instructor limits: {reference.get('limits', '')}\n\n"
        f"Student observations: {observations}\n"
        f"Student attempted name: {attempted_name}\n\n"
        "Coach the student supportively. Point out what they are doing well, guide them toward the most relevant visible features to focus on, "
        "and end with one concrete next observation or question."
    )



def guided_start(
    provider: str,
    sample_type: str,
    observations: str,
    attempted_name: str,
    source_name: str,
    focus_zone: str,
) -> str:
    reference = get_or_create_guided_reference(provider, sample_type, source_name, focus_zone)
    prompt = build_guided_start_prompt(sample_type, observations, attempted_name, source_name, focus_zone, reference)

    if provider == "OpenAI":
        messages = [
            {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
            build_openai_multimodal_user_message(prompt, st.session_state.last_image_b64),
        ]
        return call_openai(messages, json_mode=False, temperature=0.3, model=get_openai_model("guided")).strip()

    return call_gemini(
        user_text=prompt,
        image_b64=st.session_state.last_image_b64,
        system_prompt=GUIDED_SYSTEM_PROMPT,
        json_mode=False,
        temperature=0.3,
        model=get_gemini_model("guided"),
    ).strip()



def build_guided_followup_prompt(provider: str, sample_type: str, question: str, source_name: str, focus_zone: str) -> str:
    reference = get_or_create_guided_reference(provider, sample_type, source_name, focus_zone)
    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    history_lines = []
    for item in st.session_state.guided_chat_history.get(provider, []):
        speaker = "Student" if item.get("role") == "user" else "Tutor"
        history_lines.append(f"{speaker}: {item.get('content', '')}")
    history_text = "\n".join(history_lines)
    return (
        f"Continue the guided tutoring conversation for the same image.\n"
        f"User-selected sample type: {sample_type}.\n"
        f"Focus zone: {focus_zone}.\n"
        f"Source name: {source_name or 'upload'}.\n"
        f"Sample-type guidance: {guidance}\n\n"
        f"Hidden instructor reference likely identification: {reference.get('candidate', '')}\n"
        f"Hidden instructor alternate: {reference.get('alternate', '')}\n"
        f"Hidden instructor observations: {', '.join(reference.get('observations', []))}\n"
        f"Hidden instructor explanation: {reference.get('why', '')}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Student follow-up: {question}"
    )



def guided_followup(provider: str, sample_type: str, question: str, source_name: str, focus_zone: str) -> str:
    prompt = build_guided_followup_prompt(provider, sample_type, question, source_name, focus_zone)
    if provider == "OpenAI":
        messages = [
            {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
            build_openai_multimodal_user_message(prompt, st.session_state.last_image_b64),
        ]
        return call_openai(messages, json_mode=False, temperature=0.3, model=get_openai_model("guided")).strip()

    return call_gemini(
        user_text=prompt,
        image_b64=st.session_state.last_image_b64,
        system_prompt=GUIDED_SYSTEM_PROMPT,
        json_mode=False,
        temperature=0.3,
        model=get_gemini_model("guided"),
    ).strip()



def render_chat_history(history: List[Dict[str, str]], title: str) -> None:
    if not history:
        return
    st.markdown(f"### {title}")
    for item in history:
        if item.get("role") == "user":
            st.markdown(f"**You:** {item.get('content', '')}")
        else:
            st.markdown(f"**AI:** {item.get('content', '')}")



def show_image_compat(image: Image.Image, caption: str) -> None:
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image, caption=caption)



def render_provider_result(result: Dict[str, Any], provider: str) -> None:
    if result.get("error"):
        st.error(result["error"])
        return

    st.markdown(f"**Model:** `{result.get('model', '')}`")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{result.get('confidence', '')}/5")
    c2.metric("Sample type", result.get("sample_type", ""))
    st.write(f"**Likely identification:** {result.get('candidate', '')}")
    st.write(f"**Alternate:** {result.get('alternate', '')}")
    st.write("**Visible observations**")
    for item in result.get("observations", []):
        st.write(f"- {item}")
    st.write(f"**Why this fits:** {result.get('why', '')}")
    st.write(f"**Limits:** {result.get('limits', '')}")
    st.write(f"**Next thing to look at:** {result.get('next_look', '')}")



def render_point_summary(summary: Dict[str, Any]) -> None:
    st.markdown("### Best-supported working interpretation")
    c1, c2 = st.columns(2)
    c1.metric("Agreement", summary.get("agreement", ""))
    c2.metric("Recommended engine", summary.get("recommended_provider", ""))
    st.write(f"**Headline:** {summary.get('headline', '')}")
    st.write(f"**Why this is the working call:** {summary.get('notes', '')}")
    st.write("**Most useful observations across the two models**")
    for item in summary.get("observations", []):
        st.write(f"- {item}")



def export_point_text(summary: Dict[str, Any], results: Dict[str, Dict[str, Any]], source_name: str, focus_zone: str) -> str:
    lines = [
        CONFIG["title"],
        f"Source: {source_name or 'upload'}",
        f"Focus area: {focus_zone}",
        "",
        f"Working interpretation: {summary.get('headline', '')}",
        f"Agreement: {summary.get('agreement', '')}",
        f"Recommended engine: {summary.get('recommended_provider', '')}",
        f"Working note: {summary.get('notes', '')}",
        "",
        "Combined observations:",
    ]
    for item in summary.get("observations", []):
        lines.append(f"- {item}")

    for provider in PROVIDERS:
        result = results.get(provider, {})
        lines.extend(["", f"{provider} result:"])
        if result.get("error"):
            lines.append(f"Error: {result.get('error', '')}")
            continue
        lines.extend(
            [
                f"Model: {result.get('model', '')}",
                f"Sample type: {result.get('sample_type', '')}",
                f"Likely identification: {result.get('candidate', '')}",
                f"Alternate: {result.get('alternate', '')}",
                f"Confidence: {result.get('confidence', '')}/5",
                "Visible observations:",
            ]
        )
        for obs in result.get("observations", []):
            lines.append(f"- {obs}")
        lines.extend(
            [
                f"Why this fits: {result.get('why', '')}",
                f"Limits: {result.get('limits', '')}",
                f"Next thing to look at: {result.get('next_look', '')}",
            ]
        )
    return "\n".join(lines) + "\n"




def get_local_banner_path() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "giap_banner.png",
        here / "assets" / "giap_banner.png",
        Path("/mount/src/giap/giap_banner.png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def render_standard_banner(banner_path: Optional[Path], width_ratio: float = 3.0) -> None:
    if banner_path:
        left, center, right = st.columns([1.2, width_ratio, 1.2])
        with center:
            try:
                st.image(str(banner_path), use_container_width=True)
            except TypeError:
                try:
                    st.image(str(banner_path), use_column_width=True)
                except TypeError:
                    st.image(str(banner_path), width=900)


def render_top_banner() -> None:
    render_standard_banner(get_local_banner_path(), width_ratio=3.0)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px;
        line-height: 1.5;
    }
    .stApp {
        background: #F7F9FB;
        color: #111111;
    }
    .card {
        background: white;
        border: 1px solid #D7DCE2;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(16, 24, 40, 0.04);
    }
    .title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #111111;
    }
    .subtitle {
        font-size: 1.04rem;
        color: #243746;
        margin-bottom: 0.25rem;
    }
    .site {
        color: #1B5E86;
        font-size: 0.98rem;
    }
    .small {
        color: #33424F;
        font-size: 0.98rem;
    }
    a, a:visited {
        color: #114B73;
        text-decoration: underline;
    }
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    button[kind="secondary"],
    button[kind="primary"] {
        min-height: 48px;
        border-radius: 10px;
        font-weight: 700;
        width: 100%;
        background: #D9ECFF !important;
        color: #111111 !important;
        border: 1px solid #7FA8C4 !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }
    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    button[kind="secondary"]:hover,
    button[kind="primary"]:hover {
        background: #C7E3FF !important;
        color: #111111 !important;
        border: 1px solid #5F8EAF !important;
        filter: none !important;
        opacity: 1 !important;
    }
    div.stButton > button:focus,
    div[data-testid="stDownloadButton"] > button:focus,
    div[data-testid="stFormSubmitButton"] > button:focus,
    button[kind="secondary"]:focus,
    button[kind="primary"]:focus,
    div.stButton > button:focus-visible,
    div[data-testid="stDownloadButton"] > button:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:focus-visible,
    button[kind="secondary"]:focus-visible,
    button[kind="primary"]:focus-visible {
        color: #111111 !important;
        background: #C7E3FF !important;
        outline: 3px solid #0F4C75 !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }
    [data-baseweb="tab-list"] button {
        color: #111111 !important;
        font-weight: 700 !important;
    }
    [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: #D9ECFF !important;
        border-radius: 10px 10px 0 0 !important;
    }
    [data-testid="stInfo"] {
        background: #EAF6FF;
        color: #12212B;
        border: 1px solid #A8C7DD;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        color: #111111 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
render_top_banner()

st.markdown(
    f"""
    <div class="card">
        <div class="title">{CONFIG['title']}</div>
        <div class="subtitle">{CONFIG['subtitle']}</div>
        <div class="site">{CONFIG['website']}</div>
        <p class="small" style="margin-top:0.8rem;">
            Two-model point-and-click comparison plus guided tutoring that pushes the user toward better observations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Accessible, phone-friendly layout. Point and Click now runs OpenAI and Gemini side by side, then builds a neutral working interpretation from agreement and confidence."
)

point_tab, guided_tab = st.tabs(["Point and Click", "Guided Analysis"])

with point_tab:
    st.markdown(
        "**Point and Click** runs OpenAI and Gemini side by side, then shows the best-supported working interpretation without forcing a false consensus."
    )

with guided_tab:
    st.markdown(
        "**Guided Analysis** uses a hidden reference analysis plus the same image to coach the student toward the right observations."
    )

limit = get_session_limit()
used = st.session_state.analysis_calls_used
counts = st.session_state.provider_call_counts
if limit > 0:
    st.caption(f"AI calls this session: {used}/{limit} | OpenAI: {counts.get('OpenAI', 0)} | Gemini: {counts.get('Gemini', 0)}")
else:
    st.caption(f"AI calls this session: {used} | OpenAI: {counts.get('OpenAI', 0)} | Gemini: {counts.get('Gemini', 0)} | session limit: unlimited")

with st.expander("Model setup", expanded=False):
    st.write(f"**Point and Click – OpenAI:** `{get_openai_model('point')}`")
    st.write(f"**Point and Click – Gemini:** `{get_gemini_model('point')}`")
    st.write(f"**Guided tutor – OpenAI:** `{get_openai_model('guided')}`")
    st.write(f"**Guided tutor – Gemini:** `{get_gemini_model('guided')}`")
    st.caption("You can override these by adding OPENAI_MODEL / GEMINI_MODEL or task-specific model names in Streamlit secrets.")

st.markdown("### Image and sample setup")
uploaded_file = st.file_uploader(
    f"Upload a {CONFIG['image_label']}",
    type=["png", "jpg", "jpeg"],
    help="On phones, this usually lets you choose camera, photo library, or files. On laptops, upload an existing image file.",
)

image_bytes: Optional[bytes] = None
source_name = ""
if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name

selected_sample_type = st.selectbox(
    "Sample type",
    SAMPLE_TYPE_OPTIONS,
    index=SAMPLE_TYPE_OPTIONS.index(st.session_state.selected_sample_type)
    if st.session_state.selected_sample_type in SAMPLE_TYPE_OPTIONS
    else 0,
    help="Leave on Auto-detect if you want the app to infer the context from the image.",
)
st.session_state.selected_sample_type = selected_sample_type

focus_zone = st.selectbox(
    "Focus area",
    [
        "Full image",
        "Top left", "Top center", "Top right",
        "Middle left", "Center", "Middle right",
        "Bottom left", "Bottom center", "Bottom right",
    ],
    index=0,
    help="Whole image is fastest. Zone focus is useful when only one part of the image matters.",
)
st.session_state.focus_zone = focus_zone

if image_bytes:
    current_signature = image_signature(image_bytes, source_name)
    current_context_signature = f"{current_signature}|{focus_zone}|{selected_sample_type}"
    if current_signature != st.session_state.last_image_signature:
        st.session_state.last_image_signature = current_signature
        st.session_state.source_name = source_name
    if current_context_signature != st.session_state.last_context_signature:
        st.session_state.last_context_signature = current_context_signature
        reset_image_dependent_state()

    try:
        base_image = prepare_image(image_bytes)
        focused_image = crop_by_zone(base_image, focus_zone)
        displayed_image = base_image if focus_zone == "Full image" else focused_image
        displayed_caption = "Selected image" if focus_zone == "Full image" else f"Selected image: {focus_zone}"
        show_image_compat(displayed_image, displayed_caption)
        st.session_state.last_image_b64 = image_to_b64(focused_image)
    except Exception as exc:
        st.error(f"Image error: {exc}")
        st.stop()
else:
    st.warning("Add an image to begin.")
    st.stop()

with point_tab:
    st.markdown(
        "Use this when you want both engines to take a fair shot, then compare what they agree on and where they differ."
    )
    if st.button("Analyze image with OpenAI + Gemini", key="analyze_image_button"):
        st.session_state.analysis_error = ""
        results: Dict[str, Dict[str, Any]] = {}
        try:
            with st.spinner("Running both models..."):
                for provider in PROVIDERS:
                    try:
                        results[provider] = run_provider_point_analysis(
                            provider=provider,
                            image_b64=st.session_state.last_image_b64,
                            sample_type=selected_sample_type,
                            source_name=source_name,
                            focus_zone=focus_zone,
                        )
                    except Exception as provider_exc:
                        results[provider] = {"error": f"{provider} error: {provider_exc}"}
                st.session_state.point_results = results
                st.session_state.point_summary = build_consensus_summary(results)
            st.success("Comparison complete.")
        except Exception as exc:
            st.session_state.analysis_error = f"Unexpected error: {exc}"

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    if st.session_state.point_summary:
        render_point_summary(st.session_state.point_summary)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### OpenAI result")
            render_provider_result(st.session_state.point_results.get("OpenAI", {"error": "No result"}), "OpenAI")
        with col2:
            st.markdown("### Gemini result")
            render_provider_result(st.session_state.point_results.get("Gemini", {"error": "No result"}), "Gemini")

        followup_provider = st.radio(
            "Point-and-click follow-up engine",
            PROVIDERS,
            horizontal=True,
            key="point_followup_provider",
            help="Pick which model should answer your follow-up question about the same image.",
        )

        render_chat_history(
            st.session_state.point_chat_history.get(followup_provider, []),
            f"Point-and-click follow-up ({followup_provider})",
        )

        with st.form("point_followup_form", clear_on_submit=True):
            point_question = st.text_area(
                "Ask a follow-up question about this same image",
                height=110,
                placeholder="Example: What feature makes you lean toward basalt instead of andesite?",
            )
            point_submit = st.form_submit_button("Send follow-up")

        if point_submit:
            if not point_question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner(f"Answering with {followup_provider}..."):
                        reply = point_followup(followup_provider, point_question.strip())
                    history = st.session_state.point_chat_history.get(followup_provider, [])
                    history.append({"role": "user", "content": point_question.strip()})
                    history.append({"role": "assistant", "content": reply})
                    st.session_state.point_chat_history[followup_provider] = history
                    rerun_app()
                except requests.RequestException as exc:
                    st.error(f"Request error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

        export_text = export_point_text(
            st.session_state.point_summary,
            st.session_state.point_results,
            st.session_state.source_name,
            st.session_state.focus_zone,
        )
        st.download_button(
            "Download point-and-click comparison",
            data=export_text,
            file_name="giap_point_click_comparison.txt",
            mime="text/plain",
        )

with guided_tab:
    guided_provider = st.radio(
        "Guided tutor engine",
        PROVIDERS,
        horizontal=True,
        key="guided_provider",
        help="Pick which engine runs the guided coaching and follow-up chat.",
    )

    st.markdown(
        "Use this when you want the student to describe what they see first, then get coaching that nudges them toward the most diagnostic observations."
    )

    with st.form("guided_start_form", clear_on_submit=False):
        student_observations = st.text_area(
            "Student observations",
            height=140,
            placeholder="Describe what you see before naming it.",
        )
        attempted_name = st.text_input(
            "Student attempted name",
            placeholder="Example: basalt, quartz sandstone, brachiopod, plagioclase, not sure",
        )
        guided_submit = st.form_submit_button("Get guided feedback")

    if guided_submit:
        if not student_observations.strip() or not attempted_name.strip():
            st.warning("Please enter both observations and an attempted name.")
        else:
            try:
                with st.spinner(f"Coaching with {guided_provider}..."):
                    reply = guided_start(
                        provider=guided_provider,
                        sample_type=selected_sample_type,
                        observations=student_observations.strip(),
                        attempted_name=attempted_name.strip(),
                        source_name=source_name,
                        focus_zone=focus_zone,
                    )
                st.session_state.guided_chat_history[guided_provider] = [
                    {
                        "role": "user",
                        "content": (
                            f"Observations: {student_observations.strip()} | "
                            f"Attempted name: {attempted_name.strip()}"
                        ),
                    },
                    {"role": "assistant", "content": reply},
                ]
                st.success("Guided feedback ready.")
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    render_chat_history(
        st.session_state.guided_chat_history.get(guided_provider, []),
        f"Guided conversation ({guided_provider})",
    )

    with st.form("guided_followup_form", clear_on_submit=True):
        guided_followup_text = st.text_area(
            "Continue the guided analysis",
            height=110,
            placeholder="Add a new observation or ask a coaching question.",
        )
        guided_followup_submit = st.form_submit_button("Send guided follow-up")

    if guided_followup_submit:
        if not st.session_state.guided_chat_history.get(guided_provider):
            st.warning("Start guided analysis first.")
        elif not guided_followup_text.strip():
            st.warning("Enter a follow-up first.")
        else:
            try:
                with st.spinner(f"Continuing guided analysis with {guided_provider}..."):
                    reply = guided_followup(
                        provider=guided_provider,
                        sample_type=selected_sample_type,
                        question=guided_followup_text.strip(),
                        source_name=source_name,
                        focus_zone=focus_zone,
                    )
                history = st.session_state.guided_chat_history.get(guided_provider, [])
                history.append({"role": "user", "content": guided_followup_text.strip()})
                history.append({"role": "assistant", "content": reply})
                st.session_state.guided_chat_history[guided_provider] = history
                rerun_app()
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

if st.button("Clear current image conversation"):
    reset_image_dependent_state()
    st.success("Image-specific results and chat cleared.")
