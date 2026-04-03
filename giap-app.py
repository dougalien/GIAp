import base64
import hashlib
import io
import json
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="GIAp", layout="centered")

CONFIG = {
    "title": "GIAp",
    "subtitle": "Geology image analysis by We are dougalien",
    "website": "www.dougalien.com",
    "image_label": "rock, mineral, fossil, sediment, thin section, or related sample image",
    "analyst_role": "careful geology image analyst and supportive geology tutor",
    "model": "gpt-4o-mini",
    "timeout": 60,
    "max_image_size": 1400,
}

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

Return valid JSON only:
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
- Guide the student toward a better answer without shaming them.
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


def init_state() -> None:
    defaults = {
        "analysis": None,
        "analysis_error": "",
        "source_name": "",
        "focus_zone": "Full image",
        "last_image_b64": None,
        "last_image_signature": "",
        "last_context_signature": "",
        "selected_sample_type": "Auto-detect",
        "point_chat_history": [],
        "guided_chat_history": [],
        "guided_last_reply": "",
        "analysis_calls_used": 0,
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
        return st.secrets.get(name, default)
    except Exception:
        return default



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
        "candidate": str(data.get("candidate", "")).strip(),
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
    st.session_state.analysis = None
    st.session_state.analysis_error = ""
    st.session_state.point_chat_history = []
    st.session_state.guided_chat_history = []
    st.session_state.guided_last_reply = ""



def ensure_quota() -> None:
    limit = get_session_limit()
    used = st.session_state.analysis_calls_used
    if limit > 0 and used >= limit:
        raise RuntimeError(
            f"This session has reached its AI call limit ({used}/{limit}). "
            f"Increase MAX_AI_CALLS_PER_SESSION in secrets or set it to 0 for unlimited use."
        )



def increment_quota() -> None:
    st.session_state.analysis_calls_used += 1



def call_openai(messages: List[Dict[str, Any]], json_mode: bool = False, temperature: float = 0.2) -> str:
    api_key = get_secret("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    ensure_quota()

    payload: Dict[str, Any] = {
        "model": CONFIG["model"],
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
    increment_quota()
    return response.json()["choices"][0]["message"]["content"]



def build_multimodal_user_message(text: str, image_b64: str) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ],
    }



def run_point_analysis(image_b64: str, sample_type: str, source_name: str, focus_zone: str) -> Dict[str, Any]:
    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    prompt = (
        f"User-selected sample type: {sample_type}.\n"
        f"Source name: {source_name or 'upload'}.\n"
        f"Focus zone: {focus_zone}.\n"
        f"Sample-type guidance: {guidance}\n\n"
        "Analyze this image and identify the most likely geologic material or material group. "
        "Return cautious observations, a concise explanation, and one same-context next thing to look at or think about."
    )
    content = call_openai(
        [
            {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
            build_multimodal_user_message(prompt, image_b64),
        ],
        json_mode=True,
        temperature=0.1,
    )
    parsed = safe_json_loads(content)
    if not parsed:
        raise RuntimeError("Model returned unreadable JSON.")
    return normalize_analysis_result(parsed)



def render_chat_history(history: List[Dict[str, str]], title: str) -> None:
    if not history:
        return
    st.markdown(f"### {title}")
    for item in history:
        if item.get("role") == "user":
            st.markdown(f"**You:** {item.get('content', '')}")
        else:
            st.markdown(f"**AI:** {item.get('content', '')}")



def build_point_followup_messages(question: str) -> List[Dict[str, Any]]:
    if not st.session_state.last_image_b64:
        raise RuntimeError("Upload and analyze an image first.")

    result = st.session_state.analysis or {}
    context_text = (
        f"Existing point-and-click result:\n"
        f"Sample type: {result.get('sample_type', '')}\n"
        f"Likely identification: {result.get('candidate', '')}\n"
        f"Alternate: {result.get('alternate', '')}\n"
        f"Confidence: {result.get('confidence', '')}/5\n"
        f"Observations: {', '.join(result.get('observations', []))}\n"
        f"Why: {result.get('why', '')}\n"
        f"Limits: {result.get('limits', '')}\n"
        f"Next look: {result.get('next_look', '')}\n\n"
        f"Answer this follow-up question about the same image: {question}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT},
        build_multimodal_user_message(context_text, st.session_state.last_image_b64),
    ]
    for item in st.session_state.point_chat_history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    return messages



def build_guided_start_messages(
    image_b64: str,
    sample_type: str,
    observations: str,
    attempted_name: str,
    source_name: str,
    focus_zone: str,
) -> List[Dict[str, Any]]:
    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    prompt = (
        f"User-selected sample type: {sample_type}.\n"
        f"Source name: {source_name or 'upload'}.\n"
        f"Focus zone: {focus_zone}.\n"
        f"Sample-type guidance: {guidance}\n\n"
        f"Student observations: {observations}\n"
        f"Student attempted name: {attempted_name}\n\n"
        "Coach the student supportively. Point out what they are doing well, guide them toward the most relevant features to focus on, "
        "and end with one concrete next observation or question."
    )
    return [
        {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
        build_multimodal_user_message(prompt, image_b64),
    ]



def build_guided_followup_messages(question: str, sample_type: str) -> List[Dict[str, Any]]:
    if not st.session_state.last_image_b64:
        raise RuntimeError("Upload an image first.")

    guidance = SAMPLE_TYPE_GUIDANCE.get(sample_type, SAMPLE_TYPE_GUIDANCE["Auto-detect"])
    intro = (
        f"Continue the guided tutoring conversation for the same image.\n"
        f"User-selected sample type: {sample_type}.\n"
        f"Sample-type guidance: {guidance}\n"
        f"Student follow-up: {question}"
    )
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
        build_multimodal_user_message(intro, st.session_state.last_image_b64),
    ]
    for item in st.session_state.guided_chat_history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": question})
    return messages



def export_point_text(result: Dict[str, Any], source_name: str, focus_zone: str, history: List[Dict[str, str]]) -> str:
    obs_text = "\n".join(f"- {item}" for item in result.get("observations", []))
    chat_lines = []
    for item in history:
        speaker = "You" if item.get("role") == "user" else "AI"
        chat_lines.append(f"{speaker}: {item.get('content', '')}")
    chat_text = "\n".join(chat_lines)
    return (
        f"{CONFIG['title']}\n"
        f"Source: {source_name or 'upload'}\n"
        f"Focus area: {focus_zone}\n\n"
        f"Sample type: {result.get('sample_type', '')}\n"
        f"Likely identification: {result.get('candidate', '')}\n"
        f"Alternate: {result.get('alternate', '')}\n"
        f"Confidence: {result.get('confidence', '')}/5\n\n"
        f"Visible observations:\n{obs_text}\n\n"
        f"Why this fit: {result.get('why', '')}\n"
        f"Limits: {result.get('limits', '')}\n"
        f"Next thing to look at: {result.get('next_look', '')}\n\n"
        f"Follow-up chat:\n{chat_text}\n"
    )



def show_image_compat(image: Image.Image, caption: str) -> None:
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        try:
            st.image(image, caption=caption, use_column_width=True)
        except TypeError:
            st.image(image, caption=caption)


st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 17px;
    }
    .stApp {
        background: #F7F8FA;
        color: #111111;
    }
    .card {
        background: white;
        border: 1px solid #D7DCE2;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.02rem;
        color: #2F3B48;
        margin-bottom: 0.25rem;
    }
    .site {
        color: #2B5C88;
        font-size: 0.96rem;
    }
    .small {
        color: #47515D;
        font-size: 0.96rem;
    }
    div.stButton > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        min-height: 48px;
        border-radius: 10px;
        font-weight: 700;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div class="card">
        <div class="title">{CONFIG['title']}</div>
        <div class="subtitle">{CONFIG['subtitle']}</div>
        <div class="site">{CONFIG['website']}</div>
        <p class="small" style="margin-top:0.8rem;">
            One app, two modes: quick point-and-click interpretation and guided student coaching.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Accessible, phone-friendly layout. Use camera or upload. The app keeps the image context for follow-up questions."
)

limit = get_session_limit()
used = st.session_state.analysis_calls_used
if limit > 0:
    st.caption(f"AI calls this session: {used}/{limit}")
else:
    st.caption(f"AI calls this session: {used} | session limit: unlimited")

camera_file = st.camera_input("Take a photo")
uploaded_file = st.file_uploader(
    f"Or upload a {CONFIG['image_label']}",
    type=["png", "jpg", "jpeg"],
    help="On most phones, this can use the camera, photo library, or files.",
)

image_bytes: Optional[bytes] = None
source_name = ""
if camera_file is not None:
    image_bytes = camera_file.getvalue()
    source_name = getattr(camera_file, "name", "camera_capture.jpg")
elif uploaded_file is not None:
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
        show_image_compat(base_image, "Source image")
        focused_image = crop_by_zone(base_image, focus_zone)
        if focus_zone != "Full image":
            show_image_compat(focused_image, f"Focused view: {focus_zone}")
        st.session_state.last_image_b64 = image_to_b64(focused_image)
    except Exception as exc:
        st.error(f"Image error: {exc}")
        st.stop()
else:
    st.warning("Add an image to begin.")
    st.stop()


def render_analysis_result(result: Dict[str, Any]) -> None:
    st.markdown("### Point-and-click result")
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{result.get('confidence', '')}/5")
    c2.metric("Sample type", result.get("sample_type", ""))
    st.write(f"**AI summary:** {result.get('candidate', '')}")
    st.write(f"**Alternate possibility:** {result.get('alternate', '')}")
    st.write("**Visible observations**")
    for item in result.get("observations", []):
        st.write(f"- {item}")
    st.write(f"**Why this fits:** {result.get('why', '')}")
    st.write(f"**Limits:** {result.get('limits', '')}")
    st.write(f"**Next thing to look at or think about:** {result.get('next_look', '')}")


point_tab, guided_tab = st.tabs(["Point and Click", "Guided Analysis"])

with point_tab:
    st.markdown(
        "Use this when you want a quick, careful interpretation plus one same-context suggestion for what to examine next."
    )
    if st.button("Analyze image", key="analyze_image_button"):
        st.session_state.analysis_error = ""
        try:
            with st.spinner("Analyzing image..."):
                st.session_state.analysis = run_point_analysis(
                    st.session_state.last_image_b64,
                    selected_sample_type,
                    source_name,
                    focus_zone,
                )
            st.success("Analysis complete.")
        except requests.RequestException as exc:
            st.session_state.analysis_error = f"Request error: {exc}"
        except Exception as exc:
            st.session_state.analysis_error = f"Unexpected error: {exc}"

    if st.session_state.analysis_error:
        st.error(st.session_state.analysis_error)

    if st.session_state.analysis:
        render_analysis_result(st.session_state.analysis)
        render_chat_history(st.session_state.point_chat_history, "Point-and-click follow-up")

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
                    with st.spinner("Answering follow-up..."):
                        messages = build_point_followup_messages(point_question.strip())
                        reply = call_openai(messages, json_mode=False, temperature=0.2)
                    st.session_state.point_chat_history.append({"role": "user", "content": point_question.strip()})
                    st.session_state.point_chat_history.append({"role": "assistant", "content": reply.strip()})
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Request error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

        export_text = export_point_text(
            st.session_state.analysis,
            st.session_state.source_name,
            st.session_state.focus_zone,
            st.session_state.point_chat_history,
        )
        st.download_button(
            "Download point-and-click result",
            data=export_text,
            file_name="giap_point_click_result.txt",
            mime="text/plain",
        )

with guided_tab:
    st.markdown(
        "Use this when you want the student to make observations and attempt a name first, then get supportive coaching based on the same image."
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
                with st.spinner("Coaching response in progress..."):
                    messages = build_guided_start_messages(
                        st.session_state.last_image_b64,
                        selected_sample_type,
                        student_observations.strip(),
                        attempted_name.strip(),
                        source_name,
                        focus_zone,
                    )
                    reply = call_openai(messages, json_mode=False, temperature=0.3)
                st.session_state.guided_chat_history = [
                    {
                        "role": "user",
                        "content": (
                            f"Observations: {student_observations.strip()} | "
                            f"Attempted name: {attempted_name.strip()}"
                        ),
                    },
                    {"role": "assistant", "content": reply.strip()},
                ]
                st.session_state.guided_last_reply = reply.strip()
                st.success("Guided feedback ready.")
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

    render_chat_history(st.session_state.guided_chat_history, "Guided conversation")

    with st.form("guided_followup_form", clear_on_submit=True):
        guided_followup = st.text_area(
            "Continue the guided analysis",
            height=110,
            placeholder="Add a new observation or ask a coaching question.",
        )
        guided_followup_submit = st.form_submit_button("Send guided follow-up")

    if guided_followup_submit:
        if not st.session_state.guided_chat_history:
            st.warning("Start guided analysis first.")
        elif not guided_followup.strip():
            st.warning("Enter a follow-up first.")
        else:
            try:
                with st.spinner("Continuing guided analysis..."):
                    messages = build_guided_followup_messages(guided_followup.strip(), selected_sample_type)
                    reply = call_openai(messages, json_mode=False, temperature=0.3)
                st.session_state.guided_chat_history.append({"role": "user", "content": guided_followup.strip()})
                st.session_state.guided_chat_history.append({"role": "assistant", "content": reply.strip()})
                rerun_app()
            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")

if st.button("Clear current image conversation"):
    reset_image_dependent_state()
    st.success("Image-specific results and chat cleared.")
