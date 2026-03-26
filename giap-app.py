import os
import base64
import mimetypes
from io import BytesIO
from datetime import datetime

import requests
import streamlit as st
from PIL import Image

API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar-pro"

# ---------- State & helpers ----------

def init_state():
    defaults = {
        "started": False,
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "last_uploaded_signature": None,
        "last_ai_message": "",
        # Simple student flow
        "student_guess": "",
        "student_nickname": "",
        # Hidden / instructor
        "mode": "Auto",
        "model": DEFAULT_MODEL,
        "specimen_label": "",
        "context_notes": "",
        "include_auto_zoom": True,
        "zoom_fraction": 0.5,
        # Geo (stubbed for now)
        "latitude": None,
        "longitude": None,
        "location_accuracy_m": None,
        # API history (minimal, kept for possible future use)
        "api_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_app():
    keep_model = st.session_state.get("model", DEFAULT_MODEL)
    keep_mode = st.session_state.get("mode", "Auto")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    st.session_state.model = keep_model
    st.session_state.mode = keep_mode


def get_api_key():
    return os.getenv("PERPLEXITY_API_KEY")


def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type
    if not mime:
        mime = mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return raw, mime, data_uri


def update_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return

    signature = (uploaded_file.name, uploaded_file.size)
    if st.session_state.last_uploaded_signature == signature:
        return

    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.image_name = uploaded_file.name
    st.session_state.image_bytes = raw
    st.session_state.image_mime = mime
    st.session_state.image_data_uri = data_uri
    st.session_state.last_uploaded_signature = signature


def get_image_contents_for_api():
    if not st.session_state.image_bytes or not st.session_state.image_data_uri:
        return []

    contents = [
        {
            "type": "image_url",
            "image_url": {"url": st.session_state.image_data_uri},
        }
    ]

    if not st.session_state.include_auto_zoom:
        return contents

    try:
        img = Image.open(BytesIO(st.session_state.image_bytes))
        w, h = img.size
        frac = st.session_state.zoom_fraction
        frac = max(0.1, min(frac, 1.0))

        cw, ch = int(w * frac), int(h * frac)
        left = (w - cw) // 2
        top = (h - ch) // 2
        right = left + cw
        bottom = top + ch
        crop_center = img.crop((left, top, right, bottom))

        buf = BytesIO()
        fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "PNG"
        crop_center.save(buf, format=fmt)
        crop_bytes = buf.getvalue()
        b64 = base64.b64encode(crop_bytes).decode("utf-8")
        mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(fmt, "image/png")
        crop_data_uri = f"data:{mime};base64,{b64}"

        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": crop_data_uri},
            }
        )
    except Exception:
        pass

    return contents


def build_system_prompt(mode):
    mode_guidance = {
        "Auto": """
Decide which domain best fits the specimen: rock, mineral, fossil, sand/granular sediment, soil, or forensic particulate.
If the domain is unclear, say so explicitly and explain what visible evidence would help.
""",
        "Rock": """
Focus on rock identification with simple, student-friendly language. Mention texture, grain size, layering, vesicles, and foliation only if they are clearly visible.
""",
        "Mineral": """
Focus on mineral identification using easy terms: color, shine, transparency, and crystal shape. Do not overclaim species without strong visual evidence.
""",
        "Fossil": """
Focus on fossil identification in simple language: shell or body shape, symmetry, segments, and obvious patterns.
""",
        "Sand/Granular": """
Focus on sand or grains in simple terms: grain size, round vs sharp edges, and overall mix of light vs dark grains.
""",
        "Forensic": """
Focus on describing tiny particles in everyday language, being very cautious about any strong identification.
""",
    }

    return f"""
You are a very concise, friendly geology helper for an introductory lab app in 2026.

Your job:
- Look at each image and give just a few clear observations that a beginner can see.
- Compare those observations to the student's suggested name.
- If the student's name is probably right, say "Congratulations" or something similar, then briefly explain 1–3 reasons why it fits.
- If the name is probably wrong or too specific, say kindly that it may not match and suggest 1–2 visible reasons why, in simple language.
- Always keep the tone supportive and short. Avoid long paragraphs.
- Suggest exactly one simple next step if the image is hard to read (e.g., better light, different angle, add a scale like a coin or ruler).
- Do not talk like a research paper. Use short sentences and plain words.
- Do not list more than about 4 short sentences total, or 3 short bullet-style lines.

Student reading and typing should be minimal. Focus on what they can actually see.

Domain instructions:
{mode_guidance.get(mode, mode_guidance["Auto"])}
""".strip()


def build_api_messages(user_prompt_text):
    messages = [{"role": "system", "content": build_system_prompt(st.session_state.mode)}]
    content = [{"type": "text", "text": user_prompt_text}]
    images = get_image_contents_for_api()
    content.extend(images)
    messages.append({"role": "user", "content": content})
    return messages


def call_perplexity(messages):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.model,
        "messages": messages,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=180)

    if response.status_code != 200:
        raise RuntimeError(
            f"Perplexity error {response.status_code}: {response.text[:2000]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def capture_location_stub():
    st.info("Location tagging not available in this build.")


def start_simple_analysis():
    if not st.session_state.image_data_uri:
        raise RuntimeError("Please take or upload an image first.")

    captured_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lat = st.session_state.latitude
    lon = st.session_state.longitude
    acc = st.session_state.location_accuracy_m

    if lat is not None and lon is not None:
        location_line = f"Approximate location: {lat:.5f}, {lon:.5f} (±{acc or 0:.0f} m)"
    else:
        location_line = "Approximate location: [not captured]"

    student_guess = st.session_state.student_guess.strip() or "[no name entered]"
    nickname = st.session_state.student_nickname.strip()

    label = st.session_state.specimen_label.strip() or "[none]"
    notes = st.session_state.context_notes.strip() or "[none]"

    user_prompt_text = f"""
Student suggested name: {student_guess}
Student nickname (if given; you may use it once): {nickname or "[none]"}
Mode: {st.session_state.mode}
Specimen label (instructor / lab use): {label}
Capture time (approx): {captured_time}
{location_line}
Short notes from instructor (if any): {notes}

Please:
- Say in a few short sentences if the student's name seems close or not.
- If close, say congratulations and give 1–3 simple reasons.
- If not close, gently say so and give 1–2 reasons and one suggestion for a better photo or view.
Keep everything very short and friendly.
""".strip()

    messages = build_api_messages(user_prompt_text)
    reply = call_perplexity(messages)

    st.session_state.last_ai_message = reply
    st.session_state.started = True


# ---------- Streamlit UI: GIAp (point-and-click student helper) ----------

st.set_page_config(
    page_title="GIAp: Point & Click Geology Helper",
    page_icon="📱",
    layout="centered",
)
init_state()

st.title("📱 GIAp: Point & Click Geology Helper")
st.caption("Aim at a specimen, tap once, and get a short geology hint.")

with st.expander("Instructor / advanced options", expanded=False):
    st.session_state.model = st.text_input(
        "Perplexity model",
        value=st.session_state.model,
        help="Leave as-is for normal student use.",
    )
    st.session_state.mode = st.selectbox(
        "Specimen mode",
        ["Auto", "Rock", "Mineral", "Fossil", "Sand/Granular", "Forensic"],
        index=[
            "Auto",
            "Rock",
            "Mineral",
            "Fossil",
            "Sand/Granular",
            "Forensic",
        ].index(st.session_state.mode)
        if st.session_state.mode
        in ["Auto", "Rock", "Mineral", "Fossil", "Sand/Granular", "Forensic"]
        else 0,
        help="Use Auto for most lab work.",
    )
    st.session_state.include_auto_zoom = st.checkbox(
        "Send a zoomed-in center crop",
        value=st.session_state.include_auto_zoom,
        help="Helps the AI see fine textures.",
    )
    st.session_state.zoom_fraction = st.slider(
        "Zoom size (fraction of image)",
        min_value=0.2,
        max_value=0.8,
        value=float(st.session_state.zoom_fraction),
        step=0.1,
    )
    st.session_state.specimen_label = st.text_input(
        "Specimen label / sample ID (optional)",
        value=st.session_state.specimen_label,
        placeholder="e.g., Lab 3 sample A",
    )
    st.session_state.context_notes = st.text_area(
        "Short context notes (optional)",
        value=st.session_state.context_notes,
        height=80,
        placeholder="e.g., hand lens view, indoor light, no scale bar",
    )

st.markdown("---")

# 1. Analyze button at the top
st.subheader("Tap to analyze")

if st.button(
    "📷 Analyze sample",
    type="primary",
    use_container_width=True,
):
    try:
        if not st.session_state.image_data_uri:
            st.warning("First, take a photo or choose one below, then tap again.")
        else:
            start_simple_analysis()
            st.experimental_rerun()
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# 2. Current specimen and name
st.subheader("Current specimen")

if st.session_state.image_bytes:
    st.image(
        st.session_state.image_bytes,
        caption=st.session_state.image_name or "Specimen image",
        use_container_width=True,
    )
else:
    st.info("No specimen yet. Take or choose a photo below.")

st.subheader("Your best name for this specimen")
st.session_state.student_guess = st.text_input(
    "Sample name (your best guess)",
    value=st.session_state.student_guess,
    placeholder="e.g., sandstone, basalt, quartz, shell fossil",
)

with st.expander("Optional: your nickname", expanded=False):
    st.session_state.student_nickname = st.text_input(
        "Nickname (the helper may use this once)",
        value=st.session_state.student_nickname,
        placeholder="e.g., Alex",
    )

st.markdown("---")

# 3. Photo uploader, then (stub) location
st.subheader("Take or choose a photo")

uploaded_file = st.file_uploader(
    "Tap to take a photo or choose from your library",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    update_uploaded_image(uploaded_file)
    st.experimental_rerun()

st.markdown("#### Optional: tag this specimen with your location")

if st.button("📍 Get my location", use_container_width=True):
    try:
        capture_location_stub()
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# 4. Feedback
st.subheader("Quick feedback")

if not st.session_state.last_ai_message:
    st.info("Take a photo, enter your best name, then tap **Analyze sample**.")
else:
    st.markdown(st.session_state.last_ai_message)
    st.markdown("")
    st.caption("Try another angle, better light, or a scale bar, then tap again.")

st.markdown("---")

if st.button("Reset GIAp", use_container_width=True):
    reset_app()
    st.experimental_rerun()