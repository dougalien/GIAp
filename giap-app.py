import os
import base64
import mimetypes
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = "https://api.perplexity.ai/chat/completions"
DEFAULT_MODEL = "sonar-pro"

# ---------- Shared engine (GIAp) ----------

def init_state():
    defaults = {
        "started": False,
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "last_uploaded_signature": None,
        "display_messages": [],
        "model": DEFAULT_MODEL,
        # Tutoring context
        "mode": "Auto",
        "context_notes": "",
        "specimen_label": "",
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "student_name": "",
        # First reply anchor
        "first_reply": "",
        # Follow-up UI
        "clear_followup_next": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def ensure_image_data_uri():
    if st.session_state.get("image_bytes") and not st.session_state.get("image_data_uri"):
        try:
            mime = st.session_state.get("image_mime") or "image/png"
            b64 = base64.b64encode(st.session_state.image_bytes).decode("utf-8")
            st.session_state.image_data_uri = f"data:{mime};base64,{b64}"
        except Exception:
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.image_mime = None
            st.session_state.image_data_uri = None


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

    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.image_name = uploaded_file.name
    st.session_state.image_bytes = raw
    st.session_state.image_mime = mime
    st.session_state.image_data_uri = data_uri


def get_image_contents_for_api():
    if not st.session_state.image_bytes:
        return []

    try:
        img = Image.open(BytesIO(st.session_state.image_bytes))
        img.thumbnail((768, 768))

        buf = BytesIO()
        fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "JPEG"
        img.save(buf, format=fmt, quality=80)
        resized_bytes = buf.getvalue()

        b64 = base64.b64encode(resized_bytes).decode("utf-8")
        mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(fmt, "image/jpeg")
        data_uri = f"data:{mime};base64,{b64}"
    except Exception:
        data_uri = st.session_state.image_data_uri
        if not data_uri:
            return []

    return [
        {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }
    ]


def build_system_prompt(mode: str):
    mode_guidance = {
        "Auto": "Decide if this looks like a rock, mineral, fossil, sand/grain mix, soil, or tiny forensic-style particles.",
        "Rock": "Focus on rock ID: grain size, textures, clast vs crystalline, layers, vesicles, foliation if visible.",
        "Mineral": "Focus on mineral ID: color, luster, transparency, crystal shapes, and obvious cleavage/fracture.",
        "Fossil": "Focus on fossil ID: shape, symmetry, segments, shell patterns, and preservation style.",
        "Sand/Granular": "Focus on sand or grains: grain size, sorting, rounding, color mix, and visible fragments.",
        "Forensic": "Focus on tiny particles: shapes, colors, and how mixed they look. Stay very cautious.",
    }

    return f"""
You are a friendly, concise geology tutor for an intro lab.

For each specimen image, your job is:
1) First sentence: say clearly if the student's name is probably right, close, or not a good match for what you SEE.
2) Second sentence: give 1–2 very short reasons based only on visible features (color, grain size, texture, layering, etc.).
3) Third sentence: one short suggestion for a better photo or a simple next check (e.g., look for layering, check grain size, zoom in on crystals).

Rules:
- Maximum 3 short sentences total. No paragraphs, no bullet lists.
- Use plain words; assume a 100-level course.
- Be honest and cautious. If the ID is uncertain, say that clearly.
- Stay grounded in the image. Do not invent properties you cannot see (like hardness or streak).
- If an instructor's known name is provided and the photo is ambiguous, treat that as the best label and use the image as support for teaching.

Mode hint: {mode_guidance.get(mode, mode_guidance["Auto"])}
""".strip()


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

    response = requests.post(API_URL, headers=headers, json=payload, timeout=90)

    if response.status_code != 200:
        raise RuntimeError(
            f"Perplexity error {response.status_code}: {response.text[:1000]}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def build_initial_messages(user_content_text: str):
    return [
        {"role": "system", "content": build_system_prompt(st.session_state.mode)},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_content_text},
                *get_image_contents_for_api(),
            ],
        },
    ]


def start_first_analysis():
    if st.session_state.image_bytes and not st.session_state.image_data_uri:
        st.warning(
            "Image is still loading. Wait until the preview appears, then tap Analyze again."
        )
        return

    if not st.session_state.image_bytes:
        st.warning("Please upload a specimen photo first, then tap Analyze.")
        return

    label = st.session_state.specimen_label.strip() or "[no label]"
    notes = st.session_state.context_notes.strip() or "[no extra notes]"
    student_name = st.session_state.student_name.strip() or "[no name provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"
    mode = st.session_state.mode

    user_text = f"""
Look at this geology specimen photo for an intro lab at Salem State.

Student's current name for the specimen: {best_answer}
Instructor's known name (if any): {known_name}
Specimen label / sample ID: {label}
Student observations (if any): {observations}
Extra context notes: {notes}
Student name (optional, use casually at most once): {student_name}
Current mode: {mode}

Follow the 3-sentence format from your system instructions.
Keep it short, clear, and focused on what a beginning student can actually see.
If the photo is ambiguous, say so and lean on the instructor's known name as the best label.
""".strip()

    messages = build_initial_messages(user_text)
    reply = call_perplexity(messages)

    st.session_state.first_reply = reply
    st.session_state.display_messages = [
        {"role": "assistant", "content": reply}
    ]
    st.session_state.started = True


def send_followup(user_text: str):
    user_text = user_text.strip()
    if not user_text:
        return

    earlier = st.session_state.get("first_reply", "[no earlier reply stored]")
    label = st.session_state.specimen_label.strip() or "[no label]"
    notes = st.session_state.context_notes.strip() or "[no extra notes]"
    student_name = st.session_state.student_name.strip() or "[no name provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"
    mode = st.session_state.mode

    messages = [
        {"role": "system", "content": build_system_prompt(mode)},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
You are continuing a geology tutoring session about ONE specimen.

Here is your earlier full answer (do NOT deny writing this, even if it was wrong):
\"\"\"{earlier}\"\"\"

Context:
- Mode: {mode}
- Specimen label / ID: {label}
- Student's best name: {best_answer}
- Instructor's known name (if given): {known_name}
- Student observations: {observations}
- Extra notes: {notes}
- Student name (optional): {student_name}

Student's follow-up:
{user_text}

Instructions:
- If your earlier answer clashes with the instructor's known name or the student's evidence, clearly admit your earlier guess was likely wrong and explain briefly why.
- Never claim you did not say something that appears in the earlier answer above.
- Keep the reply short (2–4 short sentences), focused on what the student can see and how they can double-check in the lab.
""".strip(),
                }
            ],
        },
    ]

    st.session_state.display_messages.append({"role": "user", "content": user_text})

    reply = call_perplexity(messages)

    st.session_state.display_messages.append({"role": "assistant", "content": reply})


# ---------- GIAp UI (phone-friendly, anchored) ----------

st.set_page_config(
    page_title="GIAp: Guided Image Analyzer",
    page_icon="🪨",
    layout="centered",
)

init_state()
ensure_image_data_uri()

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .gia-header {
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .gia-subtle {
        color: #555555;
        font-size: 0.9rem;
    }
    .stButton>button {
        border-radius: 999px;
        border: 1px solid #444444;
        background-color: #f3f3f3;
        color: #111111;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #e0e0e0;
        border-color: #222222;
    }
    .gia-section {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        border: 1px solid #dddddd;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="gia-header">GIAp: Guided Image Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="gia-subtle">(Image-based tutor by We are dougalien — anchored, short feedback)</div>',
    unsafe_allow_html=True,
)
st.caption("Give the sample a name, take a photo, tap Analyze, then ask a short follow-up if you want.")

st.markdown("")

# 1. Sample info
st.markdown("### 1. Sample info")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    st.session_state.student_best_answer = st.text_input(
        "Sample name (your best guess)",
        value=st.session_state.student_best_answer,
        placeholder="e.g., sandstone, basalt, quartz, shell fossil",
    )

    st.session_state.specimen_label = st.text_input(
        "Specimen label / sample ID (optional)",
        value=st.session_state.specimen_label,
        placeholder="e.g., Lab 3 sample A",
    )

    st.session_state.student_observations = st.text_area(
        "Your quick observations (optional)",
        value=st.session_state.student_observations,
        height=80,
        placeholder="e.g., light-colored, sand-sized grains, some shell bits, looks layered...",
    )

    st.markdown("</div>", unsafe_allow_html=True)

# 2. Specimen photo
st.markdown("### 2. Specimen photo")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Tap here to take a photo or choose from your library",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=False,
        help="Use a clear, close-up photo. Avoid heavy glare when possible.",
    )

    if uploaded_file is not None:
        update_uploaded_image(uploaded_file)
        st.success("Image loaded. Scroll down to analyze.")

    if st.session_state.image_bytes:
        st.image(
            st.session_state.image_bytes,
            caption=st.session_state.image_name or "Specimen image",
            use_container_width=True,
        )
    else:
        st.info("No specimen image yet. Take or choose a photo above.")

    st.markdown("</div>", unsafe_allow_html=True)

# 3. Analyze
st.markdown("### 3. Quick analyze")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    if st.button("Analyze sample", use_container_width=True):
        try:
            with st.spinner("Looking at the specimen..."):
                start_first_analysis()
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

# 4. Tutor feedback
st.markdown("### 4. Tutor feedback")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    if not st.session_state.display_messages:
        st.info("After you add a photo and tap **Analyze sample**, feedback will appear here.")
    else:
        last_assistant = [
            m for m in st.session_state.display_messages if m["role"] == "assistant"
        ]
        if last_assistant:
            st.markdown(last_assistant[-1]["content"])
        else:
            st.markdown(st.session_state.display_messages[-1]["content"])

    st.markdown("</div>", unsafe_allow_html=True)

# 5. Follow-up question (anchored, text-only)
st.markdown("### 5. Follow-up question")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    if not st.session_state.started:
        st.info("Analyze at least one specimen before asking a follow-up question.")
    else:
        if st.session_state.get("clear_followup_next", False):
            st.session_state["followup_text"] = ""
            st.session_state["clear_followup_next"] = False

        followup = st.text_input(
            "Ask a quick question (optional)",
            key="followup_text",
            placeholder="e.g., What should I look at next to improve my ID?",
        )

        if st.button("Send follow-up", use_container_width=True):
            try:
                text = st.session_state.get("followup_text", "").strip()
                if text:
                    with st.spinner("Thinking..."):
                        send_followup(text)
                    st.session_state["clear_followup_next"] = True
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

# 6. Reset
st.markdown("### 6. Reset")
with st.container():
    st.markdown('<div class="gia-section">', unsafe_allow_html=True)

    if st.button("Clear GIAp", use_container_width=True):
        reset_app()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# 7. Advanced
with st.expander("Advanced (instructor / power user)", expanded=False):
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
    st.session_state.context_notes = st.text_area(
        "Short context notes (optional)",
        value=st.session_state.context_notes,
        height=80,
        placeholder="e.g., hand lens view, indoor light, scale bar present, wet surface, etc.",
    )
    st.session_state.known_name = st.text_input(
        "Known name from instructor (optional)",
        value=st.session_state.known_name,
        placeholder="e.g., coarse sandstone, garnet schist, gypsum",
    )
    st.session_state.student_name = st.text_input(
        "Student name (optional)",
        value=st.session_state.student_name,
        placeholder="e.g., Alex",
    )