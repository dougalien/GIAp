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

# ---------- Shared engine with GIA ----------

def init_state():
    defaults = {
        "started": False,
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "last_uploaded_signature": None,
        "display_messages": [],
        "api_history": [],
        "mode": "Auto",
        "context_notes": "",
        "specimen_label": "",
        "model": DEFAULT_MODEL,
        # Tutoring flow
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "student_name": "",
        # Zoom options
        "include_auto_zoom": False,
        "zoom_fraction": 0.5,
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
        "Auto": "Decide if this looks like a rock, mineral, fossil, sand/grain mix, soil, or tiny forensic-style particles.",
        "Rock": "Focus on rock ID: grain size, textures, layers, vesicles, foliation if clearly visible.",
        "Mineral": "Focus on mineral ID: color, luster, transparency, obvious crystal shapes or cleavage.",
        "Fossil": "Focus on fossil ID: shape, symmetry, segments, shell patterns that are easy to see.",
        "Sand/Granular": "Focus on sand or grains: grain size, round vs sharp edges, light vs dark mix.",
        "Forensic": "Focus on tiny particles: shapes, colors, and how mixed they look. Stay very cautious.",
    }

    return f"""
You are a very short, friendly geology coach for an intro lab.

Your job each time:
- First line: clearly say if the student's name is probably right, close, or not a good match.
- Then give 1–2 very short reasons based only on what a beginner can SEE in the image.
- Finish with exactly one very short suggestion for what photo change or extra check would help next.

Rules:
- Total response: at most 3 short sentences.
- Use plain words, no big jargon.
- Never write a full paragraph or a list.
- Be kind and encouraging, like a quick game result, not a long lesson.

Mode hint: {mode_guidance.get(mode, mode_guidance["Auto"])}
""".strip()

def build_api_messages():
    messages = [{"role": "system", "content": build_system_prompt(st.session_state.mode)}]

    for item in st.session_state.api_history:
        if item["role"] == "user":
            content = [{"type": "text", "text": item["content"]}]
            images = get_image_contents_for_api()
            content.extend(images)
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": item["content"]})

    return messages


def call_perplexity(messages=None):
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY in your environment.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if messages is None:
        messages = build_api_messages()

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


def start_first_analysis():
    if not st.session_state.image_data_uri:
        st.warning("Please upload an image first.")
        return

    label = st.session_state.specimen_label.strip() or "No specimen label provided"
    notes = st.session_state.context_notes.strip() or "No additional notes provided"
    student_name = st.session_state.student_name.strip() or "[no name provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"

    starter_prompt = f"""
Please analyze the uploaded specimen image for a teaching app.

Selected mode: {st.session_state.mode}
Specimen label: {label}
Student/instructor notes: {notes}
Student name (if given, use occasionally in a natural, non-repetitive way): {student_name}
Student observations so far: {observations}
Student best answer so far: {best_answer}
Known name from instructor (if any): {known_name}

Your job:
- Start with observation before interpretation.
- If this is sand or granular material, explicitly address whether the visible grains appear well sorted or poorly sorted, whether quartz is likely, whether lithic grains may be present, and what cannot be determined confidently.
- If the evidence does not support a strong ID, say so clearly.
- Sound conversational and non-repetitive, as if you are talking with the student at the lab bench.
- Use the full image for scale and any zoomed image(s) to inspect textures and fine details.
- End with exactly one open-ended question that invites the student to make or refine an observation.
""".strip()

    visible_user_text = (
        f"Please analyze this uploaded specimen.\n\n"
        f"Mode: {st.session_state.mode}\n"
        f"Label: {label}\n"
        f"Notes: {notes}"
    )

    st.session_state.api_history = [{"role": "user", "content": starter_prompt}]
    st.session_state.display_messages = [
        {"role": "user", "content": visible_user_text}
    ]

    reply = call_perplexity()

    st.session_state.api_history.append({"role": "assistant", "content": reply})
    st.session_state.display_messages.append({"role": "assistant", "content": reply})
    st.session_state.started = True


def send_followup(user_text):
    user_text = user_text.strip()
    if not user_text:
        return

    student_name = st.session_state.student_name.strip() or "[no name provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"

    followup_prompt = f"""
Student follow-up:
{user_text}

Context:
- Student name (if usable, mention naturally at most once per reply): {student_name}
- Mode: {st.session_state.mode}
- Specimen label: {st.session_state.specimen_label or "[none]"}
- Student observations: {observations}
- Student best answer: {best_answer}
- Known name from instructor: {known_name}
- Your earlier messages might include a guess that could be wrong.

Please answer as a conversational geology tutor.
Stay grounded in the uploaded image and the student's words.
If the student provides new observations or corrections, incorporate them honestly.
If the new information or known name conflicts with your earlier idea, politely explain the mismatch and keep your observations honest to the image.
Be concise (about 4–8 sentences), supportive, and vary your phrasing so it does not sound like a template.
When helpful, refer the student to specific parts of the main image or the zoomed view (e.g., "look closely at the zoomed image where the grains touch").
If the student asks for a summary or evaluation, provide it without a follow-up question.
Otherwise, end with exactly one open-ended question that nudges the student toward a next observation or comparison.
""".strip()

    st.session_state.display_messages.append({"role": "user", "content": user_text})
    st.session_state.api_history.append({"role": "user", "content": followup_prompt})

    reply = call_perplexity()

    st.session_state.api_history.append({"role": "assistant", "content": reply})
    st.session_state.display_messages.append({"role": "assistant", "content": reply})


# ---------- GIAp UI: phone-friendly shell ----------

st.set_page_config(
    page_title="GIAp: Guided Image Analyzer",
    page_icon="🪨",
    layout="centered",
)
init_state()

st.title("GIAp: Guided Image Analyzer")
st.caption("(point and click version)")
st.caption("Give the sample a name, take a photo, tap analyze, and chat about what you see.")

# 1. Sample name → student_best_answer
st.session_state.student_best_answer = st.text_input(
    "Sample name (your best guess)",
    value=st.session_state.student_best_answer,
    placeholder="e.g., sandstone, basalt, quartz, shell fossil",
)

st.markdown("---")

# 2. Take or upload photo
st.subheader("Take or upload specimen photo")

uploaded_file = st.file_uploader(
    "Tap here to take a photo or choose from your library",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=False,
)

if uploaded_file is not None:
    update_uploaded_image(uploaded_file)
    st.rerun()

if st.session_state.image_bytes:
    st.image(
        st.session_state.image_bytes,
        caption=st.session_state.image_name or "Specimen image",
        use_container_width=True,
    )
else:
    st.info("No specimen image yet. Take or choose a photo above.")

st.markdown("---")

# 3. Analyze sample (always enabled, warns if no image)
if st.button("Analyze sample", type="primary", use_container_width=True):
    try:
        start_first_analysis()
        st.rerun()
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# 4. Output: show last assistant message
st.subheader("Output")

if not st.session_state.display_messages:
    st.info("After you take a photo and tap **Analyze sample**, feedback will appear here.")
else:
    last_assistant = [m for m in st.session_state.display_messages if m["role"] == "assistant"]
    if last_assistant:
        st.markdown(last_assistant[-1]["content"])
    else:
        st.markdown(st.session_state.display_messages[-1]["content"])

st.markdown("---")

# 5. Follow-up chat (simple one-line input)
st.subheader("Follow-up chat")

if not st.session_state.started:
    st.info("Analyze at least one sample before using follow-up chat.")
else:
    followup = st.text_input(
        "Ask a quick question or add a short comment",
        value="",
        placeholder="e.g., What should I change to make this clearer?",
    )
    if st.button("Send follow-up", use_container_width=True):
        try:
            send_followup(followup)
            st.rerun()
        except Exception as e:
            st.error(str(e))

st.markdown("---")

# 6. Clear app
if st.button("Clear app", use_container_width=True):
    reset_app()
    st.rerun()

st.markdown("---")

# 7. Advanced at bottom
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
    st.session_state.student_name = st.text_input(
        "Student name (optional)",
        value=st.session_state.student_name,
        placeholder="e.g., Alex",
    )