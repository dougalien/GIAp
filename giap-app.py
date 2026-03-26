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
        "display_messages": [],
        "api_history": [],
        "mode": "Auto",
        "context_notes": "",
        "specimen_label": "",
        "model": DEFAULT_MODEL,
        "last_uploaded_signature": None,
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
        "Auto": """
Decide which domain best fits the specimen: rock, mineral, fossil, sand/granular sediment, soil, or forensic particulate.
If the domain is unclear, say so explicitly and explain what visible evidence would help.
""",
        "Rock": """
Focus on rock identification. Prioritize texture, grain size, clast vs crystalline texture, sorting, rounding, layering, vesicles, foliation, cement, and matrix.
Avoid overclaiming composition when the image does not support it.
""",
        "Mineral": """
Focus on mineral identification. Prioritize color, luster, transparency, habit, cleavage/fracture clues, crystal form, and likely hardness implications if visible.
Avoid claiming a mineral species unless the image evidence is strong.
""",
        "Fossil": """
Focus on fossil identification. Prioritize symmetry, segmentation, ornamentation, shell geometry, visible structures, preservation style, and likely fossil group.
Avoid forcing a genus/species ID from weak evidence.
""",
        "Sand/Granular": """
Focus on sand, grains, sediment, soil particles, or particulate forensic-style material.
Comment when possible on grain size class, sorting, roundness/angularity, sphericity, transparency/opacity, luster, quartz likelihood, feldspar clues, lithic fragments, heavy minerals, organic fragments, and what cannot be determined from this image alone.
Do not call it a powder or crystal substance unless the image clearly supports that language.
""",
        "Forensic": """
Focus on trace material or forensic-style particulate evidence.
Describe visible particle classes, shape variation, color variation, transparency, possible natural vs manufactured particles, contamination risk, and what follow-up observations are needed before any strong claim.
Be especially conservative.
""",
    }

    return f"""
You are a conversational geology tutor for an introductory college teaching app in 2026.

General rules:
- Sound like a patient lab instructor, not a script. Vary your wording and examples.
- Distinguish direct observation from interpretation and keep observations honest, even if they do not fully support the instructor's known name.
- Be useful, specific, cautious, and friendly.
- Do not overclaim. Admit uncertainty and change your mind if new information appears.
- When discussing geology, focus on the actual descriptive features students should observe.
- Teach the student how to look, not just what to conclude.
- If the evidence is weak, offer a small number of plausible interpretations and explain why.
- Keep responses compact (about 4–8 sentences) and clearly tied to THIS particular image and chat turn.
- Whenever it helps, explicitly connect your explanation to what the student just said.
- Avoid repeating the same examples or sentence openings you used earlier in this conversation.
- Use the full image for overall context and scale, and any zoomed image(s) to inspect fine details like textures, grain boundaries, cleavage, or fossils.
- If the student asks for a summary or evaluation, provide it in a friendly, concise way that validates what they did well and gives specific next steps.

Response style:
- Answer in 1–2 natural-sounding paragraphs.
- Finish with exactly one short, open-ended question to keep the conversation going (unless the student explicitly asks for a summary or says they are done).

Domain instructions:
{mode_guidance.get(mode, mode_guidance["Auto"])}
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
        raise RuntimeError("Please upload an image first.")

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

# 1. Sample name (maps to student_best_answer)
st.session_state.student_best_answer = st.text_input(
    "Sample name (your best guess)",
    value=st.session_state.student_best_answer,
    placeholder="e.g., sandstone, basalt, quartz, shell fossil",
)

st.markdown("---")

# 2. Take or upload photo (same pattern as GIA)
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

# 3. Analyze sample button (calls the same engine as GIA)
start_disabled = st.session_state.image_data_uri is None

if st.button("Analyze sample", type="primary", use_container_width=True, disabled=start_disabled):
    try:
        start_first_analysis()
        st.rerun()
    except Exception as e:
        st.error(str(e))

st.markdown("---")

# 4. Output (shows the latest AI + history summary style)
st.subheader("Output")

if not st.session_state.display_messages:
    st.info("After you take a photo and tap **Analyze sample**, feedback will appear here.")
else:
    # Show only the last assistant message as the main output
    last_assistant = [m for m in st.session_state.display_messages if m["role"] == "assistant"]
    if last_assistant:
        st.markdown(last_assistant[-1]["content"])
    else:
        st.markdown(st.session_state.display_messages[-1]["content"])

st.markdown("---")

# 5. Follow-up chat (uses same send_followup as GIA, but with a single input)
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

# 7. Advanced (instructor / power user) at bottom
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