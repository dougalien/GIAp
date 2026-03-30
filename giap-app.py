import streamlit as st
import requests
import base64
import json
from PIL import Image
import html

# =============================
# CONFIG
# =============================

OPENAI_MODEL = "gpt-4o"

# =============================
# PROMPTS
# =============================

SYSTEM = """
You are a geology lab assistant.

Rules:
- Only use visible features.
- If unsure, say "not observable".
- Be concise.
- Do not invent unseen properties.
"""

OBSERVER = SYSTEM + """
Return JSON:
{
  "id": "",
  "confidence": 1,
  "reason": "",
  "next": ""
}
"""

LESSON_PROMPT = """
Create a short geology lesson plan based on this material identification.

Material: {id}
Reason: {reason}

Include:
- Learning objective
- Activity
- Assessment question
- Time estimate
"""

# =============================
# HELPERS
# =============================

def encode_image(file) -> str:
    return base64.b64encode(file.getvalue()).decode()

def call_openai_json(prompt: str, img_b64: str | None = None) -> str:
    key = st.secrets["OPENAI_API_KEY"]

    content = [{"type": "text", "text": "Analyze."}]
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=60
    )

    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def call_openai_text(prompt: str) -> str:
    key = st.secrets["OPENAI_API_KEY"]

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": "You are a helpful geology education assistant. Be concise and clear."},
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=60
    )

    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def parse_observation(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return None

def render_audio_button(text_to_speak: str):
    safe_text = html.escape(text_to_speak).replace("\n", " ")
    st.components.v1.html(
        f"""
        <div>
          <button
            onclick="window.speechSynthesis.cancel(); window.speechSynthesis.speak(new SpeechSynthesisUtterance('{safe_text}'));"
            style="
              background:#1f1f1f;
              color:white;
              border:none;
              padding:0.6rem 1rem;
              border-radius:6px;
              cursor:pointer;
              font-size:14px;
            ">
            Play Audio
          </button>
        </div>
        """,
        height=55,
    )

# =============================
# PAGE SETUP
# =============================

st.set_page_config(page_title="GIAp", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
    color: #111111;
}

.stApp {
    background: #F7F7F7;
}

.main-block {
    background: white;
    border: 1px solid #D0D0D0;
    border-radius: 10px;
    padding: 1rem 1rem 0.8rem 1rem;
    margin-bottom: 1rem;
}

.pro-card {
    background: #FAFAFA;
    border: 1px solid #C8C8C8;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.lock-note {
    color: #555555;
    font-size: 0.95rem;
}

.feature-label {
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.small-note {
    color: #444444;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)

# =============================
# SESSION STATE
# =============================

if "obs" not in st.session_state:
    st.session_state.obs = None

if "lesson_plan" not in st.session_state:
    st.session_state.lesson_plan = None

# =============================
# HEADER
# =============================

st.title("GIAp")
st.caption("Earth Material Identifier")

st.markdown("""
<div class="main-block">
<p><strong>Free tier:</strong> Upload an image, analyze it, and receive a concise identification with reasoning and a next step.</p>
<p><strong>Pro tier:</strong> Teaching and workflow tools for instructors and advanced users.</p>
</div>
""", unsafe_allow_html=True)

# =============================
# PRO FEATURES AT TOP
# =============================

with st.expander("Pro Features", expanded=True):
    st.markdown("""
    <div class="pro-card">
      <div class="small-note">
      These features are visible here by design. They are currently locked in the standard app view.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Multiple model analysis**")
        st.button("Use multiple APIs", disabled=True, use_container_width=True, key="pro_multi")
        st.caption("Locked in Pro")

        st.markdown("**Upload lesson materials**")
        st.button("Upload lesson materials", disabled=True, use_container_width=True, key="pro_materials")
        st.caption("Locked in Pro")

    with c2:
        st.markdown("**Generate lesson plan**")
        st.button("Create lesson plan", disabled=True, use_container_width=True, key="pro_lesson")
        st.caption("Locked in Pro")

        st.markdown("**Save results to file**")
        st.button("Save results", disabled=True, use_container_width=True, key="pro_save")
        st.caption("Locked in Pro")

    with c3:
        st.markdown("**Analytics**")
        st.button("View analytics", disabled=True, use_container_width=True, key="pro_analytics")
        st.caption("Locked in Pro")

        st.markdown("**Free updates and added tools**")
        st.button("See roadmap", disabled=True, use_container_width=True, key="pro_updates")
        st.caption("Locked in Pro")

# =============================
# FREE TIER
# =============================

st.subheader("Free Tier")

uploaded_file = st.file_uploader(
    "Upload image of rock, mineral, or fossil",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear image for analysis."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded sample image", use_container_width=True)

    if st.button("Analyze", use_container_width=False):
        try:
            img_b64 = encode_image(uploaded_file)
            raw = call_openai_json(OBSERVER, img_b64)
            obs = parse_observation(raw)

            if not obs:
                st.error("The model returned an unreadable response.")
            else:
                st.session_state.obs = obs
                st.session_state.lesson_plan = None

        except requests.RequestException as e:
            st.error(f"Request error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# =============================
# RESULTS
# =============================

if st.session_state.obs:
    obs = st.session_state.obs

    st.subheader("Result")

    st.markdown("""
    <div class="main-block">
    """, unsafe_allow_html=True)

    st.write(f"**Likely identification:** {obs.get('id', '')}")
    st.write(f"**Confidence:** {obs.get('confidence', '')} out of 5")
    st.write(f"**Reasoning:** {obs.get('reason', '')}")
    st.info(f"Next step: {obs.get('next', '')}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Optional audio only
    st.markdown("#### Optional Audio")
    speech_text = (
        f"Likely identification: {obs.get('id', '')}. "
        f"Confidence: {obs.get('confidence', '')} out of 5. "
        f"Reasoning: {obs.get('reason', '')}. "
        f"Next step: {obs.get('next', '')}."
    )
    render_audio_button(speech_text)

    # Locked pro tools directly under result too
    st.subheader("Locked Pro Tools")

    p1, p2, p3 = st.columns(3)

    with p1:
        st.button("Use multiple APIs", disabled=True, use_container_width=True, key="result_multi")
        st.button("Upload lesson materials", disabled=True, use_container_width=True, key="result_materials")

    with p2:
        st.button("Create lesson plan", disabled=True, use_container_width=True, key="result_lesson")
        st.button("Save results", disabled=True, use_container_width=True, key="result_save")

    with p3:
        st.button("View analytics", disabled=True, use_container_width=True, key="result_analytics")
        st.button("See roadmap", disabled=True, use_container_width=True, key="result_roadmap")

# =============================
# OPTIONAL DEVELOPER TEST AREA
# =============================

with st.expander("Developer Test Area", expanded=False):
    enable_dev = st.checkbox("Enable working versions of selected Pro tools")

    if enable_dev:
        st.markdown("This section is for testing only. It is not part of the student-facing locked experience.")

        if st.session_state.obs:
            obs = st.session_state.obs

            col_a, col_b = st.columns(2)

            with col_a:
                if st.button("Generate lesson plan now", use_container_width=True):
                    prompt = LESSON_PROMPT.format(
                        id=obs.get("id", ""),
                        reason=obs.get("reason", "")
                    )
                    try:
                        st.session_state.lesson_plan = call_openai_text(prompt)
                    except requests.RequestException as e:
                        st.error(f"Request error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

            with col_b:
                st.download_button(
                    "Download current result as JSON",
                    data=json.dumps(obs, indent=2),
                    file_name="gia_result.json",
                    use_container_width=True
                )

            if st.session_state.lesson_plan:
                st.markdown("#### Lesson Plan")
                st.write(st.session_state.lesson_plan)
        else:
            st.write("Analyze an image first to test the lesson plan and save tools.")