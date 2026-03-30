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
- Only use visible features
- If unsure say "not observable"
- Be concise
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
Create a short geology lesson plan.

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

def encode(file):
    return base64.b64encode(file.getvalue()).decode()

def call_openai(prompt, img=None):
    key = st.secrets["OPENAI_API_KEY"]

    content = [{"type": "text", "text": "Analyze."}]

    if img:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )

    return r.json()["choices"][0]["message"]["content"]

# =============================
# ACCESSIBLE STYLES
# =============================

st.set_page_config(layout="wide", page_title="GIAp")

st.markdown("""
<style>
body {
    color: #111;
    background-color: #FFFFFF;
    font-size: 16px;
}

h1, h2, h3 {
    color: #111;
}

.section {
    border: 1px solid #CCCCCC;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #FFFFFF;
}

.locked {
    opacity: 0.5;
}

button:disabled {
    background-color: #E0E0E0 !important;
    color: #666 !important;
}

</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================

st.title("GIAp")
st.caption("Earth Material Identifier")

# =============================
# FREE TIER
# =============================

st.markdown("## Identify Sample")

file = st.file_uploader(
    "Upload image (rock, mineral, fossil)",
    type=["png", "jpg", "jpeg"]
)

obs = None

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded sample image", use_container_width=True)

    if st.button("Analyze"):
        b64 = encode(file)
        raw = call_openai(OBSERVER, b64)

        try:
            obs = json.loads(raw)
        except:
            st.error("Model response error")
            st.stop()

        # =============================
        # RESULT
        # =============================

        st.markdown("## Result")

        st.write(f"Likely identification: {obs['id']}")
        st.write(f"Confidence: {obs['confidence']} out of 5")
        st.write(obs["reason"])
        st.info(f"Next step: {obs['next']}")

        # =============================
        # TEXT-TO-SPEECH (ACCESSIBILITY)
        # =============================

        speech_text = f"""
        Likely identification: {obs['id']}.
        Confidence: {obs['confidence']} out of 5.
        {obs['reason']}.
        Next step: {obs['next']}.
        """

        safe_text = html.escape(speech_text)

        st.markdown("### Audio Output")

        st.components.v1.html(f"""
        <button onclick="speechSynthesis.speak(new SpeechSynthesisUtterance('{safe_text}'))">
            Play Audio
        </button>
        """, height=50)

# =============================
# PRO FEATURES (VISIBLE)
# =============================

if obs:
    st.markdown("---")
    st.markdown("## Pro Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Lesson plan generator")
        st.button("Generate lesson plan", disabled=True)
        st.caption("Available in Pro")

    with col2:
        st.markdown("Save results")
        st.button("Download results", disabled=True)
        st.caption("Available in Pro")

    # =============================
    # DEV ENABLE (FOR YOU ONLY)
    # =============================

    with st.expander("Enable Pro Features (developer)"):
        enable = st.checkbox("Enable")

        if enable:

            st.markdown("### Lesson Plan")

            if st.button("Generate lesson plan (active)"):
                prompt = LESSON_PROMPT.format(
                    id=obs['id'],
                    reason=obs['reason']
                )
                lesson = call_openai(prompt)
                st.write(lesson)

            st.markdown("### Download")

            st.download_button(
                "Download JSON",
                data=json.dumps(obs, indent=2),
                file_name="gia_result.json"
            )

# =============================
# FUTURE FEATURES
# =============================

if obs:
    st.markdown("## Planned Features")

    st.markdown("""
    - Class analytics  
    - Adaptive quizzes  
    - Assignment builder  
    - Lab report generator  
    """)