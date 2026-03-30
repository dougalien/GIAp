import streamlit as st
import requests
import base64
import json
from PIL import Image

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
# UI SETUP (ACCESSIBLE)
# =============================

st.set_page_config(layout="wide", page_title="GIAp")

st.markdown("""
<style>
body {
    color: #111;
}

h1, h2, h3 {
    color: #111;
}

.block {
    background: #FFFFFF;
    border: 1px solid #DDD;
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
}

.locked {
    opacity: 0.5;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================

st.title("🪨 GIAp")
st.write("Earth Material Identifier")

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
    st.image(img, caption="Uploaded sample", use_container_width=True)

    if st.button("Analyze"):
        b64 = encode(file)
        raw = call_openai(OBSERVER, b64)

        try:
            obs = json.loads(raw)
        except:
            st.error("Model error")
            st.stop()

        st.markdown("### Result")

        st.write(f"**Likely ID:** {obs['id']}")
        st.write(f"**Confidence:** {obs['confidence']}/5")
        st.write(obs["reason"])
        st.info(f"Next step: {obs['next']}")

# =============================
# PRO FEATURES (VISIBLE + LOCKED)
# =============================

if obs:
    st.markdown("---")
    st.markdown("## 🔒 Pro Features")

    col1, col2 = st.columns(2)

    # LESSON PLAN (WORKING)
    with col1:
        st.markdown("### 📘 Lesson Plan")
        st.button("Generate Lesson Plan", disabled=True)
        st.caption("Available in Pro")

    # SAVE RESULTS (WORKING)
    with col2:
        st.markdown("### 💾 Save Results")
        st.button("Download Results", disabled=True)
        st.caption("Available in Pro")

    # =============================
    # OPTIONAL: TURN ON REAL FEATURES FOR TESTING
    # =============================

    with st.expander("Developer Test (enable features)"):
        enable = st.checkbox("Enable Pro Features")

        if enable:

            st.markdown("### 📘 Lesson Plan")

            if st.button("Generate Lesson Plan (Active)"):
                prompt = LESSON_PROMPT.format(
                    id=obs['id'],
                    reason=obs['reason']
                )
                lesson = call_openai(prompt)

                st.write(lesson)

            st.markdown("### 💾 Save Results")

            st.download_button(
                "Download JSON",
                data=json.dumps(obs, indent=2),
                file_name="gia_result.json"
            )

            st.markdown("### 🔍 Multi-Model")
            if st.button("Run Multi-Model (placeholder)"):
                st.info("Will connect OpenAI + Claude")

# =============================
# FUTURE FEATURES
# =============================

if obs:
    st.markdown("### 🚀 Coming Soon")

    st.markdown("""
    - Class analytics  
    - Adaptive quizzes  
    - Assignment builder  
    - Lab report generator  
    """)
