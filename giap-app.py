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
- Objective
- Activity
- Assessment
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
# UI
# =============================

st.set_page_config(layout="wide", page_title="GIAp")

st.markdown("## 🪨 GIAp")
st.caption("Earth Material Identifier")

# Toggle for demo
is_pro = st.sidebar.toggle("Pro Mode (demo)", value=False)

# =============================
# FREE SECTION
# =============================

st.markdown("### Identify Sample")

file = st.file_uploader(
    "Upload image (rock, mineral, fossil)",
    type=["png", "jpg", "jpeg"]
)

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

        # ===== FREE RESULT =====
        st.markdown("### Result")

        st.write(f"**Likely ID:** {obs['id']}")
        st.write(f"**Confidence:** {obs['confidence']}/5")
        st.write(obs["reason"])
        st.info(f"Next step: {obs['next']}")

        # =============================
        # PRO PANEL (VISIBLE ALWAYS)
        # =============================

        st.markdown("---")
        st.markdown("### 🔒 Pro Tools")

        col1, col2 = st.columns(2)

        # MULTI MODEL
        with col1:
            if is_pro:
                if st.button("Run Multi-Model Validation"):
                    st.info("Coming soon")
            else:
                st.button("Run Multi-Model Validation", disabled=True)
                st.caption("Pro feature")

        # LESSON PLAN
        with col2:
            if is_pro:
                if st.button("Generate Lesson Plan"):
                    prompt = LESSON_PROMPT.format(
                        id=obs['id'],
                        reason=obs['reason']
                    )
                    lesson = call_openai(prompt)
                    st.markdown("#### Lesson Plan")
                    st.write(lesson)
            else:
                st.button("Generate Lesson Plan", disabled=True)
                st.caption("Pro feature")

        # SAVE
        if is_pro:
            st.download_button(
                "Download Results",
                data=json.dumps(obs, indent=2),
                file_name="gia_result.json"
            )
        else:
            st.button("Download Results", disabled=True)
            st.caption("Pro feature")

        # FUTURE FEATURES
        st.markdown("### 🚀 Coming Soon")
        st.markdown("""
        - Class analytics  
        - Adaptive quizzes  
        - Assignment builder  
        """)
