import streamlit as st
import requests
import base64
import json
from datetime import datetime
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

RULES:
- Only use visible features.
- If not visible → say "not observable".
- Be concise.
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
Create a short geology lesson plan based on this identification:

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
# UI STYLES (ACCESSIBLE)
# =============================

def apply_styles():
    st.markdown("""
    <style>
    .main-card {
        background: #FFFFFF;
        border: 1px solid #CCCCCC;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .answer-box {
        background: #FFFFFF;
        border-left: 5px solid #2F6B5F;
        padding: 1rem;
        color: #111111;
    }

    .locked {
        opacity: 0.5;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================
# COMPONENTS
# =============================

def render_header():
    st.markdown("""
    <div class="main-card">
        <h1>🪨 GIAp</h1>
        <div>Earth Material Identifier</div>
    </div>
    """, unsafe_allow_html=True)

def render_upload():
    return st.file_uploader("Upload image (rock, mineral, fossil)", type=["png","jpg","jpeg"])

def render_image(img):
    st.image(img, caption="Uploaded sample", use_container_width=True)

def render_free(obs):
    st.markdown(f"""
    <div class="main-card">
        <div class="answer-box">
            <strong>Likely ID:</strong> {obs['id']}<br>
            <strong>Confidence:</strong> {obs['confidence']}/5<br><br>
            {obs['reason']}
        </div>
        <div><strong>Next step:</strong> {obs['next']}</div>
    </div>
    """, unsafe_allow_html=True)

def render_pro_locked():
    st.markdown("""
    <div class="main-card locked">
        <strong>🔒 Pro Features</strong>
        <ul>
            <li>Multi-model validation</li>
            <li>Zoom analysis</li>
            <li>Follow-up chat</li>
            <li>Session saving</li>
            <li>Lesson plan generator</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_lesson_plan(obs, is_pro):
    if is_pro:
        if st.button("Generate Lesson Plan"):
            prompt = LESSON_PROMPT.format(id=obs['id'], reason=obs['reason'])
            lesson = call_openai(prompt)

            st.markdown("""
            <div class="main-card">
            <strong>Lesson Plan</strong>
            """, unsafe_allow_html=True)

            st.write(lesson)
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="main-card locked">
            🔒 Lesson Plan Generator (Pro)
        </div>
        """, unsafe_allow_html=True)

# =============================
# MAIN APP
# =============================

def main():
    st.set_page_config(page_title="GIAp", layout="wide")
    apply_styles()
    render_header()

    is_pro = st.sidebar.checkbox("Enable Pro (demo)")

    file = render_upload()

    if file:
        img = Image.open(file)
        render_image(img)

        if st.button("Analyze"):
            b64 = encode(file)
            obs_raw = call_openai(OBSERVER, b64)

            try:
                obs = json.loads(obs_raw)
            except:
                st.error("Model response error")
                return

            render_free(obs)

            if is_pro:
                st.success("Pro features unlocked")
            else:
                render_pro_locked()

            render_lesson_plan(obs, is_pro)

            render_future()

def render_future():
    st.markdown("""
    <div class="main-card locked">
        <strong>Coming Soon</strong>
        <ul>
            <li>Class analytics</li>
            <li>Adaptive quizzes</li>
            <li>Lab report generator</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================
# RUN
# =============================

if __name__ == "__main__":
    main()