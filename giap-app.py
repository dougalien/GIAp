# GIAp — Improved UI + Pro Features (Readable Output, Zoom, Chat, Save Logs)

import streamlit as st
import requests
import base64
import json
from datetime import datetime
from PIL import Image
from io import BytesIO

# =============================
# CONFIG
# =============================

OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# =============================
# PROMPTS (SAFE + CLEAN)
# =============================

SYSTEM = """
You are a geology lab assistant.

RULES:
- Only use visible features.
- If not visible → say "not observable".
- Do NOT guess chemistry, hardness, or unseen properties.
- Be concise.

Confidence:
1 = weak guess
3 = uncertain
5 = strong match
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

VALIDATOR = SYSTEM + """
Return JSON:
{
 "id": "",
 "confidence": 1,
 "agreement": "agree/partial/disagree",
 "reason": "",
 "next": ""
}
"""

JUDGE = SYSTEM + """
Return JSON:
{
 "final": "",
 "confidence": 1,
 "reply": "short student answer",
 "next": ""
}
"""

# =============================
# HELPERS
# =============================

def encode(file):
    return base64.b64encode(file.getvalue()).decode()


def call_openai(prompt, img):
    key = st.secrets["OPENAI_API_KEY"]

    payload = {
        "model": OPENAI_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                ]
            }
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )

    return json.loads(r.json()["choices"][0]["message"]["content"])


def call_claude(prompt, img):
    key = st.secrets["CLAUDE_API_KEY"]

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 800,
        "system": prompt,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}}
            ]
        }]
    }

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        json=payload
    )

    return json.loads(r.json()["content"][0]["text"])

# =============================
# STATE
# =============================

if "chat" not in st.session_state:
    st.session_state.chat = []

# =============================
# UI
# =============================

st.set_page_config(layout="wide", page_title="GIAp")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #F7F4EF 0%, #EDE6DA 100%);
    color: #1F2933;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #E5DED0 0%, #D8CFBF 100%);
}

.main-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid #C8BFAF;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 1rem;
}

.answer-box {
    background: #F8FBF8;
    border-left: 6px solid #2F6B5F;
    border-radius: 12px;
    padding: 1rem;
    color: #1F2933;
}

.next-box {
    background: #F6F1E8;
    border-left: 6px solid #8C6A43;
    border-radius: 12px;
    padding: 0.9rem;
    color: #1F2933;
}

.chat-user {
    background: #EAF2F8;
    border-radius: 12px;
    padding: 0.75rem;
    margin: 0.4rem 0;
    border: 1px solid #B8C7D9;
}

.chat-ai {
    background: #EEF5EE;
    border-radius: 12px;
    padding: 0.75rem;
    margin: 0.4rem 0;
    border: 1px solid #B9D1BF;
}

.soft-label {
    font-size: 0.84rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #5B6570;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <h1 style="margin-bottom:0.2rem;">🪨 GIAp</h1>
    <div style="color:#5B6570;">Earth Material Identifier</div>
</div>
""", unsafe_allow_html=True)

# Tier
mode = st.sidebar.radio("Tier", ["Free", "Pro"])

# Pro settings
if mode == "Pro":
    st.sidebar.subheader("Pro Controls")
    zoom = st.sidebar.slider("Zoom", 1.0, 3.0, 1.0)
    user_name = st.sidebar.text_input("User name")

# Upload
file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])

# Display image with zoom
if file:
    img = Image.open(file)

    if mode == "Pro":
        w, h = img.size
        crop = img.crop((w*(1-1/zoom)/2, h*(1-1/zoom)/2, w*(1+1/zoom)/2, h*(1+1/zoom)/2))
        st.image(crop, use_container_width=True)
        img_to_send = crop
    else:
        st.image(img, use_container_width=True)
        img_to_send = img

    if st.button("Analyze"):
        b64 = encode(file)

        obs = call_openai(OBSERVER, b64)

        if mode == "Free":
            st.markdown("""
<div class="main-card">
    <div class="soft-label">Answer</div>
    <div class="answer-box">
        <div><strong>Likely ID:</strong> """ + str(obs['id']) + """</div>
        <div style="margin-top:0.35rem;"><strong>Confidence:</strong> """ + str(obs['confidence']) + """/5</div>
        <div style="margin-top:0.6rem;">""" + str(obs['reason']) + """</div>
    </div>
    <div class="next-box" style="margin-top:0.8rem;"><strong>Next step:</strong> """ + str(obs['next']) + """</div>
</div>
""", unsafe_allow_html=True)

        else:
            val = call_claude(VALIDATOR, b64)
            judge = call_openai(JUDGE, b64)

            st.markdown("""
<div class="main-card">
    <div class="soft-label">Final Answer</div>
    <div class="answer-box">""" + str(judge['reply']) + """</div>
</div>
""", unsafe_allow_html=True)
            st.metric("Confidence", judge['confidence'])
            st.markdown("""
<div class="next-box"><strong>Next step:</strong> """ + str(judge['next']) + """</div>
""", unsafe_allow_html=True)

            with st.expander("Details"):
                st.json(obs)
                st.json(val)

            # Save to chat
            st.session_state.chat.append({
                "time": str(datetime.now()),
                "user": user_name,
                "result": judge
            })

# =============================
# CHAT
# =============================

if mode == "Pro":
    st.markdown("---")
    st.subheader("Follow-up Chat")

    msg = st.text_input("Ask a question")

    if st.button("Send"):
        st.session_state.chat.append({"user_msg": msg})

    for item in st.session_state.chat:
        if 'user_msg' in item:
            st.markdown(f"<div class='chat-user'><strong>You:</strong> {item['user_msg']}</div>", unsafe_allow_html=True)
        elif 'result' in item:
            reply = item['result'].get('reply', '')
            st.markdown(f"<div class='chat-ai'><strong>GIAp:</strong> {reply}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>{item}</div>", unsafe_allow_html=True)

# =============================
# SAVE LOG
# =============================

if mode == "Pro" and st.session_state.chat:
    st.download_button(
        "Download session",
        data=json.dumps(st.session_state.chat, indent=2),
        file_name="gia_session.json"
    )

