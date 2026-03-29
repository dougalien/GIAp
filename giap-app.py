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

st.set_page_config(layout="wide")
st.title("🪨 GIAp")

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
            st.markdown("### Answer")
            st.write(f"**Likely:** {obs['id']}")
            st.write(f"Confidence: {obs['confidence']}/5")
            st.write(obs['reason'])
            st.info(f"Next step: {obs['next']}")

        else:
            val = call_claude(VALIDATOR, b64)
            judge = call_openai(JUDGE, b64)

            st.markdown("### Final Answer")
            st.success(judge['reply'])
            st.metric("Confidence", judge['confidence'])
            st.info(f"Next: {judge['next']}")

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
        st.write(item)

# =============================
# SAVE LOG
# =============================

if mode == "Pro" and st.session_state.chat:
    st.download_button(
        "Download session",
        data=json.dumps(st.session_state.chat, indent=2),
        file_name="gia_session.json"
    )
