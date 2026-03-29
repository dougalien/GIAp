# GIAp — Clean Refactor (Low Hallucination + Free/Pro Split)

import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# =============================
# CONFIG
# =============================

OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"

# =============================
# PROMPTS (REWRITTEN)
# =============================

SYSTEM_BASE = """
You are a geology lab assistant helping students identify Earth materials.

STRICT RULES:
- Only use what is visually observable in the image.
- If something cannot be seen, say: "not observable".
- Do NOT guess hardness, chemistry, locality, or microscopic features.
- Be cautious. Avoid overconfidence.

Confidence scale:
1 = weak guess
3 = uncertain but plausible
5 = strong visual match
"""

OBSERVER_PROMPT = SYSTEM_BASE + """
ROLE: OBSERVER

Return JSON:
{
  "visible": ["short phrases"],
  "id": "best guess",
  "alt": "alternative",
  "confidence": 1,
  "reason": "short",
  "next": "simple next step"
}
"""

VALIDATOR_PROMPT = SYSTEM_BASE + """
ROLE: VALIDATOR

Be independent. Challenge the observer if needed.

Return JSON:
{
  "id": "best guess",
  "confidence": 1,
  "agreement": "agree / partial / disagree / uncertain",
  "reason": "short",
  "next": "simple next step"
}
"""

JUDGE_PROMPT = SYSTEM_BASE + """
ROLE: JUDGE

Be conservative.

Return JSON:
{
  "final": "best answer",
  "confidence": 1,
  "reply": "2-3 short sentences for a student",
  "next": "clear next step"
}
"""

# =============================
# HELPERS
# =============================

def encode_image(file):
    return base64.b64encode(file.getvalue()).decode()


def call_openai(prompt, image_b64):
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
                    {"type": "text", "text": "Analyze this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )

    return r.json()["choices"][0]["message"]["content"]


def call_claude(prompt, image_b64):
    key = st.secrets["CLAUDE_API_KEY"]

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 800,
        "system": prompt,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                    }
                }
            ]
        }]
    }

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01"
        },
        json=payload
    )

    return r.json()["content"][0]["text"]

# =============================
# UI
# =============================

st.set_page_config(page_title="GIAp", layout="wide")

st.title("🪨 GIAp — Earth Material Identifier")

# --- Free vs Pro toggle ---
mode = st.sidebar.radio("Tier", ["Free", "Pro"])

# --- Upload ---
file = st.file_uploader("Upload specimen image", type=["png","jpg"])

# --- Free UI ---
if mode == "Free":
    st.write("Simple mode: quick answer")

# --- Pro UI ---
if mode == "Pro":
    st.sidebar.subheader("Advanced")
    student_guess = st.sidebar.text_input("Your guess")
    notes = st.sidebar.text_area("Notes")

# --- Run ---
if file:
    st.image(file, use_container_width=True)

    if st.button("Analyze"):
        img_b64 = encode_image(file)

        observer = call_openai(OBSERVER_PROMPT, img_b64)

        if mode == "Free":
            st.success(observer)

        else:
            validator = call_claude(VALIDATOR_PROMPT, img_b64)
            judge = call_openai(JUDGE_PROMPT, img_b64)

            st.subheader("Final Answer")
            st.success(judge)

            with st.expander("Observer"):
                st.write(observer)

            with st.expander("Validator"):
                st.write(validator)
