import streamlit as st
import requests
import base64
import json
from PIL import Image
import html
from typing import Any, Dict, Optional, Tuple

# =========================================================
# CONFIG
# =========================================================

OPENAI_MODEL_FAST = "gpt-4o-mini"
OPENAI_MODEL_STRONG = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
GEMINI_MODEL = "gemini-2.0-flash"
PERPLEXITY_MODEL = "sonar-pro"

TIMEOUT = 75

# =========================================================
# PROMPTS
# =========================================================

SYSTEM = """
You are a careful geology image analyst.

Rules:
- Use only visible image evidence.
- Do not infer locality, chemistry, hardness, streak, taste, magnetism, or reaction unless directly visible.
- If something cannot be determined from the image alone, say so.
- Distinguish observation from interpretation.
- Be concise and specific.
""".strip()

PRIMARY_OBSERVER = SYSTEM + """

Return valid JSON only:
{
  "candidate": "best identification or most likely material group",
  "alternate": "brief alternate possibility or 'none'",
  "confidence": 1,
  "observations": ["visible feature 1", "visible feature 2", "visible feature 3"],
  "why": "brief explanation of why the visible evidence supports the candidate",
  "limits": "what cannot be determined from image alone",
  "next_test": "single best real-world follow-up test or observation"
}

Confidence scale:
1 = weak guess
2 = plausible
3 = moderate
4 = strong
5 = very strong
"""

JUDGE_PROMPT = """
You are comparing candidate geological image identifications.
Use only the evidence supplied.
Do not choose the most eloquent answer. Choose the answer best grounded in visible observations.

Return valid JSON only:
{
  "winner": "openai or claude or gemini",
  "why": "short explanation",
  "final_confidence": 1,
  "agreement": "high or moderate or low"
}
""".strip()

LESSON_PROMPT = """
Create a short geology lesson plan based on this material identification.

Material: {candidate}
Why: {why}
Observations: {observations}

Include:
- Learning objective
- Activity
- Assessment question
- Time estimate
""".strip()

# =========================================================
# STATE
# =========================================================

def init_state():
    defaults = {
        "authenticated": False,
        "login_error": "",
        "username": "",
        "obs": None,
        "lesson_plan": None,
        "uploaded_name": None,
        "raw_results": {},
        "selected_mode": "Cost-aware balanced",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================================================
# HELPERS
# =========================================================

def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def get_app_password() -> str:
    return get_secret("APP_PASSWORD", "")


def encode_image(file) -> str:
    return base64.b64encode(file.getvalue()).decode()


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        pass

    # fallback: pull first json object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        return None
    return None


def display_name() -> str:
    name = st.session_state.username.strip()
    return name if name else "Friend"


def available_providers() -> Dict[str, bool]:
    return {
        "openai": bool(get_secret("OPENAI_API_KEY", "")),
        "claude": bool(get_secret("ANTHROPIC_API_KEY", "")),
        "gemini": bool(get_secret("GEMINI_API_KEY", "")),
        "perplexity": bool(get_secret("PERPLEXITY_API_KEY", "")),
    }


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
              padding:0.65rem 1rem;
              border-radius:8px;
              cursor:pointer;
              font-size:14px;
              font-weight:600;
            ">
            Play Audio
          </button>
        </div>
        """,
        height=58,
    )


# =========================================================
# API CALLS
# =========================================================

def call_openai_json(prompt: str, img_b64: Optional[str] = None, model: str = OPENAI_MODEL_FAST) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    content = [{"type": "text", "text": "Analyze this geology image carefully."}]
    if img_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_openai_text(prompt: str, model: str = OPENAI_MODEL_FAST) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You are a concise geology education assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_claude_json(prompt: str, img_b64: str) -> str:
    key = get_secret("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in Streamlit secrets.")

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 900,
        "temperature": 0.1,
        "system": prompt,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Analyze this geology image and return only the requested JSON.",
                },
            ],
        }],
    }

    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    parts = data.get("content", [])
    return "".join(p.get("text", "") for p in parts if p.get("type") == "text")


def call_gemini_json(prompt: str, img_b64: str) -> str:
    key = get_secret("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    payload = {
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json"
        },
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": img_b64
                    }
                }
            ]
        }]
    }

    r = requests.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def call_perplexity_text(prompt: str) -> str:
    key = get_secret("PERPLEXITY_API_KEY", "")
    if not key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY in Streamlit secrets.")

    payload = {
        "model": PERPLEXITY_MODEL,
        "temperature": 0.0,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise geology helper. Use only provided content. Do not browse."
            },
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        },
        json=payload,
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =========================================================
# ROUTING LOGIC
# =========================================================

def normalize_result(data: Dict[str, Any], provider: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "candidate": str(data.get("candidate", "")).strip(),
        "alternate": str(data.get("alternate", "none")).strip(),
        "confidence": int(data.get("confidence", 1) or 1),
        "observations": data.get("observations", []) if isinstance(data.get("observations", []), list) else [],
        "why": str(data.get("why", "")).strip(),
        "limits": str(data.get("limits", "")).strip(),
        "next_test": str(data.get("next_test", "")).strip(),
    }


def provider_pass(provider: str, img_b64: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        if provider == "openai":
            raw = call_openai_json(PRIMARY_OBSERVER, img_b64, OPENAI_MODEL_FAST)
        elif provider == "claude":
            raw = call_claude_json(PRIMARY_OBSERVER, img_b64)
        elif provider == "gemini":
            raw = call_gemini_json(PRIMARY_OBSERVER, img_b64)
        else:
            return None, f"Unsupported provider: {provider}"

        parsed = safe_json_loads(raw)
        if not parsed:
            return None, f"{provider} returned unreadable JSON"
        return normalize_result(parsed, provider), None
    except Exception as e:
        return None, f"{provider}: {e}"


def build_judge_input(results: Dict[str, Dict[str, Any]]) -> str:
    packed = {
        name: {
            "candidate": r.get("candidate"),
            "alternate": r.get("alternate"),
            "confidence": r.get("confidence"),
            "observations": r.get("observations"),
            "why": r.get("why"),
            "limits": r.get("limits"),
            "next_test": r.get("next_test"),
        }
        for name, r in results.items()
    }
    return json.dumps(packed, indent=2)


def judge_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    judge_payload = (
        "Compare these candidate interpretations of the same geology image.\n\n"
        f"{build_judge_input(results)}"
    )
    raw = call_openai_json(JUDGE_PROMPT + "\n\n" + judge_payload, None, OPENAI_MODEL_STRONG)
    parsed = safe_json_loads(raw)
    if not parsed:
        return {"winner": "openai", "why": "Judge fallback used.", "final_confidence": 2, "agreement": "low"}
    return parsed


def build_consensus(primary: Dict[str, Any], secondary: Optional[Dict[str, Any]], judge: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if secondary is None:
        return {
            "candidate": primary.get("candidate", ""),
            "alternate": primary.get("alternate", "none"),
            "confidence": primary.get("confidence", 1),
            "observations": primary.get("observations", []),
            "why": primary.get("why", ""),
            "limits": primary.get("limits", ""),
            "next_test": primary.get("next_test", ""),
            "agreement": "single-model",
            "winner": primary.get("provider", "openai"),
        }

    if judge:
        winner = judge.get("winner", primary.get("provider", "openai"))
        chosen = primary if primary.get("provider") == winner else secondary
        other = secondary if chosen is primary else primary

        combined_observations = chosen.get("observations", [])[:]
        for item in other.get("observations", []):
            if item not in combined_observations and len(combined_observations) < 5:
                combined_observations.append(item)

        return {
            "candidate": chosen.get("candidate", ""),
            "alternate": chosen.get("alternate", "none"),
            "confidence": judge.get("final_confidence", chosen.get("confidence", 1)),
            "observations": combined_observations,
            "why": chosen.get("why", ""),
            "limits": chosen.get("limits", ""),
            "next_test": chosen.get("next_test", ""),
            "agreement": judge.get("agreement", "moderate"),
            "winner": winner,
        }

    return {
        "candidate": primary.get("candidate", ""),
        "alternate": primary.get("alternate", "none"),
        "confidence": primary.get("confidence", 1),
        "observations": primary.get("observations", []),
        "why": primary.get("why", ""),
        "limits": primary.get("limits", ""),
        "next_test": primary.get("next_test", ""),
        "agreement": "moderate",
        "winner": primary.get("provider", "openai"),
    }


def analyze_image(img_b64: str, mode: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    providers = available_providers()
    details: Dict[str, Any] = {"errors": [], "raw_results": {}, "mode": mode}

    if not providers["openai"]:
        raise RuntimeError("OPENAI_API_KEY is required because OpenAI is used as the primary router/judge.")

    openai_result, err = provider_pass("openai", img_b64)
    if err:
        details["errors"].append(err)
        raise RuntimeError("Primary OpenAI vision pass failed.")
    details["raw_results"]["openai"] = openai_result

    secondary_result = None
    judge = None

    if mode == "Lowest cost":
        return build_consensus(openai_result, None, None), details

    secondary_provider = None
    if mode == "Cost-aware balanced":
        if providers["gemini"]:
            secondary_provider = "gemini"
        elif providers["claude"]:
            secondary_provider = "claude"
    elif mode == "Max accuracy":
        if providers["claude"]:
            secondary_provider = "claude"
        elif providers["gemini"]:
            secondary_provider = "gemini"

    if secondary_provider:
        secondary_result, err = provider_pass(secondary_provider, img_b64)
        if err:
            details["errors"].append(err)
        elif secondary_result:
            details["raw_results"][secondary_provider] = secondary_result
            judge = judge_results({
                "openai": openai_result,
                secondary_provider: secondary_result,
            })
            details["judge"] = judge

    consensus = build_consensus(openai_result, secondary_result, judge)

    # optional explanation polish using Perplexity only as a text simplifier, not identifier
    if mode == "Max accuracy" and providers["perplexity"]:
        try:
            explain_prompt = (
                "Rewrite this geology explanation more clearly for a student. "
                "Do not add any new facts. Keep it under 90 words.\n\n"
                f"Candidate: {consensus['candidate']}\n"
                f"Observations: {consensus['observations']}\n"
                f"Why: {consensus['why']}\n"
                f"Limits: {consensus['limits']}"
            )
            polished = call_perplexity_text(explain_prompt)
            if polished:
                consensus["why"] = polished.strip()
                details["used_perplexity_for_explanation"] = True
        except Exception as e:
            details["errors"].append(f"perplexity explanation pass: {e}")

    return consensus, details


# =========================================================
# STYLES
# =========================================================

st.set_page_config(page_title="GIA", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
    color: #111111;
}

.stApp {
    background: #F5F6F8;
}

.main-card {
    background: #FFFFFF;
    border: 1px solid #D0D4DA;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.pro-card {
    background: #FAFAFA;
    border: 1px solid #C9CDD3;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.input-box {
    background: #EEF4FB;
    border: 1px solid #C9D8EA;
    border-radius: 10px;
    padding: 1rem;
    color: #111111;
}

.output-box {
    background: #F3F8F1;
    border: 1px solid #C9D8C4;
    border-radius: 10px;
    padding: 1rem;
    color: #111111;
}

.small-note {
    color: #4D5661;
    font-size: 0.95rem;
}

.brand-line {
    font-size: 1.02rem;
    color: #222222;
    margin-top: -0.2rem;
}

.brand-link {
    color: #2B5C88;
    font-size: 0.95rem;
    margin-top: 0.2rem;
}

.section-label {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}

.box-label {
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

div.stButton > button {
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
    border: 1px solid #BFC6CF;
}

div.stButton > button:disabled {
    background: #E7EAEE !important;
    color: #666666 !important;
    border: 1px solid #C7CCD3 !important;
}

div[data-baseweb="input"] > div,
textarea, input {
    border-radius: 8px !important;
}

hr {
    border: none;
    border-top: 1px solid #D8DCE2;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOGIN
# =========================================================

def render_login():
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("""
        <div class="main-card" style="margin-top:3rem;">
            <div class="section-label">1. Sign In</div>
            <h1 style="margin-bottom:0.15rem;">GIA</h1>
            <div class="brand-line"><strong>G</strong>uided <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
            <div class="brand-link">www.dougalien.com</div>
            <p class="small-note" style="margin-top:0.9rem;">
                Enter the app password to continue.
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            entered = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Enter", use_container_width=True)

        if submitted:
            actual = get_app_password()
            if not actual:
                st.session_state.login_error = "APP_PASSWORD is missing from Streamlit secrets."
            elif entered == actual:
                st.session_state.authenticated = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect password."

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

# =========================================================
# HEADER
# =========================================================

def render_header():
    st.markdown("""
    <div class="main-card">
        <h1 style="margin-bottom:0.15rem;">GIA</h1>
        <div class="brand-line"><strong>G</strong>uided <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
        <div class="brand-link">www.dougalien.com</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# ROUTING PANEL
# =========================================================

def render_router_section():
    st.markdown('<div class="section-label">2. Analysis Routing</div>', unsafe_allow_html=True)
    providers = available_providers()

    mode = st.radio(
        "Choose analysis mode",
        ["Lowest cost", "Cost-aware balanced", "Max accuracy"],
        index=["Lowest cost", "Cost-aware balanced", "Max accuracy"].index(st.session_state.selected_mode),
        horizontal=True,
    )
    st.session_state.selected_mode = mode

    st.markdown("""
    <div class="pro-card">
        <div class="small-note">
            Lowest cost = OpenAI only.<br>
            Cost-aware balanced = OpenAI first, Gemini second if available, then OpenAI judge.<br>
            Max accuracy = OpenAI first, Claude second if available, then OpenAI judge. Perplexity is used only to polish explanation text, not to identify the sample.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OpenAI", "Ready" if providers["openai"] else "Missing")
    c2.metric("Claude", "Ready" if providers["claude"] else "Missing")
    c3.metric("Gemini", "Ready" if providers["gemini"] else "Missing")
    c4.metric("Perplexity", "Ready" if providers["perplexity"] else "Missing")

# =========================================================
# USER INFO
# =========================================================

def render_user_section():
    st.markdown('<div class="section-label">3. User</div>', unsafe_allow_html=True)
    st.session_state.username = st.text_input(
        "User name (optional)",
        value=st.session_state.username,
        placeholder="Enter a user name or leave blank"
    )

# =========================================================
# UPLOAD
# =========================================================

def render_upload_section():
    st.markdown('<div class="section-label">4. Upload Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload image of rock, mineral, or fossil",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image for analysis."
    )

    if uploaded_file is not None:
        st.session_state.uploaded_name = uploaded_file.name
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded sample image", use_container_width=True)

    return uploaded_file

# =========================================================
# ANALYZE
# =========================================================

def render_analyze_section(uploaded_file):
    st.markdown('<div class="section-label">5. Analyze</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.button("Analyze", disabled=True, use_container_width=False)
        return

    if st.button("Analyze", use_container_width=False):
        try:
            img_b64 = encode_image(uploaded_file)
            consensus, details = analyze_image(img_b64, st.session_state.selected_mode)
            st.session_state.obs = consensus
            st.session_state.raw_results = details
            st.session_state.lesson_plan = None
        except requests.RequestException as e:
            st.error(f"Request error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# =========================================================
# RESULTS
# =========================================================

def render_result_section():
    if not st.session_state.obs:
        return

    obs = st.session_state.obs

    st.markdown('<div class="section-label">6. Result</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="input-box">
        <div class="box-label">{display_name()}</div>
        <div>Uploaded image ready for analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    obs_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in obs.get("observations", []))

    st.markdown(f"""
    <div class="output-box">
        <div class="box-label">GIA</div>
        <div><strong>Likely identification:</strong> {html.escape(obs.get('candidate', ''))}</div>
        <div style="margin-top:0.4rem;"><strong>Alternate:</strong> {html.escape(obs.get('alternate', ''))}</div>
        <div style="margin-top:0.4rem;"><strong>Confidence:</strong> {html.escape(str(obs.get('confidence', '')))} out of 5</div>
        <div style="margin-top:0.4rem;"><strong>Agreement:</strong> {html.escape(obs.get('agreement', ''))}</div>
        <div style="margin-top:0.4rem;"><strong>Chosen model:</strong> {html.escape(obs.get('winner', ''))}</div>
        <div style="margin-top:0.7rem;"><strong>Visible observations:</strong><ul>{obs_html}</ul></div>
        <div style="margin-top:0.7rem;"><strong>Why this fit:</strong> {html.escape(obs.get('why', ''))}</div>
        <div style="margin-top:0.7rem;"><strong>Limits:</strong> {html.escape(obs.get('limits', ''))}</div>
        <div style="margin-top:0.7rem;"><strong>Best next test:</strong> {html.escape(obs.get('next_test', ''))}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# AUDIO
# =========================================================

def render_audio_section():
    if not st.session_state.obs:
        return

    obs = st.session_state.obs

    st.markdown('<div class="section-label">7. Optional Audio</div>', unsafe_allow_html=True)

    speech_text = (
        f"Likely identification: {obs.get('candidate', '')}. "
        f"Alternate: {obs.get('alternate', '')}. "
        f"Confidence: {obs.get('confidence', '')} out of 5. "
        f"Why: {obs.get('why', '')}. "
        f"Next test: {obs.get('next_test', '')}."
    )
    render_audio_button(speech_text)

# =========================================================
# DEVELOPER TOOLS
# =========================================================

def render_dev_tools():
    st.markdown('<div class="section-label">8. Developer Tools</div>', unsafe_allow_html=True)

    with st.expander("Open developer tools", expanded=False):
        st.write("These tools are for testing only and are not part of the standard student view.")

        if not st.session_state.obs:
            st.write("Analyze an image first to test lesson plan and export tools.")
            return

        obs = st.session_state.obs
        details = st.session_state.raw_results or {}
        c1, c2 = st.columns(2)

        with c1:
            if st.button("Generate lesson plan now", use_container_width=True):
                prompt = LESSON_PROMPT.format(
                    candidate=obs.get("candidate", ""),
                    why=obs.get("why", ""),
                    observations=", ".join(obs.get("observations", []))
                )
                try:
                    st.session_state.lesson_plan = call_openai_text(prompt, OPENAI_MODEL_FAST)
                except requests.RequestException as e:
                    st.error(f"Request error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

        with c2:
            export_payload = {
                "final": obs,
                "details": details,
            }
            st.download_button(
                "Download current result as JSON",
                data=json.dumps(export_payload, indent=2),
                file_name="gia_result.json",
                use_container_width=True
            )

        if st.session_state.lesson_plan:
            st.markdown(f"""
            <div class="output-box">
                <div class="box-label">GIA</div>
                <div><strong>Lesson plan</strong></div>
                <div style="margin-top:0.7rem;">{html.escape(st.session_state.lesson_plan)}</div>
            </div>
            """, unsafe_allow_html=True)

        if details:
            st.write("Raw routing details")
            st.json(details)

# =========================================================
# MAIN
# =========================================================

if not st.session_state.authenticated:
    render_login()
    st.stop()

render_header()
render_router_section()
render_user_section()
uploaded_file = render_upload_section()
render_analyze_section(uploaded_file)
render_result_section()
render_audio_section()
render_dev_tools()
