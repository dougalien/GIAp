import base64
import html
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image

# =========================================================
# CONFIG
# =========================================================

OPENAI_MODEL_FAST = "gpt-4o-mini"
OPENAI_MODEL_STRONG = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
GEMINI_MODEL = "gemini-2.0-flash"
PERPLEXITY_MODEL = "sonar-pro"

TIMEOUT = 75
MAX_CHAT_TURNS = 8

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
""".strip()

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

SYNTHESIS_PROMPT = SYSTEM + """

You are the final synthesis step for a geology image analysis counsel.
You will receive:
- the image again
- model-by-model candidate analyses
- the judge result

Return valid JSON only:
{
  "candidate": "best identification or most likely material group",
  "alternate": "brief alternate possibility or 'none'",
  "confidence": 1,
  "observations": ["visible feature 1", "visible feature 2", "visible feature 3"],
  "why": "brief explanation grounded in visible evidence",
  "limits": "what cannot be determined from image alone",
  "next_test": "single best real-world follow-up test or observation"
}

Rules:
- Use the image as the final authority.
- Use the counsel outputs as advisors, not facts.
- Do not invent data.
- Prefer material group names over overconfident exact labels when warranted.
""".strip()

FOLLOWUP_SYSTEM = SYSTEM + """

You are answering a follow-up question about the same uploaded geology image.
Use the image and the stored analysis context.
Be concise, helpful, and honest about uncertainty.
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


def init_state() -> None:
    defaults = {
        "authenticated": False,
        "login_error": "",
        "username": "",
        "lesson_plan": None,
        "uploaded_name": None,
        "selected_view": "Compare both",
        "single_result": None,
        "single_details": {},
        "advanced_result": None,
        "advanced_details": {},
        "current_image_b64": None,
        "followup_history": [],
        "followup_draft": "",
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

    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
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



def truncate_chat(history: List[Dict[str, str]], max_turns: int = MAX_CHAT_TURNS) -> List[Dict[str, str]]:
    return history[-max_turns * 2 :]



def result_summary_for_prompt(result: Dict[str, Any], title: str) -> str:
    return (
        f"{title}\n"
        f"Candidate: {result.get('candidate', '')}\n"
        f"Alternate: {result.get('alternate', '')}\n"
        f"Confidence: {result.get('confidence', '')}/5\n"
        f"Agreement: {result.get('agreement', '')}\n"
        f"Winner: {result.get('winner', '')}\n"
        f"Observations: {', '.join(result.get('observations', []))}\n"
        f"Why: {result.get('why', '')}\n"
        f"Limits: {result.get('limits', '')}\n"
        f"Next test: {result.get('next_test', '')}\n"
    )



def build_compare_note(single_result: Dict[str, Any], advanced_result: Dict[str, Any]) -> str:
    same_candidate = single_result.get("candidate", "").strip().lower() == advanced_result.get("candidate", "").strip().lower()
    same_confidence = single_result.get("confidence") == advanced_result.get("confidence")

    if same_candidate and same_confidence:
        return "Both paths landed in essentially the same place. The advanced path mainly adds cross-checking and better auditability."
    if same_candidate:
        return "Both paths chose the same main identification, but the advanced path changed the confidence and supporting evidence."
    return "The advanced counsel changed the leading interpretation, so this is exactly the kind of sample where the side-by-side comparison is useful."



def render_audio_controls(text_to_speak: str, key: str) -> None:
    payload = json.dumps(text_to_speak)
    html_block = f"""
    <div style="border:1px solid #C9CDD3;border-radius:10px;padding:0.85rem;background:#FFFFFF;">
      <div style="font-weight:700;margin-bottom:0.5rem;">Audio playback</div>
      <div id="status_{key}" style="font-size:0.95rem;margin-bottom:0.65rem;color:#333;">Ready</div>
      <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
        <button id="play_{key}" aria-label="Play audio summary" style="min-height:44px;padding:0.6rem 0.9rem;border-radius:8px;border:1px solid #BFC6CF;background:#1F1F1F;color:#FFFFFF;cursor:pointer;">Play audio</button>
        <button id="stop_{key}" aria-label="Stop audio summary" style="min-height:44px;padding:0.6rem 0.9rem;border-radius:8px;border:1px solid #BFC6CF;background:#FFFFFF;color:#111111;cursor:pointer;">Stop audio</button>
      </div>
      <script>
        const text_{key} = {payload};
        const synth_{key} = window.speechSynthesis;
        const status_{key} = document.getElementById("status_{key}");
        const play_{key} = document.getElementById("play_{key}");
        const stop_{key} = document.getElementById("stop_{key}");

        function speak_{key}() {{
          if (!('speechSynthesis' in window)) {{
            status_{key}.textContent = 'Speech is not supported in this browser.';
            return;
          }}
          synth_{key}.cancel();
          const utter_{key} = new SpeechSynthesisUtterance(text_{key});
          utter_{key}.rate = 1.0;
          utter_{key}.pitch = 1.0;
          utter_{key}.onstart = function() {{ status_{key}.textContent = 'Playing audio…'; }};
          utter_{key}.onend = function() {{ status_{key}.textContent = 'Finished.'; }};
          utter_{key}.onerror = function() {{ status_{key}.textContent = 'Audio playback failed in this browser.'; }};
          synth_{key}.speak(utter_{key});
        }}

        play_{key}.onclick = speak_{key};
        stop_{key}.onclick = function() {{
          if ('speechSynthesis' in window) {{
            synth_{key}.cancel();
            status_{key}.textContent = 'Stopped.';
          }}
        }};
      </script>
    </div>
    """
    st.components.v1.html(html_block, height=140)



def format_speech_text(result: Dict[str, Any], label: str) -> str:
    return (
        f"{label}. "
        f"Likely identification: {result.get('candidate', '')}. "
        f"Alternate: {result.get('alternate', '')}. "
        f"Confidence: {result.get('confidence', '')} out of 5. "
        f"Agreement: {result.get('agreement', '')}. "
        f"Why: {result.get('why', '')}. "
        f"Best next test: {result.get('next_test', '')}."
    )



def normalize_result(data: Dict[str, Any], provider: str) -> Dict[str, Any]:
    confidence_raw = data.get("confidence", 1)
    try:
        confidence = int(confidence_raw)
    except Exception:
        confidence = 1
    confidence = max(1, min(5, confidence))

    observations = data.get("observations", [])
    if not isinstance(observations, list):
        observations = []
    observations = [str(x).strip() for x in observations if str(x).strip()][:5]

    return {
        "provider": provider,
        "candidate": str(data.get("candidate", "")).strip(),
        "alternate": str(data.get("alternate", "none")).strip(),
        "confidence": confidence,
        "observations": observations,
        "why": str(data.get("why", "")).strip(),
        "limits": str(data.get("limits", "")).strip(),
        "next_test": str(data.get("next_test", "")).strip(),
    }


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
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })

    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
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
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]



def call_openai_followup(question: str, image_b64: str, context_text: str, chat_history: List[Dict[str, str]]) -> str:
    key = get_secret("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": FOLLOWUP_SYSTEM},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Stored analysis context:\n"
                        f"{context_text}\n\n"
                        "Answer the follow-up question using the image and the stored context."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        },
    ]

    for item in truncate_chat(chat_history):
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": question})

    payload = {
        "model": OPENAI_MODEL_STRONG,
        "temperature": 0.2,
        "messages": messages,
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()



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
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}},
            ]
        }],
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
                "content": "You are a concise geology helper. Use only provided content. Do not browse.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# =========================================================
# ROUTING LOGIC
# =========================================================


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
        return {
            "winner": "openai",
            "why": "Judge fallback used.",
            "final_confidence": 2,
            "agreement": "low",
        }
    return parsed



def synthesize_final_result(img_b64: str, counsel_results: Dict[str, Dict[str, Any]], judge: Dict[str, Any]) -> Dict[str, Any]:
    synthesis_payload = (
        "Counsel outputs:\n"
        f"{build_judge_input(counsel_results)}\n\n"
        f"Judge result:\n{json.dumps(judge, indent=2)}"
    )
    raw = call_openai_json(SYNTHESIS_PROMPT + "\n\n" + synthesis_payload, img_b64, OPENAI_MODEL_STRONG)
    parsed = safe_json_loads(raw)
    if not parsed:
        winner = judge.get("winner", "openai")
        fallback = counsel_results.get(winner) or next(iter(counsel_results.values()))
        return normalize_result(fallback, "advanced")
    final = normalize_result(parsed, "advanced")
    final["agreement"] = judge.get("agreement", "moderate")
    final["winner"] = judge.get("winner", "openai")
    return final



def run_single_pass(img_b64: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    providers = available_providers()
    if not providers["openai"]:
        raise RuntimeError("OPENAI_API_KEY is required.")

    details: Dict[str, Any] = {"errors": [], "raw_results": {}, "mode": "Single pass"}
    result, err = provider_pass("openai", img_b64)
    if err or not result:
        details["errors"].append(err or "Single-pass analysis failed.")
        raise RuntimeError("Single-pass OpenAI vision analysis failed.")

    result["agreement"] = "single-model"
    result["winner"] = "openai"
    details["raw_results"]["openai"] = result
    return result, details



def run_advanced_counsel(img_b64: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    providers = available_providers()
    if not providers["openai"]:
        raise RuntimeError("OPENAI_API_KEY is required because OpenAI is used in the counsel and synthesis path.")

    details: Dict[str, Any] = {"errors": [], "raw_results": {}, "mode": "Advanced counsel"}
    counsel_order = ["openai"]
    if providers["claude"]:
        counsel_order.append("claude")
    if providers["gemini"]:
        counsel_order.append("gemini")

    counsel_results: Dict[str, Dict[str, Any]] = {}
    for provider in counsel_order:
        result, err = provider_pass(provider, img_b64)
        if err:
            details["errors"].append(err)
            continue
        if result:
            counsel_results[provider] = result
            details["raw_results"][provider] = result

    if not counsel_results:
        raise RuntimeError("All counsel model passes failed.")

    if len(counsel_results) == 1:
        only = next(iter(counsel_results.values()))
        only["agreement"] = "single-model"
        only["winner"] = next(iter(counsel_results.keys()))
        details["judge"] = {
            "winner": only["winner"],
            "why": "Only one provider was available.",
            "final_confidence": only["confidence"],
            "agreement": "single-model",
        }
        return only, details

    judge = judge_results(counsel_results)
    details["judge"] = judge
    final = synthesize_final_result(img_b64, counsel_results, judge)

    if providers["perplexity"]:
        try:
            explain_prompt = (
                "Rewrite this geology explanation more clearly for a student. "
                "Do not add any new facts. Keep it under 90 words.\n\n"
                f"Candidate: {final['candidate']}\n"
                f"Observations: {final['observations']}\n"
                f"Why: {final['why']}\n"
                f"Limits: {final['limits']}"
            )
            polished = call_perplexity_text(explain_prompt)
            if polished:
                final["why"] = polished.strip()
                details["used_perplexity_for_explanation"] = True
        except Exception as e:
            details["errors"].append(f"perplexity explanation pass: {e}")

    return final, details



def run_selected_analyses(img_b64: str, view: str) -> None:
    st.session_state.current_image_b64 = img_b64
    st.session_state.followup_history = []
    st.session_state.lesson_plan = None

    if view in ["Single pass only", "Compare both"]:
        single_result, single_details = run_single_pass(img_b64)
        st.session_state.single_result = single_result
        st.session_state.single_details = single_details
    else:
        st.session_state.single_result = None
        st.session_state.single_details = {}

    if view in ["Advanced counsel only", "Compare both"]:
        advanced_result, advanced_details = run_advanced_counsel(img_b64)
        st.session_state.advanced_result = advanced_result
        st.session_state.advanced_details = advanced_details
    else:
        st.session_state.advanced_result = None
        st.session_state.advanced_details = {}


# =========================================================
# STYLES
# =========================================================

st.set_page_config(page_title="GIA", layout="wide")

st.markdown(
    """
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

div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    border-radius: 8px;
    min-height: 44px;
    font-weight: 600;
    border: 1px solid #BFC6CF;
}

label, .stRadio label, .stTextInput label, .stTextArea label {
    font-weight: 600 !important;
}

hr {
    border: none;
    border-top: 1px solid #D8DCE2;
    margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# RENDERERS
# =========================================================


def render_login() -> None:
    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        st.markdown(
            """
        <div class="main-card" style="margin-top:3rem;">
            <div class="section-label">1. Sign In</div>
            <h1 style="margin-bottom:0.15rem;">GIA</h1>
            <div class="brand-line"><strong>G</strong>uided <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
            <div class="brand-link">www.dougalien.com</div>
            <p class="small-note" style="margin-top:0.9rem;">
                Enter the app password to continue.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

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



def render_header() -> None:
    st.markdown(
        """
    <div class="main-card">
        <h1 style="margin-bottom:0.15rem;">GIA</h1>
        <div class="brand-line"><strong>G</strong>uided <strong>I</strong>mage <strong>A</strong>nalysis by We are dougalien</div>
        <div class="brand-link">www.dougalien.com</div>
    </div>
    """,
        unsafe_allow_html=True,
    )



def render_accessibility_section() -> None:
    st.markdown('<div class="section-label">2. Accessibility and audio</div>', unsafe_allow_html=True)
    st.info(
        "Large text, clear labels, keyboard-friendly controls, visible status messages, and browser-based audio playback are enabled. "
        "If audio does not play, test in Chrome or Safari first because browser speech support varies."
    )



def render_controls_section() -> None:
    st.markdown('<div class="section-label">3. Analysis controls</div>', unsafe_allow_html=True)
    providers = available_providers()

    view = st.radio(
        "Choose what to run",
        ["Compare both", "Single pass only", "Advanced counsel only"],
        index=["Compare both", "Single pass only", "Advanced counsel only"].index(st.session_state.selected_view),
        horizontal=True,
        help="Compare both runs a fast single-pass result next to the advanced counsel result.",
    )
    st.session_state.selected_view = view

    st.markdown(
        """
        <div class="pro-card">
            <div class="small-note">
                Single pass = one OpenAI vision pass.<br>
                Advanced counsel = OpenAI + Claude and/or Gemini when available, plus a judge and final synthesis pass.<br>
                Follow-up chat uses the stored result and the uploaded image.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OpenAI", "Ready" if providers["openai"] else "Missing")
    c2.metric("Claude", "Ready" if providers["claude"] else "Missing")
    c3.metric("Gemini", "Ready" if providers["gemini"] else "Missing")
    c4.metric("Perplexity", "Ready" if providers["perplexity"] else "Missing")



def render_user_section() -> None:
    st.markdown('<div class="section-label">4. User</div>', unsafe_allow_html=True)
    st.session_state.username = st.text_input(
        "User name (optional)",
        value=st.session_state.username,
        placeholder="Enter a user name or leave blank",
    )



def render_upload_section():
    st.markdown('<div class="section-label">5. Upload image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload image of rock, mineral, or fossil",
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image for analysis.",
    )

    if uploaded_file is not None:
        st.session_state.uploaded_name = uploaded_file.name
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded sample image", use_container_width=True)
    return uploaded_file



def render_analyze_section(uploaded_file) -> None:
    st.markdown('<div class="section-label">6. Analyze</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.button("Run analysis", disabled=True)
        return

    if st.button("Run analysis"):
        try:
            img_b64 = encode_image(uploaded_file)
            with st.spinner("Running analysis..."):
                run_selected_analyses(img_b64, st.session_state.selected_view)
            st.success("Analysis complete.")
        except requests.RequestException as e:
            st.error(f"Request error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")



def render_result_card(result: Dict[str, Any], title: str, audio_key: str) -> None:
    st.subheader(title)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{result.get('confidence', '')}/5")
    c2.metric("Agreement", result.get("agreement", ""))
    c3.metric("Winning model", result.get("winner", ""))

    st.write(f"**Likely identification:** {result.get('candidate', '')}")
    st.write(f"**Alternate:** {result.get('alternate', '')}")
    st.write("**Visible observations:**")
    for item in result.get("observations", []):
        st.write(f"- {item}")
    st.write(f"**Why this fit:** {result.get('why', '')}")
    st.write(f"**Limits:** {result.get('limits', '')}")
    st.write(f"**Best next test:** {result.get('next_test', '')}")
    render_audio_controls(format_speech_text(result, title), audio_key)



def render_compare_section() -> None:
    single = st.session_state.single_result
    advanced = st.session_state.advanced_result
    if not single and not advanced:
        return

    st.markdown('<div class="section-label">7. Results</div>', unsafe_allow_html=True)

    tabs = ["Summary"]
    if single:
        tabs.append("Single pass")
    if advanced:
        tabs.append("Advanced counsel")
    tabs.extend(["Follow-up chat", "Developer tools"])

    tab_objects = st.tabs(tabs)
    tab_map = dict(zip(tabs, tab_objects))

    with tab_map["Summary"]:
        if single and advanced:
            st.success(build_compare_note(single, advanced))
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Single pass**")
                st.write(f"Candidate: {single.get('candidate', '')}")
                st.write(f"Confidence: {single.get('confidence', '')}/5")
                st.write(f"Agreement: {single.get('agreement', '')}")
                st.write(f"Model path: {single.get('winner', '')}")
            with c2:
                st.write("**Advanced counsel**")
                st.write(f"Candidate: {advanced.get('candidate', '')}")
                st.write(f"Confidence: {advanced.get('confidence', '')}/5")
                st.write(f"Agreement: {advanced.get('agreement', '')}")
                st.write(f"Model path: {advanced.get('winner', '')}")

            st.write("**What changed**")
            st.write(f"- Candidate changed: {'Yes' if single.get('candidate') != advanced.get('candidate') else 'No'}")
            st.write(f"- Confidence changed: {'Yes' if single.get('confidence') != advanced.get('confidence') else 'No'}")
            st.write(f"- Observation count changed: {len(single.get('observations', []))} → {len(advanced.get('observations', []))}")
        elif single:
            st.info("Single-pass analysis completed.")
            render_result_card(single, "Single pass result", "summary_single")
        elif advanced:
            st.info("Advanced counsel analysis completed.")
            render_result_card(advanced, "Advanced counsel result", "summary_advanced")

    if single and "Single pass" in tab_map:
        with tab_map["Single pass"]:
            render_result_card(single, "Single pass result", "single_pass_audio")

    if advanced and "Advanced counsel" in tab_map:
        with tab_map["Advanced counsel"]:
            render_result_card(advanced, "Advanced counsel result", "advanced_audio")
            details = st.session_state.advanced_details or {}
            raw_results = details.get("raw_results", {})
            if raw_results:
                st.write("**Counsel members**")
                for provider, result in raw_results.items():
                    with st.expander(f"{provider.title()} pass", expanded=False):
                        st.json(result)
            if details.get("judge"):
                st.write("**Judge result**")
                st.json(details["judge"])

    with tab_map["Follow-up chat"]:
        render_followup_chat()

    with tab_map["Developer tools"]:
        render_dev_tools()



def render_followup_chat() -> None:
    single = st.session_state.single_result
    advanced = st.session_state.advanced_result
    image_b64 = st.session_state.current_image_b64

    if not image_b64 or (not single and not advanced):
        st.write("Run an analysis first to enable follow-up chat.")
        return

    default_context = "Advanced counsel" if advanced else "Single pass"
    context_choice = st.radio(
        "Use context from",
        [label for label in ["Advanced counsel", "Single pass"] if (label == "Advanced counsel" and advanced) or (label == "Single pass" and single)],
        horizontal=True,
        index=0,
    )

    chosen_result = advanced if context_choice == "Advanced counsel" else single
    context_text = result_summary_for_prompt(chosen_result, context_choice)

    for msg in st.session_state.followup_history:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.write(msg["content"])

    prompt = st.chat_input(f"Ask a follow-up about the image using {default_context.lower()} context")
    if prompt:
        st.session_state.followup_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = call_openai_followup(
                        question=prompt,
                        image_b64=image_b64,
                        context_text=context_text,
                        chat_history=st.session_state.followup_history[:-1],
                    )
                except Exception as e:
                    answer = f"Follow-up chat failed: {e}"
            st.write(answer)
        st.session_state.followup_history.append({"role": "assistant", "content": answer})



def render_dev_tools() -> None:
    st.write("These tools are for testing only and are not part of the standard student view.")

    single = st.session_state.single_result
    advanced = st.session_state.advanced_result
    details = {
        "single": st.session_state.single_details,
        "advanced": st.session_state.advanced_details,
    }

    if not single and not advanced:
        st.write("Analyze an image first to test lesson plan and export tools.")
        return

    source = advanced or single
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Generate lesson plan now", use_container_width=True):
            prompt = LESSON_PROMPT.format(
                candidate=source.get("candidate", ""),
                why=source.get("why", ""),
                observations=", ".join(source.get("observations", [])),
            )
            try:
                st.session_state.lesson_plan = call_openai_text(prompt, OPENAI_MODEL_FAST)
            except requests.RequestException as e:
                st.error(f"Request error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

    with c2:
        export_payload = {
            "single_result": single,
            "single_details": st.session_state.single_details,
            "advanced_result": advanced,
            "advanced_details": st.session_state.advanced_details,
            "followup_history": st.session_state.followup_history,
        }
        st.download_button(
            "Download current session as JSON",
            data=json.dumps(export_payload, indent=2),
            file_name="gia_session.json",
            use_container_width=True,
        )

    if st.session_state.lesson_plan:
        st.write("**Lesson plan**")
        st.write(st.session_state.lesson_plan)

    with st.expander("Open raw analysis details", expanded=False):
        st.json(details)


# =========================================================
# MAIN
# =========================================================

if not st.session_state.authenticated:
    render_login()
    st.stop()

render_header()
render_accessibility_section()
render_controls_section()
render_user_section()
uploaded_file = render_upload_section()
render_analyze_section(uploaded_file)
render_compare_section()
