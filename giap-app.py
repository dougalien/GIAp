import os
import json
import base64
import mimetypes
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# =========================================================
# GIAp — Guided Image Analyzer
# by We are dougalien
# =========================================================

st.set_page_config(
    page_title="GIAp",
    page_icon="🪨",
    layout="wide",
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
DEFAULT_PERPLEXITY_MODEL = "sonar-pro"

COUNSEL_MODES = {
    "Cheap": {
        "observer": True,
        "validator": False,
        "judge": False,
        "judge_on_disagreement_only": False,
        "label": "Low cost",
    },
    "Balanced": {
        "observer": True,
        "validator": True,
        "judge": True,
        "judge_on_disagreement_only": True,
        "label": "Best value",
    },
    "Max caution": {
        "observer": True,
        "validator": True,
        "judge": True,
        "judge_on_disagreement_only": False,
        "label": "Highest caution",
    },
}


# =========================================================
# Helpers
# =========================================================

def init_state():
    defaults = {
        "authenticated": False,
        "login_error": "",
        "started": False,
        "image_name": None,
        "image_bytes": None,
        "image_mime": None,
        "image_data_uri": None,
        "last_uploaded_signature": None,
        "display_messages": [],
        "analysis_meta": {},
        "final_reply": "",
        "first_reply": "",
        "observer_json": None,
        "validator_json": None,
        "judge_json": None,
        "counsel_mode": "Balanced",
        "mode": "Auto",
        "context_notes": "",
        "specimen_label": "",
        "student_observations": "",
        "student_best_answer": "",
        "known_name": "",
        "student_name": "",
        "include_auto_zoom": True,
        "zoom_fraction": 0.5,
        "clear_followup_next": False,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "claude_model": DEFAULT_CLAUDE_MODEL,
        "perplexity_model": DEFAULT_PERPLEXITY_MODEL,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def get_secret_or_env(name: str, default: str = ""):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def app_password():
    return get_secret_or_env("APP_PASSWORD", "")


def get_openai_key():
    return get_secret_or_env("OPENAI_API_KEY", "")


def get_claude_key():
    return get_secret_or_env("CLAUDE_API_KEY", get_secret_or_env("ANTHROPIC_API_KEY", ""))


def get_perplexity_key():
    return get_secret_or_env("PERPLEXITY_API_KEY", "")


def sign_out():
    keep = {
        "counsel_mode": st.session_state.get("counsel_mode", "Balanced"),
        "mode": st.session_state.get("mode", "Auto"),
        "openai_model": st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
        "claude_model": st.session_state.get("claude_model", DEFAULT_CLAUDE_MODEL),
        "perplexity_model": st.session_state.get("perplexity_model", DEFAULT_PERPLEXITY_MODEL),
    }
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    for k, v in keep.items():
        st.session_state[k] = v
    st.session_state.authenticated = False


def reset_specimen():
    keep = {
        "authenticated": st.session_state.get("authenticated", False),
        "counsel_mode": st.session_state.get("counsel_mode", "Balanced"),
        "mode": st.session_state.get("mode", "Auto"),
        "openai_model": st.session_state.get("openai_model", DEFAULT_OPENAI_MODEL),
        "claude_model": st.session_state.get("claude_model", DEFAULT_CLAUDE_MODEL),
        "perplexity_model": st.session_state.get("perplexity_model", DEFAULT_PERPLEXITY_MODEL),
    }
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    init_state()
    for k, v in keep.items():
        st.session_state[k] = v


def file_to_data_uri(uploaded_file):
    raw = uploaded_file.getvalue()
    mime = uploaded_file.type or mimetypes.guess_type(uploaded_file.name)[0] or "image/png"
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return raw, mime, data_uri


def update_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return
    raw, mime, data_uri = file_to_data_uri(uploaded_file)
    st.session_state.image_name = uploaded_file.name
    st.session_state.image_bytes = raw
    st.session_state.image_mime = mime
    st.session_state.image_data_uri = data_uri


def ensure_image_data_uri():
    if st.session_state.get("image_bytes") and not st.session_state.get("image_data_uri"):
        try:
            mime = st.session_state.get("image_mime") or "image/png"
            b64 = base64.b64encode(st.session_state.image_bytes).decode("utf-8")
            st.session_state.image_data_uri = f"data:{mime};base64,{b64}"
        except Exception:
            st.session_state.image_bytes = None
            st.session_state.image_name = None
            st.session_state.image_mime = None
            st.session_state.image_data_uri = None


def get_image_contents_for_openai():
    if not st.session_state.image_bytes or not st.session_state.image_data_uri:
        return []

    contents = [
        {
            "type": "image_url",
            "image_url": {"url": st.session_state.image_data_uri},
        }
    ]

    if not st.session_state.include_auto_zoom:
        return contents

    try:
        img = Image.open(BytesIO(st.session_state.image_bytes))
        w, h = img.size
        frac = max(0.1, min(float(st.session_state.zoom_fraction), 1.0))
        cw, ch = int(w * frac), int(h * frac)
        left = (w - cw) // 2
        top = (h - ch) // 2
        right = left + cw
        bottom = top + ch
        crop_center = img.crop((left, top, right, bottom))

        buf = BytesIO()
        fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "PNG"
        crop_center.save(buf, format=fmt)
        crop_bytes = buf.getvalue()
        b64 = base64.b64encode(crop_bytes).decode("utf-8")
        mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }.get(fmt, "image/png")
        crop_data_uri = f"data:{mime};base64,{b64}"
        contents.append(
            {
                "type": "image_url",
                "image_url": {"url": crop_data_uri},
            }
        )
    except Exception:
        pass

    return contents


def get_image_blocks_for_anthropic():
    if not st.session_state.image_bytes:
        return []

    blocks = []

    try:
        mime = st.session_state.get("image_mime") or "image/png"
        media_type = mime if mime in ["image/jpeg", "image/png", "image/webp", "image/gif"] else "image/png"
        source_data = base64.b64encode(st.session_state.image_bytes).decode("utf-8")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": source_data,
                },
            }
        )

        if st.session_state.include_auto_zoom:
            img = Image.open(BytesIO(st.session_state.image_bytes))
            w, h = img.size
            frac = max(0.1, min(float(st.session_state.zoom_fraction), 1.0))
            cw, ch = int(w * frac), int(h * frac)
            left = (w - cw) // 2
            top = (h - ch) // 2
            right = left + cw
            bottom = top + ch
            crop_center = img.crop((left, top, right, bottom))

            buf = BytesIO()
            fmt = img.format if img.format in ["JPEG", "PNG", "WEBP"] else "PNG"
            crop_center.save(buf, format=fmt)
            crop_bytes = buf.getvalue()
            crop_media_type = {
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }.get(fmt, "image/png")

            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": crop_media_type,
                        "data": base64.b64encode(crop_bytes).decode("utf-8"),
                    },
                }
            )
    except Exception:
        pass

    return blocks


def specimen_context_block():
    label = st.session_state.specimen_label.strip() or "[no label]"
    notes = st.session_state.context_notes.strip() or "[no extra notes]"
    student_name = st.session_state.student_name.strip() or "[no name provided]"
    observations = st.session_state.student_observations.strip() or "[none entered yet]"
    best_answer = st.session_state.student_best_answer.strip() or "[none entered yet]"
    known_name = st.session_state.known_name.strip() or "[none provided]"
    mode = st.session_state.mode

    return f"""
Specimen label / sample ID: {label}
Student observations: {observations}
Student best answer: {best_answer}
Instructor known name: {known_name}
Extra notes: {notes}
Student name: {student_name}
Mode: {mode}
""".strip()


def role_system_prompt(role: str):
    base = """
You are helping with an intro geology lab image analysis workflow.
Stay grounded in visible evidence only.
Do not invent hardness, streak, chemistry, locality, or microscopic details that are not visible.
Be concise, careful, and honest.
""".strip()

    mode_hint = {
        "Auto": "Decide whether the specimen seems more like a rock, mineral, fossil, sand/granular sample, or tiny mixed particles.",
        "Rock": "Prioritize grain size, layering, clasts, crystals, foliation, vesicles, and texture.",
        "Mineral": "Prioritize luster, color, transparency, obvious cleavage or fracture, and crystal habit.",
        "Fossil": "Prioritize symmetry, shell shape, segments, ridges, repeating structures, and preservation style.",
        "Sand/Granular": "Prioritize grain size, sorting, rounding, mixed fragments, and visible color variation.",
        "Forensic": "Prioritize tiny-particle appearance and remain highly cautious.",
    }.get(st.session_state.mode, "Stay grounded in what is visibly present.")

    if role == "observer":
        return f"""
{base}

You are the OBSERVER.
Your job is to describe what is visibly present and propose the best tentative ID.

Return valid JSON only with this exact schema:
{{
  "visible_evidence": ["short item", "short item", "short item"],
  "likely_identification": "short phrase",
  "alternative_identification": "short phrase",
  "confidence": 1,
  "image_clarity": "clear / somewhat unclear / poor",
  "reasoning": "1-2 short sentences",
  "next_check": "short practical next check"
}}

Rules:
- confidence must be an integer from 1 to 5.
- Keep items short and factual.
- If the instructor known name is provided and the image is ambiguous, you may lean toward it but still note uncertainty.
- {mode_hint}
""".strip()

    if role == "validator":
        return f"""
{base}

You are the VALIDATOR.
Act independently from the observer and check whether the likely ID is well supported.

Return valid JSON only with this exact schema:
{{
  "visible_evidence": ["short item", "short item", "short item"],
  "likely_identification": "short phrase",
  "alternative_identification": "short phrase",
  "confidence": 1,
  "agreement_with_student_name": "probably right / close / not a good match / uncertain",
  "reasoning": "1-2 short sentences",
  "next_check": "short practical next check"
}}

Rules:
- confidence must be an integer from 1 to 5.
- Stay independent.
- {mode_hint}
""".strip()

    return f"""
{base}

You are the JUDGE.
You will compare the observer and validator outputs and produce a conservative student-facing result.

Return valid JSON only with this exact schema:
{{
  "agreement_level": "high / medium / low",
  "winner": "observer / validator / blended",
  "final_identification": "short phrase",
  "confidence": 1,
  "why": "1-2 short sentences",
  "next_check": "short practical next check",
  "student_reply": "2-4 short sentences, plain language, concise, student-facing"
}}

Rules:
- confidence must be an integer from 1 to 5.
- Prefer caution over confidence.
- If disagreement is substantial, use blended and explain uncertainty.
- If an instructor known name is given and the image is ambiguous, weigh it appropriately.
- {mode_hint}
""".strip()


def safe_json_loads(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def call_openai_json(system_prompt: str, user_text: str):
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    *get_image_contents_for_openai(),
                ],
            },
        ],
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("OpenAI returned non-JSON output.")
    return parsed


def call_claude_json(system_prompt: str, user_text: str):
    api_key = get_claude_key()
    if not api_key:
        raise RuntimeError("Missing CLAUDE_API_KEY or ANTHROPIC_API_KEY.")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    content_blocks = [{"type": "text", "text": user_text}, *get_image_blocks_for_anthropic()]

    payload = {
        "model": st.session_state.claude_model,
        "max_tokens": 900,
        "temperature": 0.2,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": content_blocks,
            }
        ],
    }

    resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Claude error {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    text_parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "\n".join(text_parts).strip()
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("Claude returned non-JSON output.")
    return parsed


def call_perplexity_judge(system_prompt: str, user_text: str):
    api_key = get_perplexity_key()
    if not api_key:
        raise RuntimeError("Missing PERPLEXITY_API_KEY.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.perplexity_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
            },
        ],
    }

    resp = requests.post(PERPLEXITY_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Perplexity error {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    parsed = safe_json_loads(text)
    if parsed is None:
        raise RuntimeError("Perplexity returned non-JSON output.")
    return parsed


def disagreement_score(observer, validator):
    if not observer or not validator:
        return 0

    obs_id = (observer.get("likely_identification") or "").strip().lower()
    val_id = (validator.get("likely_identification") or "").strip().lower()
    obs_alt = (observer.get("alternative_identification") or "").strip().lower()
    val_alt = (validator.get("alternative_identification") or "").strip().lower()

    score = 0

    if obs_id and val_id and obs_id != val_id:
        score += 2
    if obs_id and val_alt and obs_id == val_alt:
        score -= 1
    if val_id and obs_alt and val_id == obs_alt:
        score -= 1

    try:
        oc = int(observer.get("confidence", 0))
        vc = int(validator.get("confidence", 0))
        if oc <= 2 or vc <= 2:
            score += 1
    except Exception:
        pass

    return max(score, 0)


def build_student_reply_from_observer(observer):
    evidence = observer.get("visible_evidence", [])
    likely = observer.get("likely_identification", "uncertain")
    alt = observer.get("alternative_identification", "another possibility")
    conf = int(observer.get("confidence", 2))
    clarity = observer.get("image_clarity", "clear")
    next_check = observer.get("next_check", "Check one more visible feature in lab.")

    sentence1 = f"I can see {', '.join(evidence[:3])}." if evidence else "I can see a few useful visible features."
    if conf >= 4:
        sentence2 = f"The best fit looks like {likely}."
    elif conf == 3:
        sentence2 = f"The best fit may be {likely}, but {alt} is still plausible."
    else:
        sentence2 = f"The image looks {clarity}, but the ID is still uncertain; {likely} is only a tentative fit."

    sentence3 = f"Next, {next_check}"
    return f"{sentence1} {sentence2} {sentence3}"


def build_student_reply_from_validator(validator):
    evidence = validator.get("visible_evidence", [])
    likely = validator.get("likely_identification", "uncertain")
    alt = validator.get("alternative_identification", "another possibility")
    conf = int(validator.get("confidence", 2))
    match = validator.get("agreement_with_student_name", "uncertain")
    next_check = validator.get("next_check", "check one more visible feature in lab")

    sentence1 = f"I notice {', '.join(evidence[:3])}." if evidence else "I notice a few visible features worth checking."
    if conf >= 4:
        sentence2 = f"Your current name looks {match}, and {likely} is the strongest fit."
    elif conf == 3:
        sentence2 = f"Your current name looks {match}; {likely} may fit, but {alt} also deserves a look."
    else:
        sentence2 = f"Your current name is {match}, but the image still leaves real uncertainty."

    sentence3 = f"Next, {next_check}."
    return f"{sentence1} {sentence2} {sentence3}"


def analyze_with_counsel():
    if st.session_state.image_bytes and not st.session_state.image_data_uri:
        st.warning("Image is still loading. Wait for the preview, then try again.")
        return

    if not st.session_state.image_bytes:
        st.warning("Please upload a specimen photo first.")
        return

    context = specimen_context_block()

    observer_prompt = role_system_prompt("observer")
    validator_prompt = role_system_prompt("validator")
    judge_prompt = role_system_prompt("judge")

    observer_user = f"""
Analyze this specimen image for an intro geology lab.

{context}
""".strip()

    observer = call_openai_json(observer_prompt, observer_user)
    st.session_state.observer_json = observer

    config = COUNSEL_MODES[st.session_state.counsel_mode]
    validator = None
    judge = None

    if config["validator"]:
        validator_user = f"""
Independently validate this specimen image for an intro geology lab.

{context}
""".strip()
        validator = call_claude_json(validator_prompt, validator_user)
        st.session_state.validator_json = validator

    use_judge = False
    if config["judge"]:
        if not config["judge_on_disagreement_only"]:
            use_judge = True
        else:
            score = disagreement_score(observer, validator)
            if score >= 2:
                use_judge = True

            try:
                if int(observer.get("confidence", 0)) <= 2:
                    use_judge = True
            except Exception:
                pass
            try:
                if validator and int(validator.get("confidence", 0)) <= 2:
                    use_judge = True
            except Exception:
                pass

    if use_judge:
        judge_user = f"""
Compare these two structured geology-image analyses and produce the most cautious student-facing reply.

Specimen context:
{context}

Observer JSON:
{json.dumps(observer, ensure_ascii=False)}

Validator JSON:
{json.dumps(validator, ensure_ascii=False) if validator else "{}"}
""".strip()
        judge = call_perplexity_judge(judge_prompt, judge_user)
        st.session_state.judge_json = judge

    if judge and judge.get("student_reply"):
        final_reply = judge["student_reply"].strip()
    elif validator:
        if disagreement_score(observer, validator) >= 2:
            obs_id = observer.get("likely_identification", "an uncertain fit")
            val_id = validator.get("likely_identification", "another uncertain fit")
            next_check = validator.get("next_check") or observer.get("next_check") or "check one more visible feature in lab"
            final_reply = (
                f"I can see useful features in the photo, but there is still some uncertainty. "
                f"One reading leans toward {obs_id}, while another leans toward {val_id}. "
                f"The safest next step is to {next_check}."
            )
        else:
            final_reply = build_student_reply_from_validator(validator)
    else:
        final_reply = build_student_reply_from_observer(observer)

    st.session_state.final_reply = final_reply
    st.session_state.first_reply = final_reply
    st.session_state.display_messages = [{"role": "assistant", "content": final_reply}]
    st.session_state.analysis_meta = {
        "observer_model": st.session_state.openai_model,
        "validator_model": st.session_state.claude_model if validator else None,
        "judge_model": st.session_state.perplexity_model if judge else None,
        "counsel_mode": st.session_state.counsel_mode,
    }
    st.session_state.started = True


def followup_system_prompt():
    return """
You are a concise geology tutor continuing the SAME specimen discussion.
Stay consistent with the prior analysis unless the new user message provides a reason to revise it.
Be honest if the earlier answer was uncertain or likely wrong.
Use 2-4 short sentences.
Keep the answer grounded in visible evidence and simple next checks.
""".strip()


def send_followup(user_text: str):
    user_text = user_text.strip()
    if not user_text:
        return

    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")

    prior = st.session_state.get("first_reply", "[no earlier reply stored]")
    context = specimen_context_block()

    followup_text = f"""
You are continuing a geology tutoring session about one specimen.

Earlier anchored reply:
\"\"\"{prior}\"\"\"

Specimen context:
{context}

Student follow-up:
{user_text}

Instructions:
- Stay short.
- If the earlier answer was uncertain, say so plainly.
- If the instructor known name changes the interpretation, admit the earlier guess may have been wrong.
- Focus on what can be seen and what to check next.
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": st.session_state.openai_model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": followup_system_prompt()},
            {"role": "user", "content": followup_text},
        ],
    }

    resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI follow-up error {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    reply = data["choices"][0]["message"]["content"].strip()

    st.session_state.display_messages.append({"role": "user", "content": user_text})
    st.session_state.display_messages.append({"role": "assistant", "content": reply})


def secret_status(key_name, value):
    return "✅" if value else "—"


# =========================================================
# Styling
# =========================================================

init_state()
ensure_image_data_uri()

st.markdown(
    """
<style>
@import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&f[]=boska@400,500,700&display=swap');

:root {
    --bg: #f6f4ee;
    --panel: #fbfaf6;
    --panel-2: #f1ede5;
    --text: #23211c;
    --muted: #6f6d67;
    --line: rgba(35,33,28,.12);
    --teal: #0b6b6f;
    --teal-2: #d4e7e5;
    --accent: #8a5a2b;
    --shadow: 0 10px 30px rgba(30, 27, 21, 0.08);
    --radius: 18px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(circle at top left, rgba(11,107,111,.06), transparent 24%),
        radial-gradient(circle at bottom right, rgba(138,90,43,.06), transparent 20%),
        var(--bg);
    color: var(--text);
    font-family: 'General Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Boska', serif;
    color: var(--text);
    letter-spacing: 0.01em;
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    max-width: 1380px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f4ee 0%, #f0ece4 100%);
    border-right: 1px solid var(--line);
}

.brand-wrap {
    background: rgba(251,250,246,.72);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.1rem 1.1rem 1rem 1.1rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(8px);
}

.brand-row {
    display: flex;
    align-items: center;
    gap: .85rem;
    margin-bottom: .65rem;
}

.brand-mark {
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(145deg, #0b6b6f, #154b4e);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #f6f4ee;
    box-shadow: 0 8px 18px rgba(11,107,111,.25);
    font-size: 1.25rem;
}

.brand-title {
    font-family: 'Boska', serif;
    font-size: 1.7rem;
    line-height: 1;
    margin: 0;
}

.brand-sub {
    color: var(--muted);
    font-size: .94rem;
    margin-top: .15rem;
}

.byline {
    color: var(--muted);
    font-size: .9rem;
    margin-top: .2rem;
}

.panel {
    background: rgba(251,250,246,.88);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
}

.hero-card {
    background:
        linear-gradient(180deg, rgba(255,255,255,.65), rgba(255,255,255,.35)),
        linear-gradient(135deg, rgba(11,107,111,.09), rgba(138,90,43,.08));
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 1.25rem 1.25rem 1rem 1.25rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}

.kicker {
    display: inline-block;
    background: var(--teal-2);
    color: var(--teal);
    border-radius: 999px;
    padding: .35rem .65rem;
    font-size: .84rem;
    font-weight: 600;
    margin-bottom: .65rem;
}

.subtle {
    color: var(--muted);
}

.readout {
    border: 1px solid var(--line);
    background: rgba(255,255,255,.58);
    border-radius: 18px;
    padding: .9rem 1rem;
    margin-bottom: .8rem;
}

.smallcap {
    font-size: .78rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .28rem;
    font-weight: 700;
}

.reply-box {
    border: 1px solid rgba(11,107,111,.22);
    background: linear-gradient(180deg, rgba(212,231,229,.42), rgba(255,255,255,.68));
    border-radius: 20px;
    padding: 1rem 1rem;
    box-shadow: var(--shadow);
}

.meta-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: .7rem;
}

.meta-pill {
    background: rgba(255,255,255,.72);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: .7rem .8rem;
}

.chat-user {
    background: rgba(138,90,43,.08);
    border: 1px solid rgba(138,90,43,.18);
    border-radius: 16px;
    padding: .8rem .9rem;
    margin: .5rem 0;
}

.chat-ai {
    background: rgba(11,107,111,.08);
    border: 1px solid rgba(11,107,111,.18);
    border-radius: 16px;
    padding: .8rem .9rem;
    margin: .5rem 0;
}

div.stButton > button {
    border-radius: 14px;
    border: 1px solid var(--line);
    background: linear-gradient(180deg, #ffffff, #f3efe7);
    color: var(--text);
    font-weight: 600;
    min-height: 44px;
}

div.stButton > button:hover {
    border-color: rgba(11,107,111,.35);
    color: var(--teal);
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div,
textarea, input {
    border-radius: 14px !important;
}

hr {
    border: none;
    height: 1px;
    background: var(--line);
    margin: .9rem 0 1rem;
}

@media (max-width: 900px) {
    .meta-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# Login view
# =========================================================

if not st.session_state.authenticated:
    left, center, right = st.columns([1, 1.35, 1])

    with center:
        st.markdown(
            """
            <div class="brand-wrap" style="margin-top:3rem;">
                <div class="brand-row">
                    <div class="brand-mark">⟡</div>
                    <div>
                        <div class="brand-title">GIAp</div>
                        <div class="brand-sub">Guided Image Analyzer</div>
                    </div>
                </div>
                <div class="kicker">Sign in</div>
                <p class="subtle" style="margin:.2rem 0 .7rem 0;">A clean start for image-based geology help.</p>
                <div class="byline">by We are dougalien</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            entered = st.text_input("Password", type="password", placeholder="Enter app password")
            submit = st.form_submit_button("Enter")

        if submit:
            actual = app_password()
            if not actual:
                st.error("APP_PASSWORD is not set in Streamlit secrets or environment.")
            elif entered == actual:
                st.session_state.authenticated = True
                st.session_state.login_error = ""
                st.rerun()
            else:
                st.session_state.login_error = "Incorrect password."

        if st.session_state.login_error:
            st.error(st.session_state.login_error)

        st.markdown(
            """
            <div class="panel" style="margin-top:1rem;">
                <div class="smallcap">Secrets</div>
                <div class="subtle">Put your password and API keys in <code>.streamlit/secrets.toml</code>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-row">
                <div class="brand-mark">⟡</div>
                <div>
                    <div class="brand-title">GIAp</div>
                    <div class="brand-sub">Guided Image Analyzer</div>
                </div>
            </div>
            <div class="byline">by We are dougalien</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:.65rem;'></div>", unsafe_allow_html=True)

    st.markdown('<div class="smallcap">Counsel</div>', unsafe_allow_html=True)
    st.session_state.counsel_mode = st.radio(
        "Mode",
        list(COUNSEL_MODES.keys()),
        index=list(COUNSEL_MODES.keys()).index(st.session_state.counsel_mode),
        label_visibility="collapsed",
    )

    st.markdown('<div class="smallcap">Roles</div>', unsafe_allow_html=True)
    st.caption(f"Observer: OpenAI — {st.session_state.openai_model}")
    st.caption(f"Validator: Claude — {st.session_state.claude_model}")
    st.caption(f"Judge: Perplexity — {st.session_state.perplexity_model}")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="smallcap">Session</div>', unsafe_allow_html=True)
    if st.button("Clear specimen", use_container_width=True):
        reset_specimen()
        st.rerun()

    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="smallcap">Config</div>', unsafe_allow_html=True)
    st.caption(f"APP_PASSWORD {secret_status('APP_PASSWORD', app_password())}")
    st.caption(f"OPENAI_API_KEY {secret_status('OPENAI_API_KEY', get_openai_key())}")
    st.caption(f"CLAUDE_API_KEY {secret_status('CLAUDE_API_KEY', get_claude_key())}")
    st.caption(f"PERPLEXITY_API_KEY {secret_status('PERPLEXITY_API_KEY', get_perplexity_key())}")

    with st.expander("secrets.toml"):
        st.code(
            """APP_PASSWORD = "your_password_here"
OPENAI_API_KEY = "sk-..."
CLAUDE_API_KEY = "sk-ant-..."
PERPLEXITY_API_KEY = "pplx-..."
""",
            language="toml",
        )

# =========================================================
# Main UI
# =========================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="kicker">Image lab</div>
        <h1 style="margin:.1rem 0 .35rem 0;">GIAp</h1>
        <p class="subtle" style="margin:0 0 .35rem 0;">Short, grounded help for specimen photos.</p>
        <div class="byline">by We are dougalien</div>
    </div>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.05, 1], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Specimen")
    st.caption("Keep it simple and visible.")

    uploaded_file = st.file_uploader(
        "Upload photo",
        type=["png", "jpg", "jpeg", "webp"],
        help="Use a clear, well-lit image.",
    )

    if uploaded_file is not None:
        signature = (uploaded_file.name, uploaded_file.size)
        if signature != st.session_state.last_uploaded_signature:
            update_uploaded_image(uploaded_file)
            st.session_state.last_uploaded_signature = signature

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.mode = st.selectbox(
            "Type",
            ["Auto", "Rock", "Mineral", "Fossil", "Sand/Granular", "Forensic"],
            index=["Auto", "Rock", "Mineral", "Fossil", "Sand/Granular", "Forensic"].index(st.session_state.mode),
        )
    with c2:
        st.session_state.specimen_label = st.text_input(
            "Label",
            value=st.session_state.specimen_label,
            placeholder="Sample ID",
        )

    st.session_state.student_best_answer = st.text_input(
        "Student answer",
        value=st.session_state.student_best_answer,
        placeholder="What does the student think it is?",
    )

    st.session_state.known_name = st.text_input(
        "Known name",
        value=st.session_state.known_name,
        placeholder="Instructor answer, if known",
    )

    st.session_state.student_name = st.text_input(
        "Student name",
        value=st.session_state.student_name,
        placeholder="Optional",
    )

    st.session_state.student_observations = st.text_area(
        "Observations",
        value=st.session_state.student_observations,
        placeholder="Color, layers, shell pattern, grain size, sparkle, shape...",
        height=100,
    )

    st.session_state.context_notes = st.text_area(
        "Notes",
        value=st.session_state.context_notes,
        placeholder="Anything extra you want the app to know.",
        height=85,
    )

    z1, z2 = st.columns([1, 1])
    with z1:
        st.session_state.include_auto_zoom = st.checkbox(
            "Auto zoom",
            value=st.session_state.include_auto_zoom,
            help="Adds a center crop for texture/detail.",
        )
    with z2:
        st.session_state.zoom_fraction = st.slider(
            "Zoom",
            min_value=0.2,
            max_value=1.0,
            value=float(st.session_state.zoom_fraction),
            step=0.05,
        )

    if st.button("Analyze", type="primary", use_container_width=True):
        try:
            analyze_with_counsel()
        except Exception as e:
            st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.image_bytes:
        st.markdown('<div class="panel" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.subheader("Preview")
        st.image(st.session_state.image_bytes, caption=st.session_state.image_name, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Reply")

    if st.session_state.final_reply:
        st.markdown(
            f"""
            <div class="reply-box">
                <div class="smallcap">Student-facing reply</div>
                <div style="font-size:1.03rem; line-height:1.65;">{st.session_state.final_reply}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Upload a specimen photo and run an analysis.")

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Readout")
    meta = st.session_state.get("analysis_meta", {})
    observer_json = st.session_state.get("observer_json")
    validator_json = st.session_state.get("validator_json")
    judge_json = st.session_state.get("judge_json")

    st.markdown('<div class="meta-grid">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="meta-pill">
            <div class="smallcap">Mode</div>
            <div>{st.session_state.counsel_mode}</div>
            <div class="subtle">{COUNSEL_MODES[st.session_state.counsel_mode]['label']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    obs_id = observer_json.get("likely_identification", "—") if observer_json else "—"
    st.markdown(
        f"""
        <div class="meta-pill">
            <div class="smallcap">Observer</div>
            <div>{obs_id}</div>
            <div class="subtle">{meta.get("observer_model") or "—"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    judge_line = (
        judge_json.get("final_identification", "—")
        if judge_json else
        (validator_json.get("likely_identification", "—") if validator_json else "—")
    )
    st.markdown(
        f"""
        <div class="meta-pill">
            <div class="smallcap">Final</div>
            <div>{judge_line}</div>
            <div class="subtle">{meta.get("judge_model") or meta.get("validator_model") or "single model"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if observer_json:
        with st.expander("Observer details"):
            st.json(observer_json)
    if validator_json:
        with st.expander("Validator details"):
            st.json(validator_json)
    if judge_json:
        with st.expander("Judge details"):
            st.json(judge_json)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel" style="margin-top:1rem;">', unsafe_allow_html=True)
    st.subheader("Follow-up")

    followup = st.text_input(
        "Ask a follow-up",
        placeholder="What visible feature should the student check next?",
        key="followup_input",
    )

    if st.button("Send follow-up", use_container_width=True):
        if not st.session_state.started:
            st.warning("Run the first analysis before sending a follow-up.")
        else:
            try:
                send_followup(followup)
                st.rerun()
            except Exception as e:
                st.error(str(e))

    if st.session_state.display_messages:
        st.markdown("<hr>", unsafe_allow_html=True)
        for msg in st.session_state.display_messages:
            cls = "chat-user" if msg["role"] == "user" else "chat-ai"
            label = "You" if msg["role"] == "user" else "GIAp"
            st.markdown(
                f"""
                <div class="{cls}">
                    <div class="smallcap">{label}</div>
                    <div>{msg["content"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)