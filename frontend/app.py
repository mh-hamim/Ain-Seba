"""
AinSeba (আইনসেবা) - Streamlit Frontend
Chat interface for the Bangladesh Legal Aid Assistant.

Connects to the FastAPI backend over HTTP.

Run with:
    streamlit run frontend/app.py
"""

import json
import os
import uuid

import requests
import streamlit as st

# ============================================
# Configuration
# ============================================

# Read from the environment so the same file works locally and deployed.
# Set AINSEBA_API_URL on Streamlit Cloud / Render / Railway to the backend URL.
API_BASE_URL = os.getenv("AINSEBA_API_URL", "http://localhost:8000").rstrip("/")

# The first request after a cold start loads the reranker weights, so the
# read timeout is generous. Connect timeout stays short to fail fast when the
# backend simply is not running.
TIMEOUT = (5, 120)

st.set_page_config(
    page_title="AinSeba - আইনসেবা",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# Custom CSS
# ============================================

st.markdown("""
<style>
    .main .block-container { max-width: 900px; padding-top: 1.5rem; }

    .app-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .app-header h1 { color: #1a5276; font-size: 2rem; margin-bottom: 0.2rem; }
    .app-header p { color: #666; font-size: 0.95rem; }

    .disclaimer-banner {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #856404;
    }

    .source-card {
        background: #f8f9fa;
        border-left: 3px solid #1a5276;
        padding: 0.5rem 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
    .source-card .citation { font-weight: 600; color: #1a5276; }
    .source-card .score { color: #888; font-size: 0.78rem; }

    section[data-testid="stSidebar"] { background: #f0f4f8; }

    .stButton > button { text-align: left !important; font-size: 0.85rem !important; }

    .lang-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }
    .lang-en { background: #d4edda; color: #155724; }
    .lang-bn { background: #cce5ff; color: #004085; }
    .lang-banglish { background: #fff3cd; color: #856404; }
    .meta-line { font-size: 0.78rem; color: #888; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State
# ============================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"st_{uuid.uuid4().hex[:12]}"
if "response_language" not in st.session_state:
    st.session_state.response_language = "auto"


# ============================================
# API Helpers
# ============================================

def _payload(question: str, language, act_id, category) -> dict:
    body = {
        "question": question,
        "session_id": st.session_state.session_id,
        "language": language,
        "use_reranker": True,
    }
    if act_id:
        body["act_id"] = act_id
    if category:
        body["category"] = category
    return body


def stream_api(question: str, language=None, act_id=None, category=None):
    """
    Call the streaming endpoint and yield parsed SSE events.

    Yields dicts of the form {"type": "token"|"answer"|"sources"|"metadata"|
    "done"|"error", ...}. Falls back to the blocking endpoint if the stream
    cannot be established, so the UI still works against an older backend.
    """
    try:
        with requests.post(
            f"{API_BASE_URL}/api/query/stream",
            json=_payload(question, language, act_id, category),
            stream=True,
            timeout=TIMEOUT,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            if resp.status_code == 429:
                yield {"type": "error", "message":
                       "Rate limit exceeded. Please wait a moment and try again."}
                return
            if resp.status_code != 200:
                yield {"type": "error", "message":
                       f"Server error ({resp.status_code}): {resp.text[:300]}"}
                return

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                try:
                    yield json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue

    except requests.ConnectionError:
        yield {"type": "error", "message":
               "Cannot connect to the AinSeba API. Start it with:\n\n"
               "`uvicorn src.api.app:app --port 8000`"}
    except requests.Timeout:
        yield {"type": "error", "message":
               "Request timed out. The backend may still be warming up — try again."}
    except Exception as e:
        yield {"type": "error", "message": f"Unexpected error: {e}"}


def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def get_available_sources():
    try:
        resp = requests.get(f"{API_BASE_URL}/api/sources", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def submit_feedback(query: str, answer: str, rating: int, comment=None):
    try:
        requests.post(
            f"{API_BASE_URL}/api/feedback",
            json={
                "query": query,
                "answer": answer,
                "rating": rating,
                "comment": comment,
                "session_id": st.session_state.session_id,
            },
            timeout=5,
        )
    except Exception:
        pass


def clear_server_session():
    """Clear backend memory too, not just the browser's copy."""
    try:
        requests.delete(
            f"{API_BASE_URL}/api/session/{st.session_state.session_id}",
            timeout=5,
        )
    except Exception:
        pass


# ============================================
# Render Helpers
# ============================================

def render_sources(sources: list, key_prefix: str):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            citation = src.get("citation") or "Unknown"
            sim = src.get("similarity_score") or 0
            rerank = src.get("rerank_score") or 0
            score_text = f"similarity: {sim:.3f}"
            if rerank:
                score_text += f" | rerank: {rerank:.3f}"
            st.markdown(
                f'<div class="source-card">'
                f'<span class="citation">[{i}] {citation}</span><br>'
                f'<span class="score">{score_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_language_badge(lang: str) -> str:
    labels = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}
    css_class = f"lang-{lang}" if lang in labels else "lang-en"
    return f'<span class="lang-badge {css_class}">{labels.get(lang, lang)}</span>'


def render_meta(detected: str, translated: bool):
    extra = " (translated)" if translated else ""
    st.markdown(
        f'<div class="meta-line">Detected: {render_language_badge(detected)}{extra}</div>',
        unsafe_allow_html=True,
    )


def answer_question(question: str, act_filter, category_filter):
    """
    Run one turn: stream the answer into the placeholder, then persist it.

    Both the chat box and the example buttons route through here — the original
    duplicated this block, so any fix had to be applied twice.
    """
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Searching Bangladesh law…_")

        lang = st.session_state.response_language
        answer = ""
        sources = []
        detected = "en"
        translated = False
        error = None

        for event in stream_api(
            question,
            language=lang if lang != "auto" else None,
            act_id=act_filter,
            category=category_filter,
        ):
            etype = event.get("type")
            if etype == "metadata":
                detected = event.get("detected_language", "en")
                translated = event.get("was_translated", False)
            elif etype == "token":
                answer += event.get("content", "")
                placeholder.markdown(answer + "▌")
            elif etype == "answer":
                answer = event.get("content", "")
                placeholder.markdown(answer)
            elif etype == "sources":
                sources = event.get("sources", [])
            elif etype == "error":
                error = event.get("message", "Unknown error")
                break

        if error:
            placeholder.empty()
            st.error(error)
            st.session_state.messages.append(
                {"role": "assistant", "content": error}
            )
            return

        placeholder.markdown(answer or "No answer received.")
        render_sources(sources, key_prefix="live")
        render_meta(detected, translated)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "answer": answer,
            "sources": sources,
            "query": question,
            "detected_language": detected,
            "was_translated": translated,
        })


# ============================================
# Sidebar
# ============================================

with st.sidebar:
    st.markdown("### Settings")

    lang_options = {"Auto-detect": "auto", "English": "en", "Bangla (বাংলা)": "bn"}
    selected_lang = st.selectbox(
        "Response Language",
        options=list(lang_options.keys()),
        index=0,
        help="Auto-detect matches the language of your question.",
    )
    st.session_state.response_language = lang_options[selected_lang]

    sources_data = get_available_sources()
    act_filter = None
    category_filter = None

    if sources_data:
        acts = sources_data.get("acts", [])
        indexed_acts = [a for a in acts if a.get("indexed")]
        categories = sources_data.get("categories", [])

        if indexed_acts:
            act_names = ["All Acts"] + [a["name"] for a in indexed_acts]
            selected_act = st.selectbox("Filter by Act", act_names, index=0)
            if selected_act != "All Acts":
                act_filter = next(
                    (a["id"] for a in indexed_acts if a["name"] == selected_act), None
                )

        if categories:
            cat_names = ["All Categories"] + categories
            selected_cat = st.selectbox("Filter by Category", cat_names, index=0)
            if selected_cat != "All Categories":
                category_filter = selected_cat

    st.markdown("---")
    st.markdown("### Example Questions")

    example_questions = [
        "What is the penalty for theft?",
        "What are the maximum daily working hours?",
        "My employer hasn't paid me for 3 months. What can I do?",
        "চুরির শাস্তি কী?",
        "amar malik betan dey nai, ki korbo?",
        "What are the rules for maternity leave?",
        "What counts as criminal breach of trust?",
    ]

    # Index-based keys. The original hashed the question text, and hash() is
    # salted per process, so keys shifted between runs.
    for idx, q in enumerate(example_questions):
        if st.button(q, key=f"ex_{idx}", use_container_width=True):
            st.session_state.pending_question = q

    st.markdown("---")

    health = check_api_health()
    if health:
        doc_count = health.get("vector_store_documents", 0)
        if doc_count:
            st.success(f"API connected · {doc_count} passages indexed")
        else:
            st.warning("API connected, but the vector store is empty.")
    else:
        st.error(f"API offline at {API_BASE_URL}\n\n`uvicorn src.api.app:app --port 8000`")

    if st.button("Clear Chat", use_container_width=True):
        clear_server_session()
        st.session_state.messages = []
        st.session_state.session_id = f"st_{uuid.uuid4().hex[:12]}"
        st.rerun()


# ============================================
# Main Chat Area
# ============================================

st.markdown(
    '<div class="app-header">'
    '<h1>AinSeba আইনসেবা</h1>'
    '<p>Bangladesh Legal Aid Assistant — ask about Bangladesh law in English, Bangla, or Banglish</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer-banner">'
    '<strong>Disclaimer:</strong> This tool provides legal information for educational '
    'purposes only. It does not constitute legal advice. For specific legal matters, '
    'please consult a qualified lawyer.'
    '</div>',
    unsafe_allow_html=True,
)

# Chat history. Enumerating gives every widget a unique, stable key — the
# original derived keys from the first 50 characters of the answer, and legal
# answers frequently share an opening, which crashed Streamlit with a duplicate
# element key.
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("sources"):
                render_sources(msg["sources"], key_prefix=f"hist{idx}")
            if msg.get("detected_language"):
                render_meta(msg["detected_language"], msg.get("was_translated", False))
            if msg.get("answer"):
                cols = st.columns([1, 1, 8])
                with cols[0]:
                    if st.button("👍", key=f"fb_up_{idx}", help="Helpful"):
                        submit_feedback(msg.get("query", ""), msg["answer"], 5)
                        st.toast("Thanks for the feedback!")
                with cols[1]:
                    if st.button("👎", key=f"fb_down_{idx}", help="Not helpful"):
                        submit_feedback(msg.get("query", ""), msg["answer"], 1)
                        st.toast("Thanks — we'll work on improving!")

# Example-button question
if "pending_question" in st.session_state:
    answer_question(st.session_state.pop("pending_question"), act_filter, category_filter)
    st.rerun()

# Chat input
if question := st.chat_input("Ask a legal question in English, Bangla, or Banglish..."):
    answer_question(question, act_filter, category_filter)
    st.rerun()