"""
AinSeba (আইনসেবা) - Gradio frontend for Hugging Face Spaces.

Unlike frontend/app.py (Streamlit), this talks to the RAG chain in-process
rather than over HTTP. One service, no CORS, no cold-start handshake between
a separate frontend and backend.

The FastAPI backend in src/api/ remains the reference interface for programmatic
use; this module is the hosted demo surface.

Run locally:
    python app.py
"""

import logging
import os
import threading
import uuid

import gradio as gr

from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    USE_RERANKER,
    LAW_REGISTRY,
)

# Import the chain modules here, on the main thread, before the warm-up thread
# starts. Importing them from that thread instead races Gradio's own imports:
# both pull in pydantic, and importing a partially-initialised package from two
# threads raises "KeyError: 'pydantic.v1'" out of importlib. Object
# construction is still deferred to the background thread -- that part is slow
# but thread-safe.
try:
    from src.chain.builder import build_bilingual_chain
    from src.language.detector import detect_language
    _import_error = None
except Exception as _e:  # missing deps, broken install
    build_bilingual_chain = None
    detect_language = None
    _import_error = str(_e)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================
# Chain bootstrap
# ============================================

_chain = None
_chain_error: str | None = None
_chain_lock = threading.Lock()
_store_stats: dict = {}


def _build_chain():
    """Build the bilingual chain once. Safe to call from several threads."""
    global _chain, _chain_error, _store_stats

    with _chain_lock:
        if _chain is not None or _chain_error is not None:
            return _chain

        if _import_error is not None:
            _chain_error = f"Could not import the RAG chain: {_import_error}"
            logger.error(_chain_error)
            return None

        if not OPENAI_API_KEY:
            _chain_error = (
                "OPENAI_API_KEY is not set. Add it under "
                "Settings → Variables and secrets as a **Secret**."
            )
            logger.error(_chain_error)
            return None

        try:
            logger.info("Building chain (loading reranker may take ~30s)...")
            _chain = build_bilingual_chain(use_reranker=USE_RERANKER)

            _store_stats = _chain.rag_chain.retriever.store.get_stats()
            count = _store_stats.get("total_documents", 0)

            # builder.py swallows a reranker load failure and proceeds without
            # it, so report the object that actually exists rather than the
            # flag that was requested.
            live = getattr(_chain.rag_chain.retriever, "reranker", None) is not None
            if USE_RERANKER and not live:
                state = "REQUESTED BUT FAILED TO LOAD"
            elif live:
                state = "enabled"
            else:
                state = "disabled"

            logger.info(f"Ready. {count} passages indexed. Reranker: {state}.")
            if count == 0:
                _chain_error = (
                    "The vector store is empty. The ChromaDB index did not ship "
                    "with this deployment."
                )
        except Exception as e:
            _chain_error = f"Failed to initialise: {e}"
            logger.error(_chain_error, exc_info=True)

        return _chain


# Warm in the background so Gradio can bind its port immediately and the first
# visitor does not pay the model-loading cost.
threading.Thread(target=_build_chain, daemon=True).start()


# ============================================
# Helpers
# ============================================

LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Bangla (বাংলা)": "bn",
}

EXAMPLES = [
    "What is the penalty for theft?",
    "What are the maximum daily working hours?",
    "My employer hasn't paid me for 3 months. What can I do?",
    "How much casual leave am I entitled to?",
    "চুরির শাস্তি কী?",
    "amar malik betan dey nai, ki korbo?",
]

DISCLAIMER = (
    "**Educational information only — not legal advice.** "
    "The corpus reflects the law as printed in its source PDFs and does not "
    "track later amendments or judicial interpretation. "
    "Verify against [bdlaws.minlaw.gov.bd](http://bdlaws.minlaw.gov.bd/) and "
    "consult a qualified lawyer for any actual matter."
)


def _act_choices() -> list[str]:
    """Only offer acts that are actually in the index."""
    indexed = set(_store_stats.get("acts", []))
    names = [law["name"] for law in LAW_REGISTRY if law["id"] in indexed]
    return ["All Acts"] + sorted(names)


def _category_choices() -> list[str]:
    indexed = set(_store_stats.get("acts", []))
    cats = {law["category"] for law in LAW_REGISTRY if law["id"] in indexed}
    return ["All Categories"] + sorted(cats)


def _act_id_for(name: str) -> str | None:
    if not name or name == "All Acts":
        return None
    return next((law["id"] for law in LAW_REGISTRY if law["name"] == name), None)


def _render_sources(sources: list[dict]) -> str:
    """Format retrieved passages as a markdown block."""
    if not sources:
        return "_No sources for this response._"

    lines = []
    for i, src in enumerate(sources, 1):
        citation = src.get("citation") or "Unknown source"
        sim = src.get("similarity_score") or 0.0
        rerank = src.get("rerank_score")
        scores = f"similarity {sim:.3f}"
        if rerank:
            scores += f" · rerank {rerank:.3f}"
        preview = (src.get("text_preview") or "").strip().replace("\n", " ")
        if preview:
            preview = f"<br><small>{preview[:220]}…</small>"
        lines.append(f"**[{i}] {citation}**<br><small>{scores}</small>{preview}")

    return "\n\n".join(lines)


# ============================================
# Chat handler
# ============================================

def respond(message, history, language_label, act_label, category_label, session_id):
    """
    Stream one answer into the chat history.

    English streams token by token. A non-English target cannot be streamed —
    translation needs the finished answer — so it arrives in one piece. Both
    render identically.
    """
    history = list(history or [])

    if not message or not message.strip():
        yield history, "", ""
        return

    chain = _build_chain()
    if chain is None:
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ {_chain_error}"},
        ]
        yield history, "", ""
        return

    act_id = _act_id_for(act_label)
    category = None if category_label in (None, "All Categories") else category_label
    target = LANGUAGES.get(language_label)

    history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, "_Searching Bangladesh law…_", ""

    try:
        detection = detect_language(message)
        resolved = target or (
            detection.response_language
            if chain.default_response_language == "auto"
            else chain.default_response_language
        )

        if resolved == "en":
            english_query = message
            if detection.needs_translation:
                english_query = chain.translator.translate_query_to_english(
                    message, detection.language.value
                )

            rag = chain.rag_chain
            answer = ""
            for token in rag.stream(
                english_query,
                session_id=session_id,
                act_id=act_id,
                category=category,
                use_reranker=USE_RERANKER,
            ):
                answer += token
                history[-1]["content"] = answer
                yield history, "_Searching Bangladesh law…_", ""

            sources = getattr(rag, "last_sources", []) or []
        else:
            result = chain.query(
                question=message,
                session_id=session_id,
                response_language=resolved,
                act_id=act_id,
                category=category,
                use_reranker=USE_RERANKER,
            )
            history[-1]["content"] = result.answer
            sources = result.sources

        badge = {"en": "English", "bn": "Bangla", "banglish": "Banglish"}.get(
            detection.language.value, detection.language.value
        )
        meta = f"Detected: **{badge}**"
        if detection.needs_translation:
            meta += " · translated"

        yield history, _render_sources(sources), meta

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        history[-1]["content"] = f"⚠️ Something went wrong: {e}"
        yield history, "", ""


def clear_chat(session_id):
    """Reset both the visible transcript and the chain's server-side memory."""
    chain = _build_chain()
    if chain is not None:
        try:
            chain.rag_chain.clear_conversation(session_id)
        except Exception:
            pass
    return [], "", "", f"gr_{uuid.uuid4().hex[:12]}"


# ============================================
# Interface
# ============================================

CSS = """
.gradio-container { max-width: 1100px !important; }
#title h1 { margin-bottom: 0.1rem; }
#title p { color: #6b7280; margin-top: 0; }
footer { display: none !important; }
"""

THEME = gr.themes.Soft(primary_hue="blue")

# Gradio 6 moved `css` and `theme` off the Blocks constructor onto launch().
with gr.Blocks(title="AinSeba — Bangladesh Legal Aid") as demo:

    session_id = gr.State(f"gr_{uuid.uuid4().hex[:12]}")

    gr.Markdown(
        "# ⚖️ AinSeba (আইনসেবা)\n"
        "Ask about Bangladesh law in **English, Bangla, or Banglish** — "
        "every answer cites the sections it relies on.",
        elem_id="title",
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                avatar_images=(None, None),
            )
            meta = gr.Markdown("")

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a legal question…",
                    show_label=False,
                    scale=8,
                    autofocus=True,
                )
                send = gr.Button("Send", variant="primary", scale=1)
                clear = gr.Button("Clear", scale=1)

            gr.Examples(examples=EXAMPLES, inputs=msg, label="Try one")

        with gr.Column(scale=2):
            language = gr.Dropdown(
                choices=list(LANGUAGES.keys()),
                value="Auto-detect",
                label="Response language",
            )
            act = gr.Dropdown(
                choices=_act_choices(),
                value="All Acts",
                label="Filter by act",
            )
            category = gr.Dropdown(
                choices=_category_choices(),
                value="All Categories",
                label="Filter by category",
            )
            with gr.Accordion("Sources", open=True):
                sources_md = gr.Markdown("_Ask a question to see citations._")

    gr.Markdown(DISCLAIMER)

    inputs = [msg, chatbot, language, act, category, session_id]
    outputs = [chatbot, sources_md, meta]

    msg.submit(respond, inputs, outputs).then(lambda: "", None, msg)
    send.click(respond, inputs, outputs).then(lambda: "", None, msg)
    clear.click(clear_chat, session_id, [chatbot, sources_md, meta, session_id])

    def _refresh_filters():
        return (
            gr.update(choices=_act_choices(), value="All Acts"),
            gr.update(choices=_category_choices(), value="All Categories"),
        )

    # The store stats are only known once the background warm-up finishes, so
    # populate the filters when the page loads rather than at import time.
    demo.load(_refresh_filters, None, [act, category])


if __name__ == "__main__":
    demo.queue(max_size=32).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        css=CSS,
        theme=THEME,
    )