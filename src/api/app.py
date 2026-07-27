"""
AinSeba - FastAPI Backend
Production-ready REST API wrapping the entire RAG pipeline.

Endpoints:
    GET  /                   — Service banner + link to docs
    POST /api/query          — Submit a legal question (bilingual)
    POST /api/query/stream   — Streaming response (SSE), now with sources
    GET  /api/health         — Health check + vector store status
    GET  /api/sources        — List available law documents
    POST /api/feedback       — User feedback on response quality
    GET  /api/session/{id}   — Get conversation history for a session
    DELETE /api/session/{id} — Clear a conversation session

Run with:
    uvicorn src.api.app:app --reload --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from src.models.schemas import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    HealthResponse,
    SourceListResponse,
    FeedbackRequest,
    FeedbackResponse,
    ErrorResponse,
)
from src.api.rate_limiter import RateLimiter
from src.config import (
    OPENAI_API_KEY,
    LLM_MODEL,
    CHROMA_COLLECTION_NAME,
    LAW_REGISTRY,
    CORS_ORIGINS,
    API_RATE_LIMIT,
    API_RATE_WINDOW,
    USE_RERANKER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================
# Global State (initialized at startup)
# ============================================
_bilingual_chain = None
_feedback_store: list[dict] = []
_rate_limiter = RateLimiter(
    max_requests=API_RATE_LIMIT,
    window_seconds=API_RATE_WINDOW,
)


def _get_bilingual_chain():
    """Lazy-load the bilingual chain (warmed at startup, so normally a no-op)."""
    global _bilingual_chain
    if _bilingual_chain is None:
        from src.chain.builder import build_bilingual_chain
        _bilingual_chain = build_bilingual_chain(use_reranker=USE_RERANKER)
    return _bilingual_chain


def _get_rag_chain():
    """Lazy-load the base RAG chain (from bilingual wrapper)."""
    return _get_bilingual_chain().rag_chain


def _serialize_sources(raw_sources: list[dict]) -> list[SourceInfo]:
    """Map retriever source dicts onto the Pydantic response model."""
    return [
        SourceInfo(
            citation=src.get("citation", ""),
            act_name=src.get("act_name", ""),
            act_id=src.get("act_id", ""),
            section_number=str(src.get("section_number", "")),
            section_title=src.get("section_title", ""),
            chapter=src.get("chapter", ""),
            similarity_score=src.get("similarity_score", 0.0) or 0.0,
            rerank_score=src.get("rerank_score", 0.0) or 0.0,
        )
        for src in raw_sources
    ]


# ============================================
# App Lifespan
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.

    The chain is built eagerly here. Building it lazily on the first request
    meant the cross-encoder reranker (~90MB of torch weights) downloaded and
    loaded mid-request, so the very first question took 30-60s and often hit
    the client timeout. Paying that cost at boot keeps every request fast.
    """
    logger.info("AinSeba API starting up...")

    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set! Query endpoints will return 503.")
    else:
        try:
            chain = _get_bilingual_chain()
            count = chain.rag_chain.retriever.store.collection.count()
            logger.info(
                f"Warm-up complete. Vector store holds {count} documents. "
                f"Reranker: {'enabled' if USE_RERANKER else 'DISABLED'}."
            )
            if count == 0:
                logger.warning(
                    "Vector store is EMPTY. Run: "
                    "python -m src.vectorstore.populate --source data/processed/all_chunks_combined.json"
                )
        except Exception as e:
            logger.error(f"Warm-up failed: {e}", exc_info=True)

    yield
    logger.info("AinSeba API shutting down.")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="AinSeba API",
    description=(
        "Bangladesh Legal Aid RAG Assistant API. "
        "Ask legal questions in English, Bangla, or Banglish and receive "
        "citation-grounded answers from Bangladesh law."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    responses={
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)

# CORS middleware.
# allow_credentials=True is incompatible with the "*" wildcard: browsers reject
# the combination outright, so credentials are only enabled for an explicit
# origin list.
_wildcard = "*" in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not _wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Middleware: Request Logging
# ============================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"-> {response.status_code} ({duration:.2f}s)"
    )
    return response


# ============================================
# Helper: Rate Limit Check
# ============================================

def _check_rate_limit(request: Request):
    """Check rate limit and raise 429 if exceeded."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(client_ip):
        remaining = _rate_limiter.remaining(client_ip)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Try again in {_rate_limiter.window_seconds} seconds.",
            headers={"X-RateLimit-Remaining": str(remaining)},
        )


# ============================================
# Endpoints
# ============================================

@app.get("/", include_in_schema=False)
async def root():
    """Service banner so hitting the bare host is not a 404."""
    return {
        "service": "AinSeba API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Ask a legal question",
    description="Submit a legal question in English, Bangla, or Banglish. "
    "Returns a citation-grounded answer with source references.",
    tags=["Query"],
)
async def query(request: Request, body: QueryRequest):
    """Process a legal question through the bilingual RAG pipeline."""
    _check_rate_limit(request)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured")

    try:
        chain = _get_bilingual_chain()

        response = chain.query(
            question=body.question,
            session_id=body.session_id,
            response_language=body.language,
            act_id=body.act_id,
            category=body.category,
            use_reranker=body.use_reranker,
        )

        return QueryResponse(
            answer=response.answer,
            answer_english=response.answer_english,
            sources=_serialize_sources(response.sources),
            query_original=response.query_original,
            query_english=response.query_english,
            detected_language=response.detected_language,
            response_language=response.response_language,
            was_translated=response.was_translated,
            session_id=response.session_id,
            model=response.model,
            retrieval_count=len(response.sources),
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/query/stream",
    summary="Ask a legal question (streaming)",
    description="Submit a legal question and receive the answer as Server-Sent "
    "Events. Emits: metadata -> token* -> sources -> done.",
    tags=["Query"],
)
async def query_stream(request: Request, body: QueryRequest):
    """
    Stream a legal answer using Server-Sent Events.

    Two corrections versus the original implementation:

    1. Sources were never sent to the client, so a streaming UI showed answers
       with no citations -- the whole point of the system. A `sources` event is
       now emitted once the answer completes.

    2. The stream always emitted English. A Bangla question therefore produced
       a Bangla answer on /api/query but an English one here. Token-level
       streaming cannot be translated mid-flight, so when the target language
       is not English the endpoint runs the full bilingual query and emits the
       finished answer as a single `answer` event. The client renders both the
       same way.
    """
    _check_rate_limit(request)

    if not OPENAI_API_KEY:
        raise HTTPException(status_code=503, detail="API key not configured")

    try:
        chain = _get_bilingual_chain()

        from src.language.detector import detect_language
        detection = detect_language(body.question)

        target_lang = body.language or chain.default_response_language
        if target_lang == "auto":
            target_lang = detection.response_language

        english_query = body.question
        if detection.needs_translation:
            english_query = chain.translator.translate_query_to_english(
                body.question, detection.language.value
            )

        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        async def event_generator():
            """Generate SSE events."""
            yield sse({
                "type": "metadata",
                "detected_language": detection.language.value,
                "response_language": target_lang,
                "query_english": english_query,
                "was_translated": detection.needs_translation,
                "streaming": target_lang == "en",
            })

            try:
                if target_lang == "en":
                    # True token streaming.
                    rag_chain = chain.rag_chain
                    generator = rag_chain.stream(
                        english_query,
                        session_id=body.session_id,
                        act_id=body.act_id,
                        category=body.category,
                        use_reranker=body.use_reranker,
                    )
                    for token in generator:
                        yield sse({"type": "token", "content": token})

                    # The chain records the last retrieval, so citations can be
                    # emitted after the answer finishes.
                    # rag_chain.stream() records its retrieval on last_sources
                    # (see the two-line patch in src/chain/rag_chain.py).
                    raw_sources = getattr(rag_chain, "last_sources", []) or []
                else:
                    # Non-English target: translation needs the complete answer.
                    result = chain.query(
                        question=body.question,
                        session_id=body.session_id,
                        response_language=target_lang,
                        act_id=body.act_id,
                        category=body.category,
                        use_reranker=body.use_reranker,
                    )
                    yield sse({"type": "answer", "content": result.answer})
                    raw_sources = result.sources

                yield sse({
                    "type": "sources",
                    "sources": [s.model_dump() for s in _serialize_sources(raw_sources)],
                })
                yield sse({"type": "done"})

            except Exception as inner:
                logger.error(f"Stream failed mid-flight: {inner}", exc_info=True)
                yield sse({"type": "error", "message": str(inner)})

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # stops nginx buffering the stream
            },
        )

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and vector store status.",
    tags=["System"],
)
async def health():
    """Health check endpoint."""
    doc_count = 0
    acts = []

    try:
        chain = _get_bilingual_chain()
        stats = chain.rag_chain.retriever.store.get_stats()
        doc_count = stats.get("total_documents", 0)
        acts = stats.get("acts", [])
    except Exception as e:
        logger.warning(f"Health check could not reach the vector store: {e}")

    return HealthResponse(
        status="ok" if doc_count > 0 else "degraded",
        version="1.0.0",
        vector_store_documents=doc_count,
        vector_store_acts=acts,
        model=LLM_MODEL,
    )


@app.get(
    "/api/sources",
    response_model=SourceListResponse,
    summary="List available laws",
    description="List all law documents available in the system.",
    tags=["System"],
)
async def list_sources():
    """List available law documents and categories."""
    doc_count = 0
    acts_in_store = []

    try:
        chain = _get_bilingual_chain()
        stats = chain.rag_chain.retriever.store.get_stats()
        doc_count = stats.get("total_documents", 0)
        acts_in_store = stats.get("acts", [])
    except Exception as e:
        logger.warning(f"Source listing could not reach the vector store: {e}")

    acts = [
        {
            "id": law["id"],
            "name": law["name"],
            "category": law["category"],
            "year": law["year"],
            "priority": law["priority"],
            "indexed": law["id"] in acts_in_store,
        }
        for law in LAW_REGISTRY
    ]

    # Only offer categories the user can actually get answers from.
    indexed_categories = sorted({
        law["category"] for law in LAW_REGISTRY if law["id"] in acts_in_store
    })
    categories = indexed_categories or sorted({law["category"] for law in LAW_REGISTRY})

    return SourceListResponse(
        total_documents=doc_count,
        acts=acts,
        categories=categories,
    )


@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit feedback on a response quality (1-5 rating).",
    tags=["Feedback"],
)
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Store user feedback for quality tracking."""
    _check_rate_limit(request)

    from datetime import datetime

    _feedback_store.append({
        "timestamp": datetime.now().isoformat(),
        "query": body.query,
        "answer_preview": body.answer[:200],
        "rating": body.rating,
        "comment": body.comment,
        "session_id": body.session_id,
    })
    logger.info(f"Feedback received: rating={body.rating}, query='{body.query[:50]}'")

    return FeedbackResponse(
        status="received",
        message="Thank you for your feedback!",
    )


@app.get(
    "/api/session/{session_id}",
    summary="Get conversation history",
    description="Retrieve conversation history for a session.",
    tags=["Session"],
)
async def get_session(session_id: str):
    """Get conversation history for a session."""
    try:
        chain = _get_bilingual_chain()
        history = chain.rag_chain.get_conversation_history(session_id)
        return {"session_id": session_id, "messages": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/session/{session_id}",
    summary="Clear conversation history",
    description="Clear server-side memory for a session.",
    tags=["Session"],
)
async def clear_session(session_id: str):
    """
    Clear a session's conversation memory.

    Without this, 'Clear Chat' in the UI only wiped the browser's copy while the
    backend kept feeding the old exchanges back into every prompt.
    """
    try:
        chain = _get_bilingual_chain()
        chain.rag_chain.clear_conversation(session_id)
        return {"session_id": session_id, "status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code,
        ).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            status_code=500,
        ).model_dump(),
    )