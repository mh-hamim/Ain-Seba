"""
AinSeba - RAG Chain
Connects the retrieval pipeline to GPT-4o-mini for citation-grounded legal answers.

Uses LangChain's LCEL (LangChain Expression Language) for a clean,
composable pipeline: Query -> Condense -> Retrieve -> Format -> LLM -> Parse.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Generator

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.prompts.templates import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_context,
)
from src.chain.memory import ConversationMemory

logger = logging.getLogger(__name__)


# ============================================
# Follow-up Condensation
# ============================================

CONDENSE_PROMPT = """\
Rewrite the user's follow-up question as a standalone question that can be \
understood without the conversation history. Resolve every pronoun and \
reference ("that", "it", "those sections") using the history.

Rules:
- Output ONLY the rewritten question. No preamble, no explanation, no quotes.
- Keep it in English.
- Preserve the user's intent exactly. Do not answer the question.
- If the question already stands alone, return it unchanged.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""


# ============================================
# Response Data Model
# ============================================

@dataclass
class RAGResponse:
    """Structured response from the RAG chain."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    query: str = ""
    retrieval_count: int = 0
    model: str = ""
    session_id: str = "default"
    search_query: str = ""  # what was actually sent to the retriever

    def format_sources(self) -> str:
        """Format sources as a readable string."""
        if not self.sources:
            return "No sources referenced."
        lines = []
        for i, src in enumerate(self.sources, 1):
            citation = src.get("citation", "Unknown source")
            score = src.get("similarity_score", 0)
            lines.append(f"  [{i}] {citation} (relevance: {score:.2f})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "query": self.query,
            "retrieval_count": self.retrieval_count,
            "model": self.model,
            "session_id": self.session_id,
            "search_query": self.search_query,
        }


# ============================================
# RAG Chain
# ============================================

class LegalRAGChain:
    """
    Full RAG chain for AinSeba legal assistant.

    Pipeline:
    1. Take user question
    2. Condense follow-ups into standalone search queries
    3. Retrieve relevant legal context (via LegalRetriever from Phase 2)
    4. Format prompt with context + conversation history
    5. Send to GPT-4o-mini
    6. Parse and return structured response with source tracking
    7. Update conversation memory

    Features:
    - Citation-grounded answers with section references
    - Conversation memory for follow-up questions (sliding window, k=5)
    - History-aware retrieval so follow-ups actually find their subject
    - Streaming support for real-time responses
    - Source document tracking
    - Graceful handling of out-of-scope questions
    """

    # Words that make a question dependent on what came before it.
    _ANAPHORIC = re.compile(
        r"\b(that|this|those|these|it|its|there|then|same|above|previous|"
        r"former|latter|he|she|they|them|his|her|their|such)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        retriever,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        memory_k: int = 5,
        condense_followups: bool = True,
    ):
        """
        Args:
            retriever: LegalRetriever instance (from Phase 2).
            api_key: OpenAI API key.
            model: LLM model name.
            temperature: LLM temperature (low = more factual).
            max_tokens: Maximum response tokens.
            memory_k: Number of conversation exchanges to remember.
            condense_followups: Rewrite follow-up questions before retrieving.
        """
        self.retriever = retriever
        self.memory = ConversationMemory(k=memory_k)
        self.model_name = model
        self.condense_followups = condense_followups

        # Initialize LangChain LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        # Small, cheap model call for query rewriting. Low ceiling on output
        # because a rewritten question is one line.
        self.condenser = ChatOpenAI(
            model=model,
            temperature=0.0,
            max_tokens=120,
            api_key=api_key,
        )

        # Output parser
        self.parser = StrOutputParser()

        # Retrieval behind the most recent stream() call. query() returns its
        # sources inside RAGResponse, but stream() only yields tokens, so the
        # API has no other way to emit citations once the stream finishes.
        self.last_sources: list[dict] = []

        logger.info(
            f"RAG chain initialized (model={model}, temp={temperature}, memory_k={memory_k})"
        )

    # ----------------------------------------
    # Follow-up handling
    # ----------------------------------------

    def _is_followup(self, question: str, chat_history: list[dict]) -> bool:
        """
        Decide whether a question depends on the conversation to make sense.

        "What section covers that?" retrieves nothing useful on its own -- the
        embedding carries no legal subject at all -- so the model correctly
        reports that it has no information. Detecting these before retrieval is
        what makes multi-turn conversation work.
        """
        if not chat_history:
            return False

        words = question.split()
        if len(words) <= 4:
            return True
        return len(words) <= 15 and bool(self._ANAPHORIC.search(question))

    def _condense(self, question: str, chat_history: list[dict]) -> str:
        """
        Rewrite a follow-up into a standalone retrieval query.

        Falls back to concatenating the previous user turn, which costs nothing
        and still puts the missing subject into the embedding.
        """
        recent = chat_history[-4:]
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:400]}" for m in recent
        )

        try:
            rewritten = (self.condenser | self.parser).invoke([
                HumanMessage(content=CONDENSE_PROMPT.format(
                    history=history_text,
                    question=question,
                ))
            ]).strip().strip('"')

            if rewritten and len(rewritten) > 3:
                logger.info(f"  Condensed follow-up -> '{rewritten[:100]}'")
                return rewritten

        except Exception as e:
            logger.warning(f"  Condensation failed ({e}); falling back to concatenation")

        last_user = next(
            (m["content"] for m in reversed(chat_history) if m["role"] == "user"),
            "",
        )
        return f"{last_user} {question}".strip() if last_user else question

    def _search_query(self, question: str, chat_history: list[dict]) -> str:
        """Resolve the text that should actually be embedded for retrieval."""
        if self.condense_followups and self._is_followup(question, chat_history):
            return self._condense(question, chat_history)
        return question

    # ----------------------------------------
    # Main entry points
    # ----------------------------------------

    def query(
        self,
        question: str,
        session_id: Optional[str] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """
        Process a legal question through the full RAG pipeline.

        Args:
            question: User's legal question.
            session_id: Conversation session ID (for memory).
            act_id: Optional filter by specific act.
            category: Optional filter by legal category.
            use_reranker: Whether to use cross-encoder reranking.

        Returns:
            RAGResponse with answer, sources, and metadata.
        """
        session_id = session_id or "default"

        logger.info(f"Processing query: '{question[:80]}...'")

        # Step 1: Get conversation history, then resolve the retrieval query
        chat_history = self.memory.get_history(session_id)
        search_query = self._search_query(question, chat_history)

        # Step 2: Retrieve relevant context
        retrieval_results = self.retriever.retrieve(
            query=search_query,
            act_id=act_id,
            category=category,
            use_reranker=use_reranker,
        )
        logger.info(f"  Retrieved {len(retrieval_results)} relevant chunks")

        # Step 3: Build prompt. The user still sees their original wording; only
        # retrieval uses the rewritten form.
        user_prompt = build_user_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history,
        )

        # Step 4: Call LLM via LangChain LCEL
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        chain = self.llm | self.parser
        answer = chain.invoke(messages)

        # Step 5: Build source tracking
        sources = []
        for r in retrieval_results:
            sources.append({
                "chunk_id": r.chunk_id,
                "citation": r.citation,
                "act_name": r.act_name,
                "act_id": r.act_id,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "chapter": r.chapter,
                "similarity_score": r.similarity_score,
                "rerank_score": r.rerank_score,
                "text_preview": r.text[:200],
            })

        # Step 6: Update conversation memory
        self.memory.add_user_message(question, session_id)
        self.memory.add_assistant_message(
            content=answer,
            sources=[s["citation"] for s in sources],
            session_id=session_id,
        )

        self.last_sources = sources

        response = RAGResponse(
            answer=answer,
            sources=sources,
            query=question,
            retrieval_count=len(retrieval_results),
            model=self.model_name,
            session_id=session_id,
            search_query=search_query,
        )

        logger.info(f"  Response generated ({len(answer)} chars, {len(sources)} sources)")

        return response

    def stream(
        self,
        question: str,
        session_id: Optional[str] = None,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
        use_reranker: bool = True,
    ) -> Generator[str, None, RAGResponse]:
        """
        Stream a response token-by-token.

        Yields:
            Individual text chunks as they arrive.

        Returns:
            Final RAGResponse, available as the generator's return value and
            also mirrored on self.last_sources for callers that only iterate.
        """
        session_id = session_id or "default"

        # Resolve the retrieval query the same way query() does, so streaming
        # and non-streaming answers stay consistent on follow-ups.
        chat_history = self.memory.get_history(session_id)
        search_query = self._search_query(question, chat_history)

        # Retrieve context
        retrieval_results = self.retriever.retrieve(
            query=search_query,
            act_id=act_id,
            category=category,
            use_reranker=use_reranker,
        )

        # Build prompt
        user_prompt = build_user_prompt(
            question=question,
            retrieval_results=retrieval_results,
            chat_history=chat_history,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        # Stream from LLM
        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            if token:
                full_answer += token
                yield token

        # Build sources
        sources = []
        for r in retrieval_results:
            sources.append({
                "chunk_id": r.chunk_id,
                "citation": r.citation,
                "act_name": r.act_name,
                "act_id": r.act_id,
                "section_number": r.section_number,
                "section_title": r.section_title,
                "chapter": r.chapter,
                "similarity_score": r.similarity_score,
                "rerank_score": r.rerank_score,
            })

        # Update memory
        self.memory.add_user_message(question, session_id)
        self.memory.add_assistant_message(
            content=full_answer,
            sources=[s["citation"] for s in sources],
            session_id=session_id,
        )

        # Expose the retrieval behind this stream so the API can emit citations
        # once the tokens have finished.
        self.last_sources = sources

        return RAGResponse(
            answer=full_answer,
            sources=sources,
            query=question,
            retrieval_count=len(retrieval_results),
            model=self.model_name,
            session_id=session_id,
            search_query=search_query,
        )

    # ----------------------------------------
    # Session helpers
    # ----------------------------------------

    def get_conversation_history(
        self, session_id: Optional[str] = None
    ) -> list[dict]:
        """Get the conversation history for a session."""
        return self.memory.get_history(session_id)

    def clear_conversation(self, session_id: Optional[str] = None) -> None:
        """Clear conversation history for a session."""
        self.memory.clear(session_id)
        logger.info(f"Conversation cleared for session '{session_id or 'default'}'")

    def get_context_preview(
        self,
        question: str,
        act_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> str:
        """
        Preview what context would be retrieved for a question,
        without calling the LLM. Useful for debugging retrieval quality.
        """
        results = self.retriever.retrieve(
            query=question,
            act_id=act_id,
            category=category,
        )
        return format_context(results)