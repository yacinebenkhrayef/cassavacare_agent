"""Shared in-process RAG query logic used by the agent and /query API route."""
from __future__ import annotations

import logging
from typing import List, Optional

from api.cache import get_cached, make_cache_key, set_cached
from api.generator import generate_answer, generate_answer_async
from api.models import QueryResponse, SourceChunk
from api.retriever import retrieve
from api.timing import timed_stage

logger = logging.getLogger("cassavacare-rag")


def _sources_to_dicts(chunks: List[SourceChunk]) -> List[dict]:
    return [chunk.model_dump() for chunk in chunks]


def _cached_to_result(cached: QueryResponse) -> dict:
    sources = cached.sources
    if sources and hasattr(sources[0], "model_dump"):
        sources = [s.model_dump() for s in sources]
    return {
        "answer": cached.answer,
        "sources": sources,
        "chunks_used": cached.chunks_used,
        "timing_ms": cached.timing_ms,
    }


def query_knowledge(
    question: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
) -> dict:
    """Retrieve relevant chunks from Qdrant and synthesize an answer with Gemini."""
    timing: dict = {}
    cache_key = make_cache_key(question, top_k, source_filter)
    cached = get_cached(cache_key)
    if cached is not None:
        logger.info("Cache hit: %s", question[:80])
        return _cached_to_result(cached)

    try:
        with timed_stage("retrieval", timing):
            chunks = retrieve(question, top_k, source_filter)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        return {
            "answer": "Knowledge retrieval failed. Is Qdrant running and indexed?",
            "sources": [],
            "chunks_used": 0,
        }

    if not chunks:
        response = QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            question=question,
            chunks_used=0,
            timing_ms=timing,
        )
        set_cached(cache_key, response)
        return _cached_to_result(response)

    try:
        with timed_stage("generation", timing):
            answer = generate_answer(question, chunks)
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        return {
            "answer": f"LLM generation failed: {exc}",
            "sources": _sources_to_dicts(chunks),
            "chunks_used": len(chunks),
            "timing_ms": timing,
        }

    timing["total"] = round(sum(v for k, v in timing.items() if k != "total"), 2)
    response = QueryResponse(
        answer=answer,
        sources=chunks,
        question=question,
        chunks_used=len(chunks),
        timing_ms=timing,
    )
    set_cached(cache_key, response)
    return _cached_to_result(response)


async def query_knowledge_async(
    question: str,
    top_k: int = 5,
    source_filter: Optional[str] = None,
) -> dict:
    """Async variant for FastAPI routes — uses non-blocking Gemini generation."""
    timing: dict = {}
    cache_key = make_cache_key(question, top_k, source_filter)
    cached = get_cached(cache_key)
    if cached is not None:
        logger.info("Cache hit: %s", question[:80])
        return _cached_to_result(cached)

    try:
        with timed_stage("retrieval", timing):
            chunks = retrieve(question, top_k, source_filter)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        return {
            "answer": "Knowledge retrieval failed. Is Qdrant running and indexed?",
            "sources": [],
            "chunks_used": 0,
        }

    if not chunks:
        response = QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            question=question,
            chunks_used=0,
            timing_ms=timing,
        )
        set_cached(cache_key, response)
        return _cached_to_result(response)

    try:
        with timed_stage("generation", timing):
            answer = await generate_answer_async(question, chunks)
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        return {
            "answer": f"LLM generation failed: {exc}",
            "sources": _sources_to_dicts(chunks),
            "chunks_used": len(chunks),
            "timing_ms": timing,
        }

    timing["total"] = round(sum(v for k, v in timing.items() if k != "total"), 2)
    response = QueryResponse(
        answer=answer,
        sources=chunks,
        question=question,
        chunks_used=len(chunks),
        timing_ms=timing,
    )
    set_cached(cache_key, response)
    return _cached_to_result(response)
