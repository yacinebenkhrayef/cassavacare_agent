import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import QueryRequest, QueryResponse, SourceChunk
from api.rag_service import query_knowledge_async
from api.cache import cache_stats
from api.validation import is_suspicious, sanitize_question

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cassavacare-rag")

app = FastAPI(
    title="CassavaCare RAG API",
    description="Retrieval-Augmented Generation API for cassava leaf disease diagnosis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "CassavaCare RAG API"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # 1. Input Sanitization and Validation
    request.question = sanitize_question(request.question)

    if is_suspicious(request.question):
        logger.warning(f"Suspicious query blocked: {request.question}")
        raise HTTPException(status_code=400, detail="Query rejected by input validation.")

    if len(request.question.strip()) < 5:
        raise HTTPException(status_code=400, detail="Question is too short to process.")

    # 2. Knowledge retrieval + generation
    try:
        result = await query_knowledge_async(
            request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=503, detail="RAG service unavailable.")

    return QueryResponse(
        answer=result["answer"],
        sources=[SourceChunk(**s) for s in result["sources"]],
        question=request.question,
        chunks_used=result["chunks_used"],
        timing_ms=result.get("timing_ms"),
    )


@app.post("/sources", response_model=List[SourceChunk])
def get_sources(request: QueryRequest):
    """Return raw retrieved chunks without LLM generation. Useful for explainability."""
    from api.retriever import retrieve
    try:
        return retrieve(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))