from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.models import QueryRequest, QueryResponse
from api.retriever import retrieve
from api.generator import generate_answer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cassavacare-rag")

app = FastAPI(
    title="CassavaCare RAG API",
    description="Retrieval-Augmented Generation API for cassava leaf disease diagnosis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "CassavaCare RAG API"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    logger.info(f"Query received: {request.question}")

    try:
        chunks = retrieve(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable. Is Qdrant running?")

    if not chunks:
        return QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            question=request.question,
            chunks_used=0,
        )

    try:
        answer = generate_answer(request.question, chunks)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {str(e)}")

    logger.info(f"Answer generated. Chunks used: {len(chunks)}")
    return QueryResponse(
        answer=answer,
        sources=chunks,
        question=request.question,
        chunks_used=len(chunks),
    )

from api.models import SourceChunk
from typing import List

@app.post("/sources", response_model=List[SourceChunk])
def get_sources(request: QueryRequest):
    """Return raw retrieved chunks without LLM generation. Useful for explainability."""
    try:
        return retrieve(
            question=request.question,
            top_k=request.top_k,
            source_filter=request.source_filter,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))