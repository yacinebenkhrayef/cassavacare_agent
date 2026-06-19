import logging
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.models import QueryRequest, QueryResponse, SourceChunk
from api.retriever import retrieve
# Updated import to bring in the asynchronous generator function
from api.generator import generate_answer_async
from api.cache import make_cache_key, get_cached, set_cached, cache_stats
from api.timing import timed_stage
# New input validation imports
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

    # 2. Cache Lookup
    timing = {}
    cache_key = make_cache_key(request.question, request.top_k, request.source_filter)
    cached_response = get_cached(cache_key)
    if cached_response is not None:
        logger.info(f"Cache hit: {request.question}")
        return cached_response

    # 3. Knowledge Retrieval
    try:
        with timed_stage("retrieval", timing):
            chunks = retrieve(
                question=request.question,
                top_k=request.top_k,
                source_filter=request.source_filter,
            )
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        raise HTTPException(status_code=503, detail="Retrieval service unavailable. Is Qdrant running?")

    # Fallback if no data is found
    if not chunks:
        response = QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[], 
            question=request.question, 
            chunks_used=0, 
            timing_ms=timing,
        )
        set_cached(cache_key, response)
        return response

    # 4. LLM Response Generation
    try:
        with timed_stage("generation", timing):
            # Awaiting the async generation function
            answer = await generate_answer_async(request.question, chunks)
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM generation failed: {str(e)}")

    # Calculate performance execution times
    timing["total"] = round(sum(v for k, v in timing.items() if k != "total"), 2)

    # 5. Build Response & Cache
    response = QueryResponse(
        answer=answer, 
        sources=chunks, 
        question=request.question,
        chunks_used=len(chunks), 
        timing_ms=timing,
    )
    set_cached(cache_key, response)
    return response


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