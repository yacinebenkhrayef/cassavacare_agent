from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500,
                          example="What are the symptoms of cassava mosaic disease?")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of chunks to retrieve from Qdrant")
    source_filter: Optional[str] = Field(default=None,
                                         description="Filter by source: FAO, PubMed, IITA, Wikipedia")

class SourceChunk(BaseModel):
    text: str
    source: str
    filename: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    question: str
    chunks_used: int