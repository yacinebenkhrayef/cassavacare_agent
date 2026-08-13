# api/client.py
from dataclasses import dataclass
from typing import List, Optional

from api.rag_service import query_knowledge


@dataclass
class RAGAnswer:
    answer: str
    sources: List[dict]
    chunks_used: int
    timing_ms: Optional[dict] = None


class CassavaRAGClient:
    """In-process RAG client — retrieves from Qdrant and generates via Gemini."""

    def ask(self, question: str, top_k: int = 5, source_filter: Optional[str] = None) -> RAGAnswer:
        try:
            data = query_knowledge(question, top_k, source_filter)
            return RAGAnswer(
                answer=data["answer"],
                sources=data.get("sources", []),
                chunks_used=data.get("chunks_used", len(data.get("sources", []))),
                timing_ms=data.get("timing_ms"),
            )
        except Exception as exc:
            return RAGAnswer(
                answer=f"Knowledge service error: {exc}",
                sources=[],
                chunks_used=0,
            )