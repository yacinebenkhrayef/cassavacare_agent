import requests
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RAGAnswer:
    answer: str
    sources: List[dict]
    chunks_used: int
    timing_ms: Optional[dict] = None

class CassavaRAGClient:
    """Thin client for the CassavaCare-Agent to consume the RAG API without
    knowing anything about Qdrant, embeddings, or HTTP internals."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def ask(self, question: str, top_k: int = 5, source_filter: Optional[str] = None) -> RAGAnswer:
        try:
            resp = requests.post(
                f"{self.base_url}/query",
                json={"question": question, "top_k": top_k, "source_filter": source_filter},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return RAGAnswer(
                answer=data["answer"],
                sources=data["sources"],
                chunks_used=data["chunks_used"],
                timing_ms=data.get("timing_ms"),
            )
        except requests.exceptions.Timeout:
            return RAGAnswer(answer="The knowledge service timed out. Please try again.", sources=[], chunks_used=0)
        except requests.exceptions.RequestException as e:
            return RAGAnswer(answer=f"Knowledge service error: {e}", sources=[], chunks_used=0)

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False