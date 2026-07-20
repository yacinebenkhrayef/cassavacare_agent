"""Pydantic response models for the CassavaCare-Agent API (Phase 4, Part 3)."""
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.api.jobs import JobStatus


class JobSubmitResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class DiagnosisResult(BaseModel):
    """A superset covering all three graph exit paths from FR3: the
    low-confidence branch (needs_new_image), the healthy-leaf shortcut, and
    the full diseased-leaf pipeline. Extra keys from the raw AgentState
    (image_path, pred_class, probabilities, ...) are silently dropped —
    Pydantic ignores unlisted fields by default."""
    pred_disease: Optional[str] = None
    pred_disease_short: Optional[str] = None
    confidence: Optional[float] = None
    needs_new_image: bool = False
    gradcam_path: Optional[str] = None
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)
    weather: Optional[Dict[str, float]] = None
    weather_error: Optional[str] = None
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    final_report: str
    trace: List[str] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: Optional[DiagnosisResult] = None
    error: Optional[str] = None