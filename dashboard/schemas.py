from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiagnosisResult:
    pred_disease: Optional[str] = None
    pred_disease_short: Optional[str] = None
    confidence: Optional[float] = None
    needs_new_image: bool = False
    gradcam_path: Optional[str] = None
    rag_sources: list = field(default_factory=list)     # raw dicts — shaped in Part 3
    weather: Optional[dict] = None                       # raw dict — shaped in Part 2/3
    weather_error: Optional[str] = None
    decision: Optional[str] = None
    decision_reason: Optional[str] = None
    final_report: str = ""
    trace: list = field(default_factory=list)            # flat list[str], reasoning trace

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosisResult":
        return cls(
            pred_disease=d.get("pred_disease"),
            pred_disease_short=d.get("pred_disease_short"),
            confidence=d.get("confidence"),
            needs_new_image=d.get("needs_new_image", False),
            gradcam_path=d.get("gradcam_path"),
            rag_sources=d.get("rag_sources", []),
            weather=d.get("weather"),
            weather_error=d.get("weather_error"),
            decision=d.get("decision"),
            decision_reason=d.get("decision_reason"),
            final_report=d.get("final_report", ""),
            trace=d.get("trace", []),
        )


@dataclass
class JobStatusResult:
    job_id: str
    status: str
    result: Optional[DiagnosisResult] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "JobStatusResult":
        result = d.get("result")
        return cls(
            job_id=d.get("job_id", ""),
            status=str(d.get("status", "unknown")),
            result=DiagnosisResult.from_dict(result) if result else None,
            error=d.get("error"),
        )

    @property
    def is_terminal(self) -> bool:
        return self.status.lower() in ("completed", "failed")

    @property
    def is_failed(self) -> bool:
        return self.status.lower() == "failed"