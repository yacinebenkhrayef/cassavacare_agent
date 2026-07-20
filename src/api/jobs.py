"""Thread-safe in-memory job store for CassavaCare-Agent's async /diagnose
endpoint (Phase 4, Part 3).

Per project decision: in-memory, not persisted. Job history is lost on
server restart — acceptable for a single-process PFE demo. If this ever
needs to survive restarts, swap this class's internals for SQLite without
touching src/api/main.py's calls to it (create/get/mark_* stay the same
signatures).
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.QUEUED
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = self.created_at
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = JobRecord(job_id)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
            job.updated_at = datetime.now(timezone.utc)

    def mark_completed(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.COMPLETED
            job.result = result
            job.updated_at = datetime.now(timezone.utc)

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.FAILED
            job.error = error
            job.updated_at = datetime.now(timezone.utc)


# Module-level singleton — imported directly by src/api/main.py
job_store = JobStore()