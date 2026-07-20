"""
CassavaCare-Agent API (Phase 4, Part 3).

POST /diagnose         — submit a leaf image + city, get a job_id back (202)
GET  /diagnose/{id}     — poll job status/result
GET  /diagnose/{id}/gradcam — fetch the Grad-CAM heatmap once completed
GET  /health            — liveness check
"""
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.agent.config import ALLOWED_CONTENT_TYPES, UPLOAD_DIR
from src.agent.graph import agent_graph
from src.agent.nodes import initialize_agent_singletons
from src.api.jobs import JobStatus, job_store
from src.api.schemas import DiagnosisResult, JobStatusResponse, JobSubmitResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loads the checkpoint, RAG client, weather client, and Gemini client
    # exactly once at process startup — not at import time (flagged since
    # Part 1, fixed in nodes.py §6a this part).
    initialize_agent_singletons()
    yield
    # Nothing to tear down: Qdrant/OpenWeather/Gemini are all stateless
    # HTTP clients, no open connections to close.


app = FastAPI(title="CassavaCare-Agent API", lifespan=lifespan)


def _run_diagnosis_job(job_id: str, image_path: str, location: str) -> None:
    """Runs the full LangGraph pipeline in Starlette's background threadpool
    (see §10 note below) — safe to block here."""
    job_store.mark_running(job_id)
    try:
        result = agent_graph.invoke({"image_path": image_path, "location": location})
        job_store.mark_completed(job_id, result)
    except Exception as exc:  # last-resort catch-all: a bug in one job must
        # never leave that job stuck in "running" forever.
        job_store.mark_failed(job_id, str(exc))


@app.post("/diagnose", response_model=JobSubmitResponse, status_code=202)
async def submit_diagnosis(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    location: str = Form(...),
):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{image.content_type}'. "
                f"Allowed (FR1): {sorted(ALLOWED_CONTENT_TYPES)}."
            ),
        )

    job_id = job_store.create()
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    image_path = job_dir / image.filename
    with image_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)

    background_tasks.add_task(_run_diagnosis_job, job_id, str(image_path), location)

    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        status_url=f"/diagnose/{job_id}",
    )


@app.get("/diagnose/{job_id}", response_model=JobStatusResponse)
async def get_diagnosis_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id.")

    result_model = DiagnosisResult(**job.result) if job.result else None
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        updated_at=job.updated_at,
        result=result_model,
        error=job.error,
    )


@app.get("/diagnose/{job_id}/gradcam")
async def get_gradcam_image(job_id: str):
    job = job_store.get(job_id)
    if job is None or job.status != JobStatus.COMPLETED or not job.result:
        raise HTTPException(status_code=404, detail="No completed job with this id.")
    gradcam_path = job.result.get("gradcam_path")
    if not gradcam_path or not Path(gradcam_path).exists():
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(gradcam_path, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "ok"}