from __future__ import annotations
import time
import requests
from config import settings
from schemas import JobStatusResult
import mimetypes

class AgentAPIError(Exception):
    pass


class AgentAPIClient:
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or settings.agent_api_base_url).rstrip("/")

    def submit_diagnosis(self, image_bytes: bytes, filename: str, location: str):
        """Returns (job_id, status_url)."""
        url = f"{self.base_url}{settings.submit_path}"
        mime_type, _ = mimetypes.guess_type(filename)

        if mime_type is None:
            mime_type = "application/octet-stream"

        files = {"image": (filename, image_bytes, mime_type)}
        data = {"location": location}
        try:
            resp = requests.post(url, files=files, data=data, timeout=settings.request_timeout_s)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise AgentAPIError(f"Failed to submit image: {exc}") from exc
        payload = resp.json()
        job_id = payload.get("job_id")
        status_url = payload.get("status_url") or f"{settings.submit_path}/{job_id}"
        if not job_id:
            raise AgentAPIError(f"No job_id in submit response: {payload}")
        return job_id, status_url

    def get_status(self, status_url: str) -> JobStatusResult:
        clean_path = status_url if status_url.startswith("/") else f"/{status_url}"
        url = f"{self.base_url}{clean_path}"
        try:
            resp = requests.get(url, timeout=settings.request_timeout_s)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise AgentAPIError(f"Failed to fetch job status: {exc}") from exc
        return JobStatusResult.from_dict(resp.json())

    def wait_for_completion(self, status_url: str, on_poll=None) -> JobStatusResult:
        deadline = time.monotonic() + settings.poll_timeout_s
        while True:
            result = self.get_status(status_url)
            if on_poll:
                on_poll(result.status)
            if result.is_terminal:
                return result
            if time.monotonic() > deadline:
                raise AgentAPIError(f"Timed out waiting for job (last status: {result.status})")
            time.sleep(settings.poll_interval_s)

    def get_gradcam_bytes(self, job_id: str) -> bytes:
        """Fetches the Grad-CAM PNG once the job is completed."""
        url = f"{self.base_url}{settings.submit_path}/{job_id}{settings.gradcam_path_suffix}"
        try:
            resp = requests.get(url, timeout=settings.request_timeout_s)
        except requests.RequestException as exc:
            raise AgentAPIError(f"Failed to reach Grad-CAM endpoint: {exc}") from exc

        if resp.status_code == 404:
            try:
                detail = resp.json().get("detail", "Grad-CAM image not available.")
            except ValueError:
                detail = "Grad-CAM image not available."
            raise AgentAPIError(detail)

        try:
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise AgentAPIError(f"Failed to fetch Grad-CAM image: {exc}") from exc

        return resp.content