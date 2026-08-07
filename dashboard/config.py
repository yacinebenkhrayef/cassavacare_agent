import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    agent_api_base_url: str = os.getenv("AGENT_API_BASE_URL", "http://localhost:8000")
    submit_path: str = "/diagnose"               # confirmed: POST /diagnose
    gradcam_path_suffix: str = "/gradcam"        # confirmed: GET /diagnose/{id}/gradcam
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "60"))
    poll_interval_s: float = float(os.getenv("POLL_INTERVAL_S", "1.0"))
    poll_timeout_s: float = float(os.getenv("POLL_TIMEOUT_S", "25"))
    allowed_image_types: tuple = ("jpg", "jpeg", "png")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))


settings = Settings()