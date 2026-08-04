from __future__ import annotations
from config import settings


def validate_image_file(uploaded_file):
    """Client-side pre-check only — the API still enforces its own
    ALLOWED_CONTENT_TYPES server-side and returns 415 if that fails,
    which surfaces as an AgentAPIError."""
    if uploaded_file is None:
        return False, "No file uploaded."
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext not in settings.allowed_image_types:
        return False, f"Unsupported format '.{ext}'. Allowed: {', '.join(settings.allowed_image_types)}."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        return False, f"File too large ({size_mb:.1f} MB). Max is {settings.max_upload_mb} MB."
    return True, None