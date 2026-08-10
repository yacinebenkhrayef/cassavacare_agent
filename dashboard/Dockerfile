FROM python:3.11-slim

WORKDIR /app

# Prevent .pyc generation and ensure instant log printing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dashboard-specific requirements (excludes PyTorch/LangChain/Heavy ML packages)
COPY dashboard/requirements.txt ./requirements.txt

# Install lightweight UI dependencies using pip caching
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy dashboard frontend code
COPY dashboard/ ./dashboard/
WORKDIR /app/dashboard

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]