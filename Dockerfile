# ============================================================================
# Production Dockerfile for CKD Explainable AI Prediction Service
# ============================================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies (build essentials, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code and trained model artifacts
COPY data/ ./data/
COPY models/ ./models/
COPY reports/ ./reports/
COPY src/ ./src/
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY README.md .

# Healthcheck probe to verify service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose default port
EXPOSE 8000

# Start Uvicorn ASGI server with dynamic port binding
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
