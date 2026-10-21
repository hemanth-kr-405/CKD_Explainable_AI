"""
Proxy / Entrypoint module redirecting to backend.main.
Allows starting the server via `uvicorn app.main:app` or `uvicorn backend.main:app`.
"""

from backend.main import app

__all__ = ["app"]
