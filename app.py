"""Compatibility launcher for the relocated FastAPI application."""
import os

from backend.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("FASHION_APP_PORT", "8765")),
        reload=False,
    )
