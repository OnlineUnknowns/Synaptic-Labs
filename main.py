"""
Application entrypoint.

Starts the FastAPI service with Uvicorn using settings from environment variables.
"""

from __future__ import annotations

import logging

import uvicorn

from config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        factory=False,
        reload=False,
    )


if __name__ == "__main__":
    main()
