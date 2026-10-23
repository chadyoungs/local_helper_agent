"""Entry point for Local Helper Agent.

Run with:  python3 backend.py
Backend listens on http://127.0.0.1:5001 by default (override via HOST/PORT env).
"""

from __future__ import annotations

import logging

import uvicorn

from agent.api import build_app
from agent.settings import DEFAULT_SETTINGS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = build_app(DEFAULT_SETTINGS)

if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_SETTINGS.host, port=DEFAULT_SETTINGS.port)
