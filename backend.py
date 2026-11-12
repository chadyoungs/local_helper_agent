"""Entry point for Local Helper Agent.

Run with:  python3 backend.py
Backend listens on http://127.0.0.1:5001 by default (override via HOST/PORT env).
"""
import logging
import sys

import pydantic
import uvicorn

from agent.api import build_app
from agent.settings import DEFAULT_SETTINGS


def _check_runtime() -> None:
    """Fail with an actionable message instead of a puzzling AttributeError.

    The project uses the Pydantic v2 API (``model_dump`` / ``model_validate``);
    an old environment (pydantic 1.x / fastapi 0.85) breaks every session call
    with ``type object ... has no attribute 'model_validate'``.
    """
    major = int(pydantic.VERSION.split(".")[0])
    if major < 2:
        print(
            "This project needs pydantic>=2.5 (found %s) and fastapi>=0.110.\n"
            "Your current interpreter is: %s\n"
            "Create an isolated environment instead of upgrading a shared one:\n"
            "  python3 -m venv .venv\n"
            "  .venv/bin/python -m pip install -r requirements.txt\n"
            "  .venv/bin/python backend.py" % (pydantic.VERSION, sys.executable),
            file=sys.stderr,
        )
        raise SystemExit(2)


_check_runtime()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = build_app(DEFAULT_SETTINGS)

if __name__ == "__main__":
    uvicorn.run(app, host=DEFAULT_SETTINGS.host, port=DEFAULT_SETTINGS.port)
