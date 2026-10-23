"""Runtime configuration for Local Helper Agent.

Settings are read from environment variables with sane defaults. An optional
``.env`` file at the project root (see ``.env.example``) is loaded first so
local overrides are easy. No extra dependency is required beyond the standard
library -- we parse the dotenv file ourselves to keep the project light.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader. Does not override variables already set in the env."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Central settings object. Frozen so consumers cannot mutate it at runtime."""

    host: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PORT", 5001))

    # --- LLM ---
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0"))
    )

    # Prefer Ollama's native function-calling API. When false (or the model does
    # not emit tool calls), fall back to strict structured-JSON generation.
    enable_native_tools: bool = field(
        default_factory=lambda: _env_bool("ENABLE_NATIVE_TOOLS", False)
    )

    # --- Agent loop ---
    max_agent_steps: int = field(default_factory=lambda: _env_int("MAX_AGENT_STEPS", 5))
    json_retries: int = field(default_factory=lambda: _env_int("JSON_RETRIES", 2))
    tool_output_max_chars: int = field(
        default_factory=lambda: _env_int("TOOL_OUTPUT_MAX_CHARS", 3000)
    )

    # --- Tool sandbox / subprocess ---
    tool_timeout_seconds: int = field(
        default_factory=lambda: _env_int("TOOL_TIMEOUT_SECONDS", 120)
    )

    # --- Paths (relative to project root) ---
    workspace_root: Path = field(
        default_factory=lambda: PROJECT_ROOT / "workspace"
    )
    sessions_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "sessions")
    tools_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "tools")

    # --- CORS ---
    # "null" is the Origin header value sent by browsers when a page is opened
    # via file://. We allow it explicitly so the no-build frontend works locally.
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS",
            ["http://127.0.0.1:5001", "http://localhost:5001", "null"],
        )
    )

    # --- Derived ---
    @property
    def input_root(self) -> Path:
        return self.workspace_root / "input"

    @property
    def output_root(self) -> Path:
        return self.workspace_root / "output"

    def ensure_dirs(self) -> None:
        """Create all runtime directories if they do not exist yet."""
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()
