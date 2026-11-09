"""Environment settings."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    database_path: str = "./data/observability.db"
    log_content: bool = True
    host: str = "127.0.0.1"
    port: int = 8000


def load_settings() -> Settings:
    # An explicit path stops python-dotenv walking up into parent projects' .env files.
    load_dotenv(".env")
    defaults = Settings()
    return Settings(
        openrouter_base_url=os.getenv("OPENROUTER_BASE_URL", defaults.openrouter_base_url),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", defaults.openrouter_api_key),
        database_path=os.getenv("DATABASE_PATH", defaults.database_path),
        # Anything unrecognised turns content logging off, so a typo cannot store prompts.
        log_content=os.getenv("LOG_CONTENT", "true").strip().lower() in {"true", "1", "yes", "on"},
        host=os.getenv("HOST", defaults.host),
        port=int(os.getenv("PORT", str(defaults.port))),
    )
