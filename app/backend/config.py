from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _split_env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return default
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return values or default


@dataclass(frozen=True)
class Settings:
    cors_origins: list[str] = field(
        default_factory=lambda: _split_env_list(
            "CORS_ORIGINS",
            ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
        )
    )
    max_upload_size: int = int(os.getenv("MAX_UPLOAD_SIZE", str(50 * 1024 * 1024)))
    allowed_extensions: set[str] = field(
        default_factory=lambda: {".csv", ".tsv", ".txt"}
    )
    production_models_dir: Path = Path(
        os.getenv(
            "PRODUCTION_MODELS_DIR",
            Path(__file__).resolve().parent / "production_models",
        )
    )
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "120"))
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
    ollama_top_p: float = float(os.getenv("OLLAMA_TOP_P", "0.8"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "400"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
