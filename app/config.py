"""Runtime configuration from environment variables (12-factor style)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # never overrides variables already set in the real environment


def _csv(name: str, default: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=lambda: os.environ.get("DATABASE_URL", "sqlite:///./abstain.db"))
    llm_api_key: str | None = field(default_factory=lambda: os.environ.get("GROQ_API_KEY") or None)
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"))
    # Provider-specific request fields, e.g. {"reasoning_effort": "low"} for gpt-oss.
    llm_extra_body: dict = field(default_factory=lambda: json.loads(os.environ.get("LLM_EXTRA_BODY", '{"reasoning_effort": "low"}')))
    llm_requests_per_minute: float | None = field(
        default_factory=lambda: float(os.environ["LLM_RPM"]) if os.environ.get("LLM_RPM") else None)
    llm_samples: int = field(default_factory=lambda: int(os.environ.get("LLM_SAMPLES", "3")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "700")))
    llm_max_concurrency: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_CONCURRENCY", "4")))
    llm_budget_s: float = field(default_factory=lambda: float(os.environ.get("LLM_BUDGET_S", "25")))
    llm_cache_path: str | None = field(default_factory=lambda: os.environ.get("LLM_CACHE_PATH", ".cache/llm_responses.sqlite"))
    extraction_mode: str = field(default_factory=lambda: os.environ.get("EXTRACTION_MODE", "cascade"))
    api_key: str | None = field(default_factory=lambda: os.environ.get("ABSTAIN_API_KEY") or None)
    cors_origins: list[str] = field(default_factory=lambda: _csv("CORS_ORIGINS", "http://localhost:5173,https://abstain-kappa.vercel.app"))
    log_level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    # Hosted demo: load a sample of resolved disputes on first start so Insights is not empty.
    demo_seed: bool = field(default_factory=lambda: os.environ.get("DEMO_SEED", "").lower() in ("1", "true", "yes"))
    demo_seed_limit: int = field(default_factory=lambda: int(os.environ.get("DEMO_SEED_LIMIT", "600")))
