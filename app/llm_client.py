"""
llm_client.py

Thin wrapper exposing exactly the interface evidence_scorer.py expects:
.invoke(prompt: str, temperature: float) -> str. Kept separate from the
scorer so it stays provider-agnostic — swapping to Gemini's free tier later
means adding one small class here, not touching the scoring logic.

Import of langchain_groq is guarded, same pattern as sentence-transformers
in retrieval/index_builder.py: an optional dependency being missing should
never crash app startup, only disable the feature it backs.
"""
from __future__ import annotations
import os

from dotenv import load_dotenv

# Loaded here, not in main.py or run_eval.py individually, so every
# entrypoint that needs GROQ_API_KEY gets it automatically — main.py's
# startup event, eval/run_eval.py run standalone, or a bare python shell
# importing this module directly. load_dotenv() never overwrites a
# variable that's already set in the real environment (e.g. a Docker
# `environment:` block or a CI secret), so this is safe in every context.
load_dotenv()

try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


class GroqLLMClient:
    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None):
        if not _GROQ_AVAILABLE:
            raise RuntimeError(
                "langchain-groq is not installed. Run `pip install langchain-groq` "
                "or use the fallback heuristic (build_llm_client() will return None)."
            )

        key = api_key or os.environ.get("GROQ_API_KEY")

        if not key:
            raise ValueError(
                "GROQ_API_KEY not set — pass api_key explicitly or set the env var."
            )

        self._client = ChatGroq(
            model=model,
            api_key=key,
            max_retries=0,
        )

    def invoke(self, prompt: str, temperature: float = 0.4) -> str:
        # ChatGroq's temperature is set via .bind(), not an invoke() kwarg —
        # rebinding here keeps this class's public interface (temperature
        # as a plain argument) stable regardless of the LangChain client's API.
        response = self._client.bind(temperature=temperature).invoke(prompt)
        return response.content


def build_llm_client() -> GroqLLMClient | None:
    """Returns None (never raises) if the dependency is missing or the API
    key isn't configured, so the app boots and runs on the fallback
    heuristic — the same 'degrade, don't break' rule used throughout the
    engine layer, applied here at the very entrypoint of the LLM path."""
    try:
        return GroqLLMClient()
    except (ValueError, RuntimeError):
        return None