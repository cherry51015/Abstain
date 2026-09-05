"""
evidence_scorer.py

Produces the ONLY thing the decision engine is allowed to consume from the
LLM side: a validated EvidenceAssessment (strength, uncertainty, gaps).

Two deliberate design choices:

1. Uncertainty isn't invented from a single LLM call's confidence talk —
   that's a well-known unreliable signal (models are often fluently wrong).
   Instead we sample the LLM N times at nonzero temperature and use the
   *disagreement across samples* (self-consistency) as the uncertainty
   proxy. It's not a calibrated statistical confidence interval — that
   would need a much larger validation study — and this module says so
   explicitly rather than dressing it up as one.

2. If the API is unreachable, rate-limited, or returns unparseable output
   after retries, this NEVER raises up into the pipeline and NEVER blocks
   a demo. It falls back to a conservative, purely structural heuristic
   (evidence completeness + conflicting-evidence flag) and marks the
   result source="fallback_heuristic" so it's visible in the audit trail
   that a degraded path was used, rather than silently pretending the LLM
   scored it.
"""
from __future__ import annotations

import json
import logging
import statistics
import re
import time
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from .models import EvidenceAssessment

logger = logging.getLogger("abstain.evidence_scorer")
_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)\s*(ms|s)\b", re.IGNORECASE)


class _RawLLMScore(BaseModel):
    """Strict schema the LLM's JSON output must conform to. Anything that
    doesn't parse into this is treated as a failed sample, not patched up."""
    evidence_strength: float = Field(ge=0.0, le=1.0)
    key_gaps: list[str] = Field(default_factory=list)
    reasoning: str = ""


PROMPT_TEMPLATE = """You are assessing evidence strength for a payment dispute, \
strictly from the evidence listed — do not assume evidence that isn't stated.

Reason code: {reason_code} — {reason_label}
Required evidence for this reason code: {required_evidence}
Evidence present in this case: {present_evidence}
Evidence missing: {missing_evidence}

Return ONLY valid JSON, no markdown fences, no commentary, matching exactly:
{{"evidence_strength": <float 0.0-1.0>, "key_gaps": [<string>, ...], "reasoning": "<one or two sentences>"}}

evidence_strength should reflect how strong the case for CONTESTING is, \
given what's present vs. what's missing for this specific reason code — \
not a generic completeness percentage.
"""


class EvidenceScorer:
    def __init__(
        self,
        llm_client=None,
        n_samples: int = 1,
        temperature: float = 0.4,
        max_retries_per_sample: int = 2,   # was 1 — now safe to bump since retries actually wait
        default_backoff_s: float = 2.0,
    ):
        self.llm_client = llm_client
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_retries_per_sample = max_retries_per_sample
        self.default_backoff_s = default_backoff_s


    # ---------- public entrypoint ----------
    def assess(
        self,
        reason_code: str,
        reason_label: str,
        required_evidence: list[str],
        present_evidence: list[str],
        missing_evidence: list[str],
    ) -> EvidenceAssessment:
        if self.llm_client is None:
            logger.warning("No LLM client configured — using fallback heuristic.")
            return self._fallback(present_evidence, required_evidence)

        prompt = PROMPT_TEMPLATE.format(
            reason_code=reason_code,
            reason_label=reason_label,
            required_evidence=required_evidence,
            present_evidence=present_evidence,
            missing_evidence=missing_evidence,
        )

        samples: list[_RawLLMScore] = []
        for i in range(self.n_samples):
            result = self._sample_once(prompt, attempt_label=f"sample_{i+1}")
            if result is not None:
                samples.append(result)

        if not samples:
            logger.warning(
                "All %d LLM samples failed to parse/return — falling back to heuristic.",
                self.n_samples,
            )
            return self._fallback(present_evidence, required_evidence)

        strengths = [s.evidence_strength for s in samples]
        mean_strength = statistics.mean(strengths)
        # stdev needs >=2 points; treat a single successful sample as max uncertainty
        # within this method rather than falsely reporting zero disagreement.
        uncertainty = statistics.pstdev(strengths) if len(strengths) > 1 else 0.30
        all_gaps = sorted(set(g for s in samples for g in s.key_gaps))
        combined_reasoning = " | ".join(s.reasoning for s in samples if s.reasoning)

        return EvidenceAssessment(
            strength=round(mean_strength, 3),
            uncertainty=round(min(1.0, uncertainty), 3),
            source="llm",
            key_gaps=all_gaps,
            reasoning=combined_reasoning[:500],
        )

    # ---------- internals ----------
    def _sample_once(self, prompt: str, attempt_label: str) -> Optional[_RawLLMScore]:
        for attempt in range(self.max_retries_per_sample):
            try:
                raw_text = self.llm_client.invoke(prompt, temperature=self.temperature)
            except Exception as exc:
                logger.warning("%s attempt %d: LLM call failed (%s)", attempt_label, attempt, exc)
                if attempt < self.max_retries_per_sample - 1:
                    wait_s = self._parse_retry_after(exc) or self.default_backoff_s
                    logger.info("%s: backing off %.2fs before retry.", attempt_label, wait_s)
                    time.sleep(wait_s)
                continue

            parsed = self._parse_json_strict(raw_text)
            if parsed is None:
                logger.warning("%s attempt %d: response did not parse as valid JSON schema.",
                                attempt_label, attempt)
                continue
            return parsed
        return None
    
    @staticmethod
    def _parse_retry_after(exc: Exception) -> Optional[float]:
        match = _RETRY_AFTER_RE.search(str(exc))
        if not match:
            return None
        value, unit = match.groups()
        return float(value) / 1000.0 if unit.lower() == "ms" else float(value)

    @staticmethod
    def _parse_json_strict(raw_text: str) -> Optional[_RawLLMScore]:
        cleaned = raw_text.strip()
        # Defensive: strip accidental markdown fences even though the prompt forbids them.
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        try:
            return _RawLLMScore(**data)
        except ValidationError:
            return None

    @staticmethod
    def _fallback(present_evidence: list[str], required_evidence: list[str]) -> EvidenceAssessment:
        """Purely structural — no LLM, no network, cannot fail. This is the
        safety net that keeps the pipeline alive when the free-tier API is
        down, rate-limited, or unreachable during a live demo."""
        if not required_evidence:
            completeness = 0.5  # unknown requirement set — neutral, forces MEDIUM confidence downstream
        else:
            completeness = len(set(present_evidence) & set(required_evidence)) / len(required_evidence)
        missing = sorted(set(required_evidence) - set(present_evidence))
        return EvidenceAssessment(
            strength=round(completeness, 3),
            uncertainty=0.35,  # deliberately in the MEDIUM/LOW boundary — fallback should rarely auto-decide
            source="fallback_heuristic",
            key_gaps=missing,
            reasoning="LLM unavailable — scored from structural evidence completeness only.",
        )