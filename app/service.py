"""
Evaluation pipeline: documents -> facts -> P(win) -> decision -> counterfactual.

Extraction modes:
  rules    deterministic rules only
  llm      LLM self-consistency sampling for every case
  cascade  rules first; the LLM is called only when a document that should
           answer a relevant fact exists but the rules could not read it.
           Rule answers are kept, LLM answers fill the gaps.

Facts the rules could not read from a document that should cover them
("unread" facts) are not silently treated as unknown: when the rules result
is final (rules mode, or the fallback after an LLM failure/timeout) they
are imputed from training priors, so the P(win) spread reflects what was
not read and the value-of-information test can route the case to a human.
"""
from __future__ import annotations

import asyncio
import logging
import time
import zlib
from dataclasses import dataclass

from app import observability as obs
from app.catalog import Catalog
from app.domain import FACT_SOURCES, DisputeCase, EvidenceFacts, Merchant, ReasonCode, Tri
from app.engine.conflicts import find_conflicts
from app.engine.counterfactual import Counterfactual, find_flip
from app.engine.decision_engine import Decision, DecisionEngine
from app.extraction.aggregate import agreement, majority
from app.extraction.llm_extractor import ExtractionResult, LLMExtractor
from app.extraction.rules import extract_with_rules, unreadable_doc_types
from app.scoring.win_model import WinEstimate, WinModel

logger = logging.getLogger("abstain.service")
MODES = ("rules", "llm", "cascade")


@dataclass
class Evaluation:
    case: DisputeCase
    merchant: Merchant
    reason_code: ReasonCode
    extraction: ExtractionResult
    facts: EvidenceFacts
    fact_agreement: dict[str, float]
    unread_facts: list[str]           # imputed because a document could not be read
    estimate: WinEstimate
    contributions: dict[str, float]
    conflicts: list[str]
    decision: Decision
    counterfactual: Counterfactual
    model_version: str
    degraded_reason: str | None
    audit_log: list[str]              # every pipeline step, in order, with timings


def unread_facts(case: DisputeCase, facts: EvidenceFacts, relevant: list[str]) -> list[str]:
    """Relevant facts left unknown while a document that could answer them contains text the
    rules could not interpret. Unknowns from fully understood documents are genuine absences."""
    unreadable = unreadable_doc_types(case.documents)
    return [f for f in relevant
            if facts.get(f) == Tri.UNKNOWN and any(t in unreadable for t in FACT_SOURCES[f])]


class DisputeService:
    def __init__(
        self,
        *,
        catalog: Catalog,
        model: WinModel,
        engine: DecisionEngine,
        llm: LLMExtractor | None,
        mode: str = "cascade",
        llm_budget_s: float | None = 25.0,   # None: no overall budget (offline eval)
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.catalog = catalog
        self.model = model
        self.engine = engine
        self.llm = llm
        self.mode = mode if llm is not None else "rules"
        self.llm_budget_s = llm_budget_s

    async def evaluate(self, case: DisputeCase) -> Evaluation:
        rc = self.catalog.reason_code(case.reason_code)
        started = time.perf_counter()
        extraction, degraded = await self._extract(case, rc)
        extraction.notes.append(f"extraction finished in {(time.perf_counter() - started) * 1000:.0f} ms")
        return self.assess(case, extraction, degraded=degraded)

    def assess(self, case: DisputeCase, extraction: ExtractionResult, *, degraded: str | None = None) -> Evaluation:
        """Everything after extraction. Pure and synchronous; the eval harness
        calls it directly with oracle facts to isolate extraction error."""
        started = time.perf_counter()
        rc = self.catalog.reason_code(case.reason_code)
        merchant = self.catalog.merchant(case.merchant_id)
        audit = [f"case {case.case_id}: reason {rc.code} ({rc.category.value}), merchant {merchant.merchant_id}, "
                 f"amount ₹{case.amount_inr:,}, {len(case.documents)} document(s)"]
        audit += extraction.notes
        facts = majority(extraction.samples)
        # An LLM answering 'unknown' has read the document; the rules
        # returning 'unknown' may just mean an unfamiliar phrasing.
        unread = (unread_facts(case, facts, rc.relevant_facts)
                  if extraction.source in ("rules", "rules_fallback") else [])
        samples = self.model.impute(extraction.samples, unread, seed=zlib.crc32(case.case_id.encode()))

        def estimate(fact_samples: list[EvidenceFacts]) -> WinEstimate:
            return self.model.estimate(rc.category, fact_samples, merchant)

        known = {f: v.value for f, v in facts.model_dump().items() if v != Tri.UNKNOWN}
        audit.append(f"facts ({extraction.source}): {known or 'none established'}")
        if unread:
            audit.append(f"imputed {len(unread)} unread fact(s) from training priors: {', '.join(unread)} "
                         f"-> {len(samples)} Monte Carlo draws")
        conflicts = find_conflicts(facts, rc.relevant_facts)
        if conflicts:
            audit.append(f"contradictions: {'; '.join(conflicts)}")
        est = estimate(samples)
        audit.append(f"win model {self.model.version}: P(win) = {est.mean:.3f} ± {est.std:.3f} "
                     f"over {est.n_draws} draws (samples x ensemble members)")
        decision = self.engine.decide(case, merchant, est, conflicts)
        audit.append(f"decision: {decision.action.value} ({decision.confidence}); EV(contest) ₹{decision.ev_contest_inr:,.0f}, "
                     f"review value ₹{decision.review_value_inr:,.0f} vs cost ₹{decision.review_cost_inr:,.0f}")
        if unread:
            decision.reasons.insert(0, f"Could not read {', '.join(unread)} from the documents; "
                                       f"imputed from training priors, widening the uncertainty.")
        if degraded:
            decision.reasons.insert(0, f"Degraded run ({degraded}): rules-only extraction.")
        cf = find_flip(engine=self.engine, estimate=estimate, case=case, merchant=merchant,
                       relevant_facts=rc.relevant_facts, fact_samples=samples, baseline=decision, unread=unread)
        audit.append(f"counterfactual: {cf.note}")
        audit.append(f"assessment finished in {(time.perf_counter() - started) * 1000:.0f} ms")

        obs.DECISIONS.labels(decision.action.value, extraction.source).inc()
        return Evaluation(
            case=case, merchant=merchant, reason_code=rc, extraction=extraction, facts=facts,
            fact_agreement=agreement(extraction.samples), unread_facts=unread, estimate=est,
            contributions=self.model.contributions(rc.category, facts, merchant),
            conflicts=conflicts, decision=decision, counterfactual=cf,
            model_version=self.model.version, degraded_reason=degraded, audit_log=audit,
        )

    async def _extract(self, case: DisputeCase, rc: ReasonCode) -> tuple[ExtractionResult, str | None]:
        rules_facts = extract_with_rules(case.documents, case.cardholder_name)
        rules_result = ExtractionResult(samples=[rules_facts], source="rules")
        unreadable = sorted(t.value for t in unreadable_doc_types(case.documents))
        rules_result.notes.append(f"mode {self.mode}; rules ran; unreadable documents: {unreadable or 'none'}")
        if self.mode == "rules":
            return rules_result, None
        gaps = unread_facts(case, rules_facts, rc.relevant_facts)
        if self.mode == "cascade" and not gaps:
            rules_result.notes.append("LLM skipped: the rules read every relevant document")
            return rules_result, None

        reason = f"to read {', '.join(gaps)}" if gaps else "(llm mode)"
        try:
            llm_result = await asyncio.wait_for(self.llm.extract(case, rc), timeout=self.llm_budget_s)
        except TimeoutError:
            rules_result.notes.append(f"LLM called {reason} but exceeded the {self.llm_budget_s}s budget; "
                                      f"falling back to rules")
            return self._fallback(rules_result, "llm_timeout"), "llm_timeout"
        if not llm_result.samples:
            logger.warning("all llm samples failed", extra={"case_id": case.case_id, "errors": llm_result.errors[:3]})
            rules_result.notes.append(f"LLM called {reason}; all {llm_result.llm_calls} samples failed "
                                      f"({llm_result.errors[0][:120] if llm_result.errors else 'unknown'}); "
                                      f"falling back to rules")
            return self._fallback(rules_result, "llm_failed"), "llm_failed"

        llm_result.notes = rules_result.notes + [
            f"LLM called {reason}: {len(llm_result.samples)}/{llm_result.llm_calls} samples usable, "
            f"{llm_result.ungrounded_answers} ungrounded answer(s) discarded, "
            f"{llm_result.prompt_tokens + llm_result.completion_tokens} tokens"]
        if self.mode == "cascade":
            keep = {f: v for f, v in rules_facts.model_dump().items() if v != Tri.UNKNOWN}
            llm_result.samples = [s.model_copy(update=keep) for s in llm_result.samples]
            llm_result.source = "cascade"
            llm_result.notes.append(f"kept {len(keep)} rule-read fact(s); LLM filled the rest")
        return llm_result, None

    @staticmethod
    def _fallback(rules_result: ExtractionResult, reason: str) -> ExtractionResult:
        obs.DEGRADED.labels(reason=reason).inc()
        rules_result.source = "rules_fallback"
        return rules_result
