# Abstain

### An AI risk decision engine for chargebacks

**The LLM reads the evidence. It never touches the money.**

🔗 **Live demo:** [abstain-kappa.vercel.app](https://abstain-kappa.vercel.app/)
⚙️ **API:** [abstain-api.onrender.com](https://abstain-api.onrender.com/)

---

## Why I built this

Every chargeback automation tool I looked at was solving the wrong problem. They were all trying to answer "can we win this dispute?" — a yes/no classifier wearing an AI costume.

But that's not what a risk team actually needs. What they need answered is:

> "Given the evidence we have, the money at stake, the deadline we're up against, and this merchant's risk history — should we contest, concede, or get a human to look at this?"

That's a different problem, and it needed a different architecture.

So I built Abstain around one rule I didn't compromise on: **the LLM is never allowed to spend money.** It reads evidence and tells me how strong it is. A separate, deterministic policy engine takes that read and decides what happens next. Two layers, two jobs, never mixed.

Then I went one step further, because deciding case-by-case felt incomplete. If a merchant keeps losing disputes for the same reason every month, a system that only outputs "CONCEDE" on each individual case is missing the actual story. So Abstain doesn't stop at the dispute — it climbs to *why this merchant keeps failing*, then up again to *why the whole portfolio is bleeding money*, and tells you which one it is.

---

## The one-line pitch

**Abstain uses an LLM to understand chargeback evidence, a deterministic engine to decide what to do with it, and a diagnostic layer to explain why merchants — and the whole portfolio — keep losing.**

---

## The core idea, visually

```text
                 ┌────────────────────┐
                 │   Chargeback Case   │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │  LLM Evidence      │   ← reads, doesn't decide
                 │  Scorer            │
                 └──────────┬─────────┘
                            ▼
                 ┌────────────────────┐
                 │  Deterministic     │   ← decides, doesn't guess
                 │  Decision Engine    │
                 └──────────┬─────────┘
                 ┌──────────┼───────────┐
                 ▼          ▼           ▼
             CONTEST     CONCEDE     ESCALATE
                                        │
                                        ▼
                              ┌────────────────────┐
                              │ Human Attention     │
                              │ Ranking (by EV,      │
                              │ not FIFO)            │
                              └────────────────────┘
```

That third outcome — **ESCALATE** — is the whole philosophy of this project in one word. Most systems are built to always output an answer. I wanted a system honest enough to say "I don't know enough here, get a person." A model that can abstain is more trustworthy than one that's confidently wrong.

---

## Three layers of intelligence, not one

Most chargeback tools stop at the dispute. Abstain climbs the whole ladder.

```text
                       ABSTAIN
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
     DISPUTE           MERCHANT          PORTFOLIO
     What do we do     Why is THIS       Why are we
     with this case?   merchant          losing overall?
                        failing?
         │                │                │
         └────────────────┼────────────────┘
                          ▼
                  ACTIONABLE INSIGHT
```

**Level 1 — Dispute.** Standard operational decision. Evidence in, decision out, in real time.

**Level 2 — Merchant.** A merchant can have a perfectly normal overall chargeback rate and still be quietly bleeding money from one specific, repeated weakness:

```text
Merchant A
────────────────────────────
Primary weakness:      Customer communication evidence
Loss concentration:    7 of 10 losses
Most affected reason:  Fraud / unauthorized dispute
```

Benchmarked against the portfolio, because a raw number doesn't tell you if it's a *them* problem or an *us* problem:

```text
                    Merchant A     Portfolio
──────────────────────────────────────────────
Loss rate              43%             41%
Communication gap      70%             30%
```

Merchant A's overall loss rate is basically average. Their communication gap is more than double the portfolio's. That's not noise — that's a specific, fixable operational problem unique to this merchant.

**Level 3 — Portfolio.** If the same evidence gap shows up across multiple *unrelated* merchants, it's stopped being a merchant problem — it's systemic:

```text
PORTFOLIO LOSS ANALYSIS
────────────────────────────────────
Customer communication      7 losses
Missing supporting evidence 5 losses
Delivery verification       4 losses
```

If "customer communication" keeps topping the list across merchants who otherwise have nothing in common, the fix isn't any single merchant's process — it's *our* evidence-collection pipeline. That's a different fix, and a more valuable insight than "contest more disputes."

```text
Individual dispute → Why did THIS case fail?
        ↓
Merchant           → Why does THIS merchant repeatedly fail?
        ↓
Portfolio          → Why do MULTIPLE merchants share the same failure?
        ↓
Systemic intervention
```

---

## Why Abstain beats a naive binary decision system

A binary contest/concede system has to pick a side even when it shouldn't. Here's what that costs you, using the same three cases as the actual engine:

**Scenario 1 — Conflicting evidence.** Merchant's transaction record says one thing, the customer's photos say another. A binary system thresholds on P(win) anyway and picks a side — confidently wrong roughly as often as it's confidently right, because the underlying signal genuinely doesn't support a call. Abstain checks the `conflicting_evidence` flag before it ever looks at expected value, and escalates outright: *"Evidence signals point in different directions — escalating regardless of EV, since an automated evidence draft could misrepresent the case."* It's not smarter about the evidence. It's honest about not being able to be smart about it.

**Scenario 2 — High-value, genuinely uncertain.** A ₹15,000+ dispute where the evidence is thin in both directions. A binary system with a 50% threshold either contests (and eats the ops cost + arbitration fee if it loses) or concedes (and leaves real money on the table if it would've won) — and at this dollar amount, either mistake is expensive. Abstain's self-consistency sampling catches the disagreement across repeated LLM calls, lands in LOW confidence, and routes it to a human — who gets the case *with* the reasoning already attached, not a blank slate.

**Scenario 3 — Repeat dispute against a merchant near their chargeback-rate threshold.** Looks profitable in isolation (P(win) is fine, amount clears ops cost). A binary system that only sees this one case says CONTEST. Abstain applies the repeat-dispute cost multiplier (arbitration fees run higher) *and* the portfolio risk penalty (this merchant is close enough to their threshold that one more contest carries downstream risk), and the combined EV flips the decision to CONCEDE — a call a single-case system structurally cannot make, because it doesn't know the merchant's portfolio position exists.

None of these are edge cases I hand-picked to look good — they're the exact three gates (`conflicting_evidence`, confidence-driven escalation, portfolio-aware EV) already live in `decision_engine.py`. Try them yourself on the live demo.

---

## Under the hood: the design decisions I actually had to make

### 1. Hybrid retrieval — reused knowledge, not reinvented code

I'd built hybrid retrieval (BM25 + dense/FAISS) before, on a legal AI assistant I worked on earlier. Reason codes and evidence requirements in chargeback disputes are short, keyword-heavy, and highly specific — "13.1," "not as described," exact policy phrasing. Pure dense retrieval is great at semantic similarity but genuinely bad at exact-term matching; pure BM25 is the opposite. Instead of picking one, I combined BM25 for exact reason-code/policy-term matches with FAISS for semantic similarity across evidence descriptions — and carried the *pattern* over from my earlier project rather than starting from scratch. Different domain, same underlying retrieval problem, so I reused the understanding, not the code.

### 2. Self-consistency for uncertainty, not the model's self-reported confidence

I almost just asked the LLM "how confident are you, 0 to 1?" and used that directly. Killed that fast — LLM self-reported confidence is notoriously miscalibrated, it just *sounds* authoritative. Instead I sample the model multiple times at non-zero temperature on the same case and measure how much the evidence-strength scores disagree:

```text
Low disagreement:            High disagreement:
0.82, 0.79, 0.84              0.84, 0.51, 0.23
      ↓                              ↓
Lower uncertainty              Higher uncertainty
      ↓                              ↓
Proceed automatically          Route to human
```

This adds token cost and latency, but gives the engine a practical disagreement signal. It's explicitly treated as a **disagreement proxy**, not a calibrated statistical confidence interval.

### 3. The LLM never executes the decision — full stop

The LLM's output is a strict, schema-validated JSON object: evidence strength, key gaps, reasoning. Nothing else. It never sees dispute economics, never sees the deadline, never sees the decision thresholds. `decision_engine.py` has **zero import of any LLM client** — that's not a design-doc claim, it's a fact about the dependency graph you can check yourself.

```text
EV(contest) = P(win) × dispute_amount − operating_cost − portfolio_risk_penalty
EV(concede) = 0
```

**Why I accepted the extra plumbing:** a financial decision needs to be reproducible and debuggable without interrogating a model's reasoning trace every time something goes wrong. Same inputs, same output, every time — non-negotiable for anything touching money.

### 4. Portfolio-aware risk, not per-dispute isolation

A dispute doesn't happen in a vacuum. A merchant sitting at 0.41% against a 0.45% chargeback-rate threshold gets treated more conservatively than one sitting at 0.12%, because contesting one more marginal case has downstream risk beyond that single dispute's dollar value.

### 5. Repeat-dispute cost multiplier

Handling a dispute isn't a flat cost — repeat/arbitration disputes cost more to process. I built an explicit multiplier instead of assuming every contest is priced the same. Small detail, but it's the difference between a toy model and something that reflects real ops economics.

### 6. Counterfactual evidence — a diagnosis, not just a verdict

Every ESCALATE or CONCEDE comes with: *what specific missing evidence would flip this decision?*

```text
Current decision:  CONCEDE
Missing evidence:  Customer communication log
Counterfactual:    If obtained and evidence strength crosses
                   the threshold → decision may flip to CONTEST
```

I reused the actual `decide()` function to compute this — the counterfactual just calls it again with one field perturbed — so the "what-if" logic can never silently drift out of sync with the real decision logic.

### 7. EV-ranked human attention, not FIFO

Escalated cases get ranked by potential value, not arrival order. The point isn't removing humans from the loop — it's making sure their limited time goes to the case where a human decision actually moves the needle.

### 8. Fail loud about failing, never fail silently

Groq rate limits happen. APIs go down. When the LLM call fails after retries, the system falls back to a conservative structural heuristic based on evidence completeness — and tags the result `source: "fallback_heuristic"`. It never pretends a degraded result came from the full pipeline.

Don't have a Groq key handy? The system runs entirely on this fallback path automatically — you can test the whole decision engine end to end without spending a single API call.

---

## Architecture

```text
Abstain/
├── app/
│   ├── main.py                      # FastAPI entrypoint
│   ├── models.py                    # Pydantic schemas
│   ├── db.py                        # SQLAlchemy models + session
│   ├── llm_client.py                # provider-agnostic LLM wrapper
│   ├── retrieval/
│   │   ├── hybrid_search.py         # BM25 + FAISS retriever
│   │   └── index_builder.py
│   ├── graph/
│   │   ├── pipeline.py              # LangGraph orchestration
│   │   ├── nodes.py
│   │   └── state.py
│   ├── engine/
│   │   ├── evidence_scorer.py       # LLM evidence assessment + self-consistency
│   │   ├── decision_engine.py       # deterministic EV-based policy
│   │   ├── portfolio.py             # merchant risk adjustment
│   │   ├── attention_ranking.py     # EV-ranked escalation queue
│   │   └── counterfactual.py        # "what evidence flips this?"
│   └── reporting/
│       └── portfolio_intelligence.py # merchant + portfolio root-cause reports
├── dataset/
│   ├── generate_dataset.py          # seeded, reproducible synthetic data
│   ├── reason_codes.json
│   ├── merchants.json
│   └── disputes.json
├── eval/
│   ├── run_eval.py
│   └── eval_report.md
├── tests/
│   ├── test_decision_engine.py
│   ├── test_counterfactual.py
│   └── test_portfolio_pairs.py
├── frontend/                        # console (deployed to Vercel)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

**Pipeline flow (LangGraph state machine):**

```text
Dispute in
   ↓
Retrieve case context (hybrid BM25 + FAISS)
   ↓
LLM evidence assessment (+ self-consistency sampling)
   ↓
Deterministic decision (EV + deadline + portfolio risk)
   ↓
CONTEST / CONCEDE / ESCALATE
   ↓
Counterfactual explanation
   ↓
Merchant + portfolio intelligence aggregation
   ↓
Human attention ranking (for ESCALATE queue)
```

---

## Evaluation — methodology and results

I built the eval harness specifically to catch the failure modes a plain accuracy number hides.

**How I score a three-way decision as a binary classifier:**

* Ground truth "should contest" = the dispute was actually won
* Ground truth "should concede" = the dispute was actually lost
* **ESCALATE is excluded from precision/recall on purpose.** It's the abstention mechanism working as designed, not a missed prediction — it's reported separately as the escalation rate.

**Cost accounting, not just accuracy:**

* False-positive cost: ops cost spent contesting a case that was actually lost
* False-negative cost: recoverable amount left on the table by conceding a case that was actually winnable
* Decision-boundary experiments track how changing the abstention threshold affects precision, recall, FP cost, and escalation rate

### Most recent run

**57 cases, 56 scored**

```text
TP = 14    FP = 11
FN = 16    TN = 11

Precision = 0.56
Recall    = 0.467
Escalation rate ≈ 7%
```

### Decision-boundary experiment

To test whether the engine should be more conservative around medium-confidence decisions, I widened the EV margin from **0.25× → 0.35×**:

```text
                           margin=0.25      margin=0.35
Precision                     0.560            0.632
Recall                        0.467            0.429
FP cost                       ₹7,970           ₹5,000
Escalation rate                  7%              18%
```

The wider margin increased precision from **56.0% → 63.2%** and reduced false-positive cost by roughly **37%**, while increasing the number of cases deliberately handed to humans.

The change moved 6 cases from auto-decision into ESCALATE: **4 former false positives and 2 former true positives**. That demonstrates the intended tradeoff — the engine becomes more selective about automated decisions rather than forcing uncertain cases through.

**The important metric for Abstain is therefore not "contest everything" recovery.** The project is specifically designed to trade some automatic decisions for safer, more trustworthy automation when the evidence is uncertain.

### Why economics, not just accuracy

Contesting a ₹500 dispute that should've been conceded and conceding a ₹10,000 dispute that could've been won are not the same mistake.

Plain accuracy treats them identically.

This harness doesn't.

---

## Running it locally

```bash
pip install -r requirements.txt

# .env
GROQ_API_KEY=your_api_key

uvicorn app.main:app --reload --port 8000
# → http://127.0.0.1:8000/docs
```

No `GROQ_API_KEY`? Leave it blank — the system runs entirely on the fallback heuristic, so you can exercise the full pipeline without an API call.

```bash
# Evaluation
python eval/run_eval.py        # → eval/eval_report.md

# Tests
pytest
```

---

## Tech stack

**Backend** — Python, FastAPI, Pydantic, PostgreSQL, Docker
**Retrieval** — hybrid BM25 + FAISS
**AI/LLM** — LangChain, LangGraph orchestration, Groq, self-consistency uncertainty estimation, strict structured-output validation
**Eval** — Python
**Deployed on** — Vercel (frontend), Render (API)

---

## What I'd build next

* Calibrate uncertainty properly with a larger labeled validation set instead of relying on the disagreement proxy alone
* Historical-outcome learning to improve win-probability estimates over time
* Merchant-specific policy configuration instead of one global threshold set
* Reviewer feedback loop so human ESCALATE decisions actually improve the model
* What-if portfolio simulation before rolling out policy changes

---

## The thing I actually believe about this project

Don't automate the decision just because you can automate the prediction.

In a high-stakes workflow, the best AI system isn't the one that makes the most calls on its own. It's the one that knows the difference between when to act, when not to, and when to hand it to a person — and on top of that, tells you *why* you keep losing in the first place.
