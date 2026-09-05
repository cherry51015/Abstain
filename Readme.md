# Abstain
### An AI Risk Decision Engine for Chargebacks

**The LLM reads the evidence. It never touches the money.**

---

## Why I built this

Every chargeback automation tool I looked at was solving the wrong problem. They were all trying to answer "can we win this dispute?" — basically a yes/no classifier wearing an AI costume.

But that's not what a risk team actually needs. What they need answered is:

> "Given the evidence we have, the money at stake, the deadline we're up against, and this merchant's risk history — should we contest, concede, or get a human to look at this?"

That's a completely different problem. And it needs a completely different architecture.

So I built Abstain around one rule I didn't compromise on: **the LLM is never allowed to spend money.** It reads evidence and tells you how strong it is. A separate, deterministic policy engine takes that read and decides what happens next. Two layers, two jobs, never mixed.

And then I went one step further — because just deciding case-by-case felt incomplete. If Merchant A keeps losing disputes for the same reason every month, a system that only says "CONCEDE" on each individual case is missing the actual story. So Abstain doesn't stop at the dispute. It climbs up to *why this merchant keeps failing*, and then up again to *why the whole portfolio is bleeding money* — and tells you which one it is.

---

## The one-line pitch

**Abstain uses an LLM to understand chargeback evidence, a deterministic engine to decide what to do with it, and a diagnostic layer to explain why merchants — and the whole portfolio — keep losing.**

---

## The core idea, visually

```
                 ┌────────────────────┐
                 │   Chargeback Case   │
                 └──────────┬─────────┘
                             ▼
                 ┌────────────────────┐
                 │  LLM Evidence       │   ← reads, doesn't decide
                 │  Scorer              │
                 └──────────┬─────────┘
                             ▼
                 ┌────────────────────┐
                 │  Deterministic       │   ← decides, doesn't guess
                 │  Decision Engine      │
                 └──────────┬─────────┘
                 ┌───────────┼───────────┐
                 ▼           ▼           ▼
             CONTEST     CONCEDE     ESCALATE
                                         │
                                         ▼
                              ┌────────────────────┐
                              │ Human Attention      │
                              │ Ranking               │
                              └────────────────────┘
```

That third outcome — **ESCALATE** — is the whole philosophy of this project in one word. Most systems are built to always output an answer. I wanted a system that's honest enough to say "I don't know enough here, get a person." A model that can abstain is more trustworthy than one that's confidently wrong.

---

## What makes this different: three layers of intelligence, not one

Most chargeback tools stop at the dispute. Abstain climbs the whole ladder.

```
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

### Level 1 — Dispute: what do we do right now?

Standard operational decision. Evidence in, decision out, in real time.

### Level 2 — Merchant: why does *this* merchant keep losing?

A merchant can have a perfectly fine overall chargeback rate and still be quietly bleeding money from one specific, repeated weakness. So for every merchant, Abstain builds a diagnosis:

```
Merchant A
────────────────────────────
Primary weakness:      Customer communication evidence
Loss concentration:    7 of 10 losses
Most affected reason:  Fraud / unauthorized dispute
Pattern observed:      Transactions are well documented,
                        but communication logs are almost
                        never available.
```

And then I benchmark that merchant against the portfolio average, because a number alone doesn't tell you if it's a *them* problem or an *us* problem:

```
                    Merchant A     Portfolio
──────────────────────────────────────────────
Loss rate              43%             41%
Communication gap      70%             30%
Delivery evidence gap  10%             18%
Fraud disputes         50%             32%
```

Merchant A's overall loss rate is basically average. But their communication gap is more than double the portfolio's. That's not noise — that's a specific, fixable operational problem unique to this merchant.

### Level 3 — Portfolio: why is the *business* losing?

Zoom out further. If I see the same evidence gap showing up across multiple unrelated merchants, that's not a merchant problem anymore — that's a systemic one.

```
PORTFOLIO LOSS ANALYSIS — 23 total losses

Top recurring weaknesses
────────────────────────────────────
Customer communication      7 losses
Missing supporting evidence 5 losses
Delivery verification       4 losses
Other                       7 losses
```

If "customer communication" keeps showing up as the top reason across merchants who otherwise have nothing in common, the problem isn't any single merchant's process. It's *our* evidence-collection pipeline. That's a completely different fix — and a much more valuable insight than "contest more disputes."

```
Individual dispute → Why did THIS case fail?
        ↓
Merchant           → Why does THIS merchant repeatedly fail?
        ↓
Portfolio          → Why do MULTIPLE merchants share the same failure?
        ↓
Systemic intervention
```

This is what turns Abstain from a classifier into a risk-intelligence platform. It goes prediction → diagnosis → root cause.

---

## Under the hood: the design decisions I actually had to make

### 1. Hybrid retrieval, not pure dense — reused, not reinvented

I've built hybrid retrieval before (BM25 + dense/FAISS) on [[levi-legal-ai]], my legal AI assistant, and I knew exactly why I'd need it again here. Reason codes and evidence requirements in chargeback disputes are short, keyword-heavy, and highly specific — "3.13", "not as described," exact policy phrasing. Pure dense retrieval is great at semantic similarity but it's genuinely bad at exact-term matching, and pure BM25 is bad at paraphrase and synonymy. So instead of picking one, I combined BM25 (for exact reason-code and policy-term matches) with FAISS (for semantic similarity across evidence descriptions), and I carried that same architectural pattern over from Levi rather than reinventing it from scratch. Different domain, same underlying retrieval problem — so I reused the knowledge, not the code.

### 2. Self-consistency for uncertainty, not the model's self-reported confidence

Early on I almost just asked the LLM "how confident are you, 0 to 1?" and used that directly. I killed that idea fast — LLM self-reported confidence is notoriously miscalibrated, it just sounds authoritative. Instead I sample the model multiple times at non-zero temperature on the same case and measure how much the evidence-strength scores *disagree* with each other.

```
Low disagreement:            High disagreement:
0.82, 0.79, 0.84              0.84, 0.51, 0.23
      ↓                              ↓
Lower uncertainty              Higher uncertainty
      ↓                              ↓
Proceed automatically          Route to human
```

Tradeoff I accepted here: this costs more tokens and more latency per case than a single call. I decided that was worth it, because the whole point of this system is not making confidently wrong financial decisions. I'm explicit in the docs that this is a *disagreement proxy*, not a calibrated statistical confidence interval — I didn't want to overclaim rigor I hadn't validated at scale.

### 3. LLM never executes the decision — full stop

The LLM's output is a strict, schema-validated JSON object: evidence strength, key gaps, reasoning. Nothing else. It never sees dispute economics, never sees the deadline, never sees the decision thresholds. The deterministic engine downstream is the only thing that combines evidence strength with expected value, operating cost, and portfolio risk to actually choose CONTEST / CONCEDE / ESCALATE.

```
EV(contest) = P(win) × dispute_amount − operating_cost − portfolio_risk_penalty
EV(concede) = 0
```

I made this split on purpose even though it's more code and more plumbing than just letting the LLM output a decision directly. Why: a financial decision needs to be reproducible, testable, and debuggable without needing to interrogate a model's reasoning trace every time something goes wrong. Deterministic policy means the same inputs always produce the same output — that's non-negotiable for anything touching money.

### 4. Portfolio-aware risk, not per-dispute isolation

A dispute doesn't happen in a vacuum. If a merchant is already close to a chargeback-rate threshold, contesting one more marginal case has downstream risk beyond that single dispute's dollar value. So the decision engine pulls in merchant-level chargeback rate against threshold and adjusts the risk penalty accordingly — a merchant sitting at 0.41% against a 0.45% threshold gets treated more conservatively than one sitting at 0.12%.

### 5. Repeat-dispute cost multiplier

Handling a dispute isn't a flat operational cost — repeat disputes from the same merchant cost more to process. I built in an explicit multiplier on operating cost for repeat cases instead of assuming every contest costs the same to run. Small detail, but it's the difference between a toy model and something that reflects real ops economics.

### 6. Counterfactual evidence — turning a diagnosis into an action item

I didn't want the system to just say "this case is weak" and stop there. So every ESCALATE or CONCEDE decision comes with a counterfactual: what specific missing evidence would flip this decision if we had it?

```
Current decision:  CONCEDE
Missing evidence:  Customer communication log
Counterfactual:    If obtained and evidence strength crosses
                    the threshold → decision may flip to CONTEST
```

This turns Abstain from a passive classifier into something that tells an operator exactly what to go find.

### 7. EV-ranked human attention, not FIFO

Escalated cases don't get reviewed in the order they arrive — they get ranked by potential value, so a high-dollar uncertain case surfaces before a low-dollar uncertain one. The point isn't to remove humans from the loop. It's to make sure their time goes to the case where a human decision actually moves the needle.

### 8. Fail loud about failing, never fail silently

Groq rate limits happen. APIs go down. I didn't want a live demo — or a production system — to just crash because an external provider had a bad minute. So when the LLM call fails after retries, the system falls back to a conservative structural heuristic based on evidence completeness, and it tags the result:

```
source = "fallback_heuristic"
```

It never pretends a degraded result came from the full LLM pipeline. The audit trail always tells the truth about how a decision was actually made.

---

## Architecture

```
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
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

**Pipeline flow (LangGraph state machine):**

```
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

## Evaluation — and I'm not going to oversell it

I built an eval harness specifically to catch the failure modes a naive accuracy number would hide:

- **TP** — correctly contested a winnable dispute
- **FP** — contested one that should've been conceded
- **TN** — correctly conceded
- **FN** — conceded one that should've been contested
- **ESCALATE** is tracked separately as an abstention outcome — I didn't want to penalize the system for correctly saying "I'm not sure" the same way I'd penalize a confidently wrong call

**Current snapshot** (57 cases, 56 scored):

```
TP = 10    FP = 4
FN = 15    TN = 10

Precision = 0.714
Recall    = 0.400
Escalation rate ≈ 30%

Baseline net: ₹260,110
Engine net:   ₹181,280
FP cost: ₹3,170   FN cost: ₹18,730
```

Being straight about this: this run got hit by Groq rate limiting, so it's a mix of real LLM-scored cases and fallback-heuristic cases. I'm treating this as an engineering benchmark to debug the pipeline, not as a clean measurement of the LLM-backed system's real performance. I'd rather say that clearly than let the number imply more than it does.

**Why economics, not just accuracy:** contesting a ₹500 dispute you should've conceded and conceding a ₹10,000 dispute you could've won are not the same mistake. A plain accuracy score treats them identically. Abstain's eval tracks the actual cost of each error type, because the whole point of the system is the financial outcome, not the classification score.

---

## Running it

```bash
pip install -r requirements.txt

# .env
GROQ_API_KEY=your_api_key

uvicorn app.main:app --reload --port 8000
# → http://127.0.0.1:8000/docs
```

For local dev/testing, `GROQ_API_KEY` can stay blank — the system runs entirely on the fallback heuristic path, so you can test the full pipeline without burning API calls.

Run the eval:

```bash
python eval/run_eval.py
# → eval/eval_report.md
```

Run tests:

```bash
pytest
```

---

## Tech stack

**Backend** — Python, FastAPI, Pydantic, PostgreSQL, Docker
**Retrieval** — hybrid BM25 + FAISS
**AI/LLM** — LangChain, LangGraph orchestration, Groq, self-consistency uncertainty estimation, strict structured-output validation
**Eval** — Python + pandas

---

## What I'd build next

- Calibrate uncertainty properly with a larger labeled validation set instead of relying on the disagreement proxy alone
- Historical-outcome learning to improve win-probability estimates over time
- Merchant-specific policy configuration instead of one global threshold set
- Reviewer feedback loop so human ESCALATE decisions actually improve the model
- What-if portfolio simulation before rolling out policy changes

---

## The thing I actually believe about this project

Don't automate the decision just because you can automate the prediction.

In a high-stakes workflow, the best AI system isn't the one that makes the most calls on its own. It's the one that knows the difference between when to act, when not to, and when to hand it to a person — and on top of that, tells you *why* you keep losing in the first place.