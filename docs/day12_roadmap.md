# Day 12 — Multi-Agent Venture Intelligence System (FROZEN SPEC V1.1)

**Architecture V1.0 — Layer 12 (Multi-Agent Intelligence Layer).** Six specialist agents each
analyze a campaign from a distinct perspective, an orchestrator coordinates them on ONE warm stack,
and a consensus engine resolves their stances into a unified **committee decision**.

> **Status:** FROZEN 2026-08-04 (V1.1). Read before building any phase. Build phase-by-phase on
> `build Phase <X>` / `build the notebook`; each phase = implement **and** teach via Framework 1,
> grounded in real project numbers.

---

## 0. What changed from the draft (changelog + why)

Day 12's architecture — specialization, orchestration, consensus — is genuinely valuable and
implementable. But the draft's agent *examples* describe a VC evaluating a SaaS startup with
founders, burn rate, and TAM. **Kickstarter has none of that** (verified: no founder / team /
financial / market-size columns anywhere). The agents must be **specialist views over the real
Day 4-11 signals**, not independent data pipelines inventing data. Corrections:

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Founder Analyst: "experienced founders", "team composition", "leadership", "advisory board", "founder embeddings", "Founder-Team Gap Analysis" | **Reframe → Creator Execution view** over the *only* creator-controllable signals (title craft, goal realism, campaign structure — Day-5 drivers + Day-8 levers). Founder experience / team / leadership / advisory → **DEFERRED**. | Kickstarter is anonymous pre-launch; there is **no founder or team data at all**. This is the most-reframed agent — it evaluates *campaign craft*, not a person. |
| **C-2** | Financial Analyst: "burn rate", "revenue growth", "runway ~10 months", "unit economics", "CAC/LTV", "profitability" | **Reframe → Funding Viability view**: goal realism vs category median, funding probability, reachable optimum, goal-size risk (Day-4 + Day-8 + Day-9). Burn / runway / revenue / CAC-LTV / unit-economics → **DEFERRED**. | The only "financial" number in the dataset is the **funding goal**. No P&L, no runway. |
| **C-3** | Market Analyst: "market size", "TAM", "industry growth rate", "CAGR" | **Reframe → Category & Market-Timing view** over Day-9 `category_analysis` (category funding rate, crowding, `year_trend`, `market_timing`). TAM / market-size / CAGR / industry-growth → **DEFERRED**. | The "market" we can observe is the **Kickstarter category cohort**, not an addressable market. (Real signal exists: e.g. Games rising, timing Favorable.) |
| **C-4** | Competition Analyst: "moat", "differentiation", "network effects", "similar success/failures via Semantic Retrieval" | Keep **competitive intensity + crowding + your-position-vs-winners** (Day-9 `competition_analysis`, frame cohort). Moat / differentiation / named-competitors → **DEFERRED**. "Similar startups" = **frame cohort, NO faiss**. | Day-9 competition is real (intensity, winners' median goal). Moat needs product data we lack; faiss would re-introduce the OpenMP hazard for no honest gain (same as Days 10/11). |
| **C-5** | `multi_agent_results.pkl` | **`.json`** content-hash manifest + `verify()`, mirroring the Day-8/10/11 serializers. | Engine = configuration over models, not a binary. JSON-not-pickle is the project-wide rule. |
| **C-6** | "A single **LLM** tends to produce generic analyses" (implies LLM agents) | **Agents are DETERMINISTIC specialist analyzers** — rule-based scoring over real signals, **no LLM, no new model, no new deps**. | Consistent with Days 9/11 (deterministic templates, "can't hallucinate a competitor"). LLM agents would re-introduce hallucination — exactly what this project avoids — and violate no-new-deps. Optional LLM narration is a future extension, not the mandatory path. |

**The single most important honesty point (C-7 — read this):**
> The six agents are **NOT independent expert opinions**. They all read the **same**
> `VentureIntelligence` object (the Day-11 aggregator), so their stances are **correlated
> specialist views of one shared signal set**, not independent votes. Consensus is therefore a
> *structured aggregation of perspectives*, **not** a statistical average of independent experts —
> and the honesty note must say so. This is still valuable (it surfaces *which perspective*
> supports/opposes and where they genuinely tension), but claiming independent-expert diversity
> would be false. The "market" and "competition" agents in particular overlap (both read the
> category), so their agreement is partly structural.

**IMPROVEMENTS beyond the draft:**
- **IMP-1 — one warm faiss-free stack.** Build ONE Day-11 `IntelligenceAggregator`, aggregate the
  campaign **once**, and pass `intelligence=vi` to all six agents + orchestrator. No re-aggregation,
  one model in memory, deterministic.
- **IMP-2 — per-agent confidence from that agent's OWN signal strength** (cohort size for market,
  goal-clarity for funding, risk-severity spread for risk, etc.), so agents are genuinely
  differentiated — not all echoing the Day-11 global confidence.
- **IMP-3 — a uniform `AgentReport` contract** ({agent, perspective, score 0-100, stance, confidence,
  findings[], key_metric, deferred, caption}) so the orchestrator/consensus treat agents uniformly.
- **IMP-4 — honest consensus with conflict surfacing** — confidence-weighted stance vote → committee
  band, plus an explicit `agreement_level`, a `conflicts` list, and a **risk veto** (a High-severity
  risk agent can cap an otherwise-positive committee), with the correlation caveat in the caption.
- **IMP-5 — honesty-tiers-as-data**: each agent carries its own `deferred_*`.

---

## 1. Honesty tiers (what may appear in agent outputs)

**Tier 1 — REAL (may drive agent scores):** category funding rate / lift / crowding / `year_trend` /
`market_timing` (Day 9); competitive intensity + winners'-median-goal position (Day 9); calibrated
funding probability + threshold + readiness stage (Days 4/10); goal vs category funded-median +
goal-size risk (Days 8/9); Day-5 SHAP drivers for creator-controllable features (title, uppercase,
goal, duration); Day-9 risk list + investor objections; Day-11 prioritized actions + confidence.

**Tier 2 — DERIVED, label model-derived:** per-agent 0-100 score; per-agent stance
(positive/cautious/negative); per-agent confidence; committee band (Strong Invest / Proceed to Due
Diligence / Monitor Progress / Do Not Invest); agreement level. All "model-simulated, not causal."

**Tier 3 — DEFERRED, NEVER fabricate:** founder experience / team / leadership / advisory board /
founder embeddings; burn rate / runway / revenue / CAC-LTV / unit economics / profitability; market
size / TAM / CAGR / industry growth; competitive moat / differentiation / network effects / named
competitors; regulatory / legal / IP / operational risk; expected return / IRR / valuation.

---

## 2. The six agents (honest roles) + uniform contract

Every agent implements `analyze(raw, intelligence=vi) -> AgentReport`:

```
AgentReport = {
  agent:        "<name>",
  perspective:  "<one-line role>",
  score:        0-100,                       # perspective score (Tier 2, labeled)
  stance:       "positive" | "cautious" | "negative",
  confidence:   0.0-1.0,                      # from THIS agent's signal strength (IMP-2)
  findings:     ["<grounded finding>", ...],  # 2-4, each tied to a real number
  key_metric:   {name, value},
  deferred:     {...Tier-3 for this agent...},
  caption:      "model-simulated, not causal; <what's deferred>",
}
```

| File (draft name kept) | Honest role | Reads (real signals) | Defers |
|---|---|---|---|
| `market_agent.py` — **MarketAgent** | Category & market-timing | Day-9 category: funding rate, crowding, `year_trend`, `market_timing` | TAM / market size / CAGR |
| `founder_agent.py` — **FounderAgent** *(Creator Execution)* | Creator's controllable campaign craft | Day-5 drivers (title/uppercase/goal), Day-8 title lever | founder experience / team / leadership |
| `financial_agent.py` — **FinancialAgent** *(Funding Viability)* | Goal realism + funding odds | goal vs category median, Day-4 prob, Day-8 goal lever, Day-9 goal risk | burn / runway / revenue / CAC-LTV |
| `competition_agent.py` — **CompetitionAgent** | Cohort competition & positioning | Day-9 competition: intensity, winners'-median position | moat / differentiation / named competitors |
| `risk_agent.py` — **RiskAgent** | Risk & investor objections | Day-9 risk list + `overall_risk_level` + objections | regulatory / legal / operational / IP risk |
| `investment_agent.py` — **InvestmentAgent** | Committee synthesis | the other 5 agents + Day-10 readiness + Day-11 confidence | expected return / IRR / valuation |

**Stance derivation (deterministic):** each agent maps its score to a stance via fixed cutoffs
(e.g. `≥60 positive`, `40-60 cautious`, `<40 negative`) — documented per agent. Confidence from the
agent's own signal strength (IMP-2): e.g. MarketAgent confidence rises with cohort `n` and trend
consistency; RiskAgent confidence rises when its risks agree in severity.

---

## 3. Consensus & committee decision (Phase I)

`ConsensusEngine.decide(agent_reports) -> CommitteeDecision`:

```
CommitteeDecision = {
  committee_recommendation: "Strong Invest" | "Proceed to Due Diligence" | "Monitor Progress" | "Do Not Invest",
  overall_confidence:  0.0-1.0,             # confidence-weighted agent confidences
  agent_votes:         [{agent, stance, confidence, weight}],
  agreement_level:     0.0-1.0,             # 1 - normalized stance dispersion
  conflicts:           [{between, tension}],# surfaced disagreements
  risk_veto:           bool,                # High risk caps a positive committee
  rationale:           "<deterministic synthesis>",
  deferred:            {...},
  honesty_note:        "Correlated specialist views of one signal set, not independent experts.",
}
```

- **Weighted vote:** stance ∈ {−1, 0, +1}; committee score `= Σ wᵢ·confᵢ·stanceᵢ / Σ wᵢ·confᵢ`.
  Map to the band by cutoffs. Weights in config (InvestmentAgent/Risk weighted higher).
- **Risk veto (IMP-4):** if RiskAgent stance is negative with high confidence *and*
  `overall_risk_level == High`, cap the committee at "Monitor Progress" (a positive committee can't
  ignore a strong risk).
- **Conflict surfacing:** list agent pairs whose stances disagree by ≥ 2 levels (e.g. Market
  positive vs Financial negative), with the tension named.
- **Correlation caveat** in the honesty note (C-7).

---

## 4. Environment & reuse

- **NO NEW DEPS. NO FAISS. NO TORCH. NO OpenMP. NO LLM. Deterministic. JSON not pickle.** lightgbm +
  pandas only — same as Days 10/11. Run `PYTHONPATH=. ./.venv/bin/python`.
- **Reuse (build ONCE, inject everywhere):** ONE Day-11 `IntelligenceAggregator` → `vi`
  (`intelligence=vi` into all six agents). Agents read their slice of `vi`; no agent calls a model
  directly except via the shared stack already inside `vi`.
- **PREFLIGHT (before Phase B):** aggregate once, confirm every agent's source slice is present in
  `vi` (category_analysis, competition_analysis, risk_analysis, drivers, counterfactual, evolution,
  scenario) and that no faiss/torch import occurs.

---

## 5. Phases (build order)

- **A** — Multi-Agent AI notes (conceptual): specialization, divide-and-conquer, committee AI, and
  the honest caveat that these agents are correlated views (C-7). `docs/day12_phaseA_multi_agent_notes.md`
  + `src/agents/__init__.py`.
- **B** — `market_agent.py` **MarketAgent** (category & market-timing).
- **C** — `founder_agent.py` **FounderAgent** (creator execution; founder-background deferred).
- **D** — `financial_agent.py` **FinancialAgent** (funding viability; financials deferred).
- **E** — `competition_agent.py` **CompetitionAgent** (cohort competition).
- **F** — `risk_agent.py` **RiskAgent** (Day-9 risk + objections).
- **G** — `investment_agent.py` **InvestmentAgent** (committee synthesis over the other 5 + readiness).
- **H** — `orchestrator.py` **AgentOrchestrator** (one aggregation 
- → run 6 agents → collect reports).
- **I** — `consensus.py` **ConsensusEngine** (weighted vote + risk veto + conflict surfacing → committee).
- **J** — `serialization.py` **AgentSystemSerializer** (content-hash JSON manifest + `verify()`).
- **Notebook** — `notebooks/12_Multi_Agent_Intelligence.ipynb`: one campaign → six agent reports →
  orchestration → committee decision → serialized verify; honesty boundary shown.

---

## 6. Repository & artifacts after Day 12

```
src/agents/{__init__, market_agent, founder_agent, financial_agent, competition_agent,
            risk_agent, investment_agent, orchestrator, consensus, serialization}.py
notebooks/12_Multi_Agent_Intelligence.ipynb
artifacts/agents/{market_analysis, founder_analysis, financial_analysis, competition_analysis,
                  risk_analysis, investment_analysis, committee_decision, agent_system_engine}.json  # all JSON (C-5)
configs/preprocessing.yaml → day12.{market, founder, financial, competition, risk, investment,
                                     consensus, serialization}
docs/day12_phaseA_multi_agent_notes.md
```

**Input contract:** every agent consumes the Day-11 `VentureIntelligence` (`vi`) — keys
`prediction`, `drivers`, `counterfactual`, `copilot.{category_analysis, competition_analysis,
risk_analysis, investor_objections}`, `evolution.stage`, `scenario`, `cohort`. (⚠️ Day-8 API duality
still applies where recs are read.)

---

## 7. Non-negotiables (frozen)

1. **No fabrication** — founder / financials / market-size / moat / regulatory DEFERRED, never emitted (C-1..4).
2. **Agents are correlated specialist views of one shared `vi`** — consensus says so (C-7).
3. **Deterministic, no LLM, no new model, no new deps, faiss/torch-free** (C-6).
4. **JSON not pickle**, content-hash + `verify()` (C-5).
5. **ONE warm stack, one aggregation** threaded into all agents (IMP-1).
6. **Uniform `AgentReport` contract**; per-agent confidence from own signal strength (IMP-2/3).
7. **Risk veto + conflict surfacing** in consensus (IMP-4).
8. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-04. Supersedes the Day-12 draft. Build on `build Phase <X>`.*
