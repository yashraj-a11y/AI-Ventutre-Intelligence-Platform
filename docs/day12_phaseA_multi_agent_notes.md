# Day 12 · Phase A — Multi-Agent AI (conceptual notes)

*Layer 12 foundation. No agent code — the "why" before the "how".*

---

## 1. The problem: one reasoner produces one shape of answer

Day 11 gave the platform a single reasoning pipeline that emits one `VentureDecision`. It's
coherent — but it speaks in *one voice*. A real venture committee doesn't work that way: a market
specialist, a finance analyst, a risk officer, and an investment partner each examine the *same*
startup through a *different lens*, and the decision emerges from their deliberation. The value
isn't more data — it's more *perspectives* on the same data, and an explicit process for reconciling
them.

**Multi-agent intelligence** is the discipline of decomposing one analysis into specialized agents,
running them, and combining their outputs through a consensus mechanism. Layer 12 turns the
platform from a single reasoner into a **committee**.

---

## 2. Why specialization helps — divide and conquer

A monolithic analysis has to weigh everything at once, and the important-but-narrow signals get
averaged into the general verdict. Specialization fixes this by **assigning each agent one
responsibility** and letting it score *only* its slice:

- The **risk agent** obsesses over what could go wrong — it isn't distracted by a favorable market.
- The **market agent** reads category timing — it isn't dragged down by a shaky goal.
- The **investment agent** synthesizes, but only *after* each specialist has spoken.

Divide-and-conquer buys three things: **focus** (each agent's score reflects its lens, not a blend),
**legibility** (you can see *which perspective* supports or opposes), and **modularity** (add or
retune one agent without touching the others). A committee decision that says "market says go,
funding says wait" is far more actionable than a single "62%."

---

## 3. The six perspectives (honest roles for this dataset)

Because Kickstarter is an anonymous pre-launch snapshot, the agents analyze what *actually exists* —
each is a specialist view over a real slice of the Day-11 intelligence:

| Agent | Lens | Real signal it reads |
|---|---|---|
| **Market** | Is the category attractive & well-timed? | category funding rate, crowding, year-trend, timing |
| **Founder** *(Creator Execution)* | Is the campaign well-crafted? | title / goal / structure drivers (Day-5/8) |
| **Financial** *(Funding Viability)* | Is the goal realistic & fundable? | goal vs category median, funding probability |
| **Competition** | How crowded, and how is it positioned? | competitive intensity, position vs winners |
| **Risk** | What could sink it? | Day-9 risk list + investor objections |
| **Investment** | Given all of the above — invest? | synthesis over the other five + readiness |

The agents keep the roadmap's names but not its fictions: there is no founder-experience, burn-rate,
or TAM data, so those responsibilities are **deferred**, not fabricated (see §5).

---

## 4. The honest caveat: correlated views, not independent experts

This is the single most important idea in Layer 12, and it's where a naive multi-agent system
overclaims. In a real committee, experts have **independent information** — the finance analyst read
the books, the market analyst ran the survey. Averaging independent opinions genuinely reduces error
(the wisdom of crowds).

**Our agents do not have independent information.** They all read the *same*
`VentureIntelligence` object. The market agent and the competition agent both read the category
cohort; the financial and risk agents both key off the goal-vs-median gap. So their stances are
**correlated specialist views of one shared signal set**, not independent votes. Concretely: if two
agents agree, that agreement is *partly structural* (they're looking at overlapping evidence), not
independent confirmation.

Why build it anyway, and how to stay honest:
- It's still valuable — it **routes** one signal set into perspective-specific verdicts and surfaces
  *where the perspectives genuinely tension* (e.g. favorable market vs unrealistic goal).
- But the consensus engine must **say so** in its honesty note, and must **not** claim that agreement
  = independent corroboration or inflate confidence by treating correlated votes as independent.

Naming this limitation is what separates an honest committee architecture from theater.

---

## 5. Committee AI: orchestration and consensus

Two coordination pieces sit above the agents:

- **Orchestration** (Phase H) — aggregate the campaign *once*, then run all six agents on that one
  shared intelligence, collecting a uniform `AgentReport` from each. The orchestrator owns sequencing
  and sharing; it computes nothing itself.
- **Consensus** (Phase I) — combine the agent stances into a committee band (Strong Invest / Proceed
  to Due Diligence / Monitor Progress / Do Not Invest) via a **confidence-weighted vote**, with two
  honesty features: a **risk veto** (a strong risk stance caps an otherwise-positive committee — you
  don't invest past a red flag) and **conflict surfacing** (list the agents that genuinely disagree).

The `AgentReport` contract is uniform — `{agent, perspective, score, stance, confidence, findings,
key_metric, deferred, caption}` — so the orchestrator and consensus treat every agent identically.

---

## 6. The honesty boundary (Tier-3 deferrals)

Multi-agent systems are the *most* tempting place to fabricate, because an impressive-sounding
"Founder Analyst: experienced team, strong leadership" reads like real committee output. We refuse:
founder experience / team / leadership; burn / runway / revenue / CAC-LTV; market size / TAM / CAGR;
moat / differentiation / named competitors; regulatory / legal / operational risk; expected return /
IRR / valuation. Each agent records its own deferrals as data — the committee documents exactly what
it *cannot* assess.

---

## 7. Multi-agent orchestration — the pattern

Layer 12 is an **orchestration-plus-aggregation** pattern: specialist projections (the agents) over
a shared representation (`vi`), coordinated by a thin orchestrator, reconciled by an explicit
consensus rule. It's deterministic — no LLM — so every committee decision is reproducible and no
agent can hallucinate a competitor or a founder. The sophistication is in *decomposition and
reconciliation*, not in any single agent's cleverness.

> **Interview soundbite:** *"Layer 12 turns the single Day-11 reasoner into a committee of six
> specialist agents — market, funding, creator, competition, risk, and investment — each scoring the
> campaign through one lens, reconciled by a confidence-weighted consensus with a risk veto. The
> honest subtlety is that these agents aren't independent experts: they all read the same aggregated
> intelligence, so their stances are correlated views of one signal set, not independent votes — so
> the consensus is a structured routing of perspectives, and I say so rather than inflating
> confidence by pretending the agreement is independent. It's deterministic, no LLM, so nothing is
> hallucinated, and every agent defers what the data can't support — founder background, financials,
> market size — instead of fabricating a convincing-sounding analysis."*
