# Day 11 · Phase A — Decision Intelligence (conceptual notes)

*Layer 11 foundation. No model code — this is the "why" before the "how".*

---

## 1. The problem: nine oracles, no decision

By the end of Day 10 the platform has nine working intelligence layers:

| Layer | Answers |
|---|---|
| Prediction (D4) | "What's the funding probability?" — 24% |
| Explainability (D5) | "Which features drove that?" — goal, duration |
| Knowledge Base (D6) | "Which campaigns are similar?" |
| Pattern Mining (D7) | "What patterns separate funded from failed?" |
| Counterfactual (D8) | "What single changes would help?" |
| Objection Simulator (D9) | "What would an investor push back on?" |
| Evolution Planner (D10) | "What's the readiness stage + prep roadmap?" |
| Scenario Simulator (D10) | "How do multi-change strategies compare?" |

Each answers *one question well*. But a founder or investor doesn't have one question — they have
**one decision**: *should I launch / fund this, and if so, what do I do first?* Nine isolated
outputs don't make a decision; a person still has to hold all nine in their head and reconcile
them. That reconciliation is the missing layer.

**Decision Intelligence** is the discipline of turning many analytical signals into a single,
ranked, defensible course of action. Layer 11 is where the platform stops *analyzing* and starts
*recommending*.

---

## 2. Why synthesis must come *after* the models, not inside them

A tempting shortcut is to bake recommendations into each module. We deliberately did **not** —
and Day 11 is why. Three reasons synthesis is its own layer:

1. **Signals interact.** The counterfactual says "lower your goal"; the scenario simulator says
   "lowering the goal alone gets you to Ready, but the title tweak actually hurts"; the objection
   simulator says "investors worry about the category's crowdedness." No single module sees all
   three. Only a layer *above* them can weigh a recommendation by *funding impact AND historical
   evidence AND investor concern AND scenario gain* at once.
2. **Audience differs, intelligence doesn't.** A founder wants "do this before launch"; an investor
   wants "fund / wait." These are two **views** of the *same* aggregated intelligence — computing
   them separately inside modules would let them silently drift apart. Synthesis lets them share
   one source of truth (they can't disagree).
3. **Confidence is a property of the ensemble, not any one model.** "How sure are we?" can only be
   answered by looking across signals — does the prediction margin, the similar-campaign cohort,
   the historical patterns, and the scenario stability all point the same way? That's an
   orchestration-level question by construction.

---

## 3. The recommendation hierarchy

Layer 11 produces a **hierarchy**, not a flat list:

```
VentureDecision
├── investment_readiness        (stage + probability + readiness index)   ← the headline verdict
├── confidence                  (score + per-factor breakdown + reason)   ← how much to trust it
├── priority_improvements[]     (ranked, evidence-backed actions)         ← what to do, in order
├── risks[] / opportunities[]                                             ← context
├── founder_strategy            (view 1: pre-launch action list)
├── investor_strategy           (view 2: fund / caution / monitor / no)
└── final_strategy              (one-sentence synthesis)
```

The hierarchy mirrors how a good advisor talks: *here's the verdict, here's how confident I am,
here's what to do first, and here's the one-line summary.*

---

## 4. Evidence aggregation — every claim traceable

The non-negotiable: **every recommendation carries its evidence**. A priority action isn't "improve
your campaign" — it's "lower your goal to $5,000 (+12 points), because Games campaigns asking
≤ $5,000 funded at 55% vs 40% — supported by the Day-7 pattern with N campaigns." The aggregator's
job is to keep that provenance attached as signals flow up, so the final decision object is
**auditable end-to-end**. This is what separates a decision-intelligence system from a black box
that emits advice.

---

## 5. Confidence as a first-class output

Most ML systems return a point estimate and stop. A decision system must also say **how much to
trust it** — because a 24%→66% "Fixable" verdict backed by 500 similar campaigns and stable
scenarios deserves more weight than the same numbers backed by 3 outliers and erratic scenarios.
Day 11 elevates confidence to a first-class, *transparent* output: a weighted blend of real
agreement signals (prediction margin, cohort agreement, pattern support, scenario stability,
counterfactual stability, and 3-model agreement), each shown, none hidden. Transparency is the
point — a confidence number you can't decompose is just another black box.

---

## 6. The honesty boundary (what this layer refuses to do)

Aggregation is where fabrication is most tempting — it's easy to make the "brain" sound impressive
by having it recommend "improve customer retention" or "strengthen your competitive moat." **We
don't.** Kickstarter is a single pre-launch snapshot; there is no retention, MRR, moat, team, or
growth trajectory to observe. Layer 11 recommends *only* on the real controllable levers
(goal / duration / launch timing / title) and category-cohort evidence, and it records everything
it refuses to touch in a `deferred` block. The orchestrator inherits, and must preserve, the
honesty discipline of every layer beneath it.

Two structural consequences (from the frozen spec):
- **No faiss / no torch / no LLM / no new model.** Day 11 aggregates the *existing* frame-based
  pipelines (Days 8/9/10), all of which are faiss-free — so "similar startups" means the
  same-category cohort in the shared pandas item frame, and there is zero OpenMP concern.
- **One warm stack.** The whole ensemble shares a single loaded model; founder and investor views
  are two projections of one intelligence object.

---

## 7. AI orchestration — the pattern

Layer 11 is a textbook **orchestration** pattern: a thin coordinator that owns *sequencing,
sharing, and synthesis* while delegating all computation to specialist modules. It computes almost
nothing itself (except confidence and the audience views); its value is entirely in *composition* —
running one warm stack, threading one shared analysis, keeping evidence attached, and reconciling
signals into one ranked decision. This is the same discipline as the Day-10 pipelines, scaled up to
span the whole architecture.

> **Interview soundbite:** *"Layer 11 is the decision-intelligence layer — the point where nine
> independent analytical models become one system. Each lower layer answers a single question well,
> but a founder has one decision, not nine questions, so synthesis has to live above the models:
> only there can a recommendation be weighed by funding impact and historical evidence and investor
> concern and scenario gain at once, and only there can confidence — an ensemble property — be
> computed. I keep every recommendation traceable to its evidence, make confidence a transparent
> weighted blend of real agreement signals, and split founder versus investor advice as two views
> of one intelligence object so they can't drift apart. And it aggregates the existing frame-based
> pipelines, so it stays faiss-free and fabricates nothing — retention, moat, and revenue are
> deferred, not invented."*
