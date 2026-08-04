# Day 13 · Phase A — Report Generation (conceptual notes)

*Layer 13 foundation — the final layer. No report code; the "why" before the "how".*

---

## 1. The problem: insights exist, but nobody can read them

By the end of Day 12 the platform has produced everything: a funding probability, SHAP drivers,
patterns, counterfactuals, a readiness roadmap, a prioritized recommendation, six agent verdicts, a
committee decision. But those live as ~20 structured JSON outputs across a dozen modules. A founder
or investor doesn't want twenty outputs — they want **one professional report** they can read and
act on.

**Report generation** is the layer that turns analytical outputs into audience-specific,
decision-ready documents. It's the "face" of the platform — the only layer most users actually
touch — and it completes Architecture V1.0.

---

## 2. The defining insight: presentation, not analysis

Here is what makes Layer 13 fundamentally different from Layers 4–12, and it is the single most
important idea in this day:

> **The reporting layer generates ZERO new analysis. It only reformats existing outputs.**

Every number in a report already exists in the Day-11 `VentureDecision` or the Day-12
`CommitteeDecision`. The report doesn't *compute* the funding probability — it *displays* Day 4's.
It doesn't *decide* the recommendation — it *presents* Day 12's. It is a **pure function of upstream
outputs**.

This has a profound consequence for honesty (§3) and a practical one for design: report generators
take no model, run no inference, and are deterministic string assembly. The sophistication is
entirely in *what to include, for whom, and how to phrase it* — not in any new intelligence.

---

## 3. Honesty is inherited — the "no new claims" rule

Because reports only reformat upstream outputs, and every upstream layer is already honesty-tiered
and deferral-aware, **the reports are automatically honest — provided they add nothing.** The one
rule Layer 13 must enforce:

> **NO NEW CLAIMS.** Every line in every report must trace to a real Day 4–12 output.

This is why the draft's glossy example — "experienced founding team, high recurring revenue, weak
competitive moat, expected return" — is *rejected*: none of those correspond to any upstream output,
because the data never had founders, revenue, moats, or returns. A report cannot surface analysis
the platform never performed. Where the draft wants a section we can't back, we **defer** it (a "Not
covered — no data" appendix), never invent it.

We operationalize the rule with **provenance**: every report section carries a `source` tag ("Day-5
SHAP", "Day-12 committee", …). If a section can't name its source, it can't be in the report. This
turns "no new claims" from a good intention into a structural, auditable property — a reader can
trace any claim back to the layer that produced it.

The honest inversion of the draft's example:

| Draft (fabricated) | Frozen (real upstream source) |
|---|---|
| "Experienced founding team" | Strengths = Day-5 SHAP **positive** drivers (real features) |
| "Customer retention" weakness | Weaknesses = Day-5 SHAP **negative** drivers |
| "High recurring revenue" | Funding **goal realism** — Day-12 funding-viability agent |
| "Weak competitive moat" | **Cohort positioning** — Day-12 competition agent |
| "Expected return 3.2×" | **Deferred** — no return data |

---

## 4. Audience-specific communication

The *same* underlying analysis is reshaped for different readers — the Day-11 founder/investor-view
principle, now scaled to whole documents:

- **Founder report** — actionable and encouraging: where you stand, your strengths, your fixable
  weaknesses, and a prioritized to-do list to improve before launch.
- **Investor report** — sceptical and diligence-oriented: the committee's recommendation, the risks
  and red flags, the market and competitive read, and what's *not* known.
- **Investment memo** — a VC-style narrative culminating in an explicit investment thesis.
- **Executive summary** — five one-line numbers for a dashboard or a quick decision.

Crucially, all four are projections of *one* analysis (the shared ReportBundle), so they can't
contradict each other — the founder's "readiness 45" and the investor's "readiness 45" are the same
number, differently framed.

---

## 5. Deterministic templating, not an LLM

The draft's learning objectives mention "prompt engineering" and a "conversational" copilot — which
implies an LLM. We don't use one, for the same reason as Days 9/11/12: an LLM would re-introduce
hallucination (the exact failure this platform is built to avoid) and a new dependency. Instead,
reports are **deterministic templates** filled with real numbers, and the Venture Copilot is a
**deterministic intent-router**: it matches a question to a fixed set of intents and returns a
templated answer assembled from the bundle — with an explicit *"not available in this dataset"*
fallback for out-of-scope questions ("what's my MRR?"). Same campaign in → byte-identical report out,
which is also what makes reports content-hash serializable.

---

## 6. Multi-format export

One report dict → several formats: **JSON** (machine-readable), **Markdown** (human-readable, git-
diffable), **HTML** (styled, print-to-PDF ready). All three are produced with stdlib string
templating — **no new dependency**. PDF is offered *optionally* and *gracefully*: exported only if a
PDF library happens to be installed, otherwise skipped with a note (the HTML is print-ready). The
mandatory path never needs a new dependency.

---

## 7. Report generation — the pattern

Layer 13 is a **presentation/projection** pattern: pure functions from one shared analytical bundle
to audience-specific documents, plus a format-export step. It's the mirror image of Layer 12
(aggregation up); here we *fan out* one decision into many views. It computes nothing new, holds no
model, and hallucinates nothing — its entire value is *communication*, and its entire risk (fabricating
a nice-sounding claim) is controlled by the no-new-claims/provenance rule.

> **Interview soundbite:** *"Layer 13 is the reporting layer — the face of the platform — and the key
> idea is that it generates zero new analysis: every report is a pure, deterministic reformatting of
> the upstream decision and committee outputs for a specific audience. That makes honesty inherited
> rather than re-earned: because every prior layer is already deferral-aware, the reports are honest
> as long as they add nothing, so I enforce one rule — no new claims — with a provenance tag on every
> section, which makes the report auditable back to the layer that produced each number. It's why the
> impressive-sounding 'experienced team, high revenue, strong moat' report is rejected and deferred:
> the platform never had that data, so the report can't surface it. Reports are deterministic
> templates, not an LLM, so nothing is hallucinated and the same campaign always produces the same
> document — which is also what lets me content-hash them."*
