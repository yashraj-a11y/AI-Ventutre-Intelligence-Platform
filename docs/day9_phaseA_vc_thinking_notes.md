# Day 9 · Phase A — Understanding Venture Capital Thinking

> Conceptual foundation for Layer 8. Grounded in this project's real signals: for a $30k
> Technology campaign the platform reads prob 0.224 (below the ~0.33 threshold), SHAP flags
> the high goal + category headwind, and the counterfactual gives controllable levers. Day 9
> **reframes** those real signals into VC-style reasoning — it invents nothing.

---

## 1. How a data-driven VC evaluates a startup

A VC partner, stripped of intuition and reduced to what's *knowable from data*, asks a
sequence of hard questions: *What are the odds this succeeds? What's working against it? What
would I push back on in the meeting? What should the founder fix first?* Day 9's job is to
answer those **before** the founder walks into the room — using only signals we can defend.

The classic VC dimensions and our honest coverage:

| VC dimension | Can we assess it? | Real signal |
|---|---|---|
| Funding odds | ✅ | Day-4 calibrated probability vs threshold |
| "Goal / ask appropriateness" | ✅ | SHAP on `usd_goal_real` + Day-7 category goal norms |
| Category / segment headwind | ✅ | Day-7 category funding rate + lift; year-trend |
| Timing | ✅ | Day-7 duration / launch-timing patterns |
| Cohort comparison | ✅ | similar campaigns' funding rate (Day-7 substrate) |
| Founder / team quality | ❌ **defer** | no founder data |
| Market size (TAM/SAM/SOM/CAGR) | ❌ **defer** | no external market data |
| Named competitors / moat | ❌ **defer** | no competitor data |
| Unit economics / burn / retention | ❌ **defer** | no financial data |

**The honesty boundary of the whole day:** we assess the five we *can*, and defer the four we
can't — never inventing a founder score or a market number.

---

## 2. Day 9 computes nothing new — it reframes

Every prior layer already produced the raw material; Day 9 is a **translation layer** from ML
output to investor language:

- **Prediction** (Day 4) → *"the odds"* — "the model puts funding at 22%, below the ~33%
  bar."
- **SHAP** (Day 5) → *"what's dragging it down"* → **weaknesses / objections** (the high goal,
  the low-success category).
- **Patterns** (Day 7) → *"the evidence"* → risk/objection backing ("Technology funds at 24% vs
  40% baseline, lift 0.59").
- **Counterfactuals** (Day 8) → *"the fix"* → **suggested responses / recommendations** ("lower
  the goal to the category norm").

An objection, a weakness, a risk, and a fixable lever are **the same real signal** viewed four
ways — which is what makes the copilot coherent rather than a pile of disconnected lists.

---

## 3. Objections, risk, and SWOT are one reframing

- **Investor objection** — a weakness *phrased as a question a VC would ask*: "Your $30k goal
  is high for Technology, where campaigns like yours fund at ~10%. Why this ask?"
- **Risk** — the same weakness *quantified*: "Category-fit risk: HIGH — lift 0.59, n=21k."
- **SWOT weakness** — the same signal *as a bullet*.
- **Recommendation** — the same signal *as an action*: "lower your goal to the category norm."

So the copilot's outputs cross-reference: a top objection is a top risk is a SWOT weakness is a
counterfactual lever. Coherence by construction.

---

## 4. Why deterministic templates, not an LLM

The draft imagines an LLM "reasoning over" the outputs. We deliberately **don't** use one:

- **Reproducibility** — the same campaign always yields the same brief; an LLM is stochastic.
- **Honesty** — a template map *cannot* hallucinate a competitor ("Microsoft") or a market
  number ("CAGR 23%"); it can only emit sentences about signals that exist. That is the entire
  point of the day's discipline.
- **Cost / offline** — no API key, no per-call cost, runs anywhere.

The "VC-like reasoning" comes from the *mapping* (real signal → investor framing), tiered by
honesty — not from a language model's fluency. An LLM layer is optional future work; it would
add polish at the cost of determinism and the risk of invention.

---

## 5. Severity from magnitude, not vibes

Every objection/risk gets a **severity grounded in a real magnitude**, never a hand-set label:

- probability gap below threshold → higher severity;
- larger SHAP contribution (e.g. the goal's −0.23) → higher severity;
- pattern lift further from 1 (Technology 0.59) → higher severity.

So "HIGH severity" means "the number is large," and it's auditable back to the source layer.

---

## 6. The honesty caption (every Day-9 output)

> These are **model predictions and historical associations reframed as investor-style
> reasoning** — not a real investor's judgement, and not verified market or competitor facts.
> Founder, market-size, named-competition, and financial assessments are **deferred** (the data
> doesn't support them).

---

## 7. Interview soundbite

> *"Day 9 makes the platform reason like a data-driven VC — but it computes nothing new, it
> reframes. The prediction becomes 'the odds,' the SHAP negatives become the objections and
> risks, the patterns become the evidence, and the counterfactuals become the suggested
> responses — so an objection, a weakness, a risk, and a fixable lever are literally the same
> real signal viewed four ways, which makes the brief coherent. I deliberately don't use an LLM:
> a template map over the real outputs is reproducible and, crucially, can't hallucinate a
> competitor like Microsoft or a market CAGR — it can only speak about signals that exist. And I
> only assess the dimensions the data supports — funding odds, goal appropriateness, category and
> timing headwinds, cohort comparison — while explicitly deferring founder, market-size, named
> competition, and financials, with severity always grounded in a real magnitude, not a label."*
