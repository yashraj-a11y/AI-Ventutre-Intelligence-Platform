# Day 8 · Phase A — Understanding Counterfactual AI

> Conceptual foundation for Layer 7. Grounded in this project's real Day-4 calibrated model
> and real controllable features. Every what-if below is simulated by perturbing a **raw**
> campaign and re-running the actual model — verified, not hypothetical.

---

## 1. Prediction vs counterfactual reasoning

The whole platform so far answers *"what will happen?"* Day 8 answers *"what should I change?"*

| | Prediction (Day 4) | Counterfactual (Day 8) |
|---|---|---|
| Question | will this campaign fund? | what change would raise its odds? |
| Output | one probability (e.g. 0.23) | ranked, actionable improvements |
| Direction | descriptive | **prescriptive / decision intelligence** |
| Example | "funding probability 23%" | "lower your goal to ~$14k → the model estimates 48%" |

A prediction is a verdict; a counterfactual is *advice*. Turning the first into the second is
what separates a venture-intelligence platform from a classifier.

---

## 2. What a counterfactual actually is here

A counterfactual is a **modified version of the input** and the model's response to it:

> *"If this campaign had asked for $14k instead of $57k (everything else equal), the model's
> estimate would be 0.048 instead of 0.023."*

We produce it by **perturbing the raw campaign** (`goal`, `deadline`, `launched`, `name`) and
re-running the Day-4 `PredictionEngine`, which recomputes every engineered feature through the
Day-3 pipeline. Perturb the *inputs*, not the engineered columns — so coupled features
(`usd_goal_real`, `goal_per_day`, duration, timing, title-stats) stay consistent automatically.

---

## 3. Only controllable features count

A recommendation is useless — or dishonest — if the founder can't act on it.

- **Mutable (we recommend):** `goal`, `campaign_duration`, launch timing, title wording. A
  founder controls all of these before launch.
- **Immutable (never recommend):** `country`, `currency`. "Relocate to the US to raise your
  odds" is gaming a model, not advice. The engine is *structurally* barred from suggesting these.
- **Soft:** `category` — surfaced only as an optional repositioning note, never a trick.
- **Deferred (don't exist):** revenue, retention, burn, team, founder experience, market size,
  traction, CAC/LTV, funding stage. The draft's "add a technical co-founder" is fiction — there
  is no such data, so the engine never claims it.

This is the honesty boundary of the whole day: **simulate only what a founder can truly change.**

---

## 4. Feature perturbation & sensitivity analysis

Sensitivity analysis asks: *how much does the output move when one input moves?* We sweep each
mutable feature over feasible values and record Δp. Real, measured (a $15k Film campaign, base
p = 0.802):

- lower goal to 30% → **Δ +0.040**
- 30-day duration → **Δ −0.011** (not every change helps — honesty means reporting that too)

The features with the largest |Δp| are the **levers**; the goal is consistently the biggest,
matching what SHAP (Day 5), retrieval (Day 6), and pattern mining (Day 7) all found.

---

## 5. Cumulative effects must be *re-simulated*, not summed

The tempting cascade — "58% → 69% → 82% → 89%" — implies you can **add** individual gains.
You can't. Features interact, and gains saturate near the probability ceiling. Measured:

```
Δ(lower goal)        = +0.040
Δ(30-day duration)   = −0.011
naive sum            = +0.029   → implied p 0.831   ← WRONG
re-simulated together =  +0.005  →           p 0.806   ← honest
interaction gap                   = 0.024 overstatement
```

So the honest cumulative probability is obtained by running the model **once on the jointly-modified
campaign** — never by adding deltas. Every step of an improvement plan is a fresh model call on
the cumulative campaign.

---

## 6. Model-simulated ≠ causal guarantee (the central honesty point)

A counterfactual shows how the **model's estimate** responds — the model trained on 265k
historical campaigns. It is **decision support, not a promise.** Two campaigns with identical
features can have different real outcomes; the model gives the historical *average*.

Evidence-strength tiers make this explicit:

- **Strong** — `goal`: model responds **and** a historical pattern corroborates (small-goal
  campaigns funded ~1.8× more) **and** there's a plausible *mechanism* (a lower all-or-nothing
  target is mechanically easier to hit).
- **Suggestive** — timing / title: the model responds and a weak pattern exists, but the model
  may be leaning on confounds; framed as "may help," not "will."

Every recommendation carries the caption: *the model estimates this change would…, based on
historical association, not a guarantee.*

---

## 7. Interview soundbite

> *"Day 8 turns predictions into decisions. A counterfactual is a modified campaign and the
> model's response — I perturb the raw inputs a founder can actually control, goal, duration,
> launch timing, title, and re-run the real Day-4 calibrated model so every coupled feature
> recomputes consistently. I only ever recommend mutable features — never 'move to the US,'
> which is gaming, not advice — and I defer everything the data doesn't have, like revenue or
> founder experience. The subtle rigor is cumulative effects: you can't add individual gains,
> because features interact and probabilities saturate — I measured a naive sum overstating the
> true joint effect by two-and-a-half points, so I re-simulate every plan jointly. And I frame it
> as model-simulated, not causal: the goal lever is strongest because the model, a historical
> pattern, and a real mechanism all agree, while a title tweak is only suggestive."*
