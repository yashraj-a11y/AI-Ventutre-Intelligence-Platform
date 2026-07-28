# Day 5 · Phase A — Understanding Explainable AI

> Conceptual foundation for Layer 5. Grounded in this project's real Day-4 model:
> a tuned + calibrated **LightGBM** predicting P(funding success) from 865 pre-launch
> features (800 title TF-IDF tokens + 65 tabular). Test **PR-AUC 0.6883**, **recall 0.813
> @ threshold 0.329**. No-Information Rate to beat: **59.62%** accuracy.

---

## 1. The problem Phase A solves

Day 4 shipped a model that outputs a number: *"this campaign has a 72% chance of
funding."* That number is **actionable only if it is trustworthy**, and it is trustworthy
only if we can answer the follow-up question every stakeholder asks:

> **"Why?"**

- A **founder** asks: *what should I change to improve my odds?* A bare probability tells
  them nothing to act on.
- An **investor** asks: *what evidence backs this score?* "Trust the black box" is not
  diligence.
- A **regulator / auditor** asks: *is the model using a legitimate signal, or a proxy for
  something it shouldn't (leakage, bias)?*
- **We (the builders)** ask: *did the model learn real structure, or an artifact?*

Explainability is the layer that turns a score into **evidence + a lever**.

---

## 2. Black-box vs white-box models

| | White-box | Black-box |
|---|---|---|
| **Definition** | Structure is directly readable | Structure is opaque; needs post-hoc tools |
| **Our example** | `logistic_regression.joblib` | `best_model.joblib` (LightGBM, 300 trees) |
| **How to read it** | Inspect the 865 coefficients directly | Attribute predictions with SHAP |
| **Accuracy here** | val PR-AUC **0.626** | test PR-AUC **0.688** |
| **Interpretability** | Native (a coefficient *is* the explanation) | Requires an approximation layer |

Concretely in this project: the logistic model *is* a weighted sum, so reading it is trivial
— `category_freq` carries coefficient **+5.96**, the single dominant driver, meaning
"common category → higher funding odds." But that readable model is **6 PR-AUC points weaker**
than the LightGBM. The LightGBM captures interactions and non-linearities the linear model
can't — and pays for it by being unreadable. That gap *is* the interpretability/accuracy
trade-off, made numeric.

---

## 3. The interpretability ↔ accuracy trade-off

```
accuracy
  ^
  |                          ● LightGBM (0.688)  ← deployed, needs SHAP
  |                 ● RandomForest (0.664)
  |        ● LogReg (0.626)  ← natively readable
  |   ● DummyClassifier (floor)
  +--------------------------------------------------> interpretability
      (opaque)                                (transparent)
```

- More flexible models fit more structure → **higher accuracy**, **lower native
  interpretability**.
- The resolution is **not** to pick a side. It is: **deploy the accurate model, then bolt on
  a post-hoc explainability layer** (SHAP) that recovers per-prediction reasons *without*
  sacrificing the accuracy. That is exactly the Day-5 architecture — keep the LightGBM,
  explain it with SHAP, and *cross-check* those explanations against the white-box LogReg
  coefficients. Agreement between the two = stronger evidence.

---

## 4. Global vs local explanations

Two fundamentally different questions, two different tools:

| | **Global** | **Local** |
|---|---|---|
| **Question** | "What drives funding *in general*?" | "Why *this* campaign?" |
| **Scope** | Whole model, all 865 features | One row |
| **Audience** | Product/strategy, model validation | The individual founder |
| **Tool (Day 5)** | `feature_importances_` (gain), mean\|SHAP\|, LogReg coefficients | SHAP values for one instance |
| **Phase** | B, C, F | E, F (waterfall) |
| **Example output** | "category commonness and goal size dominate across all campaigns" | "*Your* goal is high for your category (−0.4), but you launched in a strong month (+0.2)" |

Key intuition: **global = the average story; local = one campaign's story.** They can disagree
— a feature that barely matters globally can be decisive for one specific campaign. A founder
needs the local story; model validation needs the global one.

---

## 5. Why VCs / founders need *evidence*, not scores

A venture-intelligence product that only emits "72%" is a **magic-8-ball**. The value is in
the explanation because:

1. **It's a decision aid, not an oracle.** A founder can't change a probability, but they
   *can* change a goal amount, a launch month, or a title word. Explanations point at the
   **levers**.
2. **It exposes the model's honesty.** SHAP will show us if the model leans on something
   dumb (e.g. a single TF-IDF token) — a check we *want* before trusting it.
3. **It sets the honesty boundary of the whole product.** Our explanations may cite
   **only real features** — goal, category, country, timing, title tokens. The dataset has
   **no** founder/team/market/revenue columns, so any "experienced founder" or "strong
   market" narrative would be **fabricated**. Phase A locks this rule in: *explain what the
   model actually saw (campaign metadata + title text), never invent drivers.*

---

## 6. The honesty caption (applies to every Day-5 explanation)

> Explanations describe **associations the model learned from campaign metadata and title
> text** — not ground-truth causal drivers of success. "Feature X pushed the probability up"
> means "in the training data, campaigns like this tended to succeed," **not** "changing X
> will cause funding."

This distinction (correlational attribution vs causal claim) is the single most important
thing to say out loud in an interview about explainability — it's what separates a careful
practitioner from someone overselling SHAP.

---

## 7. Interview soundbite

> *"We deployed the accurate model, not the readable one — LightGBM at 0.688 PR-AUC over
> logistic regression at 0.626 — and recovered interpretability post-hoc with SHAP instead of
> trading accuracy for it. I explain globally with gain-based and mean-|SHAP| importance, and
> locally with per-instance SHAP values, and I cross-check both against the white-box logistic
> coefficients so agreement between an independent linear model and the tree attributions is my
> evidence the story is real. Crucially, every explanation cites only features the model
> actually saw — campaign metadata and title text — and is framed as a learned association, not
> a causal driver."*
