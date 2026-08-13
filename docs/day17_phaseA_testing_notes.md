# Day 17 · Phase A — Testing & Validation in Machine Learning (conceptual notes)

*Production Quality Engineering. These notes recap why an AI platform needs testing at every layer, and
ground the concepts in the **real, already-running** test suite before Day 17 extends it.*

---

## 0. Context — what already exists

The platform already has a **48-test pytest suite** (`tests/`, 8 files) with a `conftest.py` that (a)
sets `OMP_NUM_THREADS=1` before any import (faiss + torch collide otherwise) and (b) builds **tiny
hand-made fixtures** — a 10-row dataframe with deliberate edge cases, **not** the 55 MB real dataset. So
Day 17 is *extension*: add the validation the suite is missing (model, retrieval, explainability, e2e,
benchmark) and a CI pipeline — without touching the 48 tests that already pass.

---

## 1. Why ML testing differs from software testing

Traditional software is deterministic: same input → same output, and you assert the output *equals* the
known-correct answer. ML breaks two of those assumptions:

- **The artifact depends on data + randomness, not just code.** A bug can hide in the *data* (a shifted
  category, a null where a number belongs) with the code perfectly correct. So you must test **data**,
  not just functions.
- **The output is probabilistic and often has no known-correct value.** What *is* the "correct" funding
  probability for a campaign? There isn't one. So you can't assert `predict(x) == 0.734`.

That's why ML testing adds layers software testing never needed: **data validation, model validation,
retrieval validation, explainability validation** — each answering "is this behaving sensibly?" rather
than "does it equal the expected constant?"

---

## 2. The test pyramid (and how this repo already uses it)

- **Unit** — many, fast, isolated. The existing tests run on the 10-row fixture, so they're
  millisecond-fast and deterministic (48 tests collect in 0.25 s). Each row exists to exercise one
  behaviour (a missing column, a duplicate, a bad dtype).
- **Integration** — fewer; several modules together (e.g. the preprocessing pipeline end to end).
- **End-to-end** — fewest, slowest; the whole platform on one input (Day-14 `/analyze`).

Broad-and-fast at the bottom, narrow-and-slow at the top. Day 17 fills in the middle and top.

---

## 3. The oracle problem — test *properties*, not exact values

For ML outputs with no known-correct answer, you assert **properties that must hold** regardless of the
exact number. The existing tests already do this — they check `report.passed`, `n_rows == 10`, a check
has status `CRITICAL` — behaviour, not magic constants. Day 17 extends the idea to the models:

- **model:** test PR-AUC **≥ 0.68**, ROC-AUC **≥ 0.77** (thresholds, not `== 0.6883`).
- **retrieval:** category-consistency **lift ≥ 3×**, kNN-AUC **> 0.5** (proxy properties).
- **explainability:** SHAP **additivity** — contributions + base value ≈ the model's output — a property
  that must hold for *any* input.
- **reproducibility:** re-training twice yields **identical** metrics (Day-16 `verify_reproduction`).

Asserting `f1 == 0.6664` would be brittle — a library patch shifts the 4th decimal and the suite goes
red for no real regression. Asserting `f1 ≥ 0.63` (clears the LogReg baseline floor) catches real
breakage without false alarms.

---

## 4. Validating without ground truth (the honesty thread)

Some things have a real label to test against; some don't — and the honest move is to know which.

- **The model has a label:** the campaign's `target` (funded/failed) is ground truth, so accuracy /
  F1 / ROC-AUC are *legitimate* — you can genuinely test whether predictions are right.
- **Retrieval has NO label:** there is no ground-truth "campaign X is similar to Y" set, so
  Precision@K against relevance labels would be **fabrication**. Day 15 already solved this with proxy
  metrics grounded in labels we *do* have (does the neighbour's real outcome predict the query's?). Day
  17 asserts thresholds on those proxies — it does **not** invent a similarity oracle.

Same discipline as every layer: test what's real, and refuse to test against a truth that doesn't exist.

---

## 5. The CI reality — fast lane vs heavy lane

Continuous Integration runs tests on every push. But this platform's **data (`*.csv`), models
(`*.joblib`), and the 407 MB FAISS index are all git-ignored** — they're not in the repo, so a GitHub
Actions runner cannot see them, and rebuilding them (SBERT-encoding 265,171 campaigns) is impractical in
CI. Pretending the full suite runs in CI would be a lie.

The honest design is a **two-lane split via pytest markers**:

- **fast lane** (`pytest -m "not heavy"`) — the fixture-based unit + data-validation tests. No real
  artifacts needed, so it runs in CI on every push, in seconds.
- **heavy lane** (`-m heavy`) — model / retrieval / explainability / API / e2e / benchmark tests that
  load the real data, models, and index. Run **locally**, documented as such.

CI guarantees the fast, artifact-independent logic never regresses; the heavy suite guards the rest
locally. Honest CI is knowing which tests *can* run where.

---

## 6. What Day 17 defers (never fabricated)

- **Neural-Network validation** — no NN was ever trained (torch is SBERT-only); validating one would be
  inventing a model.
- **Precision@K / Recall@K vs similarity labels** — no such labels exist (§4).
- **Live drift detection** — a concept requiring a production data stream we don't have; noted, not built.
- **Full-artifact CI + at-scale load testing** — heavy; marked and local.

---

## 7. Testing — the pattern

Production ML testing is **layered validation of behaviour**: fast unit tests on tiny fixtures at the
base, data/model/retrieval/explainability validation in the middle, one end-to-end test at the top, and
a CI pipeline that runs what it honestly can. The discipline is asserting **properties and thresholds**
(reproducible, clears the baseline, additive, schema-valid) instead of frozen decimals, and being honest
about what has a ground truth to test against and what doesn't.

> **Interview soundbite:** *"ML testing differs from software testing because the artifact depends on
> data and randomness, not just code, and most outputs have no known-correct value — so I test
> properties, not constants. I assert the model clears its baseline and reproduces exactly, that SHAP
> explanations are additive, and that the data schema holds — never that F1 equals 0.6664, which a
> library patch would break for no real reason. I'm also honest about ground truth: the model has real
> labels so its metrics are legitimate, but retrieval has no similarity labels, so I validate proxy
> metrics rather than invent a Precision@K oracle. And my CI is a fast lane only — the data, models, and
> a 407 MB index are git-ignored, so CI runs the fixture-based tests and the artifact-heavy suite runs
> locally. Pretending the full suite runs in CI would be the dishonest option."*
