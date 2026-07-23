# Business Insights

Descriptive, decision-oriented findings from the **Kickstarter Projects** dataset.
Each entry pairs an **observation** with its **evidence**, a **confidence** level, and
the **future validation** that would upgrade it from correlation to a modeled,
explainable result. These are hypotheses to test, not conclusions.

---

## Category Intelligence

**Observation:**
Historical success rates differ substantially across categories. The best-performing
category (**Dance, 65.4%**) succeeds at nearly **3×** the rate of the worst
(**Technology, 23.8%**), against an overall decided success rate of **40.4%**.

**Evidence:**
Category-level success rate analysis on the decided subset (successful/failed only,
331,675 campaigns; success = reached goal). Success rate by `main_category`:

| Main Category | Success Rate | Campaigns |
| ------------- | ------------ | --------- |
| Dance | 65.4% | 3,573 |
| Theater | 63.8% | 10,242 |
| Comics | 59.1% | 9,878 |
| Music | 52.7% | 45,949 |
| Art | 44.9% | 25,641 |
| Games | 43.9% | 28,521 |
| Film & Video | 41.8% | 56,527 |
| Design | 41.6% | 25,364 |
| Publishing | 34.7% | 35,445 |
| Photography | 34.1% | 9,689 |
| Fashion | 28.3% | 19,775 |
| Food | 27.6% | 22,054 |
| Crafts | 27.1% | 7,818 |
| Journalism | 24.4% | 4,149 |
| Technology | 23.8% | 27,050 |

Pattern: creative / performance categories (Dance, Theater, Comics, Music) cluster
well above average; capital-intensive or product/hardware categories (Technology, Food,
Fashion) cluster well below. All 15 categories carry thousands of campaigns, so the
differences are not small-sample noise.

**Confidence:**
Medium (descriptive analysis only). This is an unadjusted success rate — it does not
control for goal size, funding duration, timing, or other confounders. High-performing
categories may simply have smaller typical goals.

**Future Validation:**
- **Logistic Regression coefficients** — measure each category's effect on success *after*
  controlling for goal, duration, and other features.
- **SHAP values** — quantify each category's marginal contribution to individual
  predictions (`src/explainability/`).
- **Pattern Mining Engine** — surface category × (goal band / timing) combinations that
  systematically over- or under-perform (`src/pattern_mining/`).

---

## Funding Goal Intelligence

**Observation:**
Success probability **falls monotonically as the funding goal rises**. Campaigns asking
for under $1,000 succeed **53.6%** of the time; those asking for $500K+ succeed just
**3.1%** — a 17× collapse. Successful campaigns have a median goal of **$3,837** vs
**$7,500** for failed ones.

**Evidence:**
Success rate by funding-goal band (decided subset, native currency):

| Goal Band | Success Rate | Campaigns |
| --------- | ------------ | --------- |
| 0–1K | 53.6% | 58,119 |
| 1K–5K | 46.3% | 111,757 |
| 5K–10K | 39.3% | 58,629 |
| 10K–50K | 31.1% | 79,624 |
| 50K–100K | 17.8% | 13,341 |
| 100K–500K | 10.6% | 8,359 |
| 500K+ | 3.1% | 1,846 |

**Confidence:**
Medium–High. The trend is strong, monotonic, and backed by large samples in every band.
`goal` is a legitimate **pre-launch** feature (no leakage), which makes this actionable.

**Future Validation:**

- **Logistic Regression on `log(goal)`** — quantify the per-unit effect on success odds.
- **SHAP dependence plot** — confirm the monotonic goal → success relationship.
- **Recommendation Engine** — suggest a realistic goal band to founders (`src/recommendation/`).

---

## Campaign Duration Intelligence

**Observation:**
**Shorter campaigns win.** Campaigns running 0–15 days succeed **51.3%** of the time,
versus **26.7%** for 45–60 day campaigns and just **22.7%** for 90+ days. The median
campaign runs **29 days**.

**Evidence:**
Success rate by campaign-duration bucket (decided subset):

| Duration (days) | Success Rate |
| --------------- | ------------ |
| 0–15 | 51.3% |
| 15–30 | 40.8% |
| 30–45 | 44.9% |
| 45–60 | 26.7% |
| 60–90 | 38.4% |
| 90+ | 22.7% |

Over-long windows likely signal weak planning or low momentum rather than causing failure.

**Confidence:**
Medium. Duration is set at launch (**no leakage**), but the relationship is confounded
with goal size and category. The 45–60 dip warrants investigation.

**Future Validation:**

- **Logistic Regression** — test duration effect after controlling for goal and category.
- **Pattern Mining Engine** — find duration × goal-band sweet spots (`src/pattern_mining/`).
- **Scenario Simulator** — model "what if the campaign ran 30 days instead of 60".

---

## Campaign Timing Intelligence

**Observation:**
Launch timing carries a **small but consistent** signal. **Tuesday** launches succeed
most (42.5%) and **Saturday** least (37.5%); **spring months** (March 42.7%) beat
**December** (37.1%) and **July** (36.5%). Launch volume peaked in **2015 (77,300
campaigns)** then declined through 2017.

**Evidence:**
Success rate by launch weekday (decided): Tue 42.5% > Mon 40.7% > Wed 40.6% > Sun 40.4%
> Thu 40.2% > Fri 38.8% > Sat 37.5%. By month, spring (Feb–Apr, ~42–43%) outperforms
mid-summer and December (~37%).

**Confidence:**
Low–Medium. Effects are real but small (a few percentage points) and likely confounded
with category and goal mix.

**Future Validation:**

- **Add temporal features** (`launch_month`, `launch_weekday`, `launch_quarter`) to the model.
- **Feature importance / SHAP** — verify timing survives once stronger features are present.
- **Recommendation Engine** — advise optimal launch windows.

---

## Text Intelligence

**Observation:**
Campaign **titles carry weak positive signal**. Successful titles average **36 characters**
(≈6 words) vs **33 characters** (≈5.4 words) for failed. Titles containing a **number**
succeed **50%** of the time vs **39%** without; an **exclamation mark** lifts success to
48% vs 40%.

**Evidence:**
Mean title metrics by outcome and success-rate-by-signal analysis (decided subset).
Success rises with title length up to the 60–80 char band (41.8%) before tailing off.
⚠️ **Data-quality caveat:** ~23,000 titles literally contain "(canceled)" — an outcome
marker that must be stripped before any NLP model to avoid leakage.

**Confidence:**
Low. Signals are weak and descriptive; raw titles need cleaning before they can be trusted.

**Future Validation:**

- **NLP pipeline** — TF-IDF / embeddings on cleaned titles (`src/features/`, Weeks 2–5).
- **Logistic Regression / SHAP** — measure whether text features add lift over tabular ones.
- **Venture Copilot** — surface title-quality suggestions to founders (`src/agents/`).

---

## Success & Failure Archetypes

**Observation:**
Successful and failed campaigns form **distinguishable profiles**. The **successful
archetype**: a lower goal (**median $3,837 USD**), a creative category (**Music**),
US-based, with a slightly longer title (~36 chars). The **failed archetype**: a higher
goal (**$7,500**), a capital-intensive category (**Film & Video**), and a shorter title.

**Evidence:**
Median numeric profile and modal categorical profile by outcome (decided subset, Part F
of `01_EDA.ipynb`). Successful campaigns systematically pair **modest goals** with
**high-performing categories**; failures pair **ambitious goals** with **capital-intensive
categories**.

**Confidence:**
Medium. Archetypes are descriptive composites, not a trained classifier — they combine
several individually-confirmed signals (goal, category, text) into a profile.

**Future Validation:**

- **Clustering** — confirm the archetypes emerge unsupervised (`src/pattern_mining/`).
- **Classification model** — turn the archetype into a probability (`src/models/`).
- **Counterfactual analysis** — "what minimal change flips a failed profile to successful?"
  (`src/counterfactual/`).

---

*All rates computed from `data/raw/ks-projects-201801.csv` via `notebooks/01_EDA.ipynb`
(Parts A–G). Related: [statistical_summary.md](statistical_summary.md),
[eda_observations.md](eda_observations.md), [eda_report.md](eda_report.md),
[data_quality_report.md](data_quality_report.md).*
