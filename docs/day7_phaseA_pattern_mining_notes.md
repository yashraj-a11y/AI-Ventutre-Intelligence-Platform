# Day 7 · Phase A — Understanding Pattern Mining

> Conceptual foundation for Layer 6. Grounded in this project's real train knowledge base:
> **265,171 campaigns**, base funding rate **0.4038**. Every pattern below is a measured
> association over **real** features (category, goal band, country, timing) — never founder /
> market / revenue signals (which don't exist).

---

## 1. Prediction vs Retrieval vs Pattern Mining

Three different questions, three layers:

| | Question | Layer | Output |
|---|---|---|---|
| **Prediction** (Day 4) | "Will *this* campaign fund?" | 4 | one calibrated probability |
| **Retrieval** (Day 6) | "Which past campaigns are *like* this one?" | 2–3 | K similar campaigns + outcomes |
| **Pattern Mining** (Day 7) | "What *repeatedly* separates funded from failed, across *all* campaigns?" | 6 | ranked, reusable patterns |

Prediction is a point estimate; retrieval is a local lookup; **pattern mining is global,
descriptive knowledge** — the recurring regularities that hold across the whole population.
It answers "why do campaigns like mine tend to succeed/fail?" with evidence, not a score.

---

## 2. Frequent itemsets & association rules

An **itemset** is a set of feature-values that co-occur, e.g. `{main_category=Dance,
goal_band=small}`. An **association rule** is `{antecedent items} → {consequent}`, here
`{Dance, small goal} → funded`. Pattern mining searches for itemsets that appear often and
whose rules toward the outcome are strong.

The three classical measures, defined on the real rule `{Dance, small goal} → funded`:

- **Support** — how often the itemset occurs: `P(Dance ∧ small goal) = 0.67%` (n=1,770).
  *Is the pattern common enough to matter?*
- **Confidence** — how often the rule holds when the antecedent does:
  `P(funded | Dance ∧ small goal) = 0.711`. *When these features appear, how often funded?*
- **Lift** — confidence relative to the base rate: `0.711 / 0.4038 = 1.76`.
  *Does the pattern change the odds, or just restate the average?*

---

## 3. Why lift needs a base rate

Confidence alone lies. A rule with 40% confidence sounds meaningful — until you remember the
**base funding rate is already 40.38%**. That rule tells you *nothing*: it's the average.

$$\text{lift} = \frac{\text{confidence}}{\text{base rate}} = \frac{P(\text{funded}\mid\text{items})}{P(\text{funded})}$$

- **lift > 1** → the pattern makes funding *more* likely (success-leaning).
- **lift = 1** → the pattern is uninformative (equals the base rate).
- **lift < 1** → the pattern makes funding *less* likely (failure-leaning).

Real, measured (train):

| Pattern | Confidence | Lift | Reading |
|---|---:|---:|---|
| `main_category = Dance` | 0.648 | **1.60** | strong success signal |
| `main_category = Technology` | 0.237 | **0.59** | strong failure signal |
| `goal_band = very small` | 0.551 | 1.36 | success-leaning |
| `goal_band = very large` | 0.099 | **0.25** | severe failure signal |
| `{Dance, small goal} → funded` | 0.711 | **1.76** | strongest combo |

Lift is what makes a pattern *informative* rather than a restatement of the average.

---

## 4. The multiple-testing trap (why rigor is mandatory)

With 15 main-categories, 159 categories, 22 countries, 5 goal bands, and timing features,
there are **thousands** of possible rules. Mine enough of them and some will look spectacular
**purely by chance** — especially on small samples.

Measured proof, on real categories:

- **Chiptune**: confidence 0.81, **lift 2.02** — but **n = 27**. Exciting, and almost certainly
  noise.
- **Dance**: confidence 0.65, lift 1.60, **n = 2,866**. Less flashy, far more trustworthy.

The small-sample categories (n < 40) swing wildly from lift 0.83 to 2.02; the large-support
ones (n ≥ 500) sit in a stable band. So a discovered pattern must clear **three** bars, not one:

1. **Support** — enough campaigns to be real (we require ≥ ~500).
2. **Effect size** — lift far enough from 1 (|lift − 1| ≥ 0.15).
3. **Significance** — a chi-square / Fisher test the deviation isn't chance, with a
   **Bonferroni-adjusted** threshold that accounts for how many rules we tested.

Skipping this turns pattern mining into a machine for confident nonsense.

---

## 5. Correlation vs causation (the honesty boundary)

Every pattern here is a **historical association**, never a proven cause. "Dance campaigns
funded at 65%" means *in this data, Dance campaigns tended to succeed* — driven by confounds we
can't separate (Dance projects ask for small goals, have tight communities, low production
cost). It does **not** mean "relabel your project as Dance to get funded."

So Day-7 explanations (Phase G) always say *associated with*, never *causes*, and carry a
mandatory caption. This is the same discipline as the Day-5 SHAP captions and Day-6 evidence
captions — pattern mining is the layer where the temptation to over-claim causation is
strongest, so the guard is strictest.

---

## 6. What we mine (real) vs defer (fabricated)

- **Mine** (real, discretized items): category / main_category, country, currency, goal band,
  launch year / month / quarter / weekday / weekend, campaign-duration band, title-length /
  word-count bands, has-number.
- **Defer** (no data — never mined): founders, TAM/market size, traction, recurring revenue,
  retention, CAC/LTV, burn rate, PMF, funding rounds/stage, team, business model, and **growth
  trajectories** (Kickstarter records one outcome, not a time series — so "growth pattern
  mining" is reframed to *temporal/timing* patterns).

---

## 7. Interview soundbite

> *"Pattern mining is the global, descriptive layer — not 'will this campaign fund' but 'what
> repeatedly separates funded from failed across the whole population.' I mine association rules
> and rank them by support, confidence, and lift, where lift is confidence over the base rate —
> that's the crucial one, because with a 40% base funding rate a 40%-confidence rule is pure
> noise. Real example: Dance campaigns with small goals funded at 71% versus 40% baseline, a lift
> of 1.76. The rigor that matters is guarding against multiple testing: mining thousands of rules
> guarantees some look great by chance, so I require minimum support, a real effect size, and a
> significance test with a Bonferroni adjustment — a Chiptune category showed lift 2.02 but on 27
> campaigns, which I'd reject, versus Dance at lift 1.60 on nearly 3,000. And every pattern is
> framed as a historical association, not a cause."*
