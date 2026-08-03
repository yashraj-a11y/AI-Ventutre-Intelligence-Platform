# Day 10 · Phase A — Understanding Startup Evolution

> Conceptual foundation for Layers 9 & 10. Grounded in this project's real signals: a Film
> campaign asking $38k reads a 0.035 funding probability (below the ~0.33 bar), and the
> platform can simulate named scenarios that lift it to 0.38 ("READY") — all through the real
> Day-4 model, over features a founder actually controls.

---

## 1. The startup lifecycle — and why our data can't see it

The classic startup lifecycle is a *temporal journey*:

    Idea → Validation → MVP → Early Traction → PMF → Growth → Scaling → Series A → Expansion

Modeling it requires **observing a company over time** — did it build an MVP, acquire users,
reach product-market fit? Our dataset is **Kickstarter campaigns**: a *single pre-launch
snapshot* per campaign, with an outcome (funded/failed). We never see a campaign "acquire its
first 100 users" or "reach PMF." So the lifecycle, and a 12-month growth roadmap through it,
are **unobservable** here — and inventing them (MVP built, retention improved, founder added)
would be fabrication.

**The honest reframe:** we can't model *where a company is in its life*, but we *can* model
**how ready this campaign is to fund, and what pre-launch changes would help.**

---

## 2. Funding-readiness stages (the honest "stage")

Instead of a fabricated lifecycle stage, we classify a **funding-readiness stage** from the
model itself — the calibrated probability, the decision threshold (~0.33), and the **optimized**
probability (the best the Day-8 counterfactual can reach):

| Stage | Condition | Meaning |
|---|---|---|
| **Not ready** | even optimized < bar | controllable tweaks can't clear the threshold |
| **Fixable** | as-is < bar, optimized ≥ bar | below the bar now, but the right changes get there |
| **Ready** | as-is ≥ bar | already above the funding bar |
| **Strong** | well above bar | comfortably fundable |

This is a *real* maturity signal — derived from the model, not asserted. The Film campaign
(0.035 as-is → 0.38 optimized) is **Fixable**: not ready today, but reachable.

---

## 3. Pre-launch preparation, not a 12-month journey

A founder *does* prepare over time — but for a Kickstarter campaign that's **pre-launch
preparation** (days/weeks): set the right goal, choose the campaign length, pick a launch
window, refine the title. Those are the **controllable levers** (Day 8), sequenced as a prep
checklist — honest and actionable. We do **not** fabricate a 12-month post-launch growth
timeline ("acquire users → improve retention → raise pre-seed"), because we have no post-launch
data.

---

## 4. Scenario analysis — the new capability

Day 8 produced **one** greedy-optimal plan. Day 10 generates **multiple named scenarios** —
combinations of controllable changes — simulates each through the real model, and **compares**
them, so a founder can weigh discrete strategies:

    Current (as-is)               0.035
    S1: lower goal                0.080  (+0.045)
    S2: goal + shorter duration   0.375  (+0.340)  READY
    S3: goal + duration + Tuesday 0.384  (+0.350)  READY
    S4: all controllable changes  0.358  (+0.323)  READY

The comparison shows the **marginal value** of each added change — and honestly, that some
don't help (S4 adds a title tweak that *lowers* the probability vs S3). That's decision
support: see which changes actually matter, not just the maximum.

---

## 5. Dynamic prediction & the composition trap

Each scenario is a **fresh model call** on the modified campaign — never a sum of individual
gains (features interact; Day 8's lesson). But there's a subtler trap unique to *multi-variable*
scenarios: **how you compose the changes matters.**

The controllable changes touch overlapping raw fields — both "shorten duration" and "launch on
Tuesday" edit `deadline`. Naively merging Day-8's independently-computed edits **double-writes
`deadline`** and corrupts the result:

- naive merge, {goal + duration + weekday}: **0.375 → 0.059** (a *regression* — adding a change
  made it worse, nonsensically);
- **canonical apply-order** (goal → shift `launched` to Tuesday → set `deadline` = shifted-launched
  + duration → title): **0.384** (monotone, sensible).

So scenarios must be composed with a **dependency-aware apply-order that recomputes derived
fields**, not a dict merge. This is the core technical discipline of the day.

---

## 6. The honesty boundary (every Day-10 output)

> Readiness stages, roadmaps, and scenarios are **model-simulated what-ifs over features a
> founder controls (goal, duration, launch timing, title)** — historical associations, not
> causal guarantees. Startup lifecycle stages, growth milestones (users/retention/MRR), founder
> and financial dimensions, and a 12-month growth timeline are **deferred** — the data can't
> observe them.

---

## 7. Interview soundbite

> *"Day 10 turns the platform into a mentor — but honestly. The startup lifecycle, idea to MVP to
> Series A, requires observing a company over time, and Kickstarter is a single pre-launch
> snapshot, so I can't model it — instead I classify a funding-readiness stage from the model
> itself: not-ready, fixable, ready, or strong, based on the probability and the best the
> counterfactual can reach. The new capability is scenario comparison: Day 8 gave one optimal
> plan, but Day 10 generates multiple named combinations of controllable changes, simulates each
> through the real model, and shows their marginal value — including when a change doesn't help.
> The key technical catch is composition: the controllable changes touch overlapping raw fields —
> duration and launch-weekday both edit the deadline — so a naive merge double-writes it and I
> measured a nonsensical regression, a scenario getting worse when I added a helpful change. The
> fix is a dependency-aware apply-order that recomputes derived fields, which restores monotone,
> sensible results."*
