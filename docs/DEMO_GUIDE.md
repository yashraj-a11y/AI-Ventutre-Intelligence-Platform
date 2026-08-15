# Demo Guide

A script for demonstrating the platform end to end: what to start, what to click, what to say, what
will be on screen, and which questions to expect. Numbers quoted here were **captured from the real
system** with the committed model artifacts (2026-11-03); they are deterministic for this model version.

**Honest scope.** The UI steps were checked by driving the real app with headless Chrome (full Founder
and Investor flows, mobile width, no browser errors) and reviewing the resulting screenshots — but not
by a person clicking through it. Rehearse once and correct any wording that differs. Screenshots are in
[`docs/assets/screenshots/`](assets/screenshots/); the screen recording and one terminal shot are still
yours ([§9](#9-screenshot-and-recording-shot-list)).

| Version | Length | Use |
|---|---|---|
| Lightning | 90 s | recruiter screen, LinkedIn video |
| **Core demo** | **5 min** | interview, portfolio walkthrough (§5) |
| Extended | 10–12 min | adds API, patterns, architecture, tests (§6) |

---

## 1. What the demo proves

A founder submits a campaign and gets a calibrated probability, the reasons, similar past campaigns, what
to change, and a report; an investor gets a committee verdict and due-diligence report — all from real
data, with the platform openly refusing to invent what its data cannot support.

## 2. Say this first (30 seconds)

1. "The data is ~330,000 Kickstarter campaigns from **2009 to 2017**. It has title, category, goal, dates —
   no founder, team, revenue or market data."
2. "So the model is **honest but modest**: ROC-AUC 0.78. Pre-launch fields only partly decide an outcome."
3. "Factors are **associations the model learned, not causes**."
4. "Anything the data can't support is shown as **deferred** — never invented."

Opening with the limits makes every later number more credible.

## 3. Pre-flight checklist

```bash
# Terminal 1 — backend
cd AI-Venture-Intelligence-Platform
OMP_NUM_THREADS=1 CORS_ORIGINS=http://localhost:5173 .venv/bin/uvicorn src.api.main:app --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

- [ ] `curl localhost:8000/ready` returns `{"status":"ready"}`.
- [ ] **Warm up retrieval** — the first `/retrieve` takes ~8–12 s: submit the demo campaign once (or `curl -X POST 'localhost:8000/retrieve?k=3' -H 'Content-Type: application/json' -d @docs/assets/campaign_A.json`) before the audience arrives.
- [ ] Browser at `http://localhost:5173`, window at least 1024 px wide (below that the form stacks above the results), zoom 100%.
- [ ] Close other heavy apps; the stack uses ~1.1 GB.
- [ ] Have `docs/assets/demo_campaigns.json` open to copy values from (`campaign_A.json`, `campaign_B.json`, `campaign_D.json` hold the request bodies).

## 4. The demo campaigns

All three are **fictional** (no match in the dataset, so retrieval cannot show "itself") and dated **inside
the 2009–2017 training range**, where the model's year signal is valid.

| | Campaign | Goal | Dates | Probability | Model label | Committee verdict (confidence) |
|---|---|--:|---|--:|---|---|
| **A** (main) | *Lumen: solar smart garden planter* — Technology / Gadgets | $25,000 | 2017-03-06 → 2017-04-05 | **40.2%** | successful | **Do Not Invest** (61%) |
| B (strong) | *Hollow Kingdom: a cooperative dungeon card game* — Games / Tabletop Games | $8,000 | 2016-09-06 → 2016-10-06 | **82.1%** | successful | **Strong Invest** (78%) |
| D (weak) | *Atlas One: modular home robot* — Technology / Hardware | $120,000 | 2017-01-10 → 2017-03-11 | **7.5%** | failed | **Do Not Invest** (86%) |

**Campaign A is the main demo** because it sits near the 0.329 threshold: there is room to improve, so
the counterfactuals and scenarios have something to show. Use B and D as a quick contrast ("same form,
different outcome") in the Investor portal.

> **Why not the form's default ("Artificer")?** It is a real, failed 2015 campaign in the dataset, so
> retrieval would show its near-twin as a "similar campaign" — misleading for a demo.

### Enter campaign A in the form

| Field | Value |
|---|---|
| Campaign title | `Lumen: solar smart garden planter` |
| Main category | Technology |
| Sub-category | `Gadgets` |
| Funding goal (USD) | `25000` |
| Currency / Country | USD / US |
| Launch | `2017-03-06 10:00` |
| Deadline | `2017-04-05 10:00` |

---

## 5. Core demo — 5 minutes (Founder, then Investor)

### Step 1 · Home and sign-in (20 s)

- **Do:** open the landing page → *Founder Portal* (or *Sign in* → Founder).
- **Say:** "One platform, two roles. The login is an auth-ready stub — real structure, no passwords yet."
- **Don't claim:** real authentication.

### Step 2 · Submit the campaign (20 s)

- **Do:** fill the form with campaign A; submit.
- **Say:** "The form offers *only* the fields the model was trained on. There's no field for founder
  experience or revenue — the API would reject them with a 422, so the form doesn't ask."
- **Screen:** cards load progressively with spinners.

### Step 3 · Prediction (30 s) — card "Prediction"

- **Screen:** probability **40%** in **amber**, a bar with a threshold marker at 33%, tiles *Decision
  threshold: Above* (cut-off 33%), *Startup score 40*, *Failure risk 60%*, *Investment score 40*, and a
  caption: *"Above the decision threshold (33%), which is tuned to catch funded campaigns, but under 50% —
  more likely to fail than to fund."*
- **Say:** "40% is a *calibrated* probability. The marker is the decision threshold, 0.329, which I tuned
  for recall — so 40% sits just above it, and the colour says so: amber means 'above the cut-off but under
  50%'. Failure risk and the scores are simple functions of that one probability."
- **Colours:** red = below the threshold (campaign D, 7%); amber = between threshold and 50% (A, 40%);
  green = above 50% (B, 82%).

### Step 4 · Why — SHAP (45 s) — card "Why — SHAP feature contributions"

- **Screen:** a bar chart (green up / red down) and, below it, a **waterfall** from the baseline to this
  campaign's score. Biggest lifts: `campaign_duration` (+0.69), the title words *smart* and *garden*, and
  `launch_weekday`. Biggest drags: `uppercase_ratio` (−0.52), `usd_goal_real` (−0.24), `goal_per_day`, and
  the word *solar*.
- **Say:** "These are SHAP values in log-odds. The waterfall starts at the average campaign (−0.70) and
  ends at this one (−0.35); the 'all other features' bar is everything not in the top list, so the chart
  always reconciles. These are associations in 2009–2017 data, not causes — `uppercase_ratio` is a good
  example of a model quirk, not advice."

### Step 5 · Scenario simulator (45 s) — card "Scenario simulator"

- **Do:** drag the **goal** slider from $25,000 down to ~$10,000; watch the probability update live.
- **Screen:** the live prediction moves; below it, *Suggested scenarios* (best: **Lower goal → about 43–44%**) and
  *Counterfactuals*: **add a number to the title +9.1 pts (→ 49%)**, **launch on a Tuesday +4.3 pts**,
  **lower the goal to $10,000 +3.3 pts**.
- **Say:** "Only real levers — goal, duration, launch day, title. Revenue, retention and team size aren't
  sliders because the data has none. And it's the *model's* answer: a counterfactual says what the model
  would predict, not what would happen."
- **Nuance to mention:** combining all changes does **not** add up — "lower goal + shorten duration +
  Tuesday + title number" scores **31%**, *below* lowering the goal alone (about 43.5%), because a 30-day duration
  is itself a positive for this campaign. Levers interact.

### Step 6 · Historical evidence (30 s) — card "Historical evidence — similar campaigns"

- **Screen:** two tables — *similar campaigns that funded* (e.g. "Smart Solar Garden: Customize Light for
  Any…", 2017, similarity 0.70) and *that failed* (e.g. "Smart Technology Planter", 2014, 0.65) — with
  category, goal and similarity.
- **Say:** "Found by sentence embeddings and an exact FAISS search over ~265,000 campaigns. It's evidence
  for the founder, *not* an input to the prediction — retrieval alone predicts outcomes only weakly
  (ROC-AUC 0.69–0.71)."

### Step 7 · The honesty boundary (20 s) — card "Scores the platform defers"

- **Screen:** six dashed grey cards — Business-Model, Founder Strength, Scalability, Market Fit, Growth
  Potential, Market Size (TAM / SAM) — each marked *deferred*. A note beneath says a category-level opportunity
  signal from Kickstarter history *is* computed (in the report's Market Analysis).
- **Say:** "A dashboard that showed 'Founder Strength: 78' would be lying with a nicer font. The data has
  no basis for it, so the backend returns nothing and the UI shows the gap." *(This is the strongest
  moment in the demo — slow down here.)*

### Step 8 · Recommendations and report (30 s) — cards "Recommendations & roadmap", "Founder Intelligence Report"

- **Screen:** *Readiness stage* **Ready**, *Readiness index* **47**, *Optimized prob.* **53%** (from 40%),
  *Confidence* **69%**; priority improvements: **1** lower the goal to $10,000, **2** launch on Tuesday,
  **3** add a number to the title. Below, an 8-section report (Executive Summary, Scorecard, Strengths,
  Weaknesses, Priority Improvements, Evolution Roadmap, Scenario Analysis, Next Milestones), each with a
  grey **source badge** naming the layer that produced it.
- **Do:** click **Print / Save as PDF** to show the export.
- **Say:** "Every section says where it came from. The PDF is the browser's print dialog — there's no
  server-side PDF."

### Step 9 · Switch to the Investor portal (60 s)

- **Do:** *Investor* in the header → the form **resets to the default campaign**, so re-enter campaign A and submit.
- **Screen:** card *Committee verdict*: **Do Not Invest**, *Overall confidence* **61%**; *Comparable
  campaigns* (same tables); *Investor Due-Diligence Report* with 10 sections.
- **Do:** scroll to **Risks & Red Flags**: "Overall risk: High. Modeled failure risk 60%", and four
  investor objections, e.g. *"Your $25,000 goal is high for Technology, where funded campaigns typically
  ask a median of $10,000. Why this ask?"* and *"Among campaigns most like yours… only 22% funded. What
  sets you apart from the 7,940 that came before?"*
- **Do (if time):** scroll to **Investment Decision**: the six agents' votes — Market, Competition, Risk
  and Investment **negative**; Founder and Financial **positive**; committee score **−0.51**, agreement 67%.
- **Say:** "Six rule-based agents vote, weighted by confidence, with a veto for the risk agent. It's a
  deterministic consensus over real signals — no LLM. Objections are rules over the same data: a high
  goal, a weak category, a crowded segment."
- **Contrast (20 s):** resubmit **B** → *Strong Invest* (78%); **D** → *Do Not Invest* (86%).

---

## 6. Extended segments (10–12 min total)

### 6a · The API (2 min)

```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @docs/assets/campaign_A.json | python3 -m json.tool

# Honesty enforced at the boundary — an invented field is rejected:
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"name":"x","main_category":"Technology","usd_goal_real":5000,"launched":"2017-03-06 10:00:00","deadline":"2017-04-05 10:00:00","founder_experience_years":7}'
# -> 422 "Extra inputs are not permitted"
```

Open `http://localhost:8000/docs` (Swagger) and expand `/analyze`.

### 6b · The copilot's honest refusal (1 min)

`/analyze` returns a `copilot` block with eight pre-built Q&A turns — *"Why is my funding probability
low?"*, *"What should I improve first?"*, *"Which campaigns are most similar to mine?"*, *"Should an
investor invest?"*, and one deliberate trap, **"What is my MRR?"**, which it declines: *"That isn't
available — this platform has no mrr data for Kickstarter campaigns (they're anonymous, pre-launch)."*

```bash
curl -s -X POST localhost:8000/analyze -H 'Content-Type: application/json' -d @docs/assets/campaign_A.json \
  | python3 -c "import sys,json; [print(t['question'],'\n ->',t['answer'][:160],'\n') for t in json.load(sys.stdin)['copilot']['turns']]"
```

The copilot is **not rendered in the UI** and takes no free-text input — it is a fixed set of turns.

### 6c · Pattern mining (1 min) — notebook, not UI

There is no live pattern panel (no `/patterns` endpoint). Open `notebooks/07_Pattern_Mining.ipynb` or
`artifacts/patterns/ranked_patterns.csv`. Example rows: *Comics ∧ a very small goal* → success,
confidence 0.77, lift 1.92 (n = 1,461); *Food ∧ a very large goal* → failure, funded 20 of 997. Say:
"Association rules with significance testing — recurring patterns, not causes — which feed the
counterfactual engine's category-typical goals."

### 6d · Architecture and engineering (2 min)

Show [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) §5 (serving) and §9 (decision intelligence). Mention:
one warm stack, LightGBM-before-FAISS load order, 95 tests, MLflow registry, evidence-ledger discipline.

### 6e · Performance, candidly (1 min)

Show [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md) §3 and §12. Lead with the weaknesses — the 2017 data
ceiling, the random split, precision 0.565.

---

## 7. Questions to expect, and honest answers

**Q1. "Above the threshold — so will it fund?"**
No. "Above the decision threshold (0.329)" means the model flags it, and the threshold is tuned to catch
successes (recall 0.81, precision 0.56). A 40% probability is *more likely to fail than fund*, which is why
the UI shows it in amber with a caption. (An earlier version of the tile read "Predicted: Funded" at 40%;
it was reworded for exactly this reason.)

**Q2. "The readiness stage says Ready, but the committee says Do Not Invest."**
They answer different questions. *Readiness* measures whether the campaign is close to the model's
threshold and what could be fixed (53% reachable). The *committee* weighs category headwinds (Technology
funds at 24% historically), a goal 2.5× the category's funded median, and risk. A founder can be "fixable"
and still a weak investment today. *The founder page's "Strategy: Proceed with Caution" line comes from the
recommendation engine; the committee's "Do Not Invest" is a separate six-agent vote — two mechanisms, not
yet reconciled in the UI.*

**Q3. "Why does a capital-letters ratio matter?"**
It doesn't *cause* anything. The model found a correlation in 2009–2017 titles. SHAP explains the model,
not the world — this is exactly the kind of quirk the "associations, not causes" caveat is for.

**Q4. "Is this an LLM?"**
No. Objections, agents and reports are rule-based over real signals; there is no external API or secret.

**Q5. "Would it work on a campaign launched today?"**
The data ends in 2017 and the evaluation split is random, so scores likely overstate performance on new
campaigns. Say so before they ask ([PERFORMANCE_REPORT.md §12](PERFORMANCE_REPORT.md#12-weaknesses-and-limits)).

**Q6. "Where's the neural network?" / "Where's founder strength?"**
Not built / deferred — no data basis. Both are listed in the README's limitations and the research roadmap.

**Q7. "Did you deploy it?"**
Docker is configured but was never built (no Docker on the dev machine). Say exactly that.

**Q8. "How accurate is it?"**
ROC-AUC 0.78, PR-AUC 0.69, accuracy 67% versus 59.6% for always guessing "failed".

---

## 8. What not to claim

- That the UI is production-hardened, authenticated, or load-tested.
- That "above the decision threshold" means a campaign *will* fund.
- That counterfactuals describe what *would happen* to a real campaign.
- Founder, team, revenue, market-fit or scalability insight — deferred.
- A neural network, an LLM, or a deployed container.
- That pattern mining is a live feature of the app.

---

## 9. Screenshot and recording shot list

**Screenshots are captured** — 19 images in [`docs/assets/screenshots/`](assets/screenshots/), taken from the
running app by [`scripts/capture_screenshots.mjs`](../scripts/capture_screenshots.mjs) (headless Chrome,
1440 px wide at 1.5× scale, campaign A). Re-run it after any UI change; see
[assets/README.md](assets/README.md) for the full list. Retake any by hand if you prefer your own browser chrome.

| # | File | Shows |
|--:|---|---|
| 1 | `01-home.png` | landing page with both portal buttons |
| 2 | `02-founder-submitted.png` | form (campaign A) beside the first result cards |
| 3 | `03-prediction.png` | probability bar, threshold marker, four tiles, caption |
| 4 | `04-shap-bar-waterfall.png` | bar chart **and** waterfall |
| 5 | `05-scenario-simulator.png` | goal slider at $10,000, suggested scenarios, counterfactuals |
| 6 | `06-retrieval-tables.png` | similar-funded and similar-failed tables |
| 7 | `07-deferred-scores.png` | the six dashed "deferred" cards |
| 8 | `08-recommendations.png` | readiness tiles and priority improvements |
| 9 | `09-founder-report.png` | report with source badges |
| 10 | `10-investor-verdict.png` | verdict badge and confidence |
| 11 | `11-investor-risks.png` | Risks & Red Flags with objections |
| 12 | `12-investor-committee-votes.png` | the agent vote table |
| 13 | `13-api-docs.png` | Swagger `/docs` |
| 15 | `15-mobile.png` | Founder page at 390 px wide |
| 14 | *(not captured)* `14-422-rejection.png` | **you:** a terminal showing the rejected invented field (§6a) |

**Screen recording (≈3 min, 1080p) — yours to record:** 0:00 say-this-first (voice-over on the home page) · 0:20 submit A ·
0:45 prediction + SHAP · 1:30 slider drag (the best visual) · 1:55 retrieval · 2:10 **deferred cards**
(pause 4 s) · 2:25 switch to Investor → verdict → red flags · 2:55 close on the Performance Report's
weakness list. Use a clean browser profile, hide bookmarks, and keep the cursor slow.

---

## 10. Known rough edges

Found while preparing this guide — worth fixing or knowing about before presenting.

1. **Copilot is API-only** — returned by `/analyze`, not displayed.
2. **`uppercase_ratio` as a top negative driver** looks odd to a non-technical audience; be ready (Q3).
3. **Pattern mining and the startup catalog are not in the UI** (no endpoints).
4. **The form's default is a real dataset campaign** ("Artificer"), and it resets when you switch portals, so re-enter campaign A each time.
5. **Founder "Strategy" says *Proceed with Caution* while the Investor portal says *Do Not Invest*** for
   campaign A. They come from two different mechanisms (the recommendation engine's investor strategy versus
   the six-agent committee) and are not reconciled in the UI. Be ready to explain it (Q2).
6. **Provenance badges and honesty notes use internal build labels** such as "Day-12 committee" and
   "Days 4-10". They are accurate but read as jargon to an outside viewer.
7. **Rehearse once by hand:** the UI was verified with headless Chrome, not by a person clicking through it.

## 11. Mapping to the roadmap's demo flow

| Roadmap demo step | Where it is shown |
|---|---|
| Founder submits startup | UI — form (8 real fields; no upload) |
| Prediction | UI — Prediction card |
| SHAP explanation | UI — bar chart + waterfall |
| Semantic retrieval | UI — Historical evidence |
| Pattern mining | Notebook 07 / `ranked_patterns.csv` (no live panel) |
| Counterfactual analysis | UI — Scenario simulator |
| Investor objections | UI — Investor report, *Risks & Red Flags* |
| Evolution roadmap | UI — Founder report, *Evolution Roadmap* |
| Scenario simulation | UI — slider + suggested scenarios |
| Recommendations | UI — Recommendations & roadmap |
| Founder report / Investor report | UI — report viewer, print to PDF |

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Spinner for ~10 s on the first run | one-time retrieval warm-up — do the pre-flight warm-up call |
| Red error box / CORS message | restart the backend with `CORS_ORIGINS=http://localhost:5173` |
| "Not ready" / 503 | the backend is still loading models — wait for `/ready` |
| Results differ from this guide | the model or artifacts changed — re-run the three campaigns through `/analyze` and update §4 and the captured values |
| Form refuses to submit | goal must be > 0 and the deadline after launch |
