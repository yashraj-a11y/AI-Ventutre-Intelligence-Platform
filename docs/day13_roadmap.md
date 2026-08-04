# Day 13 — Professional Report Generation & Venture Copilot (FROZEN SPEC V1.1)

**Architecture V1.0 — Layer 13 (Report Generation).** The final layer: turn every prior layer's
outputs into audience-specific, decision-ready reports (Founder / Investor / Memo / Executive
Summary) + a deterministic Q&A Venture Copilot, exported to JSON / Markdown / HTML. This completes
Architecture V1.0.

> **Status:** FROZEN 2026-08-04 (V1.1). Read before building any phase. Build phase-by-phase on
> `build Phase <X>` / `build the notebook`; each phase = implement **and** teach via Framework 1,
> grounded in real project numbers.

---

## 0. The central principle (read first)

**Reporting is a PRESENTATION layer — it generates ZERO new analysis.** Every report is a
deterministic *reformatting* of the Day 4–12 outputs (the Day-11 `VentureDecision` + the Day-12
`CommitteeDecision` + their underlying signals). This is the key to Day-13 honesty:

> **The honesty is INHERITED. If a report only surfaces what Days 4–12 already produced — all of
> which are honesty-tiered and deferral-aware — then the report is automatically honest.** The one
> rule Day 13 must enforce is **NO NEW CLAIMS**: every line in every report must trace to a real
> upstream output. Any draft section with no backing Day 4–12 output is **deferred**, not invented.

So Day 13 introduces **no new model, no LLM, no new fabrication risk** — only formatting.

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Report example: "Experienced founding team", "High recurring revenue", "Customer retention" weakness, "Weak competitive moat", "Delay fundraising until $20K MRR", "Expected Return", "Comparable Funding Rounds", "Financial Assessment" (P&L) | **DEFERRED — never generated.** Reports surface ONLY real upstream outputs: Strengths/Weaknesses = Day-5 SHAP ± drivers; Founder Assessment = Day-12 creator-execution agent; Financial Assessment = Day-12 funding-viability agent; Market/Competition/Risk = Day-12 agents; readiness/confidence/priority = Days 10/11; patterns = Day-7. Expected return / comparable rounds / MRR / moat / team → deferred appendix. | No founder/revenue/retention/moat/return data exists (Days 4–12 established this). Reporting can't add data the analysis never had. |
| **C-2** | Export JSON / Markdown / HTML / **PDF** | **Mandatory: JSON + Markdown + HTML** (deps-free string templates). **PDF = OPTIONAL**, gracefully gated on an optional dep (export if `reportlab`/`weasyprint` present, else skip with a note; HTML is print-to-PDF ready). | `reportlab`/`weasyprint`/`pdfkit`/`fpdf` are all **missing** — PDF would need a NEW DEP. NO-NEW-DEPS stays intact; `jinja2` is present but we use plain f-strings for zero coupling. |
| **C-3** | Venture Copilot answers "conversationally"; "prompt engineering for structured reports" | **Deterministic intent-router Q&A** — a fixed set of question intents (keyword match) → templated answers filled with real Day 4–12 numbers, with an **"I can't answer that from the data" fallback**. **No LLM.** | Consistent with Days 9/11/12 (deterministic, no hallucination). An LLM would re-introduce fabrication + a new dep. "Prompt engineering" → deterministic templating. |
| **C-4** | Copilot module = `src/copilot/venture_copilot.py` | **New file `src/reporting/copilot.py`** (class `VentureCopilotQA`). **DO NOT touch `src/copilot/venture_copilot.py`** — it already exists (Day-5 `VentureCopilot` brief, 8.7 KB). | Writing the draft's path would clobber committed Day-5 code. The Q&A copilot is a distinct, new thing. |
| **C-5** | `founder_report.pkl` implied; serialization | **JSON content-hash manifest + `verify()`**, mirroring the Day-8/10/11/12 serializers. Report artifacts are JSON/MD/HTML. | Project-wide JSON-not-pickle rule. |
| **C-6** | Each report re-runs the pipeline | **ONE warm faiss-free stack**: the reporting pipeline builds ONE `cf`, shares it into the Day-11 `RecommendationPipeline` and the Day-12 `ConsensusEngine`, runs each once, and assembles a **ReportBundle** the generators format. | Days 11 & 12 are faiss-free; sharing keeps one model in memory. Report generators are pure functions of the bundle — no model access. |

**IMPROVEMENTS beyond the draft:**
- **IMP-1 — provenance on every section.** Each report section carries a `source` tag (e.g.
  "Day-5 SHAP", "Day-12 committee") so the report is *auditable* — you can trace any claim to its
  layer. This operationalizes the "no new claims" rule.
- **IMP-2 — deferred appendix as data.** Every report ends with an explicit "Not covered (no data)"
  list, so the report documents its own honesty boundary to the reader.
- **IMP-3 — deterministic, reproducible reports.** Same campaign → byte-identical report (enables
  content-hash serialization).
- **IMP-4 — graceful optional PDF.** PDF export is attempted only if an optional dep is importable;
  otherwise the HTML (print-ready) is the deliverable and a note is recorded. Mandatory path never
  needs a new dep.
- **IMP-5 — copilot honesty fallback.** The Q&A copilot answers only from real outputs and returns a
  clear "not available in this dataset" for out-of-scope questions (e.g. "what's my MRR?").

---

## 2. Honesty tiers (what may appear in reports)

**Tier 1 — REAL (surface freely, with provenance):** funding probability + threshold (Day 4); SHAP ±
drivers = Strengths/Weaknesses (Day 5); success/failure pattern evidence (Day 7); counterfactual
improvements (Day 8); readiness stage + roadmap + milestones + scenarios (Day 10); prioritized
actions + confidence + readiness index + founder/investor strategy (Day 11); six agent reports +
committee decision + votes/conflicts/veto (Day 12).

**Tier 2 — DERIVED, label model-derived:** startup score (= Day-11 readiness index); executive
one-liners; report prose assembled from the above. All "model-simulated, not causal."

**Tier 3 — DEFERRED, NEVER fabricate (deferred appendix):** experienced team / founder background;
revenue / MRR / retention / burn / runway / CAC-LTV; competitive moat / differentiation / named
competitors; expected return / IRR / valuation / comparable funding rounds; market size / TAM.

---

## 3. The ReportBundle (shared input to all generators)

The pipeline assembles this **once**, on one warm stack; every generator is a pure function of it:

```
ReportBundle = {
  campaign:   {name, main_category, usd_goal_real, country},
  decision:   <Day-11 VentureDecision>,        # readiness, confidence, priority_improvements,
                                               #   recommendations, risks, opportunities,
                                               #   founder_strategy, investor_strategy, final_strategy, deferred
  committee:  <Day-12 CommitteeDecision>,      # committee_recommendation, overall_confidence,
                                               #   agent_reports[6], agent_votes, conflicts, risk_veto
  drivers:    <Day-5 SHAP top_positive / top_negative>,   # from decision's aggregated vi
}
```

Report generators: `generate(bundle) -> structured report dict` (with per-section `source` tags +
`deferred` appendix). Export turns a report dict into JSON / Markdown / HTML.

---

## 4. Phases (build order)

- **A** — Report-generation notes (conceptual): presentation vs analysis, the "no new claims"
  inheritance principle, audience-specific communication, deterministic templating (not LLM).
  `docs/day13_phaseA_reporting_notes.md` + `src/reporting/__init__.py`.
- **B** — `founder_report.py` **FounderReportGenerator**. Sections (all real): executive summary
  (Day-11 final_strategy), startup score + funding probability + investment readiness (Days 4/10/11),
  strengths / weaknesses (Day-5 ±drivers), success/failure patterns (Day-7 evidence), counterfactual
  improvements + priority recommendations (Days 8/11), evolution roadmap + scenarios + next
  milestones (Day-10), deferred appendix.
- **C** — `investor_report.py` **InvestorReportGenerator**. Sections: executive summary, investment
  recommendation + confidence (Day-12 committee), market / founder-assessment / financial-assessment
  / competitive-landscape (Day-12 agents), similar startups (cohort), risks + red flags (Days 9/12),
  opportunities (Day-11), investment decision (Day-12). **Deferred**: expected return, comparable
  rounds.
- **D** — `investment_memo.py` **InvestmentMemoGenerator**. VC-style memo: exec summary, startup
  overview, market opportunity, competitive position, founder (creator) analysis, financial (funding)
  analysis, risk assessment, **investment thesis** (deterministic synthesis of committee + readiness
  + top strengths/risks), recommendation.
- **E** — `executive_summary.py` **ExecutiveSummaryGenerator**. Dashboard one-liners: startup score,
  funding probability, risk level, committee recommendation, confidence.
- **F** — `copilot.py` **VentureCopilotQA** (deterministic intent-router Q&A). Fixed intents: "why
  did probability change / what to improve first / which are similar / why do similar fail / biggest
  risk / what before fundraising" → templated answers from the bundle. Honesty fallback for
  out-of-scope. **Not** `src/copilot/venture_copilot.py` (C-4).
- **G** — `export.py` **ReportExporter**. `to_json` / `to_markdown` / `to_html` (deps-free);
  `to_pdf` optional-graceful (C-2).
- **H** — `pipeline.py` **ReportingPipeline**. One `cf` → run Day-11 + Day-12 → assemble
  `ReportBundle` → generate all four reports + copilot → export. The unified entry point.
- **I** — `serialization.py` **ReportingSerializer**. Content-hash JSON manifest + `verify()`, pinning
  the Day-4/11/12 dependencies + the report artifacts.
- **Notebook** — `notebooks/13_Report_Generation.ipynb`: one campaign → founder report → investor
  report → memo → executive summary → copilot Q&A → export samples → serialized verify; deferred
  appendix + provenance shown.

---

## 5. Repository & artifacts after Day 13

```
src/reporting/{__init__, founder_report, investor_report, investment_memo, executive_summary,
               copilot, export, pipeline, serialization}.py
notebooks/13_Report_Generation.ipynb
artifacts/reports/{founder_report, investor_report, investment_memo, executive_summary,
                   copilot_log, reporting_engine}.json  (+ .md / .html exports; .pdf optional)
configs/preprocessing.yaml → day13.{founder, investor, memo, executive, copilot, export,
                                     pipeline, serialization}
docs/day13_phaseA_reporting_notes.md
```

**Input contract:** the `ReportBundle` (§3). `RecommendationPipeline.run(raw) → VentureDecision`
(Day 11); `ConsensusEngine.run(raw) → CommitteeDecision` (Day 12). Both faiss-free; share one `cf`.
Note: the copilot lives at `src/reporting/copilot.py`, NOT the pre-existing Day-5
`src/copilot/venture_copilot.py` (C-4). The empty `src/reports/` dir is vestigial — use
`src/reporting/`.

---

## 6. Non-negotiables (frozen)

1. **NO NEW CLAIMS** — reports surface only real Day 4–12 outputs; team/revenue/moat/return/
   comparable-rounds deferred (C-1). Honesty is inherited + enforced by provenance (IMP-1).
2. **No new deps on the mandatory path** — JSON/MD/HTML via stdlib string templates; PDF optional-
   graceful (C-2). No faiss/torch.
3. **Deterministic, no LLM, no new model** — copilot is intent-routing with an honesty fallback
   (C-3, IMP-5).
4. **Don't clobber Day-5 `venture_copilot.py`** — Q&A copilot is a new file (C-4).
5. **JSON not pickle**, content-hash + `verify()` (C-5).
6. **ONE warm faiss-free stack**, ReportBundle assembled once (C-6).
7. Teach every phase via Framework 1, grounded in real numbers; end with an interview soundbite.

---

*Frozen 2026-08-04. Supersedes the Day-13 draft. Completes Architecture V1.0. Build on `build Phase <X>`.*
