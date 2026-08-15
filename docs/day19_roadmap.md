# Day 19 — Frontend, UX & AI Venture Intelligence Dashboard (FROZEN SPEC V1.1)

**Roadmap V1.1 — the user-facing layer.** A React + TypeScript frontend (Founder + Investor portals)
over the EXISTING FastAPI backend — displaying only what the backend **returns** and submitting only
what it **accepts**, with the platform's honesty boundary made a first-class UI element.

> **🔴 READ THIS FIRST — the backend is already honest; the draft's UI is not.** `CampaignRequest`
> accepts **only** Kickstarter fields (title, category, goal, country, currency, launch/deadline) with
> `extra="forbid"` — so a form asking for "founder info / financials / market / pitch-deck PDF" would be
> **422-rejected**. `VentureScoreGenerator` returns only real scores (funding_probability, failure_risk,
> startup_score, investment_score, investment_readiness — all derived from the calibrated probability)
> and returns "Business-Model / Founder-Strength / Scalability / Market-Fit" as **deferred (value=None)**.
> The frontend must mirror this: real inputs, real outputs, and the **deferred boundary shown, not faked**.

> **✅ Unlike Day 18, the toolchain IS available** (node v25, npm 11) — so every phase is **built +
> typechecked** (`npm run build`, `tsc --noEmit`), grounded against the real API response shapes.

> **Status:** FROZEN 2026-08-17 (V1.1). Build phase-by-phase on `build Phase <X>`; each phase =
> implement **and** teach via Framework 1, grounded in the real API contract.

---

## 0. The REAL API contract (what the UI consumes)

| Endpoint | Returns (real fields) |
|---|---|
| `POST /predict` | `funding_probability`, `predicted_label`, `decision_threshold` |
| `POST /explain` | `funding_probability`, `top_positive[]`, `top_negative[]`, `honesty_note` |
| `POST /retrieve?k=` | `similar[]`, `similar_successful[]`, `similar_failed[]`, `honesty_note` |
| `POST /recommend` | `investment_readiness`, `confidence`, `priority_improvements[]`, `investor_strategy`, `final_strategy`, `deferred`, `honesty_note` |
| `POST /scenario` · `POST /counterfactual` | `{campaign, result, honesty_note}` (GenericResponse) |
| `POST /report/founder` · `/report/investor` | `report_type`, `sections[]` (each with a `source`), `deferred_appendix`, `honesty_note` |
| `POST /analyze` | `committee_recommendation`, `overall_confidence`, `reports{founder,investor,memo,executive}`, `copilot`, `honesty_note` |
| `GET /health` · `/live` · `/ready` | status + diagnostics (Day 18) |

**Submission contract (the ONLY form fields):** `name`, `main_category`, `category?`, `currency`,
`country`, `usd_goal_real`, `launched`, `deadline`. Anything else → 422.

---

## 1. What changed from the draft (changelog + why)

| # | Draft said | 🔧 Frozen correction | Why |
|---|---|---|---|
| **C-1** | Submission form: founder info, **financials, market, pitch-deck PDF** | **Form = the real `CampaignRequest` fields only** (title/category/goal/country/currency/dates). No file upload (backend is JSON, `extra="forbid"`). | Those fields are 422-rejected; the platform has never had founder/financial/market/PDF inputs. |
| **C-2** | Dashboard scores: **Business-Model, Founder-Strength, Market-Fit, Scalability, Growth-Potential, Market-Opportunity** | **Show the REAL scores** (funding_probability, failure_risk, startup_score=100·p, investment_score, investment_readiness). Render the fabricated ones as **visible "deferred — no data basis" cards** (the backend already returns them as `value=None`). | The backend's `VentureScoreGenerator` explicitly defers these; the UI surfaces the boundary rather than inventing values. |
| **C-3** | Endpoints `/semantic_search`, `/simulate`, `/investor_questions`, `/report` | **Real routes:** `/retrieve`, `/scenario`, `/analyze` (investor objections live inside it), `/report/{founder,investor}`. | Those draft names don't exist (same correction as Days 17–18). |
| **C-4** | Scenario sliders: **MRR, Retention, Team Size, Burn Rate, Founder Experience, Market Size** | **Real levers only:** funding **goal**, **campaign duration**, **launch weekday**, **title** tweaks → `/scenario` + `/counterfactual`. | These are the actual controllable features (Days 8/10); MRR/retention/burn/team/market don't exist in the data. |
| **C-5** | "TF-IDF (2,500)" + "Neural Networks" | **TF-IDF = 800; NN deferred.** | Repeated boilerplate; corrected on Days 16–18. |
| **C-6** | Pattern-mining panel; investor **search / filter / rank a startup catalog** | **DEFER both** — there is **no live `/patterns` endpoint** (patterns are static artifacts) and **no catalog/search endpoint**. Both portals are **submit → analyze** (founder vs investor *framing*), not a multi-startup CRM. | The UI consumes live endpoints; inventing a catalog/pattern API would be fabrication. |
| **C-7** | Export **PDF** | **Client-side print-to-PDF** of the rendered report (browser), or the backend's HTML. No PDF endpoint is claimed. | `/report` returns JSON sections; PDF is a client concern. |
| **C-8** | (implied) frontend just works | **CORS:** the backend denies cross-origin by default (Day 18). Dev needs `CORS_ORIGINS=http://localhost:5173`; document it. API base via `VITE_API_BASE_URL`. | Day-18 hardening means the frontend origin must be explicitly allowed. |

**IMPROVEMENTS:** IMP-1 **honesty boundary as UI** — every view renders its `honesty_note` /
`deferred_*` (deferred cards, provenance badges). IMP-2 **typed against the real schemas** (TS
interfaces mirror `responses.py`). IMP-3 each phase **builds + typechecks** (`npm run build`, `tsc`).
IMP-4 **auth-ready, not auth-built** — role state + protected-route wrapper stubs, no JWT implemented.
IMP-5 responsive (Tailwind), loading/error/retry states in the API client.

---

## 2. Honesty tiers

**Tier 1 — REAL:** the 8 live endpoints + their exact response shapes, the real submission fields, the
real scores (all functions of the calibrated `p`), every `honesty_note` / `deferred` boundary the
backend already emits.
**Tier 2 — DERIVED (labeled):** the React components, charts (built from real `top_positive/negative`,
`similar_*`, etc.), routing, the typed API client — **build/typecheck-verified**; live integration is a
user step (run backend + `npm run dev`).
**Tier 3 — DEFERRED, never fabricated:** founder/financial/market/pitch-deck inputs (422'd),
Business-Model/Founder-Strength/Scalability/Market-Fit/Growth/Market-Opportunity *values* (backend
defers them), a live pattern-mining panel (no endpoint), startup catalog search/filter/rank (no
endpoint), real authentication/JWT (architecture-ready only), Neural Network.

---

## 3. Phases (build order)

- **A** — **Architecture notes + scaffold** (`docs/day19_phaseA_frontend_notes.md`, `frontend/`): Vite
  React-TS, Tailwind, React Router, Axios, Recharts. `npm create vite` + install + `npm run build`.
- **B** — **Typed API client** (`frontend/src/api/client.ts` + `types.ts`): Axios client for the 8 real
  endpoints; TS interfaces mirroring `responses.py`; base from `VITE_API_BASE_URL`; error handling,
  loading, retry, timeouts. Types grounded by hitting the real endpoints.
- **C** — **Layout, routing, submission form** (`components/`, `routes/`): the real `CampaignRequest`
  form (validated client-side to match `extra="forbid"`), portal shell + role toggle (auth-ready),
  responsive nav.
- **D** — **Founder Portal** (`pages/FounderPortal.tsx`): submit → `/predict` (real scores + **deferred
  score cards**) + `/recommend` (readiness, priority improvements, roadmap) + founder report link.
- **E** — **Explainability (SHAP)** (`components/Explainability/`): `/explain` → contribution bar chart
  (top_positive/negative) + waterfall (base_value + contributions) + honesty note.
  *(Built. `/explain` now also returns `base_value` + `margin` — both already computed by the Day-5
  `LocalExplainer`, previously dropped by the service layer; additive, optional fields.)*
- **F** — **Scenario Simulator** (`components/ScenarioSimulator/`): **real levers** (goal / duration /
  launch weekday / title) → `/scenario` + `/counterfactual`; show updated probability + reason + the
  deferred boundary. No MRR/retention/team sliders.
- **G** — **Retrieval Dashboard** (`components/Retrieval/`): `/retrieve` → similar successful/failed
  tables (name, category, similarity, reason) + benchmarks + `deferred_benchmarks` (honesty).
- **H** — **Report Viewer** (`pages/ReportViewer.tsx`): `/report/{founder,investor}` + `/analyze` →
  render `sections[]` with **per-section `source` provenance** + `deferred_appendix`; print-to-PDF.
- **I** — **Investor Portal** (`pages/InvestorPortal.tsx`): investor framing — investor report +
  committee verdict/`overall_confidence` (`/analyze`) + comparables (`/retrieve`). Catalog
  search/filter/rank **deferred**.
- **J** — **Auth-ready architecture** (`routes/`, `hooks/useAuth`): role state (founder/investor),
  `ProtectedRoute` wrapper + token slot — **stubbed, not implemented** (no JWT).
- **K** — **Docs** (`frontend/README.md` + component docs): purpose / props / endpoints / state /
  navigation / responsive / example.
- **Notebook** — `notebooks/19_Frontend_Dashboard.ipynb`: verify the frontend **builds** (`npm run
  build` via subprocess), document the architecture + the real API contract the UI consumes, show the
  live API responses (TestClient) the components render; honest that a Python notebook can't run React.

---

## 4. Environment & repo

- **Toolchain present:** node v25, npm 11 → each phase `npm run build` + `tsc --noEmit`. **No Python
  deps.** Frontend deps (react, typescript, vite, tailwindcss, react-router-dom, axios, recharts;
  @tanstack/react-query optional) via npm.
- **New tree:** `frontend/` (its own `package.json`, `tsconfig`, `vite.config.ts`, `src/`, `public/`).
  Gitignore `frontend/node_modules/` + `frontend/dist/`.
- **CORS:** set `CORS_ORIGINS=http://localhost:5173` on the backend for dev (Day-18 deny-by-default).
- **New files:** `frontend/**`, `docs/day19_phaseA_frontend_notes.md`, `notebooks/19_Frontend_Dashboard.ipynb`.
  **No changes** to `src/` unless the UI uncovers a real backend defect.

---

## 5. Non-negotiables (frozen)

1. **Submit only real fields; display only real outputs** — no founder/financial/market/PDF inputs, no
   fabricated scores (show the backend's deferred boundary) (C-1/C-2).
2. **Real routes + real scenario levers** — `/retrieve`, `/scenario`, `/analyze`, `/report/{...}`; goal
   / duration / launch / title, not MRR/retention/team (C-3/C-4).
3. **Honesty boundary is a UI element** — render every `honesty_note` / `deferred_*` (IMP-1).
4. **Typed against real schemas; build + typecheck every phase** (IMP-2/3).
5. **Auth-ready, not auth-built**; pattern/catalog panels deferred (no endpoints) (C-6, IMP-4).
6. **No new ML** — the UI presents existing outputs; NN deferred, TF-IDF 800 (C-5).
7. Teach every phase via Framework 1, grounded in the real contract; end with an interview soundbite.

---

*Frozen 2026-08-17. Supersedes the Day-19 draft. Day 19 = a React+TS Founder/Investor frontend over the
existing FastAPI backend, honest to the real inputs/outputs and rendering the deferred boundary. Next:
Day 20 — Final Release, Docs, Portfolio.*
