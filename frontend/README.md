# AI Venture Intelligence — Frontend

A React + TypeScript SPA (Founder & Investor portals) over the FastAPI backend. It is a **thin, typed
presenter**: zero ML logic in the client — it submits a campaign, receives typed JSON, and renders it,
with the platform's honesty boundary (`honesty_note`, `deferred`, per-section `source`) shown as UI,
never faked.

## Stack

React 19 · TypeScript 6 (strict, `verbatimModuleSyntax`) · Vite 8 · Tailwind CSS v4 · React Router ·
Axios · Recharts. Build = `tsc -b && vite build`, so **every build typechecks**.

## Running it

```bash
cd frontend
npm install
cp .env.example .env            # VITE_API_BASE_URL=http://localhost:8000 (default)
npm run dev                     # http://localhost:5173
npm run build                   # typecheck + production bundle -> dist/
```

**The backend must allow the dev origin.** Day-18 CORS is deny-by-default, so start the API with:

```bash
CORS_ORIGINS=http://localhost:5173 OMP_NUM_THREADS=1 uvicorn src.api.main:app --port 8000
```

`VITE_API_BASE_URL` points the client at the API (defaults to `http://localhost:8000`).

## Project structure

```
frontend/src/
├── api/
│   ├── client.ts        # typed Axios client (one fn per endpoint) + ApiError + retry
│   └── types.ts         # TS interfaces mirroring the backend response schemas
├── hooks/
│   ├── useApi.ts        # run an API call, track {data, loading, error}
│   ├── useDebounced.ts  # debounce a value (scenario sliders)
│   └── useAuth.tsx      # auth context/provider/hook (stub login)
├── components/
│   ├── ui.tsx           # Card, Button, HonestyNote, Loading, ErrorBox, Stat
│   ├── CampaignForm.tsx # the real CampaignRequest fields only
│   ├── ProtectedRoute.tsx
│   ├── Layout/Header.tsx
│   ├── Dashboard/       # PredictionPanel, DeferredScores, RecommendationPanel
│   ├── Explainability/  # ExplanationPanel (SHAP chart, code-split)
│   ├── ScenarioSimulator/
│   ├── Retrieval/       # RetrievalDashboard
│   ├── Reports/         # ReportViewer
│   ├── FounderAnalysis.tsx
│   └── Investor/InvestorAnalysis.tsx
├── pages/               # Home, Login, FounderPortal, InvestorPortal
├── App.tsx              # shell + routes
└── main.tsx             # BrowserRouter + AuthProvider
```

## API client

One typed function per real endpoint (`api.predict`, `api.explain`, `api.retrieve`, `api.recommend`,
`api.scenario`, `api.counterfactual`, `api.reportFounder`, `api.reportInvestor`, `api.analyze`,
`api.health/ready/live`). Errors are normalized to `ApiError` (status + backend detail); requests
**retry once on network/5xx only — never on 4xx** (a 422 is deterministic). The TS interfaces were
grounded by hitting the live backend, so a field the API doesn't return is a compile error.

## Component map

| Component | Purpose | Endpoint(s) | Key props / state |
|---|---|---|---|
| `CampaignForm` | submit the real Kickstarter fields | — (validates client-side) | `onSubmit(CampaignRequest)` |
| `PredictionPanel` | probability + real derived scores | `/predict` | `data: PredictionResponse` |
| `DeferredScores` | render the deferred score boundary | — (static honesty) | — |
| `ExplanationPanel` | SHAP contribution bar chart (code-split) | `/explain` | `data: ExplanationResponse` |
| `ScenarioSimulator` | live what-if over real levers | `/predict` (debounced) + `/scenario` | `campaign`, `baseline` |
| `RetrievalDashboard` | similar funded/failed + neighbor benchmark | `/retrieve` | `data: RetrievalResponse` |
| `RecommendationPanel` | readiness + priority roadmap | `/recommend` | `data: RecommendationResponse` |
| `ReportViewer` | render report sections by `kind` + print | `/report/{founder,investor}` | `campaign`, `variant` |
| `InvestorAnalysis` | committee verdict + comparables + report | `/analyze`, `/retrieve`, `/report/investor` | `campaign` |
| `FounderAnalysis` | orchestrates the founder dashboard | predict/explain/retrieve/recommend/report | `campaign` |

## Honesty boundary in the UI

The platform's honesty discipline is rendered, not hidden:

- the **submission form** offers only the real fields — no founder/financial/market/PDF inputs (the API
  `422`s them);
- **deferred scores** (Business-Model, Founder-Strength, Scalability, Market-Fit, Growth, Market-Opp.)
  render as greyed "deferred — no data basis" cards, not fabricated gauges;
- **scenario levers** are the real controllable features (goal, duration, launch weekday, title) — no
  MRR/retention/team sliders;
- every `honesty_note` renders via `HonestyNote`; each **report section shows its `source`** provenance;
  the `deferred_appendix` is shown; retrieval benchmarks are computed from real neighbors, not invented.

## State management

Local component state (`useState`) for the submitted campaign and form; `useApi` for request state; a
single React **context** (`useAuth`) for role/token. No global store is needed — data flows top-down
from the submitted campaign into the analysis components.

## Auth-ready (not auth-built)

`AuthProvider` / `useAuth` / `ProtectedRoute` / role state / token slot / localStorage persistence are
real; only `login()` is a **stub** (no credentials, no JWT). To make it real, replace `login()` with a
`POST /auth/login` that returns a JWT and decode the role — routes, guards, and components are unchanged.

## Explainability charts

`ExplanationPanel` draws the contribution bar chart and a waterfall: baseline (`base_value`) → each
top contribution → "All other features" (the remainder, so the bars always land exactly on `margin`) →
this campaign's log-odds. Both come from `/explain`; the waterfall hides itself if the backend omits
`base_value`/`margin`.

## Deferred (not built — no backend support)

Startup catalog search/filter/rank (no catalog endpoint), a live pattern-mining panel (no `/patterns`
endpoint), real JWT authentication, and a server-side PDF export (export is the browser's print dialog).
Each is named, not faked — the same discipline as the backend.
