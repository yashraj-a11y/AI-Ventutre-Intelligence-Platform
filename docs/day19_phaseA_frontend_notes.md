# Day 19 · Phase A — Frontend Architecture (conceptual notes)

*The user-facing layer. These notes recap the frontend concepts and ground them in **this** platform's
real backend before Day 19 builds the Founder/Investor portals.*

---

## 0. Context — a working backend, no interface

Days 1–18 built a production FastAPI backend: 10+ routes, 13 layers, honesty-tiered JSON. But a founder
or investor doesn't `curl /analyze` — they need a **product**. Day 19 builds a React + TypeScript
frontend that *presents* those outputs. **No new ML, no new data** — the frontend can only show what the
backend returns and submit what it accepts.

---

## 1. The stack (and why)

| Choice | Why |
|---|---|
| **React + TypeScript** | component model + types that mirror the backend's response schemas — a wrong field is a compile error, not a runtime surprise |
| **Vite** | fast dev server + `tsc -b && vite build` for a typechecked production bundle |
| **Tailwind CSS** | utility-first responsive styling without a separate CSS architecture |
| **React Router** | client-side routing for Founder / Investor / Report views (and the auth-ready structure) |
| **Axios** | one typed HTTP client with interceptors for errors / loading / retry |
| **Recharts** | render SHAP contributions, score gauges, similarity tables from the *real* arrays |

**The toolchain is present** (node v25, npm 11), so — unlike the Docker day — every phase is
**built + typechecked** (`npm run build`), not just written.

---

## 2. Client–server split — the frontend is a thin presenter

```
Browser (React SPA)  ──HTTP/JSON──▶  FastAPI backend  ──▶  13-layer engine
   renders, no ML                     the intelligence
```

The SPA holds **zero ML logic**. It sends a campaign, receives typed JSON, and renders it. Every score,
contribution, and similar-campaign row is computed server-side; the frontend's job is *presentation and
interaction*, never inference. This keeps the honesty boundary in one place (the backend) and the UI
honest by construction — it can't invent a number the API didn't send.

---

## 3. The contract is the source of truth (types mirror `responses.py`)

The API's response schemas *are* the frontend's type definitions. `PredictionResponse`,
`ExplanationResponse` (`top_positive[]`/`top_negative[]`), `RecommendationResponse`
(`investment_readiness`, `priority_improvements[]`, `deferred`), `ReportResponse` (`sections[]` with a
`source`, `deferred_appendix`) — each becomes a TypeScript interface. Typing against the real contract
means the UI *cannot* reference a field the backend doesn't return, and a backend change surfaces as a
type error. This is the frontend version of the platform's "grounded in real numbers" rule.

---

## 4. Honesty is a first-class UI element

Every backend response carries a `honesty_note` and/or a `deferred` boundary. The frontend **renders
them**, it doesn't hide them:

- the submission form offers **only** the real fields (title, category, goal, country, currency, dates)
  — the platform has no founder/financial/market/pitch-deck inputs, so the form doesn't either;
- deferred scores (Business-Model, Founder-Strength, Scalability, Market-Fit) render as visible
  **"deferred — no data basis"** cards, not fabricated gauges;
- each report section shows its **`source`** provenance; the `deferred_appendix` is shown, not dropped.

A dashboard that invented a "Founder Strength: 78" gauge would be lying with a nicer font. The UI's
credibility comes from showing the boundary the backend already draws.

---

## 5. What the frontend deliberately does NOT do (deferred)

- No **file/PDF upload** or founder/financial/market inputs (the backend `422`s them).
- No **live pattern-mining panel** (no `/patterns` endpoint — patterns are static artifacts).
- No **startup catalog** search / filter / rank (no catalog endpoint) — both portals are
  *submit → analyze* with different framing, not a multi-startup CRM.
- No **real authentication** — the structure is *auth-ready* (role state, a protected-route wrapper),
  but no JWT is implemented.

---

## 6. Frontend architecture — the pattern

A typed React SPA that is a **thin, honest presenter** of a fixed API contract: types mirror the
backend schemas, all ML stays server-side, and the platform's honesty boundary (`honesty_note`,
`deferred`, per-section `source`) is rendered as UI rather than hidden. Responsive by default,
built + typechecked every phase, auth-ready but not auth-built.

> **Interview soundbite:** *"The frontend is a thin, typed presenter — zero ML logic, it just sends a
> campaign and renders the backend's typed JSON. I derive the TypeScript interfaces from the API's
> response schemas, so the UI literally can't reference a field the backend doesn't return and a
> contract change is a compile error. The most important design decision is that honesty is a UI
> element: the form offers only the real Kickstarter fields because the backend 422-rejects anything
> else, and the scores the platform can't support render as visible 'deferred — no data basis' cards
> rather than invented gauges. A dashboard that shows a fabricated 'Founder Strength: 78' is lying with
> a nicer font — the UI's credibility comes from rendering the same boundary the backend draws."*
