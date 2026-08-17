// Landing page: what the platform does, how, how well (held-out test numbers), and what it does NOT do.
// Every figure here is a measured result from docs/PERFORMANCE_REPORT.md — nothing aspirational.
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth, type Role } from "../hooks/useAuth";
import { ServerStatusPill } from "../components/ui";

const METRICS = [
  { value: "331,464", label: "real Kickstarter campaigns", hint: "decided outcomes, 2009–2017" },
  { value: "0.776", label: "ROC-AUC on held-out test", hint: "66,293 campaigns, scored once" },
  { value: "81%", label: "of funded campaigns caught", hint: "recall at the 0.329 threshold" },
  { value: "865", label: "features per campaign", hint: "800 title terms + 65 tabular" },
];

const STEPS = [
  { n: "1", title: "Describe the campaign", body: "Title, category, goal, country, currency and dates: the fields a campaign really has before launch." },
  { n: "2", title: "The models analyse it", body: "A calibrated LightGBM model scores it, SHAP explains the score, and the simulators test what changes would help." },
  { n: "3", title: "Act on the results", body: "Founders get a prioritised improvement plan. Investors get a committee verdict and a due-diligence report." },
];

const FEATURES: { title: string; body: string; icon: ReactNode }[] = [
  { title: "Calibrated prediction", body: "A funding probability you can read as a probability, with a decision threshold tuned to catch funded campaigns.", icon: <path d="M4 19h16M7 15l3-4 3 2 4-6" /> },
  { title: "SHAP explanations", body: "Which features pushed this campaign's score up or down, shown as a bar chart and a waterfall.", icon: <path d="M5 6h10M5 12h14M5 18h7" /> },
  { title: "What-if simulator", body: "Move the goal, duration, launch day or title and watch the probability update live.", icon: <path d="M4 8h10m4 0h2M4 16h4m4 0h8M16 6v4M10 14v4" /> },
  { title: "Prioritised recommendations", body: "Counterfactual changes ranked by how much they lift the odds, each backed by historical evidence.", icon: <path d="M5 12l4 4L19 6" /> },
  { title: "Six-agent committee", body: "Market, founder, financial, competition, risk and investment agents vote, with a risk veto.", icon: <path d="M8 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm8 0a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM3 20c0-3 2.5-5 5-5s5 2 5 5m-2-5c.8-.6 2-1 3-1 2.5 0 5 2 5 5" /> },
  { title: "Founder & investor reports", body: "Structured reports where each section names its source, ready to print or save as PDF.", icon: <path d="M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6" /> },
];

const LIMITS = [
  "No founder, team, revenue or market data exists in the dataset, so Business Model, Founder Strength, Scalability, Market Fit, Growth Potential and market size are shown as deferred, never invented.",
  "The model learned from 2009–2017 campaigns. Results for a campaign launched today are likely optimistic.",
  "Feature effects are associations the model learned, not causal guarantees.",
];

const STACK = ["LightGBM", "scikit-learn", "SHAP", "SBERT + FAISS", "FastAPI", "React + TypeScript", "Tailwind CSS", "MLflow"];

function Icon({ children }: { children: ReactNode }) {
  return (
    <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600">
      <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {children}
      </svg>
    </span>
  );
}

/** Static preview of a real result (fictional demo campaign "Lumen", see docs/DEMO_GUIDE.md). */
function PreviewCard() {
  const drivers = [
    { label: "Campaign duration", w: 92, up: true },
    { label: "Title uppercase ratio", w: 70, up: false },
    { label: "Funding goal", w: 33, up: false },
    { label: "Title term 'smart'", w: 20, up: true },
  ];
  return (
    <div className="relative rounded-2xl border border-slate-200 bg-white p-6 shadow-xl shadow-indigo-100">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Sample result</p>
          <p className="mt-1 font-semibold text-slate-900">Lumen: solar smart garden planter</p>
          <p className="text-xs text-slate-500">Technology · $25,000 goal · 30 days</p>
        </div>
        <span className="text-3xl font-bold text-amber-600">40%</span>
      </div>
      <div className="relative mt-4 h-2.5 rounded-full bg-slate-100">
        <div className="h-full w-[40%] rounded-full bg-amber-500" />
        <div className="absolute top-[-3px] h-4 w-0.5 bg-slate-800" style={{ left: "33%" }} />
      </div>
      <p className="mt-2 text-xs text-slate-500">Above the 33% decision threshold, but under 50%: borderline.</p>

      <p className="mt-5 text-xs font-semibold uppercase tracking-wider text-slate-400">Biggest drivers</p>
      <ul className="mt-2 space-y-2">
        {drivers.map((d) => (
          <li key={d.label} className="flex items-center gap-3 text-sm">
            <span className="w-40 shrink-0 truncate text-slate-600">{d.label}</span>
            <span className="h-2 flex-1 rounded-full bg-slate-100">
              <span className={`block h-full rounded-full ${d.up ? "bg-emerald-500" : "bg-rose-500"}`} style={{ width: `${d.w}%` }} />
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-4 text-[11px] text-slate-400">Fictional demo campaign. Bars show relative size of SHAP effects.</p>
    </div>
  );
}

export default function Home() {
  const { login } = useAuth();
  const navigate = useNavigate();
  // The sign-in is a demo stub (role choice, no password), so the CTAs skip straight into a portal.
  const enter = (role: Role) => {
    login(role);
    navigate(`/${role}`);
  };

  return (
    <div>
      {/* Hero */}
      <section className="relative overflow-hidden bg-gradient-to-b from-indigo-50/80 via-white to-slate-50">
        <div className="pointer-events-none absolute -top-40 right-0 h-96 w-96 rounded-full bg-indigo-200/40 blur-3xl" />
        <div className="relative mx-auto grid max-w-6xl items-center gap-12 px-4 py-16 md:py-24 lg:grid-cols-2">
          <div>
            <ServerStatusPill />
            <h1 className="mt-5 text-4xl font-extrabold tracking-tight text-slate-900 sm:text-5xl">
              Know a campaign's odds <span className="text-indigo-600">before it launches.</span>
            </h1>
            <p className="mt-5 max-w-xl text-lg text-slate-600">
              AI Venture Intelligence predicts whether a crowdfunding campaign will get funded, explains why,
              and shows which changes would raise its chances. It's trained on 331k real Kickstarter campaigns.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <button onClick={() => enter("founder")} className="rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-200 transition hover:bg-indigo-700">
                I'm a founder →
              </button>
              <button onClick={() => enter("investor")} className="rounded-xl bg-white px-6 py-3 text-sm font-semibold text-slate-800 ring-1 ring-slate-300 transition hover:ring-slate-400">
                I'm an investor
              </button>
            </div>
            <p className="mt-4 text-xs text-slate-400">No sign-up. Example campaigns are one click away.</p>
          </div>
          <PreviewCard />
        </div>
      </section>

      {/* Metrics */}
      <section className="border-y border-slate-200 bg-white">
        <div className="mx-auto grid max-w-6xl grid-cols-2 gap-6 px-4 py-10 lg:grid-cols-4">
          {METRICS.map((m) => (
            <div key={m.label}>
              <p className="text-3xl font-bold tracking-tight text-slate-900">{m.value}</p>
              <p className="mt-1 text-sm font-medium text-slate-700">{m.label}</p>
              <p className="text-xs text-slate-400">{m.hint}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="mx-auto max-w-6xl scroll-mt-20 px-4 py-20">
        <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">How it works</p>
        <h2 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">From campaign details to a decision in seconds</h2>
        <div className="mt-10 grid gap-6 md:grid-cols-3">
          {STEPS.map((s) => (
            <div key={s.n} className="rounded-2xl border border-slate-200 bg-white p-6">
              <span className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-600 text-sm font-bold text-white">{s.n}</span>
              <h3 className="mt-4 font-semibold text-slate-900">{s.title}</h3>
              <p className="mt-2 text-sm text-slate-500">{s.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="bg-white py-20">
        <div className="mx-auto max-w-6xl px-4">
          <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">What you get</p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">One analysis, many angles</h2>
          <div className="mt-10 grid gap-x-8 gap-y-10 sm:grid-cols-2 lg:grid-cols-3">
            {FEATURES.map((f) => (
              <div key={f.title}>
                <Icon>{f.icon}</Icon>
                <h3 className="mt-4 font-semibold text-slate-900">{f.title}</h3>
                <p className="mt-2 text-sm text-slate-500">{f.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Audiences */}
      <section className="mx-auto max-w-6xl px-4 py-20">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="rounded-2xl bg-indigo-600 p-8 text-white">
            <p className="text-xs font-semibold uppercase tracking-wider text-indigo-200">For founders</p>
            <h3 className="mt-2 text-2xl font-bold">Fix the campaign before you launch it</h3>
            <ul className="mt-5 space-y-2 text-sm text-indigo-100">
              <li>• Funding probability and what drives it</li>
              <li>• Live what-if simulator for goal, duration, launch day and title</li>
              <li>• Ranked improvements with the expected gain</li>
              <li>• A founder report you can save as PDF</li>
            </ul>
            <button onClick={() => enter("founder")} className="mt-7 rounded-xl bg-white px-5 py-2.5 text-sm font-semibold text-indigo-700 hover:bg-indigo-50">
              Open the founder portal
            </button>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white p-8">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">For investors</p>
            <h3 className="mt-2 text-2xl font-bold text-slate-900">Screen a campaign like a committee would</h3>
            <ul className="mt-5 space-y-2 text-sm text-slate-600">
              <li>• Verdict from six rule-based agents, with a risk veto</li>
              <li>• Overall confidence and each agent's reasoning</li>
              <li>• Risks, red flags and likely investor objections</li>
              <li>• A 10-section due-diligence report</li>
            </ul>
            <button onClick={() => enter("investor")} className="mt-7 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white hover:bg-slate-800">
              Open the investor portal
            </button>
          </div>
        </div>
      </section>

      {/* Honest limits */}
      <section className="bg-white py-20">
        <div className="mx-auto grid max-w-6xl gap-10 px-4 lg:grid-cols-[1fr_1.4fr]">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">Honest by design</p>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">What it won't pretend to know</h2>
            <p className="mt-4 text-slate-500">
              If the data can't support a number, the platform says so instead of making one up.
            </p>
          </div>
          <ul className="space-y-4">
            {LIMITS.map((l) => (
              <li key={l} className="flex gap-3 rounded-xl bg-amber-50 p-4 text-sm text-amber-900 ring-1 ring-amber-200">
                <span className="mt-0.5 font-bold">!</span>
                <span>{l}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      {/* Stack + CTA */}
      <section className="mx-auto max-w-6xl px-4 py-20 text-center">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Built with</p>
        <div className="mt-4 flex flex-wrap justify-center gap-2">
          {STACK.map((s) => (
            <span key={s} className="rounded-full bg-white px-3 py-1 text-sm text-slate-600 ring-1 ring-slate-200">{s}</span>
          ))}
        </div>
        <h2 className="mt-16 text-3xl font-bold tracking-tight text-slate-900">Try it on a campaign</h2>
        <p className="mt-3 text-slate-500">Pick an example or describe your own. Analysis takes a few seconds.</p>
        <div className="mt-8 flex flex-wrap justify-center gap-3">
          <button onClick={() => enter("founder")} className="rounded-xl bg-indigo-600 px-6 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700">
            Start as a founder
          </button>
          <button onClick={() => enter("investor")} className="rounded-xl bg-white px-6 py-3 text-sm font-semibold text-slate-800 ring-1 ring-slate-300 hover:ring-slate-400">
            Start as an investor
          </button>
        </div>
      </section>
    </div>
  );
}
