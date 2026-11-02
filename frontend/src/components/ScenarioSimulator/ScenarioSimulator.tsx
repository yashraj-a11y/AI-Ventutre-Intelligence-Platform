// Day 19 · Phase F — interactive what-if over the REAL controllable levers: funding goal, campaign
// duration, launch weekday, and a title number. Moving a control rebuilds the campaign and re-calls
// /predict live (debounced) — a genuine slider→prediction loop. There are NO MRR/retention/team/burn
// sliders: those features don't exist in the data. We also show the backend's own best scenarios.
import { useMemo, useState } from "react";
import { api } from "../../api/client";
import type { CampaignRequest } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { useDebounced } from "../../hooks/useDebounced";
import { ErrorBox, HonestyNote, Loading } from "../ui";

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];

interface RankedScenario { rank: number; name: string; probability: number; gain: number; crosses_bar: boolean; stage: string }
interface ScenarioResult { ranked_scenarios?: RankedScenario[]; best_scenario?: { name: string; probability: number }; honesty_note?: string }

function parse(dt: string): Date { return new Date(dt.replace(" ", "T")); }
function fmt(d: Date): string {
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:00`;
}
function withNumber(name: string, on: boolean): string {
  const stripped = name.replace(/\s+\d+$/, "");
  return on ? (/\d/.test(name) ? name : `${stripped} 2`) : stripped;
}

export default function ScenarioSimulator({ campaign, baseline }: { campaign: CampaignRequest; baseline?: number }) {
  const base = useMemo(() => {
    const launched = parse(campaign.launched);
    const deadline = parse(campaign.deadline);
    const days = Math.max(1, Math.round((deadline.getTime() - launched.getTime()) / 86_400_000));
    return { launched, days, weekday: launched.getDay() };
  }, [campaign]);

  const [goal, setGoal] = useState(campaign.usd_goal_real);
  const [days, setDays] = useState(base.days);
  const [weekday, setWeekday] = useState(base.weekday);
  const [addNumber, setAddNumber] = useState(/\d/.test(campaign.name));

  const modified = useMemo<CampaignRequest>(() => {
    const launched = new Date(base.launched);
    launched.setDate(launched.getDate() + (weekday - base.weekday)); // shift to chosen weekday
    const deadline = new Date(launched);
    deadline.setDate(deadline.getDate() + days);
    return {
      ...campaign,
      name: withNumber(campaign.name, addNumber),
      usd_goal_real: Math.round(goal),
      launched: fmt(launched),
      deadline: fmt(deadline),
    };
  }, [campaign, base, goal, days, weekday, addNumber]);

  const debounced = useDebounced(modified);
  const sim = useApi(api.predict, debounced);          // live re-prediction of the modified campaign
  const scen = useApi(api.scenario, campaign);         // the backend's own scenario exploration (once)

  const current = sim.data?.funding_probability;
  const result = scen.data?.result as unknown as ScenarioResult | undefined;

  const slider = "w-full accent-indigo-600";
  const label = "flex items-center justify-between text-xs font-medium text-slate-600";

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <div className={label}><span>Funding goal</span><span className="text-slate-900">${Math.round(goal).toLocaleString()}</span></div>
          <input type="range" min={1000} max={100000} step={1000} value={goal} onChange={(e) => setGoal(Number(e.target.value))} className={slider} />
        </div>
        <div>
          <div className={label}><span>Campaign duration</span><span className="text-slate-900">{days} days</span></div>
          <input type="range" min={1} max={60} step={1} value={days} onChange={(e) => setDays(Number(e.target.value))} className={slider} />
        </div>
        <div>
          <div className={label}><span>Launch weekday</span><span className="text-slate-900">{WEEKDAYS[weekday]}</span></div>
          <select value={weekday} onChange={(e) => setWeekday(Number(e.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-1.5 text-sm">
            {WEEKDAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
          </select>
        </div>
        <label className="flex items-center gap-2 self-end text-sm text-slate-700">
          <input type="checkbox" checked={addNumber} onChange={(e) => setAddNumber(e.target.checked)} className="accent-indigo-600" />
          Add a number to the title
        </label>
      </div>

      {/* baseline vs simulated */}
      <div className="rounded-lg bg-slate-50 p-4">
        <div className="flex items-center justify-around text-center">
          <div>
            <div className="text-xs text-slate-500">Baseline</div>
            <div className="text-xl font-bold text-slate-700">{baseline != null ? `${Math.round(baseline * 100)}%` : "—"}</div>
          </div>
          <div className="text-slate-400">→</div>
          <div>
            <div className="text-xs text-slate-500">Simulated</div>
            <div className="text-2xl font-bold text-indigo-600">
              {sim.loading ? "…" : current != null ? `${Math.round(current * 100)}%` : "—"}
            </div>
          </div>
          {baseline != null && current != null && (
            <div>
              <div className="text-xs text-slate-500">Change</div>
              <div className={`text-xl font-bold ${current >= baseline ? "text-emerald-600" : "text-rose-600"}`}>
                {current >= baseline ? "+" : ""}{Math.round((current - baseline) * 100)} pts
              </div>
            </div>
          )}
        </div>
        {sim.error && <ErrorBox message={sim.error} />}
      </div>

      {/* the backend's own best scenarios */}
      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Suggested scenarios (model-explored)</h4>
        {scen.loading && <Loading label="Exploring scenarios…" />}
        {scen.error && <ErrorBox message={scen.error} />}
        <ul className="space-y-2">
          {result?.ranked_scenarios?.slice(0, 4).map((s) => (
            <li key={s.rank} className="flex items-center justify-between rounded-lg border border-slate-200 p-2.5 text-sm">
              <span className="text-slate-700">{s.name}</span>
              <span className={`ml-3 shrink-0 font-semibold ${s.crosses_bar ? "text-emerald-600" : "text-slate-500"}`}>
                {Math.round(s.probability * 100)}%{s.crosses_bar ? " ✓" : ""}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-slate-400">
        Levers are the real controllable features (goal, duration, timing, title). Revenue, retention,
        team, and burn are deferred — the dataset has no such signals.
      </p>
      <HonestyNote note={result?.honesty_note} />
    </div>
  );
}
