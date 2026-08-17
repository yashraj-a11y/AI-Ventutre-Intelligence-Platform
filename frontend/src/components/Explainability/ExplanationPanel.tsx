// Day 19 · Phase E — SHAP explanation: a diverging contribution bar chart (the standard local-SHAP
// view) in log-odds margin space, plus a waterfall (base value -> each contribution -> "all other
// features" -> this campaign's margin) when /explain returns base_value + margin. Green = up, red = down.
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useState } from "react";
import type { Contribution, ExplanationResponse } from "../../api/types";
import { useIsPhone } from "../../hooks/useMediaQuery";
import { featureLabel } from "../../lib/featureLabels";
import { HonestyNote } from "../ui";

function shortLabel(c: Contribution): string {
  const s = featureLabel(c.descriptor || c.feature);
  return s.length > 30 ? `${s.slice(0, 29)}…` : s;
}

interface Step { label: string; range: [number, number]; value: number; kind: "base" | "up" | "down" | "total" }

// base -> contributions (largest |effect| first) -> remainder (features not in the top lists) -> margin.
function waterfall(data: ExplanationResponse): Step[] | null {
  if (data.base_value == null || data.margin == null) return null;
  const steps: Step[] = [{ label: "Baseline (avg. campaign)", range: [0, data.base_value], value: data.base_value, kind: "base" }];
  let run = data.base_value;
  const shown = [...data.top_positive, ...data.top_negative]
    .map((c) => ({ label: shortLabel(c), v: Number(c.contribution) }))
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  for (const c of shown) {
    steps.push({ label: c.label, range: [Math.min(run, run + c.v), Math.max(run, run + c.v)], value: c.v, kind: c.v >= 0 ? "up" : "down" });
    run += c.v;
  }
  const rest = data.margin - run;
  if (Math.abs(rest) > 1e-4) {
    steps.push({ label: "All other features", range: [Math.min(run, data.margin), Math.max(run, data.margin)], value: rest, kind: rest >= 0 ? "up" : "down" });
  }
  steps.push({ label: "This campaign (log-odds)", range: [0, data.margin], value: data.margin, kind: "total" });
  return steps;
}

const FILL = { base: "#64748b", total: "#4f46e5", up: "#10b981", down: "#f43f5e" } as const;

/** Phone layout of the contribution chart: label above a diverging bar, so neither gets squeezed. */
function ContributionList({ rows }: { rows: { label: string; value: number }[] }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const sorted = [...rows].sort((a, b) => b.value - a.value);
  return (
    <ul className="space-y-2.5">
      {sorted.map((r, i) => {
        const w = `${(Math.abs(r.value) / max) * 50}%`;
        const up = r.value >= 0;
        return (
          <li key={`${r.label}-${i}`}>
            <div className="flex items-baseline justify-between gap-3 text-xs">
              <span className="text-slate-700">{r.label}</span>
              <span className={`shrink-0 font-semibold tabular-nums ${up ? "text-emerald-700" : "text-rose-700"}`}>
                {up ? "+" : ""}{r.value.toFixed(2)}
              </span>
            </div>
            <div className="relative mt-1 h-2 rounded-full bg-slate-100">
              <span className="absolute inset-y-0 left-1/2 w-px bg-slate-300" />
              <span
                className={`absolute inset-y-0 rounded-full ${up ? "bg-emerald-500" : "bg-rose-500"}`}
                style={up ? { left: "50%", width: w } : { right: "50%", width: w }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

export default function ExplanationPanel({ data }: { data: ExplanationResponse }) {
  const rows = [...data.top_negative, ...data.top_positive]
    .map((c) => ({ label: shortLabel(c), value: Number(c.contribution) }))
    .sort((a, b) => a.value - b.value);

  const steps = waterfall(data);
  const topUp = data.top_positive[0];
  const topDown = data.top_negative[0];
  const [showWaterfall, setShowWaterfall] = useState(false);
  const isPhone = useIsPhone();

  return (
    <div>
      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        {topUp && (
          <div className="rounded-xl bg-emerald-50 p-3 ring-1 ring-emerald-100">
            <p className="text-xs font-medium text-emerald-700">Biggest lift</p>
            <p className="mt-0.5 font-semibold text-emerald-900">{featureLabel(topUp.descriptor || topUp.feature)}</p>
          </div>
        )}
        {topDown && (
          <div className="rounded-xl bg-rose-50 p-3 ring-1 ring-rose-100">
            <p className="text-xs font-medium text-rose-700">Biggest drag</p>
            <p className="mt-0.5 font-semibold text-rose-900">{featureLabel(topDown.descriptor || topDown.feature)}</p>
          </div>
        )}
      </div>

      {isPhone ? <ContributionList rows={rows} /> : (
      <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 30)}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={190} tick={{ fontSize: 12, fill: "#334155" }} axisLine={false} tickLine={false} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} cursor={{ fill: "#f1f5f9" }} />
          <Bar dataKey="value" radius={2}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.value >= 0 ? "#10b981" : "#f43f5e"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      )}

      <div className="mt-2 flex justify-center gap-6 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm bg-emerald-500" /> pushes funding up</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm bg-rose-500" /> pushes funding down</span>
      </div>

      {steps && (
        <button
          onClick={() => setShowWaterfall((s) => !s)}
          className="no-print mt-5 text-sm font-medium text-indigo-600 hover:text-indigo-800"
          aria-expanded={showWaterfall}
        >
          {showWaterfall ? "Hide" : "Show"} how the score builds up (waterfall) {showWaterfall ? "▴" : "▾"}
        </button>
      )}

      {steps && showWaterfall && (
        <div className="mt-4">
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Waterfall — how the score builds up (log-odds)</h4>
          <ResponsiveContainer width="100%" height={Math.max(240, steps.length * 30)}>
            <BarChart data={steps} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} axisLine={false} tickLine={false} tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis type="category" dataKey="label" width={isPhone ? 120 : 190} tick={{ fontSize: isPhone ? 10 : 12, fill: "#334155" }} axisLine={false} tickLine={false} />
              <ReferenceLine x={0} stroke="#94a3b8" />
              <Tooltip
                formatter={(_v, _n, item) => {
                  const v = (item.payload as Step).value;
                  return `${v >= 0 ? "+" : ""}${v.toFixed(3)}`;
                }}
                cursor={{ fill: "#f1f5f9" }}
              />
              <Bar dataKey="range" radius={2}>
                {steps.map((st, i) => (
                  <Cell key={i} fill={FILL[st.kind]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="text-center text-xs text-slate-400">
            Margin {data.margin?.toFixed(2)} = baseline {data.base_value?.toFixed(2)} + all feature contributions; the
            calibrated probability is a monotone function of this margin.
          </p>
        </div>
      )}

      <HonestyNote note={data.honesty_note} />
    </div>
  );
}
