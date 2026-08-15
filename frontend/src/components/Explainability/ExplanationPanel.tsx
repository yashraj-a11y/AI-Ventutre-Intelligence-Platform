// Day 19 · Phase E — SHAP explanation: a diverging contribution bar chart (the standard local-SHAP
// view) in log-odds margin space, plus a waterfall (base value -> each contribution -> "all other
// features" -> this campaign's margin) when /explain returns base_value + margin. Green = up, red = down.
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Contribution, ExplanationResponse } from "../../api/types";
import { HonestyNote } from "../ui";

function shortLabel(c: Contribution): string {
  const s = c.descriptor || c.feature;
  return s.length > 32 ? `${s.slice(0, 31)}…` : s;
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

export default function ExplanationPanel({ data }: { data: ExplanationResponse }) {
  const rows = [...data.top_negative, ...data.top_positive]
    .map((c) => ({ label: shortLabel(c), value: Number(c.contribution) }))
    .sort((a, b) => a.value - b.value);

  const steps = waterfall(data);
  const topUp = data.top_positive[0];
  const topDown = data.top_negative[0];

  return (
    <div>
      <p className="mb-3 text-sm text-slate-600">
        {topUp && <>Biggest lift: <span className="font-medium text-emerald-700">{topUp.descriptor || topUp.feature}</span>. </>}
        {topDown && <>Biggest drag: <span className="font-medium text-rose-700">{topDown.descriptor || topDown.feature}</span>.</>}
      </p>

      <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 30)}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(2)} />
          <YAxis type="category" dataKey="label" width={190} tick={{ fontSize: 11 }} />
          <ReferenceLine x={0} stroke="#94a3b8" />
          <Tooltip formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} cursor={{ fill: "#f1f5f9" }} />
          <Bar dataKey="value" radius={2}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.value >= 0 ? "#10b981" : "#f43f5e"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-2 flex justify-center gap-6 text-xs text-slate-500">
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm bg-emerald-500" /> pushes funding up</span>
        <span className="flex items-center gap-1"><span className="h-2 w-3 rounded-sm bg-rose-500" /> pushes funding down</span>
      </div>

      {steps && (
        <div className="mt-6">
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Waterfall — how the score builds up (log-odds)</h4>
          <ResponsiveContainer width="100%" height={Math.max(240, steps.length * 30)}>
            <BarChart data={steps} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 8 }}>
              <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(1)} />
              <YAxis type="category" dataKey="label" width={190} tick={{ fontSize: 11 }} />
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
