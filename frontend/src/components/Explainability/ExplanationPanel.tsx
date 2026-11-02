// Day 19 · Phase E — SHAP explanation: a diverging contribution bar chart (the standard local-SHAP
// view) in log-odds margin space, plus the honesty note. Green = pushed funding up, red = down.
import {
  Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Contribution, ExplanationResponse } from "../../api/types";
import { HonestyNote } from "../ui";

function shortLabel(c: Contribution): string {
  const s = c.descriptor || c.feature;
  return s.length > 32 ? `${s.slice(0, 31)}…` : s;
}

export default function ExplanationPanel({ data }: { data: ExplanationResponse }) {
  const rows = [...data.top_negative, ...data.top_positive]
    .map((c) => ({ label: shortLabel(c), value: Number(c.contribution) }))
    .sort((a, b) => a.value - b.value);

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

      <HonestyNote note={data.honesty_note} />
    </div>
  );
}
