// Day 19 · Phase D — recommendations: readiness, confidence, and the prioritized improvement roadmap.
import type { RecommendationResponse } from "../../api/types";
import { HonestyNote, Stat } from "../ui";

export default function RecommendationPanel({ data }: { data: RecommendationResponse }) {
  const r = data.investment_readiness;
  const c = data.confidence;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Readiness stage" value={r.stage} />
        <Stat label="Readiness index" value={Math.round(r.readiness_index)} />
        <Stat label="Optimized prob." value={`${Math.round(r.optimized_probability * 100)}%`} hint={`from ${Math.round(r.funding_probability * 100)}%`} />
        <Stat label="Confidence" value={typeof c.score === "number" ? `${Math.round(c.score * 100)}%` : String(c.score)} />
      </div>

      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">Priority improvements</h4>
        <ol className="space-y-2">
          {data.priority_improvements.slice(0, 5).map((imp, i) => (
            <li key={i} className="flex gap-3 rounded-xl border border-slate-200 p-4 transition hover:border-indigo-200">
              <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">{i + 1}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-3">
                  <span className="text-sm font-semibold text-slate-800">{imp.action}</span>
                  <span className="shrink-0 rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-semibold uppercase text-indigo-700 ring-1 ring-indigo-100">
                    {imp.priority_label}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">{imp.reason}</p>
                {typeof imp.expected_impact_points === "number" && (
                  <span className="mt-2 inline-flex rounded-full bg-emerald-50 px-2 py-0.5 text-[11px] font-semibold text-emerald-700">
                    +{imp.expected_impact_points} pts expected
                  </span>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {data.final_strategy && (
        <p className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">
          <span className="font-semibold">Strategy:</span> {data.final_strategy}
        </p>
      )}
      <HonestyNote note={data.honesty_note} />
    </div>
  );
}
