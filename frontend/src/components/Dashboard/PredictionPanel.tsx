// Day 19 · Phase D — prediction + the REAL venture scores (all functions of the calibrated p).
import type { PredictionResponse } from "../../api/types";
import { Stat } from "../ui";

export default function PredictionPanel({ data }: { data: PredictionResponse }) {
  const p = data.funding_probability;
  const thr = data.decision_threshold;
  const funded = data.predicted_label === "successful";
  const pct = Math.round(p * 100);

  return (
    <div>
      {/* probability bar with the decision threshold marker */}
      <div className="mb-4">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-sm font-medium text-slate-600">Funding probability</span>
          <span className={`text-2xl font-bold ${funded ? "text-emerald-600" : "text-rose-600"}`}>{pct}%</span>
        </div>
        <div className="relative h-3 w-full overflow-hidden rounded-full bg-slate-200">
          <div className={`h-full ${funded ? "bg-emerald-500" : "bg-rose-500"}`} style={{ width: `${pct}%` }} />
          <div className="absolute top-0 h-3 w-0.5 bg-slate-800" style={{ left: `${thr * 100}%` }} title={`threshold ${thr}`} />
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-slate-400">
          <span>0%</span>
          <span>decision threshold {Math.round(thr * 100)}%</span>
          <span>100%</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Predicted" value={funded ? "Funded" : "Not funded"} />
        <Stat label="Startup score" value={Math.round(100 * p)} hint="100 · probability" />
        <Stat label="Failure risk" value={`${Math.round((1 - p) * 100)}%`} hint="1 − probability" />
        <Stat label="Investment score" value={Math.round(100 * p)} hint="100 · probability" />
      </div>
    </div>
  );
}
