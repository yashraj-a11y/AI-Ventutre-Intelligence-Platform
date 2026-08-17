// Day 19 · Phase D — prediction + the REAL venture scores (all functions of the calibrated p).
import type { PredictionResponse } from "../../api/types";
import { Gauge, Stat } from "../ui";

export default function PredictionPanel({ data }: { data: PredictionResponse }) {
  const p = data.funding_probability;
  const thr = data.decision_threshold;
  const aboveThreshold = data.predicted_label === "successful";
  const pct = Math.round(p * 100);
  const thrPct = Math.round(thr * 100);

  // "predicted_label" only says which side of the decision threshold p falls on. The threshold (tuned for
  // recall) is below 50%, so a campaign can be "above threshold" and still be more likely to fail than fund.
  // Colour and wording keep those three zones apart instead of calling a 40% campaign "Funded".
  const zone = !aboveThreshold ? "below" : p >= 0.5 ? "likely" : "borderline";
  const tone = {
    below: { hex: "#f43f5e", bar: "bg-rose-500", badge: "bg-rose-50 text-rose-700 ring-rose-200", title: "Unlikely to fund" },
    borderline: { hex: "#f59e0b", bar: "bg-amber-500", badge: "bg-amber-50 text-amber-800 ring-amber-200", title: "Borderline" },
    likely: { hex: "#10b981", bar: "bg-emerald-500", badge: "bg-emerald-50 text-emerald-700 ring-emerald-200", title: "Likely to fund" },
  }[zone];
  const caption = {
    below: `Below the ${thrPct}% decision threshold: the model does not flag this campaign as likely to fund.`,
    borderline: `Above the ${thrPct}% decision threshold, which is tuned to catch funded campaigns, but under 50%. It is still more likely to fail than to fund.`,
    likely: `Above both the ${thrPct}% decision threshold and 50%.`,
  }[zone];

  return (
    <div>
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-center">
        <Gauge value={p} color={tone.hex} label="to fund" />
        <div className="w-full flex-1">
          <span className={`inline-flex rounded-full px-3 py-1 text-sm font-semibold ring-1 ${tone.badge}`}>{tone.title}</span>
          <p className="mt-3 text-sm text-slate-600">{caption}</p>

          {/* probability bar with the decision threshold marker */}
          <div className="relative mt-4 h-2.5 w-full rounded-full bg-slate-100">
            <div className={`h-full rounded-full ${tone.bar}`} style={{ width: `${pct}%` }} />
            <div className="absolute -top-1 h-4.5 w-0.5 rounded bg-slate-800" style={{ left: `${thr * 100}%` }} title={`decision threshold ${thr}`} />
          </div>
          <div className="relative mt-1.5 h-4 text-[11px] text-slate-400">
            <span className="absolute left-0">0%</span>
            <span className="absolute -translate-x-1/2 font-medium text-slate-600" style={{ left: `${thr * 100}%` }}>threshold {thrPct}%</span>
            <span className="absolute right-0">100%</span>
          </div>
        </div>
      </div>

      <div className="mt-6 grid grid-cols-3 gap-3">
        <Stat label="vs. threshold" value={aboveThreshold ? "Above" : "Below"} hint={`cut-off ${thrPct}%`} />
        <Stat label="Failure risk" value={`${Math.round((1 - p) * 100)}%`} hint="1 − probability" />
        <Stat label="Startup / investment score" value={`${pct}/100`} hint="both = 100 × probability" />
      </div>
    </div>
  );
}
