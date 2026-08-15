// Day 19 · Phase D — prediction + the REAL venture scores (all functions of the calibrated p).
import type { PredictionResponse } from "../../api/types";
import { Stat } from "../ui";

export default function PredictionPanel({ data }: { data: PredictionResponse }) {
  const p = data.funding_probability;
  const thr = data.decision_threshold;
  const aboveThreshold = data.predicted_label === "successful";
  const pct = Math.round(p * 100);

  // "predicted_label" only says which side of the decision threshold p falls on. The threshold (tuned for
  // recall) is below 50%, so a campaign can be "above threshold" and still be more likely to fail than fund.
  // Colour and wording keep those three zones apart instead of calling a 40% campaign "Funded".
  const zone = !aboveThreshold ? "below" : p >= 0.5 ? "likely" : "borderline";
  const tone = {
    below: { text: "text-rose-600", bar: "bg-rose-500" },
    borderline: { text: "text-amber-600", bar: "bg-amber-500" },
    likely: { text: "text-emerald-600", bar: "bg-emerald-500" },
  }[zone];
  const caption = {
    below: `Below the decision threshold (${Math.round(thr * 100)}%): the model does not flag this campaign.`,
    borderline: `Above the decision threshold (${Math.round(thr * 100)}%), which is tuned to catch funded campaigns, but under 50% — more likely to fail than to fund.`,
    likely: `Above both the decision threshold (${Math.round(thr * 100)}%) and 50%.`,
  }[zone];

  return (
    <div>
      {/* probability bar with the decision threshold marker */}
      <div className="mb-4">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-sm font-medium text-slate-600">Funding probability</span>
          <span className={`text-2xl font-bold ${tone.text}`}>{pct}%</span>
        </div>
        <div className="relative h-3 w-full overflow-hidden rounded-full bg-slate-200">
          <div className={`h-full ${tone.bar}`} style={{ width: `${pct}%` }} />
          <div className="absolute top-0 h-3 w-0.5 bg-slate-800" style={{ left: `${thr * 100}%` }} title={`threshold ${thr}`} />
        </div>
        <div className="mt-1 flex justify-between text-[10px] text-slate-400">
          <span>0%</span>
          <span>decision threshold {Math.round(thr * 100)}%</span>
          <span>100%</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Decision threshold" value={aboveThreshold ? "Above" : "Below"} hint={`cut-off ${Math.round(thr * 100)}%`} />
        <Stat label="Startup score" value={Math.round(100 * p)} hint="100 · probability" />
        <Stat label="Failure risk" value={`${Math.round((1 - p) * 100)}%`} hint="1 − probability" />
        <Stat label="Investment score" value={Math.round(100 * p)} hint="100 · probability" />
      </div>
      <p className="mt-3 text-xs text-slate-500">{caption}</p>
    </div>
  );
}
