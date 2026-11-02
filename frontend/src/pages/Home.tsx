// Day 19 · Phase C — landing / role chooser.
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-16 text-center">
      <h1 className="text-4xl font-bold tracking-tight text-slate-900">AI Venture Intelligence</h1>
      <p className="mx-auto mt-4 max-w-xl text-slate-600">
        Submit a campaign and get a calibrated funding prediction, a SHAP explanation, similar
        historical campaigns, recommendations, and a full report — every number sourced from the model,
        with the honesty boundary shown, never faked.
      </p>
      <div className="mt-8 flex justify-center gap-4">
        <Link to="/founder" className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-indigo-700">
          Founder Portal
        </Link>
        <Link to="/investor" className="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50">
          Investor Portal
        </Link>
      </div>
    </div>
  );
}
