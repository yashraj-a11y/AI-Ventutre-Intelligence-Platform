// Day 19 · Phase D — the honesty boundary, made visible. The platform's VentureScoreGenerator
// DEFERS these scores (returns value=None) because the Kickstarter dataset has no basis for them.
// We render them as greyed "deferred" cards rather than fabricating gauges — a dashboard that invented
// a "Founder Strength: 78" would be lying with a nicer font.
const DEFERRED = [
  "Business-Model Score",
  "Founder Strength",
  "Scalability Score",
  "Market Fit",
  "Growth Potential",
  "Market Size (TAM / SAM)",
];

export default function DeferredScores() {
  return (
    <div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {DEFERRED.map((name) => (
          <div key={name} className="rounded-lg border border-dashed border-slate-300 bg-slate-50 p-3 text-center">
            <div className="text-lg font-semibold text-slate-300">—</div>
            <div className="mt-1 text-xs font-medium text-slate-500">{name}</div>
            <div className="mt-0.5 text-[10px] uppercase tracking-wide text-slate-400">deferred</div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-400">
        These have no data basis in the Kickstarter dataset (no founder / team / revenue / market signals),
        so the platform defers them rather than fabricating values. (A category-level opportunity signal
        from Kickstarter history <em>is</em> computed — see the report's Market Analysis; market size is what is deferred.)
      </p>
    </div>
  );
}
