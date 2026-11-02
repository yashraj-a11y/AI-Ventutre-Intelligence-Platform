// Day 19 · Phase G — historical evidence: the similar successful / failed campaigns from /retrieve,
// plus a benchmark computed CLIENT-SIDE from the returned neighbors (funded rate, count) — honest
// aggregates of real data, not a fabricated "industry benchmark" (the /retrieve response has none).
import type { RetrievalResponse, SimilarCampaign } from "../../api/types";
import { HonestyNote, Stat } from "../ui";

function SimilarTable({ rows, tone }: { rows: SimilarCampaign[]; tone: "success" | "fail" }) {
  const head = tone === "success" ? "text-emerald-700" : "text-rose-700";
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className={`border-b border-slate-200 text-xs uppercase tracking-wide ${head}`}>
            <th className="py-2 pr-3">Campaign</th>
            <th className="py-2 pr-3">Category</th>
            <th className="py-2 pr-3 text-right">Goal</th>
            <th className="py-2 pr-3 text-right">Similarity</th>
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 5).map((s) => (
            <tr key={s.campaign_id} className="border-b border-slate-100 align-top">
              <td className="py-2 pr-3">
                <div className="font-medium text-slate-800">{s.title}</div>
                {s.reason && <div className="text-xs text-slate-400">{s.reason}</div>}
              </td>
              <td className="py-2 pr-3 text-slate-600">{s.main_category ?? s.category ?? "—"}</td>
              <td className="py-2 pr-3 text-right text-slate-600">
                {typeof s.usd_goal_real === "number" ? `$${Math.round(s.usd_goal_real).toLocaleString()}` : "—"}
              </td>
              <td className="py-2 pr-3 text-right font-medium text-slate-700">{Math.round(s.similarity * 100)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RetrievalDashboard({ data }: { data: RetrievalResponse }) {
  const neighbors = data.similar ?? [];
  const funded = neighbors.filter((s) => s.outcome === "successful" || s.target === 1).length;
  const fundedRate = neighbors.length ? Math.round((funded / neighbors.length) * 100) : 0;

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Neighbors" value={neighbors.length} />
        <Stat label="Funded (of neighbors)" value={`${fundedRate}%`} hint={`${funded}/${neighbors.length}`} />
        <Stat label="Failed (of neighbors)" value={`${100 - fundedRate}%`} />
      </div>

      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-emerald-700">Similar campaigns that funded</h4>
        {data.similar_successful.length ? <SimilarTable rows={data.similar_successful} tone="success" />
          : <p className="text-sm text-slate-400">No similar funded campaigns retrieved.</p>}
      </div>

      <div>
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-rose-700">Similar campaigns that failed</h4>
        {data.similar_failed.length ? <SimilarTable rows={data.similar_failed} tone="fail" />
          : <p className="text-sm text-slate-400">No similar failed campaigns retrieved.</p>}
      </div>

      <p className="text-xs text-slate-400">
        Benchmarks are computed from the retrieved neighbors (real campaigns), not a fabricated industry
        index. Founder / revenue / market comparables are deferred — the knowledge base has none.
      </p>
      <HonestyNote note={data.honesty_note} />
    </div>
  );
}
