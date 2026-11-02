// Day 19 · Phase I — Investor analysis: the Day-12 committee verdict + overall confidence (/analyze),
// comparable campaigns (/retrieve, reusing the retrieval dashboard), and the due-diligence report
// (reusing the Phase-H ReportViewer with variant="investor"). There is NO catalog search/filter/rank —
// the backend has no catalog endpoint; the investor evaluates one campaign at a time.
import { api } from "../../api/client";
import type { AnalysisResponse, CampaignRequest } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import ReportViewer from "../Reports/ReportViewer";
import RetrievalDashboard from "../Retrieval/RetrievalDashboard";
import { Card, ErrorBox, HonestyNote, Loading, Stat } from "../ui";

function verdictTone(rec: string): string {
  const s = rec.toLowerCase();
  if (s.includes("caution")) return "bg-amber-100 text-amber-800";
  if (s.includes("proceed") || s.includes("invest")) return "bg-emerald-100 text-emerald-800";
  if (s.includes("pass") || s.includes("decline") || s.includes("avoid")) return "bg-rose-100 text-rose-800";
  return "bg-slate-100 text-slate-700";
}

function VerdictPanel({ data }: { data: AnalysisResponse }) {
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <span className={`rounded-lg px-4 py-2 text-lg font-bold ${verdictTone(data.committee_recommendation)}`}>
          {data.committee_recommendation}
        </span>
        <div className="min-w-32 flex-1">
          <Stat label="Overall confidence" value={`${Math.round(data.overall_confidence * 100)}%`} />
        </div>
      </div>
      <p className="text-xs text-slate-400">
        Verdict from the Day-12 multi-agent committee over the Day-11 unified intelligence — a
        deterministic consensus, not an LLM opinion.
      </p>
      <HonestyNote note={data.honesty_note} />
    </div>
  );
}

export default function InvestorAnalysis({ campaign }: { campaign: CampaignRequest }) {
  const an = useApi(api.analyze, campaign);
  const retr = useApi(api.retrieve, campaign);

  return (
    <div className="space-y-6">
      <Card title="Committee verdict">
        {an.loading && <Loading label="Running the committee…" />}
        {an.error && <ErrorBox message={an.error} />}
        {an.data && <VerdictPanel data={an.data} />}
      </Card>

      <Card title="Comparable campaigns">
        {retr.loading && <Loading label="Finding comparables…" />}
        {retr.error && <ErrorBox message={retr.error} />}
        {retr.data && <RetrievalDashboard data={retr.data} />}
      </Card>

      <Card title="Investor Due-Diligence Report">
        <ReportViewer campaign={campaign} variant="investor" />
      </Card>
    </div>
  );
}
