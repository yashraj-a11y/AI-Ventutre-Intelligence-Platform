// Day 19 · Phase I — Investor analysis: the Day-12 committee verdict + overall confidence (/analyze),
// comparable campaigns (/retrieve, reusing the retrieval dashboard), and the due-diligence report
// (reusing the Phase-H ReportViewer with variant="investor"). There is NO catalog search/filter/rank —
// the backend has no catalog endpoint; the investor evaluates one campaign at a time.
import { api } from "../../api/client";
import type { AnalysisResponse, CampaignRequest } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import ReportViewer from "../Reports/ReportViewer";
import SimilarCampaigns from "../Retrieval/SimilarCampaigns";
import { Card, ErrorBox, Gauge, HonestyNote, Loading, SectionNav } from "../ui";

// Committee verdicts: Strong Invest / Proceed to Due Diligence / Monitor Progress / Do Not Invest.
// Order matters: "Do Not Invest" contains "invest", so the negative cases must be tested first.
function verdictTone(rec: string): { cls: string; hex: string } {
  const s = rec.toLowerCase();
  if (s.includes("do not") || s.includes("pass") || s.includes("decline") || s.includes("avoid")) return { cls: "bg-rose-50 text-rose-800 ring-rose-200", hex: "#f43f5e" };
  if (s.includes("monitor") || s.includes("caution")) return { cls: "bg-amber-50 text-amber-800 ring-amber-200", hex: "#f59e0b" };
  if (s.includes("strong") || s.includes("proceed") || s.includes("invest")) return { cls: "bg-emerald-50 text-emerald-800 ring-emerald-200", hex: "#10b981" };
  return { cls: "bg-slate-50 text-slate-700 ring-slate-200", hex: "#64748b" };
}

function VerdictPanel({ data }: { data: AnalysisResponse }) {
  const tone = verdictTone(data.committee_recommendation);
  return (
    <div className="space-y-4">
      <div className="flex flex-col items-center gap-6 sm:flex-row">
        <Gauge value={data.overall_confidence} color={tone.hex} label="confidence" />
        <div className="w-full flex-1 text-center sm:text-left">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">Committee verdict</p>
          <p className={`mt-2 inline-flex rounded-xl px-4 py-2 text-xl font-bold ring-1 ${tone.cls}`}>
            {data.committee_recommendation}
          </p>
          <p className="mt-3 text-sm text-slate-500">
            Six rule-based agents (market, founder, financial, competition, risk, investment) vote on the
            platform's analysis, with a risk veto. A deterministic consensus, not an LLM opinion.
          </p>
        </div>
      </div>
      <HonestyNote note={data.honesty_note} />
    </div>
  );
}

const SECTIONS = [
  { id: "verdict", label: "Verdict" },
  { id: "comparables", label: "Comparables" },
  { id: "report", label: "Due-diligence report" },
];

export default function InvestorAnalysis({ campaign }: { campaign: CampaignRequest }) {
  const an = useApi(api.analyze, campaign);

  return (
    <div>
      <SectionNav items={SECTIONS} />
      <div className="space-y-6">
        <Card id="verdict" title="Investment committee" subtitle="Verdict and confidence from six independent agents">
          {an.loading && <Loading label="Running the committee…" />}
          {an.error && <ErrorBox message={an.error} />}
          {an.data && <VerdictPanel data={an.data} />}
        </Card>

        <Card id="comparables" title="Comparable campaigns" subtitle="Real Kickstarter campaigns that look like this one">
          <SimilarCampaigns campaign={campaign} label="Finding comparables…" />
        </Card>

        <Card id="report">
          <ReportViewer campaign={campaign} variant="investor" />
        </Card>
      </div>
    </div>
  );
}
