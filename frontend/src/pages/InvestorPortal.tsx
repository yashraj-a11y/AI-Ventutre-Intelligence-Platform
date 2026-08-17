// Day 19 · Investor Portal — evaluate ONE campaign (committee verdict, comparables, due-diligence
// report). Same submit → analyze flow as the Founder portal, investor framing. No startup catalog:
// the backend has no search/filter/rank-across-campaigns endpoint.
import { useRef, useState } from "react";
import type { CampaignRequest } from "../api/types";
import CampaignForm from "../components/CampaignForm";
import InvestorAnalysis from "../components/Investor/InvestorAnalysis";
import { Card, EmptyState, PageHeader, ServerWakeBanner } from "../components/ui";

export default function InvestorPortal() {
  const [campaign, setCampaign] = useState<CampaignRequest | null>(null);
  const results = useRef<HTMLDivElement>(null);
  // Below the lg breakpoint the results sit under the form, out of view: bring them up on submit.
  const submit = (c: CampaignRequest) => {
    setCampaign(c);
    if (window.matchMedia("(max-width: 1023px)").matches)
      setTimeout(() => results.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-10">
      <PageHeader eyebrow="Investor portal" title="Should this campaign get your backing?">
        Run a six-agent committee over the campaign for a verdict and confidence, then read the full
        due-diligence report.
      </PageHeader>
      <ServerWakeBanner />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <Card title="Campaign to evaluate">
            <CampaignForm onSubmit={submit} submitLabel="Run due diligence" />
          </Card>
        </div>

        <div ref={results} className="scroll-mt-20">
          {campaign ? (
            <InvestorAnalysis campaign={campaign} />
          ) : (
            <EmptyState
              title="Your due diligence will appear here"
              items={["Committee verdict with overall confidence", "Each agent's vote and reasoning", "Risks, red flags and investor objections", "10-section due-diligence report"]}
            />
          )}
        </div>
      </div>
    </div>
  );
}
