// Day 19 · Founder Portal — submit a campaign, then render its live AI analysis (Phases D–H plug in).
import { useRef, useState } from "react";
import type { CampaignRequest } from "../api/types";
import CampaignForm from "../components/CampaignForm";
import FounderAnalysis from "../components/FounderAnalysis";
import { Card, EmptyState, PageHeader, ServerWakeBanner } from "../components/ui";

export default function FounderPortal() {
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
      <PageHeader eyebrow="Founder portal" title="How likely is your campaign to get funded?">
        Get a funding probability, the reasons behind it, a live what-if simulator, ranked improvements and a
        full founder report.
      </PageHeader>
      <ServerWakeBanner />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <Card title="Your campaign">
            <CampaignForm onSubmit={submit} />
          </Card>
        </div>

        <div ref={results} className="scroll-mt-20">
          {campaign ? (
            <FounderAnalysis campaign={campaign} />
          ) : (
            <EmptyState
              title="Your analysis will appear here"
              items={["Funding probability vs. the decision threshold", "What pushes the score up or down", "What-if simulator and ranked improvements", "Founder report, printable as PDF"]}
            />
          )}
        </div>
      </div>
    </div>
  );
}
