// Day 19 · Investor Portal — evaluate ONE campaign (committee verdict, comparables, due-diligence
// report). Same submit → analyze flow as the Founder portal, investor framing. No startup catalog:
// the backend has no search/filter/rank-across-campaigns endpoint.
import { useState } from "react";
import type { CampaignRequest } from "../api/types";
import CampaignForm from "../components/CampaignForm";
import InvestorAnalysis from "../components/Investor/InvestorAnalysis";
import { Card } from "../components/ui";

export default function InvestorPortal() {
  const [campaign, setCampaign] = useState<CampaignRequest | null>(null);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h2 className="mb-1 text-2xl font-bold text-slate-900">Investor Portal</h2>
      <p className="mb-6 text-sm text-slate-500">
        Evaluate a campaign: committee verdict, confidence, comparable campaigns, and a due-diligence report.
      </p>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,26rem)_1fr]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <Card title="Evaluate a campaign">
            <CampaignForm onSubmit={setCampaign} />
          </Card>
        </div>

        <div>
          {campaign ? (
            <InvestorAnalysis campaign={campaign} />
          ) : (
            <Card>
              <p className="py-8 text-center text-sm text-slate-400">Submit a campaign to run due diligence.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
