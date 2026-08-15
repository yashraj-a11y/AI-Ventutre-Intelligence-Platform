// Day 19 · Founder Portal — submit a campaign, then render its live AI analysis (Phases D–H plug in).
import { useState } from "react";
import type { CampaignRequest } from "../api/types";
import CampaignForm from "../components/CampaignForm";
import FounderAnalysis from "../components/FounderAnalysis";
import { Card } from "../components/ui";

export default function FounderPortal() {
  const [campaign, setCampaign] = useState<CampaignRequest | null>(null);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h2 className="mb-1 text-2xl font-bold text-slate-900">Founder Portal</h2>
      <p className="mb-6 text-sm text-slate-500">
        Submit a campaign for a funding prediction, explanation, recommendations, and scenarios.
      </p>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
        <div className="lg:sticky lg:top-20 lg:self-start">
          <Card title="Submit your campaign">
            <CampaignForm onSubmit={setCampaign} />
          </Card>
        </div>

        <div>
          {campaign ? (
            <FounderAnalysis campaign={campaign} />
          ) : (
            <Card>
              <p className="py-8 text-center text-sm text-slate-400">
                Submit a campaign to see its AI analysis.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
