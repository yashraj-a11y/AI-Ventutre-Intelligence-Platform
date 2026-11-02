// Day 19 · Phase D — Founder analysis orchestrator: /predict + /recommend, with loading/error states.
import { lazy, Suspense } from "react";
import { api } from "../api/client";
import type { CampaignRequest } from "../api/types";
import { useApi } from "../hooks/useApi";
import DeferredScores from "./Dashboard/DeferredScores";
import PredictionPanel from "./Dashboard/PredictionPanel";
import RecommendationPanel from "./Dashboard/RecommendationPanel";
import ReportViewer from "./Reports/ReportViewer";
import RetrievalDashboard from "./Retrieval/RetrievalDashboard";
import ScenarioSimulator from "./ScenarioSimulator/ScenarioSimulator";
import { Card, ErrorBox, Loading } from "./ui";

// Recharts is heavy — code-split the chart panel into its own chunk, loaded only when shown.
const ExplanationPanel = lazy(() => import("./Explainability/ExplanationPanel"));

export default function FounderAnalysis({ campaign }: { campaign: CampaignRequest }) {
  const pred = useApi(api.predict, campaign);
  const expl = useApi(api.explain, campaign);
  const retr = useApi(api.retrieve, campaign);
  const rec = useApi(api.recommend, campaign);

  return (
    <div className="space-y-6">
      <Card title="Prediction">
        {pred.loading && <Loading label="Scoring campaign…" />}
        {pred.error && <ErrorBox message={pred.error} />}
        {pred.data && <PredictionPanel data={pred.data} />}
      </Card>

      <Card title="Why — SHAP feature contributions">
        {expl.loading && <Loading label="Explaining prediction…" />}
        {expl.error && <ErrorBox message={expl.error} />}
        {expl.data && (
          <Suspense fallback={<Loading label="Rendering chart…" />}>
            <ExplanationPanel data={expl.data} />
          </Suspense>
        )}
      </Card>

      <Card title="Scenario simulator — what would move the needle?">
        <ScenarioSimulator campaign={campaign} baseline={pred.data?.funding_probability} />
      </Card>

      <Card title="Historical evidence — similar campaigns">
        {retr.loading && <Loading label="Retrieving similar campaigns…" />}
        {retr.error && <ErrorBox message={retr.error} />}
        {retr.data && <RetrievalDashboard data={retr.data} />}
      </Card>

      <Card title="Scores the platform defers (honesty boundary)">
        <DeferredScores />
      </Card>

      <Card title="Recommendations & roadmap">
        {rec.loading && <Loading label="Building recommendations…" />}
        {rec.error && <ErrorBox message={rec.error} />}
        {rec.data && <RecommendationPanel data={rec.data} />}
      </Card>

      <Card title="Founder Intelligence Report">
        <ReportViewer campaign={campaign} variant="founder" />
      </Card>
    </div>
  );
}
