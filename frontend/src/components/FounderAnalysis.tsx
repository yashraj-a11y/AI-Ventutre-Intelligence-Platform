// Day 19 · Phase D — Founder analysis orchestrator: /predict + /recommend, with loading/error states.
import { lazy, Suspense } from "react";
import { api } from "../api/client";
import type { CampaignRequest } from "../api/types";
import { useApi } from "../hooks/useApi";
import DeferredScores from "./Dashboard/DeferredScores";
import PredictionPanel from "./Dashboard/PredictionPanel";
import RecommendationPanel from "./Dashboard/RecommendationPanel";
import ReportViewer from "./Reports/ReportViewer";
import SimilarCampaigns from "./Retrieval/SimilarCampaigns";
import ScenarioSimulator from "./ScenarioSimulator/ScenarioSimulator";
import { Card, ErrorBox, Loading, SectionNav } from "./ui";

// Recharts is heavy — code-split the chart panel into its own chunk, loaded only when shown.
const ExplanationPanel = lazy(() => import("./Explainability/ExplanationPanel"));

const SECTIONS = [
  { id: "prediction", label: "Prediction" },
  { id: "drivers", label: "Drivers" },
  { id: "simulator", label: "What-if" },
  { id: "recommendations", label: "Recommendations" },
  { id: "similar", label: "Similar campaigns" },
  { id: "report", label: "Report" },
];

export default function FounderAnalysis({ campaign }: { campaign: CampaignRequest }) {
  const pred = useApi(api.predict, campaign);
  const expl = useApi(api.explain, campaign);
  const rec = useApi(api.recommend, campaign);

  return (
    <div>
      <SectionNav items={SECTIONS} />
      <div className="space-y-6">
        <Card id="prediction" title="Funding prediction" subtitle="Calibrated probability that this campaign reaches its goal">
          {pred.loading && <Loading label="Scoring campaign…" />}
          {pred.error && <ErrorBox message={pred.error} />}
          {pred.data && <PredictionPanel data={pred.data} />}
        </Card>

        <Card id="drivers" title="What drives the score" subtitle="SHAP contributions: green pushes the odds up, red pulls them down">
          {expl.loading && <Loading label="Explaining prediction…" />}
          {expl.error && <ErrorBox message={expl.error} />}
          {expl.data && (
            <Suspense fallback={<Loading label="Rendering chart…" />}>
              <ExplanationPanel data={expl.data} />
            </Suspense>
          )}
        </Card>

        <Card id="simulator" title="What-if simulator" subtitle="Move the levers and watch the probability update live">
          <ScenarioSimulator campaign={campaign} baseline={pred.data?.funding_probability} />
        </Card>

        <Card id="recommendations" title="Recommendations & roadmap" subtitle="Changes ranked by how much they would lift your odds">
          {rec.loading && <Loading label="Building recommendations…" />}
          {rec.error && <ErrorBox message={rec.error} />}
          {rec.data && <RecommendationPanel data={rec.data} />}
        </Card>

        <Card id="similar" title="Similar past campaigns" subtitle="Real Kickstarter campaigns that look like yours">
          <SimilarCampaigns campaign={campaign} />
        </Card>

        <Card id="limits" title="What the data can't score" subtitle="Shown as deferred instead of being invented">
          <DeferredScores />
        </Card>

        <Card id="report">
          <ReportViewer campaign={campaign} variant="founder" />
        </Card>
      </div>
    </div>
  );
}
