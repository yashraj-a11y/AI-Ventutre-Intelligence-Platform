// Day 19 · Phase B — TypeScript interfaces mirroring the backend schemas (src/api/schemas/*.py),
// grounded against live responses. Loosely-typed dicts (`dict[str, Any]` on the backend) use index
// signatures so the UI can read known fields with types but never pretends the shape is fixed.

/** The ONLY submission fields the backend accepts (extra="forbid" => anything else is 422'd). */
export interface CampaignRequest {
  name: string;
  main_category: string;
  category?: string;
  currency?: string; // default "USD"
  country?: string; // default "US"
  usd_goal_real: number;
  goal?: number;
  launched: string; // "YYYY-MM-DD HH:MM:SS"
  deadline: string;
}

export interface PredictionResponse {
  campaign: Record<string, unknown>;
  funding_probability: number;
  predicted_label: "successful" | "failed" | string;
  decision_threshold: number;
}

/** One signed SHAP contribution (log-odds margin space). */
export interface Contribution {
  feature: string;
  group: "tabular" | "text" | string;
  contribution: number;
  feature_value: number;
  descriptor: string;
}
export interface ExplanationResponse {
  funding_probability: number;
  predicted_label: string;
  top_positive: Contribution[];
  top_negative: Contribution[];
  honesty_note?: string | null;
}

export interface SimilarCampaign {
  rank: number;
  similarity: number;
  campaign_id: number;
  title: string;
  category?: string;
  main_category?: string;
  goal?: number;
  usd_goal_real?: number;
  launch_year?: number;
  outcome?: "successful" | "failed" | string;
  target?: number;
  reason?: string;
  [k: string]: unknown;
}
export interface RetrievalResponse {
  query: Record<string, unknown>;
  similar: SimilarCampaign[];
  similar_successful: SimilarCampaign[];
  similar_failed: SimilarCampaign[];
  honesty_note?: string | null;
}

export interface InvestmentReadiness {
  stage: string;
  funding_probability: number;
  optimized_probability: number;
  threshold: number;
  readiness_index: number;
}
export interface PriorityImprovement {
  feature: string;
  category: string;
  action: string;
  priority_label: string;
  expected_impact_points: number;
  reason: string;
  rank: number;
  [k: string]: unknown;
}
export interface Confidence {
  score: number;
  reason: string;
  factors?: unknown;
}
export interface RecommendationResponse {
  campaign: Record<string, unknown>;
  investment_readiness: InvestmentReadiness;
  confidence: Confidence;
  priority_improvements: PriorityImprovement[];
  investor_strategy: Record<string, unknown>;
  final_strategy: string;
  deferred: Record<string, unknown>;
  honesty_note: string;
}

/** /scenario and /counterfactual both return this envelope (result shape differs per endpoint). */
export interface GenericResponse {
  campaign: Record<string, unknown>;
  result: Record<string, unknown>;
  honesty_note?: string | null;
}

export interface ReportSection {
  id: string;
  title: string;
  source: string; // provenance badge ("Day-N ...") — no-new-claims discipline
  kind: string;
  body: unknown;
}
export interface DeferredAppendix {
  note: string;
  items: unknown;
}
export interface ReportResponse {
  report_type: string;
  campaign: Record<string, unknown>;
  sections: ReportSection[];
  deferred_appendix: DeferredAppendix;
  honesty_note: string;
}

export interface AnalysisResponse {
  campaign: Record<string, unknown>;
  committee_recommendation: string;
  overall_confidence: number;
  reports: Record<string, unknown>;
  copilot: Record<string, unknown>;
  honesty_note: string;
}

export interface StatusResponse {
  status: string;
  [k: string]: unknown;
}
