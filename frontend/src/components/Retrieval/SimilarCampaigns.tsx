// Similar-campaign evidence (/retrieve), shared by the Founder and Investor views. On a lightweight
// deployment the API has no FAISS/SBERT loaded (/health → components.retriever=false), so we show the
// "turned off" note without calling /retrieve at all — no doomed request, no console error.
import { api } from "../../api/client";
import type { CampaignRequest } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { useRetrieverEnabled } from "../../hooks/useServerStatus";
import { ErrorBox, Loading, RetrievalUnavailable } from "../ui";
import RetrievalDashboard from "./RetrievalDashboard";

function Retrieve({ campaign, label }: { campaign: CampaignRequest; label: string }) {
  const retr = useApi(api.retrieve, campaign);
  if (retr.loading) return <Loading label={label} />;
  if (retr.error) return retr.status === 503 ? <RetrievalUnavailable /> : <ErrorBox message={retr.error} />;
  return retr.data ? <RetrievalDashboard data={retr.data} /> : null;
}

export default function SimilarCampaigns({ campaign, label = "Retrieving similar campaigns…" }: { campaign: CampaignRequest; label?: string }) {
  const enabled = useRetrieverEnabled();
  if (enabled === undefined) return <Loading label={label} />;
  if (!enabled) return <RetrievalUnavailable />;
  return <Retrieve campaign={campaign} label={label} />;
}
