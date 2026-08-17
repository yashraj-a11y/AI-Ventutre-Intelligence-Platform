// Day 19 · Phase B — typed Axios client for the 8 real backend endpoints.
// - base URL from VITE_API_BASE_URL (dev needs the backend's CORS_ORIGINS to include this origin);
// - errors normalized to ApiError (message + status + backend detail);
// - retry once on network / 5xx only — NEVER on 4xx (a 422 is the caller's fault, not transient).

import axios, { AxiosError, type AxiosInstance } from "axios";
import type {
  AnalysisResponse,
  CampaignRequest,
  ExplanationResponse,
  GenericResponse,
  PredictionResponse,
  RecommendationResponse,
  ReportResponse,
  RetrievalResponse,
  StatusResponse,
} from "./types";

// Trailing slashes stripped: "https://api/" + "/ready" would request "//ready" (a 404).
export const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** A normalized API error the UI can render (status + backend-provided detail). */
export class ApiError extends Error {
  status?: number;
  detail?: unknown;
  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const http: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  // /analyze is ~1s warm, but a free-tier host can take 1-2 minutes to wake from sleep — a shorter
  // timeout would fail the first request after idle.
  timeout: 150_000,
  headers: { "Content-Type": "application/json" },
});

function toApiError(err: AxiosError): ApiError {
  const status = err.response?.status;
  const data = err.response?.data as
    | { error?: string; detail?: unknown; message?: string }
    | undefined;
  const raw = data?.error ?? data?.message ?? data?.detail ?? err.message;
  const msg = typeof raw === "string" ? raw : JSON.stringify(raw);
  return new ApiError(msg || "request failed", status, data);
}

async function withRetry<T>(fn: () => Promise<T>, retries = 1): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    const err = e as AxiosError;
    const status = err.response?.status;
    const retriable = status === undefined || status >= 500; // network or server error only
    if (retries > 0 && retriable) return withRetry(fn, retries - 1);
    throw toApiError(err);
  }
}

const post = <T>(path: string, body: unknown): Promise<T> =>
  withRetry(async () => (await http.post<T>(path, body)).data);
const get = <T>(path: string): Promise<T> =>
  withRetry(async () => (await http.get<T>(path)).data);

/** The typed API surface — one function per real endpoint. */
export const api = {
  predict: (c: CampaignRequest) => post<PredictionResponse>("/predict", c),
  explain: (c: CampaignRequest) => post<ExplanationResponse>("/explain", c),
  retrieve: (c: CampaignRequest, k = 5) =>
    post<RetrievalResponse>(`/retrieve?k=${k}`, c),
  recommend: (c: CampaignRequest) => post<RecommendationResponse>("/recommend", c),
  scenario: (c: CampaignRequest) => post<GenericResponse>("/scenario", c),
  counterfactual: (c: CampaignRequest) =>
    post<GenericResponse>("/counterfactual", c),
  reportFounder: (c: CampaignRequest) =>
    post<ReportResponse>("/report/founder", c),
  reportInvestor: (c: CampaignRequest) =>
    post<ReportResponse>("/report/investor", c),
  analyze: (c: CampaignRequest) => post<AnalysisResponse>("/analyze", c),
  health: () => get<StatusResponse>("/health"),
  ready: () => get<StatusResponse>("/ready"),
  live: () => get<StatusResponse>("/live"),
};

export type Api = typeof api;
