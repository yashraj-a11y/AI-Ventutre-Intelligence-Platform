// Backend readiness, shared app-wide. The demo API runs on a free host that sleeps when idle and takes
// up to ~1-2 minutes to wake, so the UI says so instead of looking broken. One poller for the whole app
// (a tiny external store), started on first use.
import { useSyncExternalStore } from "react";
import { BASE_URL } from "../api/client";

export type ServerStatus = "checking" | "waking" | "ready" | "down";

const SLOW_MS = 3_000;        // no answer this fast -> it's probably waking up
const RETRY_MS = 5_000;
const GIVE_UP_MS = 180_000;   // after 3 minutes, report it as down

let status: ServerStatus = "checking";
let retriever: boolean | undefined;      // similar-campaign search loaded? (from /health once ready)
let started = false;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

// Lightweight deployments skip FAISS/SBERT; /health says so, letting the UI skip /retrieve (a 503).
async function loadCapabilities() {
  try {
    const r = await fetch(`${BASE_URL}/health`, { cache: "no-store" });
    const body = await r.json();
    retriever = body?.components?.retriever !== false;
  } catch {
    retriever = true;                    // unknown: fall back to just calling /retrieve
  }
  notify();
}

function set(next: ServerStatus) {
  if (next === status) return;
  status = next;
  notify();
  if (next === "ready") void loadCapabilities();
  if (next === "down") { retriever = true; notify(); }
}

async function ping(): Promise<boolean> {
  try {
    const r = await fetch(`${BASE_URL}/ready`, { cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

async function poll() {
  const t0 = Date.now();
  const slow = setTimeout(() => status === "checking" && set("waking"), SLOW_MS);
  while (Date.now() - t0 < GIVE_UP_MS) {
    if (await ping()) {
      clearTimeout(slow);
      set("ready");
      return;
    }
    if (status === "checking") set("waking");
    await new Promise((r) => setTimeout(r, RETRY_MS));
  }
  clearTimeout(slow);
  set("down");
}

function subscribe(l: () => void) {
  listeners.add(l);
  if (!started) {
    started = true;
    void poll();
  }
  return () => listeners.delete(l);
}

export function useServerStatus(): ServerStatus {
  return useSyncExternalStore(subscribe, () => status, () => status);
}

/** true/false once known (after the server is ready); undefined while still checking. */
export function useRetrieverEnabled(): boolean | undefined {
  return useSyncExternalStore(subscribe, () => retriever, () => retriever);
}
