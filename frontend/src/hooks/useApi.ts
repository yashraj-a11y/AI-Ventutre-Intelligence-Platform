// Day 19 · Phase D — run a typed API call for a given input, tracking loading/error/data.
// `fn` is a stable reference from the api object and `input` (the campaign) changes on each submit,
// so the effect re-runs exactly when there's a new campaign — no infinite loop.
import { useEffect, useState } from "react";
import { ApiError } from "../api/client";

export interface AsyncState<T> {
  data?: T;
  loading: boolean;
  error?: string;
}

function message(e: unknown): string {
  if (e instanceof ApiError) return e.status ? `${e.status}: ${e.message}` : e.message;
  return e instanceof Error ? e.message : String(e);
}

export function useApi<I, T>(fn: (input: I) => Promise<T>, input: I): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({ loading: true });
  useEffect(() => {
    let alive = true;
    setState({ loading: true });
    fn(input).then(
      (data) => { if (alive) setState({ data, loading: false }); },
      (err) => { if (alive) setState({ loading: false, error: message(err) }); },
    );
    return () => { alive = false; };
  }, [fn, input]);
  return state;
}
