// Day 19 · Phase C — the submission form. Offers ONLY the fields the backend accepts
// (CampaignRequest, extra="forbid"). There is deliberately NO founder/financial/market/pitch-deck
// input — the backend 422-rejects anything else, so the form never asks for it.
import { useState, type FormEvent } from "react";
import type { CampaignRequest } from "../api/types";
import { COUNTRIES, CURRENCIES, MAIN_CATEGORIES } from "../lib/categories";
import { Button } from "./ui";

interface Props {
  onSubmit: (c: CampaignRequest) => void;
  loading?: boolean;
}

const FIELD = "mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";
const LABEL = "block text-xs font-medium text-slate-600";

export default function CampaignForm({ onSubmit, loading }: Props) {
  const [form, setForm] = useState({
    name: "Artificer — a deck-building board game",
    main_category: "Games",
    category: "",
    currency: "USD",
    country: "US",
    usd_goal_real: "25000",
    launched: "2016-05-01T12:00",
    deadline: "2016-05-31T12:00",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const set = (k: keyof typeof form, v: string) => setForm((f) => ({ ...f, [k]: v }));

  function validate(): CampaignRequest | null {
    const e: Record<string, string> = {};
    if (!form.name.trim()) e.name = "Campaign title is required.";
    const goal = Number(form.usd_goal_real);
    if (!Number.isFinite(goal) || goal <= 0) e.usd_goal_real = "Goal must be a positive number.";
    const launched = new Date(form.launched);
    const deadline = new Date(form.deadline);
    if (Number.isNaN(launched.getTime())) e.launched = "Invalid launch date.";
    if (Number.isNaN(deadline.getTime())) e.deadline = "Invalid deadline.";
    if (!e.launched && !e.deadline && deadline <= launched) e.deadline = "Deadline must be after launch.";
    setErrors(e);
    if (Object.keys(e).length) return null;
    const fmt = (s: string) => `${s.replace("T", " ")}:00`; // datetime-local -> "YYYY-MM-DD HH:MM:SS"
    return {
      name: form.name.trim(),
      main_category: form.main_category,
      category: form.category.trim() || undefined,
      currency: form.currency,
      country: form.country,
      usd_goal_real: goal,
      launched: fmt(form.launched),
      deadline: fmt(form.deadline),
    };
  }

  function handleSubmit(ev: FormEvent) {
    ev.preventDefault();
    const c = validate();
    if (c) onSubmit(c);
  }

  const err = (k: string) => errors[k] && <p className="mt-1 text-xs text-red-600">{errors[k]}</p>;

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className={LABEL}>Campaign title</label>
        <input className={FIELD} value={form.name} onChange={(e) => set("name", e.target.value)} />
        {err("name")}
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label className={LABEL}>Main category</label>
          <select className={FIELD} value={form.main_category} onChange={(e) => set("main_category", e.target.value)}>
            {MAIN_CATEGORIES.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className={LABEL}>Sub-category (optional)</label>
          <input className={FIELD} value={form.category} onChange={(e) => set("category", e.target.value)} placeholder="e.g. Tabletop Games" />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <div>
          <label className={LABEL}>Funding goal (USD)</label>
          <input className={FIELD} inputMode="decimal" value={form.usd_goal_real} onChange={(e) => set("usd_goal_real", e.target.value)} />
          {err("usd_goal_real")}
        </div>
        <div>
          <label className={LABEL}>Currency</label>
          <select className={FIELD} value={form.currency} onChange={(e) => set("currency", e.target.value)}>
            {CURRENCIES.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
        <div>
          <label className={LABEL}>Country</label>
          <select className={FIELD} value={form.country} onChange={(e) => set("country", e.target.value)}>
            {COUNTRIES.map((c) => <option key={c}>{c}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4">
        <div>
          <label className={LABEL}>Launch</label>
          <input type="datetime-local" className={FIELD} value={form.launched} onChange={(e) => set("launched", e.target.value)} />
          {err("launched")}
        </div>
        <div>
          <label className={LABEL}>Deadline</label>
          <input type="datetime-local" className={FIELD} value={form.deadline} onChange={(e) => set("deadline", e.target.value)} />
          {err("deadline")}
        </div>
      </div>

      <div className="flex items-center gap-3 pt-1">
        <Button type="submit" disabled={loading}>{loading ? "Analyzing…" : "Analyze campaign"}</Button>
        <span className="text-xs text-slate-400">Real Kickstarter fields only — no founder/financial/market inputs (the model has none).</span>
      </div>
    </form>
  );
}
