// Day 19 · Phase C — the submission form. Offers ONLY the fields the backend accepts
// (CampaignRequest, extra="forbid"). There is deliberately NO founder/financial/market/pitch-deck
// input — the backend 422-rejects anything else, so the form never asks for it.
import { useState, type FormEvent } from "react";
import type { CampaignRequest } from "../api/types";
import { COUNTRIES, CURRENCIES, MAIN_CATEGORIES } from "../lib/categories";
import { EXAMPLES } from "../lib/examples";
import { Button } from "./ui";

interface Props {
  onSubmit: (c: CampaignRequest) => void;
  loading?: boolean;
  submitLabel?: string;
}

const FIELD = "mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/30";
const LABEL = "block text-xs font-medium text-slate-600";

// "YYYY-MM-DD HH:MM:SS" (API) -> "YYYY-MM-DDTHH:MM" (datetime-local input)
const toLocal = (s: string) => s.slice(0, 16).replace(" ", "T");

export default function CampaignForm({ onSubmit, loading, submitLabel = "Analyze campaign" }: Props) {
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

  // Fill the form with an example and analyze it straight away (one click from empty to results).
  function applyExample(c: CampaignRequest) {
    setForm({
      name: c.name,
      main_category: c.main_category,
      category: c.category ?? "",
      currency: c.currency ?? "USD",
      country: c.country ?? "US",
      usd_goal_real: String(c.usd_goal_real),
      launched: toLocal(c.launched),
      deadline: toLocal(c.deadline),
    });
    setErrors({});
    onSubmit(c);
  }

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
        <p className={LABEL}>Try an example <span className="font-normal text-slate-400">(fictional campaigns)</span></p>
        <div className="mt-1.5 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              type="button"
              onClick={() => applyExample(ex.campaign)}
              className="rounded-lg bg-indigo-50 px-2.5 py-1.5 text-left text-xs font-medium text-indigo-700 ring-1 ring-indigo-100 transition hover:bg-indigo-100"
              title={ex.campaign.name}
            >
              {ex.label}
              <span className="block text-[10px] font-normal text-indigo-500">{ex.blurb}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="border-t border-slate-100 pt-4">
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
          <label className={LABEL}>Goal (USD)</label>
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

      <div className="space-y-2 pt-1">
        <Button type="submit" disabled={loading}>{loading ? "Analyzing…" : submitLabel}</Button>
        <p className="text-xs text-slate-400">
          Only the fields a Kickstarter campaign has before launch. The model has no founder, financial or market data.
        </p>
      </div>
    </form>
  );
}
