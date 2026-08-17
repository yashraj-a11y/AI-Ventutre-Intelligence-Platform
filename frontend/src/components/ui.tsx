// Day 19 — tiny shared UI primitives (Tailwind). Kept minimal; components compose these.
import type { ReactNode } from "react";
import { useServerStatus } from "../hooks/useServerStatus";

interface CardProps {
  title?: string;
  subtitle?: string;
  id?: string;               // anchor for the section menu
  action?: ReactNode;        // e.g. a button aligned with the title
  children: ReactNode;
  className?: string;
}

export function Card({ title, subtitle, id, action, children, className = "" }: CardProps) {
  return (
    <section id={id} className={`scroll-mt-32 rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm shadow-slate-200/50 sm:p-6 ${className}`}>
      {(title || action) && (
        <div className="mb-5 flex items-start justify-between gap-4">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

/** Sticky in-page menu for the long analysis pages. */
export function SectionNav({ items }: { items: { id: string; label: string }[] }) {
  return (
    <nav className="no-print sticky top-14.25 z-10 -mx-1 mb-6 overflow-x-auto rounded-xl bg-slate-50/90 px-1 py-2 backdrop-blur">
      <div className="flex w-max gap-1.5">
        {items.map((i) => (
          <a key={i.id} href={`#${i.id}`} className="rounded-full bg-white px-3 py-1.5 text-xs font-medium text-slate-600 ring-1 ring-slate-200 transition hover:text-indigo-700 hover:ring-indigo-300">
            {i.label}
          </a>
        ))}
      </div>
    </nav>
  );
}

/** Circular gauge for a 0–1 value. */
export function Gauge({ value, color, size = 132, label }: { value: number; color: string; size?: number; label?: string }) {
  const r = 52, c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="10" />
        <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${pct * c} ${c}`} className="transition-[stroke-dasharray] duration-700" />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold tracking-tight text-slate-900">{Math.round(pct * 100)}%</span>
        {label && <span className="text-[11px] font-medium text-slate-500">{label}</span>}
      </div>
    </div>
  );
}

/** Brand mark: a rising line in an indigo tile (same as the favicon). */
export function Logo({ className = "h-8 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <rect width="32" height="32" rx="8" fill="#4f46e5" />
      <path d="M8 22.5 13.5 16l4 3.5L24 10" fill="none" stroke="#fff" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="24" cy="10" r="2.2" fill="#a5b4fc" />
    </svg>
  );
}

/** Page title block used by the portals. */
export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-8">
      <p className="text-xs font-semibold uppercase tracking-wider text-indigo-600">{eyebrow}</p>
      <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-900">{title}</h1>
      {children && <p className="mt-2 max-w-2xl text-sm text-slate-500">{children}</p>}
    </div>
  );
}

/** Placeholder before the first submission: says what the analysis will contain. */
export function EmptyState({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="rounded-2xl border-2 border-dashed border-slate-200 bg-white/60 px-6 py-12 text-center">
      <svg viewBox="0 0 24 24" className="mx-auto h-10 w-10 text-indigo-300" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19h16M7 15l3-4 3 2 4-6" />
      </svg>
      <p className="mt-3 font-semibold text-slate-700">{title}</p>
      <p className="mt-1 text-sm text-slate-400">Pick an example or fill in the form, then press the button.</p>
      <ul className="mx-auto mt-6 grid max-w-md gap-2 text-left text-sm text-slate-500 sm:grid-cols-2">
        {items.map((i) => (
          <li key={i} className="flex gap-2">
            <span className="text-indigo-500">✓</span>
            {i}
          </li>
        ))}
      </ul>
    </div>
  );
}

const STATUS_PILL = {
  checking: { dot: "bg-slate-400", text: "Checking demo server…" },
  waking: { dot: "bg-amber-500 animate-pulse", text: "Demo server waking up…" },
  ready: { dot: "bg-emerald-500", text: "Demo server online" },
  down: { dot: "bg-rose-500", text: "Demo server unreachable" },
} as const;

/** Small live status pill (landing page / header). */
export function ServerStatusPill({ className = "" }: { className?: string }) {
  const s = STATUS_PILL[useServerStatus()];
  return (
    <span className={`inline-flex items-center gap-2 rounded-full bg-white/80 px-3 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200 ${className}`}>
      <span className={`h-2 w-2 rounded-full ${s.dot}`} />
      {s.text}
    </span>
  );
}

/** Explains a slow first request on the free host instead of leaving a spinner unexplained. */
export function ServerWakeBanner() {
  const status = useServerStatus();
  if (status === "ready" || status === "checking") return null;
  return status === "waking" ? (
    <div className="no-print mb-6 flex items-start gap-3 rounded-xl bg-amber-50 px-4 py-3 text-sm text-amber-900 ring-1 ring-amber-200">
      <span className="mt-0.5 h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-amber-300 border-t-amber-600" />
      <span>
        <span className="font-semibold">Waking up the demo server.</span> It runs on a free tier that sleeps
        when idle, so the first analysis can take about a minute. Everything after that is fast.
      </span>
    </div>
  ) : (
    <div className="no-print mb-6 rounded-xl bg-rose-50 px-4 py-3 text-sm text-rose-800 ring-1 ring-rose-200">
      <span className="font-semibold">The demo server isn't responding.</span> Please try again in a few minutes.
    </div>
  );
}

export function Button({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className="inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {children}
    </button>
  );
}

/** The platform's honesty boundary, rendered. Use for honesty_note / deferred callouts. */
export function HonestyNote({ note }: { note?: string | null }) {
  if (!note) return null;
  return (
    <p className="mt-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 ring-1 ring-amber-200">
      <span className="font-semibold">Honesty:</span> {note}
    </p>
  );
}

/** Skeleton placeholder with a label, so loading cards keep their shape instead of collapsing. */
export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div role="status" className="space-y-3 py-1">
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-200 border-t-indigo-500" />
        {label}
      </div>
      <div className="h-3 w-11/12 animate-pulse rounded-full bg-slate-100" />
      <div className="h-3 w-3/4 animate-pulse rounded-full bg-slate-100" />
      <div className="h-3 w-1/2 animate-pulse rounded-full bg-slate-100" />
    </div>
  );
}

/** Similar-campaign search is off on small hosts (LOAD_RETRIEVER=false → /retrieve answers 503). */
export function RetrievalUnavailable() {
  return (
    <p className="rounded-md bg-slate-50 px-3 py-2 text-sm text-slate-600 ring-1 ring-slate-200">
      Similar-campaign search is turned off in this lightweight deployment (it needs ~700 MB more memory
      for the SBERT model and FAISS index). Run the project locally to see it.
    </p>
  );
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-md bg-red-50 px-3 py-2 text-sm text-red-700 ring-1 ring-red-200">
      <span className="font-semibold">Request failed:</span> {message}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-center">
      <div className="text-xl font-bold text-slate-900 sm:text-2xl">{value}</div>
      <div className="mt-1 text-xs font-medium text-slate-500">{label}</div>
      {hint && <div className="mt-0.5 text-[10px] text-slate-400">{hint}</div>}
    </div>
  );
}
