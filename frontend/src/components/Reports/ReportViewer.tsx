// Day 19 · Phase H — report viewer. Renders /report/{founder,investor} sections by `kind`, with each
// section's `source` provenance badge (the no-new-claims discipline made visible) and the
// deferred_appendix. "Print / Save as PDF" uses the browser's print dialog (client-side, no PDF endpoint).
import { api } from "../../api/client";
import type { CampaignRequest, ReportResponse, ReportSection } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { Button, ErrorBox, HonestyNote, Loading } from "../ui";

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function isTable(v: unknown): v is { columns: unknown[]; rows: unknown[][] } {
  return isRecord(v) && Array.isArray(v.columns) && Array.isArray(v.rows);
}

function SectionBody({ kind, body }: { kind: string; body: unknown }) {
  if (kind === "text" && typeof body === "string")
    return <p className="whitespace-pre-line text-sm text-slate-600">{body}</p>;

  if (kind === "list" && Array.isArray(body))
    return (
      <ul className="list-disc space-y-1 pl-5 text-sm text-slate-600">
        {body.map((x, i) => <li key={i}>{String(x)}</li>)}
      </ul>
    );

  if (kind === "metrics" && isRecord(body))
    return (
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        {Object.entries(body).map(([k, v]) => (
          <div key={k} className="rounded-lg bg-slate-50 p-2 text-center">
            <div className="text-sm font-semibold text-slate-800">{String(v)}</div>
            <div className="mt-0.5 text-[11px] text-slate-500">{k}</div>
          </div>
        ))}
      </div>
    );

  if (kind === "table" && isTable(body))
    return (
      <>
      {/* phones: one card per row ("column: value"), since 3-4 columns squeeze text into slivers */}
      <div className="space-y-2 sm:hidden print:hidden">
        {body.rows.map((row, i) => (
          <dl key={i} className="rounded-lg bg-slate-50 p-3 text-sm">
            {row.map((cell, j) => (
              <div key={j} className="flex gap-3 py-0.5">
                <dt className="w-24 shrink-0 text-xs font-medium uppercase tracking-wide text-slate-400">{String(body.columns[j] ?? "")}</dt>
                <dd className="min-w-0 flex-1 text-slate-700">{String(cell)}</dd>
              </div>
            ))}
          </dl>
        ))}
      </div>
      <div className="hidden overflow-x-auto sm:block print:block">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
              {body.columns.map((c, i) => <th key={i} className="py-1.5 pr-3">{String(c)}</th>)}
            </tr>
          </thead>
          <tbody>
            {body.rows.map((row, i) => (
              <tr key={i} className="border-b border-slate-100">
                {row.map((cell, j) => <td key={j} className="py-1.5 pr-3 text-slate-600">{String(cell)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </>
    );

  return <pre className="overflow-auto rounded bg-slate-900 p-2 text-xs text-slate-100">{JSON.stringify(body, null, 2)}</pre>;
}

// The backend's provenance labels carry internal build-stage names ("Day-12 committee", "Days 4-12 …").
// Keep the meaningful part for readers; the full label stays available on hover.
function sourceLabel(source: string): string {
  const s = source.replace(/\bDays?[- ]?\d+(\s*[-–/]\s*\d+)*\s*/gi, "").trim();
  if (!/^[a-z]/i.test(s)) return "Platform analysis";   // label was only stage numbers
  return s[0].toUpperCase() + s.slice(1);
}

function Section({ section }: { section: ReportSection }) {
  return (
    <section className="border-b border-slate-100 py-4 last:border-0">
      <div className="mb-2 flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-slate-900">{section.title}</h3>
        <span className="shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-medium text-slate-500" title={`Source: ${section.source}`}>
          {sourceLabel(section.source)}
        </span>
      </div>
      <SectionBody kind={section.kind} body={section.body} />
    </section>
  );
}

export default function ReportViewer({ campaign, variant }: { campaign: CampaignRequest; variant: "founder" | "investor" }) {
  const fn = variant === "founder" ? api.reportFounder : api.reportInvestor;
  const rep = useApi<CampaignRequest, ReportResponse>(fn, campaign);

  if (rep.loading) return <Loading label="Generating report…" />;
  if (rep.error) return <ErrorBox message={rep.error} />;
  if (!rep.data) return null;
  const data = rep.data;
  const items = Array.isArray(data.deferred_appendix?.items) ? (data.deferred_appendix.items as unknown[]) : [];

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-bold text-slate-900">{data.report_type}</h2>
        <span className="no-print"><Button onClick={() => window.print()}>Print / Save as PDF</Button></span>
      </div>

      <div>{data.sections.map((s) => <Section key={s.id} section={s} />)}</div>

      {items.length > 0 && (
        <div className="mt-4 rounded-lg border border-dashed border-slate-300 bg-slate-50 p-3">
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Deferred appendix</p>
          <p className="mt-1 text-xs text-slate-500">{data.deferred_appendix.note}</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {items.map((it, i) => (
              <span key={i} className="rounded-full bg-white px-2 py-0.5 text-[11px] text-slate-400 ring-1 ring-slate-200">{String(it)}</span>
            ))}
          </div>
        </div>
      )}

      <HonestyNote note={data.honesty_note} />
    </div>
  );
}
