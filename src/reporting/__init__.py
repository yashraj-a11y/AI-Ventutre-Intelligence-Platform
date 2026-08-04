"""Layer 13 — Professional Report Generation (Day 13). The final layer of Architecture V1.0.

Turn every prior layer's outputs into audience-specific, decision-ready documents — a Founder
Intelligence Report, an Investor Due Diligence Report, an Investment Memo, an Executive Summary — a
deterministic Q&A Venture Copilot, and multi-format export (JSON / Markdown / HTML).

Modules
-------
- ``founder_report``     — FounderReportGenerator: the founder-facing intelligence report.
- ``investor_report``    — InvestorReportGenerator: the investor due-diligence report.
- ``investment_memo``    — InvestmentMemoGenerator: a VC-style investment memo.
- ``executive_summary``  — ExecutiveSummaryGenerator: dashboard one-liners.
- ``copilot``            — VentureCopilotQA: deterministic intent-router Q&A (NOT an LLM; distinct
                           from the pre-existing Day-5 ``src/copilot/venture_copilot.py``).
- ``export``             — ReportExporter: JSON / Markdown / HTML (PDF optional-graceful).
- ``pipeline``           — ReportingPipeline: one warm stack -> ReportBundle -> all reports.
- ``serialization``      — ReportingSerializer: content-hash JSON manifest + verify().

Honesty — the defining principle of this layer
-----------------------------------------------
Reporting is **presentation only**: it generates **zero new analysis**. Every report is a
deterministic *reformatting* of the Day-11 ``VentureDecision`` + Day-12 ``CommitteeDecision`` + their
underlying signals. So the honesty is **inherited** — the one rule is **NO NEW CLAIMS**: every
report line traces to a real Day 4-12 output (tagged with its ``source``). Anything the analysis
never had — founder background, revenue, moat, expected return — goes in a **deferred appendix**,
never fabricated. Deterministic, no LLM, no new model, no faiss. Model-simulated, not causal.
"""
