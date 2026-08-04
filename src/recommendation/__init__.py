"""Layer 11 — Unified Recommendation & Strategy Engine (Day 11).

The orchestration layer: aggregate every prior intelligence layer (Days 4–10) into ONE
evidence-backed, prioritized, audience-specific **venture decision object**. This is the first
point at which the whole architecture operates as a single AI system rather than nine independent
modules.

Modules
-------
- ``intelligence_aggregator``  — Phase B: collect all prior-layer outputs into one profile.
- ``recommendation_generator`` — Phase C: strategic recommendations (real categories only).
- ``prioritization``           — Phase D: rank by cross-layer impact + evidence + confidence.
- ``founder_strategy``         — Phase E: founder-facing view of the shared intelligence.
- ``investor_strategy``        — Phase F: investor-facing band (Strong / Caution / Monitor / No).
- ``confidence``               — Phase G: transparent confidence from real agreement signals.
- ``pipeline``                 — Phase H: the unified VentureDecision entry point.
- ``serialization``            — Phase I: content-hash JSON manifest + verify().

Honesty
-------
Deterministic aggregation over REAL signals — **no LLM, no new trained model, no faiss/torch**.
Recommendations come only from controllable levers (goal / duration / launch timing / title) and
category-cohort evidence; retention / revenue / moat / team / growth-trajectory items are
**deferred**, never fabricated. Model-simulated, not causal.
"""
