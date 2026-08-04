"""Layer 12 — Multi-Agent Venture Intelligence System (Day 12).

Six specialist agents each analyze a campaign from a distinct perspective; an orchestrator runs
them on ONE warm stack; a consensus engine resolves their stances into a unified **committee
decision**. This emulates an investment committee — market, funding, creator, competition, risk,
and investment views deliberating before a verdict.

Modules
-------
- ``market_agent``      — MarketAgent: category attractiveness & market timing.
- ``founder_agent``     — FounderAgent (Creator Execution): the creator's controllable campaign craft.
- ``financial_agent``   — FinancialAgent (Funding Viability): goal realism & funding odds.
- ``competition_agent`` — CompetitionAgent: cohort competition & positioning.
- ``risk_agent``        — RiskAgent: risks & investor objections.
- ``investment_agent``  — InvestmentAgent: committee synthesis over the other five.
- ``orchestrator``      — AgentOrchestrator: one aggregation -> run all agents -> collect reports.
- ``consensus``         — ConsensusEngine: weighted stance vote + risk veto + conflict surfacing.
- ``serialization``     — AgentSystemSerializer: content-hash JSON manifest + verify().

Honesty
-------
Deterministic specialist analyzers — **no LLM, no new model, no faiss/torch**. Every agent reads
its slice of the **same** Day-11 ``VentureIntelligence`` object, so the agents are **correlated
specialist views of one shared signal set, not independent expert opinions** — consensus is a
structured aggregation of perspectives, and says so. Founder background, financials, market size,
moat, and regulatory risk are **deferred**, never fabricated. Model-simulated, not causal.
"""
