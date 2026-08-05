"""Day 14 — Service layer.

Thin business-logic wrappers that separate the API endpoints from the ML pipelines. Each service
takes the shared :class:`~src.core.model_loader.ModelRegistry`, calls the relevant EXISTING pipeline
(no reimplementation), and returns a plain dict the route serializes:

- ``prediction_service``      → Day-4 ``PredictionEngine``
- ``explainability_service``  → Day-5 ``LocalExplainer``
- ``retrieval_service``       → Day-6 ``VentureRetriever`` (faiss)
- ``counterfactual_service``  → Day-8 ``CounterfactualPipeline``
- ``scenario_service``        → Day-10 ``ScenarioPipeline``
- ``recommendation_service``  → Day-11 ``RecommendationPipeline``
- ``report_service``          → Day-13 report generators
- ``pipeline_service``        → Day-13 ``ReportingPipeline`` (the ``/analyze`` orchestrator)

The service layer exists so routes stay dumb (validate → call → respond) and the ML pipelines stay
web-framework-agnostic (testable without HTTP).
"""
