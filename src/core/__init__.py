"""Day 14 — Backend core.

Cross-cutting backend infrastructure for the API layer:

- ``model_loader`` — the singleton ``ModelRegistry``: builds ONE warm stack (the shared Day-8
  ``CounterfactualPipeline`` → Day-11/12/13 pipelines) and the Day-6 faiss retriever, once, in the
  right load order (lightgbm before faiss).
- ``exceptions`` — the backend error taxonomy (ArtifactError / InferenceError) + FastAPI handlers.

Config and logging are **reused** from :mod:`src.utils.config` and :mod:`src.utils.logger` — the
same modules all 13 layers already use — not re-implemented here.
"""
