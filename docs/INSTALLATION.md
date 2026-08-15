# Installation

How to run the platform locally. For containers see [DEPLOYMENT.md](DEPLOYMENT.md).

## What was verified

Developed and tested on **macOS on Apple Silicon (arm64, Darwin 25), Python 3.14.2, Node 25 / npm 11**. The 95
tests and every API endpoint were run on that machine with the trained artifacts already present.
A **clean-clone rebuild of every artifact from the raw CSV has not been re-run for this release** — the
rebuild order below is taken from the project's notebooks, not from a fresh end-to-end run.

## 1. Prerequisites

| Need | Notes |
|---|---|
| Python 3.14 | `requirements.txt` is pinned for it |
| Node 20+ / npm | frontend only (built here with Node 25) |
| ~5 GB disk, ~2 GB free RAM | the warm stack is ~1 GB; torch and FAISS are large |
| macOS: OpenMP | `brew install libomp` (LightGBM / XGBoost need it) |

## 2. Python environment

```bash
git clone https://github.com/yashraj-a11y/AI-Ventutre-Intelligence-Platform.git AI-Venture-Intelligence-Platform
cd AI-Venture-Intelligence-Platform
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_nltk_data.py     # stopwords / wordnet; handles the macOS SSL quirk
```

## 3. Data

Download the Kaggle dataset "Kickstarter Projects" and place it at:

```
data/raw/ks-projects-201801.csv          # 378,661 rows
```

The CSV is not in the repository.

## 4. Trained artifacts

Large and regenerable files are git-ignored (the FAISS index alone is ~407 MB). The API needs, at minimum:

- `artifacts/models/calibrated_model.joblib`, `artifacts/models/model_metadata.json`
- `artifacts/feature_pipeline/`, `artifacts/splits/`
- `artifacts/vector_db/faiss.index`, `artifacts/embeddings/startup_embeddings.npy`
- `data/processed/kickstarter_processed.csv`

To regenerate them, run the notebooks in order (`jupyter nbconvert --execute` or open them):
`01_EDA` → `02_Preprocessing` → `03_Feature_Engineering` → `04_Model_Training` → `05_Explainability` →
`06_Venture_Knowledge_Base` → `07_Pattern_Mining` → … → `16_MLOps`. Model training can also be re-run through
`TrainingPipeline` in `src/training/train_pipeline.py` (used in notebook 16). The SBERT encoding of ~265K campaigns is the slow step.

## 5. Run the backend

```bash
OMP_NUM_THREADS=1 CORS_ORIGINS=http://localhost:5173 \
  uvicorn src.api.main:app --port 8000
```

`OMP_NUM_THREADS=1` stops FAISS and torch fighting over OpenMP. `CORS_ORIGINS` lets the browser frontend
call the API (default is deny). Startup loads the models, so give it several seconds.

```bash
curl localhost:8000/ready        # {"status":"ready"} once loaded
open http://localhost:8000/docs  # interactive API docs
```

The first `/retrieve` call is slow (~12 s, model warm-up); later calls are fast.

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173
```

The API base defaults to `http://localhost:8000`; override with `VITE_API_BASE_URL` in `frontend/.env`.
`npm run build` typechecks and bundles.

## 7. Verify

```bash
pytest -m "not heavy"   # 53 tests, ~4 s, no artifacts needed (this is what CI runs)
pytest                  # 95 tests, ~5.5 min, needs data + artifacts
ruff check src tests
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| Hang or crash importing torch / faiss | set `OMP_NUM_THREADS=1` |
| `CERTIFICATE_VERIFY_FAILED` on NLTK download | use `scripts/download_nltk_data.py`, not a bare `nltk.download` |
| LightGBM / XGBoost `libomp` error (macOS) | `brew install libomp` |
| Browser shows a CORS error | start the backend with `CORS_ORIGINS=http://localhost:5173` |
| `/ready` returns `503` | still loading, or an artifact is missing — check the log |
| `422` on a request | the body has a field outside the real campaign schema |
