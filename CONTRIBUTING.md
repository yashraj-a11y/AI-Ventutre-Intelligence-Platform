# Contributing

Thanks for looking at the project. A few rules keep it honest and working.

## Setup

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install ruff
python scripts/download_nltk_data.py
cd frontend && npm install
```

Trained artifacts and the dataset are git-ignored; see [docs/INSTALLATION.md](docs/INSTALLATION.md).

## Before you open a change

```bash
ruff check src tests            # config in ruff.toml (real errors only)
pytest -m "not heavy"           # fast lane — what CI runs (53 tests, seconds)
pytest                          # full suite — needs data + artifacts (95 tests, ~5.5 min)
cd frontend && npm run build    # typechecks, then bundles
```

A change is ready when the full suite is green (or you say which heavy tests you couldn't run).

## Conventions

- **Layered modules.** Each layer in `src/` has a `pipeline.py` and a `serialization.py`; the API talks to
  them only through `src/services/`.
- **No fabricated outputs.** If the data cannot support a value, return it as *deferred* with a stated
  basis, not a number. Every API response carries an `honesty_note` and/or a `deferred` block, and the UI
  renders them.
- **Request contract.** `CampaignRequest` accepts only the Kickstarter fields and forbids extras. Don't
  widen it without data to back the new field.
- **Reproducibility.** Set `OMP_NUM_THREADS=1` when FAISS and torch share a process (the tests do).
- **Commits.** Short, descriptive subject in the imperative or past tense, one logical change each.

## Reporting issues

Include the endpoint or module, the request you sent, the response or traceback, and your Python and
OS versions.

## License

By contributing you agree your work is released under the [MIT License](LICENSE).
