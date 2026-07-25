"""Download the NLTK data required by Day-3 Phase G (text preprocessing).

Idempotent. Run once after creating the virtualenv:

    .venv/bin/python scripts/download_nltk_data.py

Why this script exists instead of a bare ``nltk.download(...)``:

1. **macOS framework-Python SSL.** The python.org framework build does not use
   the system keychain, so NLTK's HTTPS download fails with
   ``CERTIFICATE_VERIFY_FAILED``. We fix it *correctly* — pointing the default
   SSL context at ``certifi``'s CA bundle — rather than disabling verification.
2. **wordnet zip extraction.** ``nltk.download('wordnet')`` occasionally leaves
   ``wordnet.zip`` unextracted, so ``nltk.data.find('corpora/wordnet')`` fails
   even though the download "succeeded". We unzip it if the directory is absent.
"""

from __future__ import annotations

import ssl
import zipfile
from pathlib import Path

import nltk

# Resources needed by Phase G: stopwords, WordNet lemmatizer, punkt tokenizer.
# punkt_tab is required by NLTK >= 3.9.
REQUIRED = ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]

VERIFY = [
    ("stopwords", "corpora/stopwords"),
    ("wordnet", "corpora/wordnet"),
    ("punkt", "tokenizers/punkt"),
    ("punkt_tab", "tokenizers/punkt_tab"),
]


def _use_certifi_ca() -> None:
    """Keep SSL verification ON but trust certifi's CA bundle (macOS fix)."""
    try:
        import certifi
    except ImportError:
        return  # fall back to default context; may fail on framework Python
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )


def _ensure_wordnet_extracted() -> None:
    """Unzip corpora/wordnet.zip if the directory did not get extracted."""
    for root in map(Path, nltk.data.path):
        zip_path = root / "corpora" / "wordnet.zip"
        dir_path = root / "corpora" / "wordnet"
        if zip_path.exists() and not dir_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(zip_path.parent)
            print(f"  extracted {zip_path} -> {dir_path}")


def main() -> None:
    _use_certifi_ca()

    print("Downloading NLTK data:")
    for pkg in REQUIRED:
        ok = nltk.download(pkg, quiet=True)
        print(f"  {pkg:12s} {'OK' if ok else 'FAILED'}")

    _ensure_wordnet_extracted()

    print("Verifying resources resolve:")
    missing = []
    for name, path in VERIFY:
        try:
            nltk.data.find(path)
            print(f"  {name:12s} OK")
        except LookupError:
            print(f"  {name:12s} MISSING")
            missing.append(name)

    if missing:
        raise SystemExit(f"NLTK data still missing: {missing}")
    print("All Phase-G NLTK data present.")


if __name__ == "__main__":
    main()
