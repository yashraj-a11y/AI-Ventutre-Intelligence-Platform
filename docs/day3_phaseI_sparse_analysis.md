# Day 3 — Phase I: Sparse vs Dense Analysis

_Generated 2026-07-25T15:00:06.319531+00:00 · rows: 265,171_

## Per-block memory

| block | cols | density | dense | sparse | savings |
|---|--:|--:|--:|--:|--:|
| numeric | 14 | 100.000% | 29.7 MB | 44.5 MB | 1× |
| categorical | 52 | 7.692% | 110.3 MB | 13.8 MB | 8× |
| text | 800 | 0.267% | 1697.1 MB | 7.9 MB | 216× |
| **combined** | **866** | **2.325%** | **1837.1 MB** | **65.1 MB** | **28×** |

## Fusion strategy

- The **text** block is ~99.7% zeros; densifying the combined matrix would cost **1837 MB** vs **65 MB** sparse (**28×**).
- `numpy.hstack` / `numpy.concatenate` **cannot** mix a dense `ndarray` with a `scipy.sparse` matrix — different storage types.
- **Strategy (Phase J):** promote the small dense numeric block to sparse (`scipy.sparse.csr_matrix`), then `scipy.sparse.hstack([text, categorical, numeric])` → one CSR matrix.
- Do **not** StandardScale the sparse blocks (centering fills the zeros); scaling stays on the dense numeric block only.
