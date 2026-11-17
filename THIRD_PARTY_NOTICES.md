# Third-party notices

## Cambridge parallel-HSMM implementation

- Source: https://github.com/am2234/parallel-hsmm-murmur
- Pinned commit: `5f9d14ee7f9215bf8cf523341241ee9d79be79ae`
- Copyright: 2022, University of Cambridge, as stated in upstream source files.
- License: GAP Available Source License v1.0 (academic noncommercial use).

The baseline adapters and the legacy boundary semantics used by this research derive from that implementation. Original attribution, modification notices and warranty disclaimers remain in the distributed files. The full license is provided in `LICENSE` and `phase9/HSMM_LICENSE.md`.

On 2026-11-17, a dated redistribution notice was added to `phase10/src/path_bounds.py`, `query_gap_distillation.py`, `query_training_common.py`, `segment_projection_baseline.py` and `selection_aware_bounds.py`. No numerical logic was changed. The original and release hashes are listed in `provenance/source_export.json`.

The upstream repository is fetched separately by `scripts/fetch_upstream.py`. It is not vendored into this release. The fetcher checks its commit and clean working tree. Retain the upstream notices if redistributing that code.

## CirCor DigiScope

Dataset: CirCor DigiScope v1.0.3, https://doi.org/10.13026/tshs-mw03. The original repository states the Open Data Commons Attribution License v1.0. No recordings or annotations are included here. Obtain data from PhysioNet and cite the original dataset and associated study.

## Python dependencies

NumPy, SciPy, Numba, llvmlite, PyTorch, pandas, scikit-learn and matplotlib are installed separately under their respective licenses. This repository does not replace or alter those licenses.
