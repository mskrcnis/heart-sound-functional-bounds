# Reproducibility

## What can be run from a fresh clone

1. `python3 scripts/verify_repository.py` verifies the deposited-file SHA-256 manifest and independently recalculates key aggregate results. It uses the Python standard library. Historical audit receipts are treated as records, not as freshly rerun audits.
2. `python scripts/run_tests.py` executes 11 synthetic tests: exhaustive graph enumeration, score/confidence mismatch, tied paths, fixed-query and selection-aware bounds, finite-difference gradients and numerical device consistency where CUDA is available.
3. `python examples/synthetic_bounds.py` demonstrates one deterministic synthetic instance. Its output is not an empirical study result.
4. `python scripts/reproduce_figures.py` reconstructs the four figures, eight table fragments and two CSV summaries from `analysis/evidence/`, then compares the table and CSV bytes against the deposited files. It writes only under ignored `build/`.
5. `python scripts/fetch_upstream.py` obtains and verifies the pinned upstream code. `python scripts/run_tests.py --with-upstream` adds two synthetic topology/training-helper tests. It does not read participant data.

The optional original test `test_upstream_capture_matches_saved_potentials` needs a recording-derived archive that is not distributed. It is not counted among the release's data-free tests.

## Numerical environment

The historical bounds and training environment was:

| Component | Version |
| --- | --- |
| Python | 3.8.20 |
| NumPy | 1.24.3 |
| Numba / llvmlite | 0.58.1 / 0.41.1 |
| SciPy | 1.10.1 |
| PyTorch | 2.4.1+cu121 |
| pandas | 2.0.3 |
| scikit-learn | 1.3.2 |
| CUDA / GPU | 12.1 / NVIDIA RTX 3090 |

The release requirements pin the corresponding package versions and matplotlib 3.7.5. The installation example uses Python 3.10, which is supported by these package versions; historical Python 3.8 is retained here to identify the original run environment. An installation example is not a claim that retraining on another interpreter or device reproduces the exact checkpoint bytes.

The original manuscript figure-preparation environment can differ from the training environment; table/CSV comparisons are exact while raster rendering may vary with fonts, matplotlib and operating system libraries.

Set deterministic execution variables before numerical checks or research reruns:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Fresh/warm distillation and teacher-target materialization explicitly require CUDA in the preserved scripts. The default synthetic suite is CPU-compatible.

## Provenance and source changes

`provenance/source_export.json` maps each copied artifact to its original research path and original/exported SHA-256. Five phase10 modules received dated license headers for redistribution. Their computational implementation is unchanged. Other copied source/configuration/protocol/evidence files are unchanged.

`provenance/release_manifest.json` records deposited file hashes, excluding its own self-reference. The file verifier rejects missing or modified listed files. It does not turn matching bytes into proof of scientific correctness.

Several experiment scripts reject changed source hashes and refuse existing output directories. Consequently the public header additions must not be silently substituted into an old frozen run. Historical receipts retain their original hashes. The [input guide](experimental_inputs.md) distinguishes historical replay from a fresh rerun.

## Data transformations

The breadth evidence retains 768 numerical records and arbitrary package-local group codes 1–48. Original recording identifiers, original group identifiers, witness filenames and witness hashes were removed before the manuscript bundle was prepared. This supports recomputing aggregate counts and widths but not reconstruction of the original latent paths or mapping back to recordings. The file's historical name includes `anonymized`; no formal anonymization guarantee is asserted.

The public repository contains neither waveforms nor model weights. Obtain source data through PhysioNet and follow the original data license. Full experimental dependencies are listed separately in [experimental inputs](experimental_inputs.md).
