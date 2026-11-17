# Experimental inputs and conditional rerun procedure

The release supports synthetic algorithm checks and reconstruction of publication
tables and figures from the bundled evidence. Repeating the original experiments
requires additional recordings, fitted models, posterior arrays and provenance
records. Those assets are not included. The commands below describe that
conditional workflow; they are not an end-to-end reproduction claim for a fresh
download of this repository.

All paths below are relative to the repository root. Run experiments in a separate
working copy. Preserve the released evidence and the original research outputs.

## Public sources

The recordings came from the public training release of the **CirCor DigiScope
Phonocardiogram Dataset, version 1.0.3**, by Oliveira et al. (2022),
[DOI: 10.13026/tshs-mw03](https://doi.org/10.13026/tshs-mw03).
Obtain data and its license from the
[official PhysioNet release](https://physionet.org/content/circor-heart-sound/1.0.3/).
The dataset publication is
[DOI: 10.1109/JBHI.2021.3137048](https://doi.org/10.1109/JBHI.2021.3137048).
Downloading these public files does not supply the study-specific splits,
checkpoints or posterior ledgers listed below.

The duration/topology construction and teacher architecture depend on the
[parallel-hsmm-murmur source](https://github.com/am2234/parallel-hsmm-murmur/tree/5f9d14ee7f9215bf8cf523341241ee9d79be79ae),
fixed at commit `5f9d14ee7f9215bf8cf523341241ee9d79be79ae`:

```bash
mkdir -p phase9/third_party
git clone https://github.com/am2234/parallel-hsmm-murmur.git phase9/third_party/parallel-hsmm-murmur
git -C phase9/third_party/parallel-hsmm-murmur checkout --detach 5f9d14ee7f9215bf8cf523341241ee9d79be79ae
```

Use a clean checkout and retain upstream notices. Its GAP Available Source
License v1.0 permits academic noncommercial use under its stated terms; see
[`phase9/HSMM_LICENSE.md`](../phase9/HSMM_LICENSE.md). The phase10 potential
capture uses `src/segmenter.py`; teacher target materialization also uses
`src/neural_networks.py`.

## Assets required outside this release

### Initial 24-person bounds and window analyses

Supply the original, hash-matching files:

```text
phase9/results/metadata/split_manifest.csv
phase9/results/physiological_distill_v1/seed_20261107_alpha_1p0/evaluation/decoder_records.json
phase9/results/physiological_distill_v1/seed_20261107_alpha_1p0/evaluation/posteriors/<referenced array files>
```

The ledger identifies the posterior archive filenames and their SHA-256 values.
`materialize_development.py` selects one canonical-site record for each of 24
linked people using the declared deterministic selection rule, checks the saved
array hashes, and writes:

```text
phase10/results/development_inputs_v1/selection_manifest.json
phase10/results/development_inputs_v1/materialization.json
phase10/results/development_inputs_v1/00.npz ... 23.npz
```

These generated inputs are required by the subsequent initial bounds, baseline,
fixed-query and selection-aware window analyses. They are also used by one
data-dependent integration test. This stage reads saved posteriors and selection
metadata; it does not train a predictor.

### 48-person breadth analysis

Supply `phase9/results/metadata/record_inventory.csv` and the initial selection
manifest above. For each of the following eight directories, supply
`decoder_records.json` and every posterior archive referenced for the selected
48 people at both the 20-second and full-record budgets:

```text
phase9/results/hsmm_int8_control_v1/cpu_float/
phase9/results/hsmm_int8_control_v1/cpu_int8/
phase9/results/physiological_distill_v1/seed_20261107_alpha_0p0/evaluation/
phase9/results/physiological_distill_v1/seed_20261107_alpha_1p0/evaluation/
phase9/results/physiological_distill_v1/seed_20261108_alpha_0p0/evaluation/
phase9/results/physiological_distill_v1/seed_20261108_alpha_1p0/evaluation/
phase9/results/physiological_distill_v1/seed_20261109_alpha_0p0/evaluation/
phase9/results/physiological_distill_v1/seed_20261109_alpha_1p0/evaluation/
```

Posterior archives reside under each directory's `posteriors/` subdirectory.
The breadth selection excludes the original 24 people. Eight predictors and two
budgets yield 768 repeated record conditions from 48 people, not 768 independent
participants. The audit additionally reads the witness archives created by the
breadth run.

### Fresh-start distillation

Supply the following original assets:

```text
phase9/results/metadata/manifest.json
phase9/results/metadata/record_inventory.csv
phase9/results/metadata/split_manifest.csv
phase9/results/metadata/<every CSV named by manifest.json output_sha256>
phase9/data/circor_metadata/SHA256SUMS.txt
phase9/data/development_wav/<the selected 120 original WAV files>
phase9/results/hsmm_baseline_v1/run.json
phase9/results/hsmm_baseline_v1/models/fold_0.pt ... fold_4.pt
phase9/results/hsmm_training_cache_v1/manifest.json
phase9/results/hsmm_training_cache_v1/records.csv
phase9/HSMM_PROTOCOL.md
phase10/results/development_inputs_v1/selection_manifest.json
```

`phase9/HSMM_PROTOCOL.md` is required by the teacher run's input-hash verification
and is not part of this minimal source export. The supplied teacher run must have
`status: complete`, the matching baseline configuration, and the original
input/checkpoint hashes. `physiological_distill.json` records the five expected
checkpoint hashes. The cache manifest and `records.csv` establish inner-fold
membership; this phase10 materializer does not read the old training cache's
annotation-gap-collapsed feature arrays or annotation files.

The selection contains 80 fitting people, 16 monitoring people and the same 24
development people as the initial bounds analysis. Every waveform is checked
against the official checksum before its six-second prefix is used. Generated
continuous features and teacher targets are stored in:

```text
phase10/results/query_gap_inputs_v1/selection.json
phase10/results/query_gap_inputs_v1/manifest.json
phase10/results/query_gap_inputs_v1/000.npz ... 119.npz
```

Fresh training produces nine final models: three arms across three seeds.
Independent reinference additionally requires these final checkpoints, saved
development posterior archives, per-record metrics and training receipts. The
five original teacher checkpoints remain necessary for the teacher checks.

### Warm-start distillation

Supply or first generate the complete fresh-start results under
`phase10/results/query_gap_training_v1/`, including `run.json`, `summary.json`,
`independent_audit.json`, and each seed's emission-KL `final.pt` and run records.
Warm-start training also uses the same complete query-input cache. It initializes
four arms from each of the three matching emission-KL checkpoints and produces
12 additional models. Its audit reads those checkpoints, posteriors, metrics,
training orders and nominal-target records.

The released phase9 files provide the imports needed by phase10; they are not a
complete release of all earlier model-training, metadata-preparation and INT8
evaluation workflows. This repository does not reconstruct the missing
study-specific phase9 artifacts from the public data alone.

## Frozen hashes and release-source provenance

[`provenance/source_export.json`](../provenance/source_export.json) records each
exported file's original and release SHA-256 values. Release license notices were
added to five phase10 files:

```text
src/path_bounds.py
src/query_gap_distillation.py
src/query_training_common.py
src/segment_projection_baseline.py
src/selection_aware_bounds.py
```

These paths are relative to `phase10/`. The notices change file hashes even
though the computational implementation is unchanged. Historical run and cache
manifests refer to the original file bytes. Consequently:

- To audit an original frozen run, use an isolated copy with the original source
  versions and inputs matching its receipts. Validate original source files
  against the `original_sha256` values in the export manifest.
- To repeat the workflow with the release source, materialize fresh phase10
  inputs and generate new run receipts from the release files. Preserve the
  original phase9 inputs and their required hashes. Report this as a new run
  under the released source version.

Do not edit old receipts, substitute new hashes into historical manifests, or
disable validation to make a mismatched run pass. Existing evidence is retained
as the record of the original experiments; new execution may differ with
software, hardware or numerical-library versions.

## Conditional execution order

Use a fresh working copy with the external inputs above. The commands assume
`python` selects the scientific environment. Original phase10 execution used
Python 3.8.20, NumPy 1.24.3, Numba 0.58.1, SciPy 1.10.1, pandas 2.0.3,
scikit-learn 1.3.2 and PyTorch 2.4.1+cu121 with CUDA 12.1. Cython is additionally
needed for the `pyximport` import in `materialize_development.py`; the later
potential-capture helper stubs out that import. Plotting requires matplotlib.
The distillation materializers, training drivers and reinference audits use CUDA
explicitly. Synthetic algorithm tests can run on CPU.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

First verify the mathematical implementation and produce the initial study:

```bash
python phase10/tests/test_path_bounds.py
python phase10/tests/test_cheap_queries.py
python phase10/tests/test_selection_aware.py
python phase10/tests/test_query_gap_distillation.py
python phase10/src/materialize_development.py
python phase10/src/run_development_bounds.py
python phase10/src/audit_and_compare.py
python phase10/src/cheap_queries_and_winners.py
python phase10/src/run_selection_aware.py
python phase10/src/audit_selection_windows.py
```

With the original teacher assets and waveforms in place, run the controlled
distillation studies in this order:

```bash
python phase10/tests/test_query_training.py
python phase10/src/materialize_query_training.py
python phase10/src/train_query_pilot.py
python phase10/src/audit_query_pilot.py
python phase10/src/train_query_warmstart.py
python phase10/src/audit_query_warmstart.py
```

With all eight frozen posterior collections in place, run the breadth analysis:

```bash
python phase10/src/run_functional_breadth.py
python phase10/src/audit_functional_breadth.py
```

This breadth stage does not depend on the newly fitted distillation models; its
inputs are the earlier frozen phase9 predictors. Optional historical reporting
scripts can then render their corresponding completed result directories:

```bash
python phase10/src/report_feasibility.py
python phase10/src/report_query_pilots.py
python phase10/src/report_functional_breadth.py
```

Most experimental drivers refuse an existing output directory, and some training
code resumes only when its saved source/configuration receipts match. Never
remove an evidence directory to force a rerun. Use a new working copy and retain
its logs, input selection, source hashes, numerical checks and all results,
including unsuccessful arms.

The released aggregate checks do not reinfer the neural models, remeasure runtime,
or revalidate every saved witness against its original posterior array. Synthetic
checks establish implementation behavior on their declared test cases. Neither
kind of check establishes clinical validity, a new optimization principle, or
superiority of support-gap distillation; both recorded distillation gates failed.
