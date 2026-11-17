"""Separate, frozen traceback-index sensitivity; never fits a neural model.

Research derivative of Cambridge 2022 code, modified 2026-11-07. Distributed under
GAP Available Source License v1.0, without warranty; see ../HSMM_LICENSE.md.
Default invocation verifies source/build only. --evaluate explicitly evaluates
the five completed original checkpoints. The original driver/vendor are unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import hsmm_baseline as baseline

ROOT = baseline.ROOT
EXPECTED = {
    'src/hsmm_baseline.py': '3f58575db60040bf95ac6d342339d7044693e1605d72af3cce642ef557383de0',
    'configs/hsmm_baseline.json': 'c743053cf9f6d66fc7809eb4286e3d98a1b21ea95b72b2a176f08821bbfbd919',
    'HSMM_PROTOCOL.md': '85b5cf5c319a847f7593dfd0e1a1a3e0b23524e2cf1738e5d3aafa0b05f6c29f',
    'HSMM_TRACEBACK_PROTOCOL.md': 'bc605839961060d862a70178d30194fef35149cdeff21456283ce124497ae68a',
    'third_party/parallel-hsmm-murmur/viterbi_hmm.pyx': 'c17f8ddf04ba572ae123218ad73eb1d2124492f972933e2b48aabb926eaa980f',
}
DERIVATIVE_HEADER = (
    '# Local derivative, modified 2026-11-07: preserve the old endpoint predecessor\n'
    '# during traceback. Forward recurrence and all other decoder arithmetic unchanged.\n'
    '# Source commit: 5f9d14ee7f9215bf8cf523341241ee9d79be79ae.\n'
    '# Retains GAP Available Source License v1.0; see ../HSMM_LICENSE.md.\n'
)
DECODE_ERRORS = (ValueError, FloatingPointError, ZeroDivisionError, IndexError)
_CORRECTED = None


def expected_corrected_source(original):
    replacements = [
        ('    cdef int current_state = -1\n',
         '    cdef int current_state = -1\n    cdef int previous_state = -1\n'),
        ('    while t > 0:\n        d = psi_duration[t, current_state]\n',
         '    while t > 0:\n        previous_state = psi[t, current_state]\n        d = psi_duration[t, current_state]\n'),
        ('        current_state = psi[t, current_state]\n',
         '        current_state = previous_state\n'),
    ]
    for old, new in replacements:
        if original.count(old) != 1:
            raise ValueError('Unexpected upstream traceback source')
        original = original.replace(old, new)
    return DERIVATIVE_HEADER + original


def verify_sources():
    for relative, expected in EXPECTED.items():
        if baseline.digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    original = (baseline.UPSTREAM / 'viterbi_hmm.pyx').read_text()
    if (ROOT / 'src/hsmm_traceback_corrected.pyx').read_text() != expected_corrected_source(original):
        raise ValueError('Corrected decoder differs from the frozen narrow patch')
    return {relative: baseline.digest(ROOT / relative) for relative in [
        *EXPECTED, 'src/hsmm_traceback_sensitivity.py', 'src/hsmm_traceback_corrected.pyx']}


def load_corrected_segmenter(cfg):
    """Load isolated segmenter globals; never rebind the author's decoder."""
    global _CORRECTED
    verify_sources()
    _, author_segmenter, _ = baseline.load_author_modules(cfg)
    if _CORRECTED is not None:
        return _CORRECTED
    build = ROOT / 'reviews/method/hsmm_traceback_build'
    build.mkdir(parents=True, exist_ok=True)
    from pyximport.pyximport import build_module
    name = 'hsmm_traceback_corrected'
    binary = build_module(name, str(ROOT / 'src' / (name + '.pyx')),
                          pyxbuild_dir=str(build), language_level=3)
    if not Path(binary).resolve().is_relative_to(build.resolve()):
        raise ValueError('Corrected extension outside declared build directory')
    extension_spec = importlib.util.spec_from_file_location(name, binary)
    extension = importlib.util.module_from_spec(extension_spec)
    extension_spec.loader.exec_module(extension)
    segmenter_spec = importlib.util.spec_from_file_location(
        'hsmm_isolated_corrected_segmenter', baseline.UPSTREAM / 'src/segmenter.py')
    segmenter = importlib.util.module_from_spec(segmenter_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        segmenter_spec.loader.exec_module(segmenter)
    finally:
        sys.dont_write_bytecode = old_bytecode
    original_binding = author_segmenter.viterbi_hmm
    segmenter.viterbi_hmm = extension
    assert author_segmenter.viterbi_hmm is original_binding
    assert segmenter.double_duration_viterbi.__globals__ is segmenter.__dict__
    _CORRECTED = segmenter, str(binary)
    return _CORRECTED


def validate_posterior(posterior):
    array = np.asarray(posterior)
    if (array.ndim != 2 or array.shape[1] != 5 or not len(array)
            or not np.isfinite(array).all() or (array < 0).any()
            or not np.allclose(array.sum(axis=1), 1., atol=1e-5, rtol=1e-5)):
        raise ValueError('Expected a finite nonnegative five-state probability matrix')
    return np.ascontiguousarray(array, dtype=np.float32)


def posterior_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def decode_summary(segmenter, posterior, fs, kwargs):
    """Use the same declared numerical-failure convention as the base adapter."""
    try:
        decoded = segmenter.double_duration_viterbi(posterior, fs, **kwargs)
        healthy, murmur = float(decoded[2]), float(decoded[3])
        if not np.isfinite([healthy, murmur]).all():
            raise FloatingPointError('Nonfinite HSMM confidence')
        return decoded, {'failed': False, 'score': murmur - healthy,
                         'quality': max(healthy, murmur), 'healthy_confidence': healthy,
                         'murmur_confidence': murmur, 'murmur_model': decoded[4],
                         'reason': None}, None
    except DECODE_ERRORS as exc:
        return None, {'failed': True, 'score': None, 'quality': None,
                      'healthy_confidence': None, 'murmur_confidence': None,
                      'murmur_model': None, 'reason': f'hsmm_failure:{type(exc).__name__}'}, exc


class DecoderPairProxy:
    """Persist one exact posterior and return the corrected member of the pair."""
    def __init__(self, author_segmenter, corrected_segmenter, output):
        self.author = author_segmenter
        self.corrected = corrected_segmenter
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = []

    def double_duration_viterbi(self, posterior, fs, **kwargs):
        try:
            array = validate_posterior(posterior)
        except ValueError:
            # Preserve an event even when validation rejects an all-zero matrix;
            # evaluate() will retain the record through its existing fallback.
            array = np.asarray(posterior, dtype=np.float32)
            path = self.output / f'{len(self.events):05d}.npz'
            np.savez_compressed(path, posterior=array)
            failure = {'failed': True, 'score': None, 'quality': None,
                       'healthy_confidence': None, 'murmur_confidence': None,
                       'murmur_model': None, 'reason': 'hsmm_failure:ValueError'}
            event = {'ordinal': len(self.events), 'array': path.name,
                     'array_file_sha256': baseline.digest(path),
                     'posterior_raw_sha256': posterior_digest(array),
                     'posterior_shape': list(array.shape), 'author': failure,
                     'corrected': dict(failure), 'author_decoder_seconds': 0.,
                     'corrected_decoder_seconds': 0., 'posterior_validation_failed': True}
            baseline.write_json(path.with_suffix('.json'), event)
            self.events.append(event)
            raise
        before = posterior_digest(array)
        index = len(self.events)
        started = time.perf_counter()
        author, author_summary, _ = decode_summary(self.author, array, fs, kwargs)
        author_seconds = time.perf_counter() - started
        if posterior_digest(array) != before:
            raise RuntimeError('Author decoder mutated the shared posterior')
        started = time.perf_counter()
        corrected, corrected_summary, error = decode_summary(self.corrected, array, fs, kwargs)
        corrected_seconds = time.perf_counter() - started
        if posterior_digest(array) != before:
            raise RuntimeError('Corrected decoder mutated the shared posterior')
        arrays = {'posterior': array}
        for name, decoded in [('author', author), ('corrected', corrected)]:
            if decoded is not None:
                arrays[name + '_healthy_states'] = np.asarray(decoded[0])
                arrays[name + '_murmur_states'] = np.asarray(decoded[1])
        path = self.output / f'{index:05d}.npz'
        np.savez_compressed(path, **arrays)
        event = {'ordinal': index, 'array': path.name, 'array_file_sha256': baseline.digest(path),
                 'posterior_raw_sha256': before, 'posterior_shape': list(array.shape),
                 'author': author_summary, 'corrected': corrected_summary,
                 'author_decoder_seconds': author_seconds, 'corrected_decoder_seconds': corrected_seconds}
        baseline.write_json(path.with_suffix('.json'), event)
        self.events.append(event)
        if error is not None:
            raise error
        return corrected


def associate_records(events, record_rows):
    """Connect proxy calls to the original evaluate() traversal, retaining failures."""
    event_index = 0
    associated = []
    for row in record_rows:
        reason = row['feature_guards']['reason']
        reaches_decoder = reason is None or reason.startswith('hsmm_failure:')
        value = {'record': row['record'], 'role': row['role'],
                 'budget_seconds': row['budget_seconds'],
                 'acquired_seconds': row['acquired_seconds']}
        if reaches_decoder:
            if event_index >= len(events):
                raise ValueError('Missing saved decoder event')
            value.update(events[event_index])
            event_index += 1
            for key in ['failed', 'score', 'quality', 'healthy_confidence', 'murmur_confidence', 'murmur_model']:
                if value['corrected'][key] != row[key]:
                    raise ValueError(f'Corrected proxy/result mapping differs: {key}')
        else:
            value.update(ordinal=None, array=None, author={k: row[k] for k in [
                'failed', 'score', 'quality', 'healthy_confidence', 'murmur_confidence', 'murmur_model']})
            value['author']['reason'] = reason
            value['corrected'] = dict(value['author'])
        associated.append(value)
    if event_index != len(events):
        raise ValueError('Unmapped saved decoder events')
    return associated


def compare_author_replay(associated, original_rows, tolerance=1e-6):
    index = {(r['record'], str(r['budget_seconds'])): r for r in original_rows}
    if len(index) != len(original_rows) or len(associated) != len(original_rows):
        raise ValueError('Author replay record denominator differs')
    differences, maximum = [], 0.
    for row in associated:
        key = (row['record'], str(row['budget_seconds']))
        expected = index.pop(key)
        actual = row['author']
        for name in ['failed', 'murmur_model']:
            if actual[name] != expected[name]:
                differences.append({'record': key[0], 'budget_seconds': key[1], 'field': name})
        if actual['reason'] != expected['feature_guards']['reason']:
            differences.append({'record': key[0], 'budget_seconds': key[1], 'field': 'reason'})
        for name in ['score', 'quality', 'healthy_confidence', 'murmur_confidence']:
            if actual[name] is None or expected[name] is None:
                equal = actual[name] is expected[name]
            else:
                difference = abs(float(actual[name]) - float(expected[name]))
                maximum = max(maximum, difference)
                equal = difference <= tolerance
            if not equal:
                differences.append({'record': key[0], 'budget_seconds': key[1], 'field': name})
        if row['role'] != expected['role'] or row['acquired_seconds'] != expected['acquired_seconds']:
            differences.append({'record': key[0], 'budget_seconds': key[1], 'field': 'role_or_cost'})
    return {'status': 'passed' if not differences else 'failed', 'records': len(associated),
            'absolute_tolerance': tolerance, 'maximum_absolute_difference': maximum,
            'differences': differences}


def load_completed_models(source, cfg, device):
    source = Path(source)
    run = json.loads((source / 'run.json').read_text())
    if run['status'] != 'complete' or run['config'] != cfg or run['reserved_predictions'] != 0:
        raise ValueError('A complete, matching, reserved-free original run is required')
    for relative, expected in run['input_sha256'].items():
        if baseline.digest(ROOT / relative) != expected:
            raise ValueError(f'Source-run input changed: {relative}')
    # Reuse the metadata guard without reading any waveforms or annotations.
    visits, _, _ = baseline.read_metadata()
    cache = ROOT / cfg['training_cache_directory']
    cache_manifest = json.loads((cache / 'manifest.json').read_text())
    for filename in ['records.csv', 'inner_folds.csv']:
        if baseline.digest(cache / filename) != cache_manifest['files_sha256'][filename]:
            raise ValueError('Training membership cache changed')
    records = pd.read_csv(cache / 'records.csv')
    if (not records.role.eq('train').all() or not records.visit_label.isin(['Present', 'Absent']).all()
            or not set(records.patient_id) <= set(visits.loc[visits.role.eq('train'), 'patient_id'])):
        raise ValueError('Checkpoint provenance is outside known-label training')
    neural, author, _ = baseline.load_author_modules(cfg)
    models, hashes = [], {}
    for fold in range(cfg['inner_folds']):
        path = source / 'models' / f'fold_{fold}.pt'
        hashes[path.name] = baseline.digest(path)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        expected_fit = set(records.loc[records.inner_fold.ne(fold), 'group_id'])
        expected_valid = set(records.loc[records.inner_fold.eq(fold), 'group_id'])
        if (checkpoint['fold'] != fold or checkpoint['architecture'] != cfg['architecture']
                or set(checkpoint['fit_group_ids']) != expected_fit
                or set(checkpoint['inner_validation_group_ids']) != expected_valid
                or expected_fit & expected_valid):
            raise ValueError('Checkpoint fold/group provenance differs')
        model = neural.instantiate_model(cfg['architecture'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(device).eval()
        models.append(model)
    if len(models) != 5:
        raise ValueError('Exactly five original checkpoint models are required')
    return models, author, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--source-directory', type=Path, default=ROOT / 'results/hsmm_baseline_v1')
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'results/hsmm_traceback_sensitivity_v1')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    sources = verify_sources()
    cfg = json.loads((ROOT / 'configs/hsmm_baseline.json').read_text())
    corrected, binary = load_corrected_segmenter(cfg)
    if not args.evaluate:
        print(json.dumps({'status': 'sources_and_corrected_build_verified', 'evaluation_performed': False,
                          'training_performed': False, 'sources_sha256': sources,
                          'corrected_extension_sha256': baseline.digest(binary)}, indent=2))
        return
    torch.set_num_threads(cfg['torch_threads'])
    baseline.reset_seed(cfg['seed'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if args.output_directory.exists():
        raise ValueError('Refusing to overwrite sensitivity results')
    models, author, checkpoints = load_completed_models(args.source_directory, cfg, device)
    original_records_path = args.source_directory / 'record_predictions.json'
    original_records_sha = baseline.digest(original_records_path)
    original_run_sha = baseline.digest(args.source_directory / 'run.json')
    args.output_directory.mkdir(parents=True)
    proxy = DecoderPairProxy(author, corrected, args.output_directory / 'posteriors')
    run = {'status': 'running', 'started_unix': time.time(), 'training_performed': False,
           'source_directory': str(args.source_directory.resolve()), 'config': cfg,
           'checkpoint_sha256': checkpoints, 'source_run_sha256': original_run_sha,
           'author_record_predictions_sha256': original_records_sha, 'sources_sha256': sources,
           'corrected_extension_sha256': baseline.digest(binary), 'device': str(device),
           'reserved_predictions': 0, 'heldout_annotations_read': 0,
           'reporting': 'Report original and corrected variants regardless of direction; no selection.',
           'primary_report': str((args.source_directory / 'metrics.csv').resolve()),
           'sensitivity_report': str((args.output_directory / 'metrics.csv').resolve()),
           'timing_scope': 'Repeated neural evaluation, both decoders, and serialization; not deployment latency.'}
    baseline.write_json(args.output_directory / 'run.json', run)
    try:
        baseline.evaluate(models, proxy, cfg, args.output_directory, device)
        current_rows = json.loads((args.output_directory / 'record_predictions.json').read_text())
        associated = associate_records(proxy.events, current_rows)
        baseline.write_json(args.output_directory / 'paired_decoder_records.json', associated)
        if baseline.digest(original_records_path) != original_records_sha:
            raise ValueError('Original record predictions changed during sensitivity evaluation')
        comparison = compare_author_replay(associated, json.loads(original_records_path.read_text()))
        baseline.write_json(args.output_directory / 'author_replay_verification.json', comparison)
        if comparison['status'] != 'passed':
            raise ValueError('Author replay differs; comparative interpretation is blocked')
        if verify_sources() != sources:
            raise ValueError('Sensitivity source changed during evaluation')
        if baseline.digest(args.source_directory / 'run.json') != original_run_sha:
            raise ValueError('Original run changed during evaluation')
        for filename, expected in checkpoints.items():
            if baseline.digest(args.source_directory / 'models' / filename) != expected:
                raise ValueError('Original checkpoint changed during evaluation')
        run.update(status='complete', posterior_arrays=len(proxy.events),
                   elapsed_seconds=time.time() - run['started_unix'])
    except Exception as exc:
        run.update(status='failed', error=repr(exc), elapsed_seconds=time.time() - run['started_unix'])
        raise
    finally:
        baseline.write_json(args.output_directory / 'run.json', run)


if __name__ == '__main__':
    main()
