"""Saved-posterior HSMM score reconstruction and deterministic developmental figures.

No waveform, annotation, torch, decoder, or model interface is imported. Default
invocation verifies source only; --audit requires both completed decoder runs.
This is fixed-input numerical score accounting, not causal/localization evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import hsmm_site_replay as replay

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('author', 'corrected')
SITES = ('AV', 'PV', 'TV', 'MV')
MODES = ('published', 'score', 'score_quality')
MAPS = {'Healthy': (0, 1, 2, 3), 'Holosystolic': (0, 4, 2, 3),
        'Early-systolic': (0, 1, 2, 3, 4), 'Mid-systolic': (0, 1, 2, 3, 4)}
CLASS_NAMES = ('S1', 'Systole', 'S2', 'Diastole', 'Murmur')
CONFIDENCE_ATOL = 1e-12
ROUNDING_ATOL = 5e-7
AUTHOR_REPLAY_ATOL = 1e-6
EXPECTED = {
    'src/hsmm_site_replay.py': '411b72bd7302fac048c6c110135b909c89880947f0d4f0e48b23f09ba652b239',
    'src/hsmm_baseline.py': '3f58575db60040bf95ac6d342339d7044693e1605d72af3cce642ef557383de0',
    'src/hsmm_traceback_sensitivity.py': 'f2341e096385d3e3982f2016c5d3b0270d14b59138af4020dbd41d0b727817ee',
    'configs/hsmm_baseline.json': 'c743053cf9f6d66fc7809eb4286e3d98a1b21ea95b72b2a176f08821bbfbd919',
    'third_party/parallel-hsmm-murmur/src/segmenter.py': '4191848a42bfb89f4613f8276bf50db5997c5a29dc4c9dddd942e11221201790',
    'reviews/method/PHYSIOLOGICAL_STUDENT_GATE.md': '18a34c9ab651cfbb2349c32c710b0a2e4d7f692b54e0ecc5919c2a218a1f87bb',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def json_finite(value):
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def verify_sources(root=ROOT):
    for rel, expected in EXPECTED.items():
        if digest(root / rel) != expected:
            raise ValueError(f'Frozen dependency changed: {rel}')
    return {str(root / rel): expected for rel, expected in EXPECTED.items()}


def frame_centers(n):
    return (200 / 2 + np.arange(n, dtype=np.float64) * 80) / 4000


def mapped_path(path, model, n):
    if model not in MAPS:
        raise ValueError('Unknown selected topology')
    path = np.asarray(path)
    mapping = np.asarray(MAPS[model], dtype=np.int64)
    if (path.shape != (n,) or not np.issubdtype(path.dtype, np.integer)
            or (path < 0).any() or (path >= len(mapping)).any()):
        raise ValueError('Invalid local decoded state path')
    return mapping[path]


def account(posterior, healthy_path, murmur_path, summary):
    """Reconstruct selected emission means without invoking any decoder."""
    p = np.asarray(posterior)
    if (p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 5 or not len(p)
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1., atol=1e-5, rtol=1e-5)):
        raise ValueError('Expected saved float32 five-class posterior')
    if replay.boolean(summary['failed']):
        raise ValueError('A failed record cannot have a score explanation')
    h = mapped_path(healthy_path, 'Healthy', len(p))
    m = mapped_path(murmur_path, summary['murmur_model'], len(p))
    hp = p[np.arange(len(p)), h]
    mp = p[np.arange(len(p)), m]
    h32, m32 = float(hp.mean()), float(mp.mean())
    score32 = m32 - h32  # Original code casts each mean to Python float first.
    signed64 = mp.astype(np.float64) - hp.astype(np.float64)
    signed_mean64 = float(signed64.mean())
    signed_mean32 = float((mp - hp).mean())
    for key, actual in [('healthy_confidence', h32), ('murmur_confidence', m32),
                        ('score', score32), ('quality', max(h32, m32))]:
        if not math.isclose(float(summary[key]), actual, rel_tol=0., abs_tol=CONFIDENCE_ATOL):
            raise ValueError(f'Saved {key} does not reconstruct from selected paths')
    residual = score32 - signed_mean64
    if abs(residual) > ROUNDING_ATOL or abs(score32 - signed_mean32) > ROUNDING_ATOL:
        raise ValueError('Signed mean discrepancy exceeds frozen float32 tolerance')
    frames = pd.DataFrame({'frame': np.arange(len(p)), 'center_seconds': frame_centers(len(p)),
                           'healthy_local_state': healthy_path, 'murmur_local_state': murmur_path,
                           'healthy_original_class': h, 'murmur_original_class': m,
                           'healthy_selected_probability': hp, 'murmur_selected_probability': mp,
                           'signed_contribution_float64': signed64,
                           'normalized_contribution_float64': signed64 / len(p)})
    stats = {'frames': len(p), 'healthy_mean_float32': h32, 'murmur_mean_float32': m32,
             'reconstructed_stored_score': score32, 'signed_mean_float64': signed_mean64,
             'signed_mean_float32': signed_mean32, 'rounding_residual': residual,
             'sum_normalized_contributions': float((signed64 / len(p)).sum()),
             'positive_frames': int((signed64 > 0).sum()), 'negative_frames': int((signed64 < 0).sum()),
             'zero_frames': int((signed64 == 0).sum())}
    return stats, frames


def visit_account(site_rows, thresholds):
    """Report max-score witness and every distinct referral mechanism separately."""
    if not set(site_rows) <= set(SITES):
        raise ValueError('Unknown site')
    for row in site_rows.values():
        replay.validate_site(row)
    valid = [s for s in SITES if s in site_rows and not replay.boolean(site_rows[s]['failed'])]
    winner = max(valid, key=lambda s: float(site_rows[s]['score'])) if valid else None
    reasons = {}
    for mode in MODES:
        witnesses = []
        if not site_rows:
            witnesses.append({'site': None, 'reason': 'empty_inventory'})
        for s in SITES:
            if s not in site_rows:
                continue
            row = site_rows[s]
            if replay.boolean(row['failed']):
                witnesses.append({'site': s, 'reason': 'failed_site', 'detail': row.get('reason')})
                continue
            score = float(row['score'])
            if (mode == 'published' and score > 0) or (mode != 'published' and score >= thresholds[mode]):
                witnesses.append({'site': s, 'reason': 'positive_score' if mode == 'published' else 'score_threshold'})
            if mode in ('published', 'score_quality') and float(row['quality']) < .65:
                witnesses.append({'site': s, 'reason': 'low_quality'})
        reasons[mode] = {'referred': bool(witnesses), 'witnesses': witnesses,
                         'reason': ' | '.join(f"{w['site'] or 'visit'}:{w['reason']}" for w in witnesses)
                         or 'all_available_sites_nonreferral'}
    return {'winning_site': winner, 'winning_score': float(site_rows[winner]['score']) if winner else None,
            'minimum_valid_quality': min(float(site_rows[s]['quality']) for s in valid) if valid else None,
            'failed': not site_rows or any(replay.boolean(r['failed']) for r in site_rows.values()),
            'available_sites': [s for s in SITES if s in site_rows], 'decisions': reasons}


def select_cases(author, corrected):
    """Frozen examples: validation only; selection never depends on visual appeal."""
    def selected(frame):
        frame = frame.copy()
        frame['budget_seconds'] = frame.budget_seconds.map(replay.budget_key)
        frame = frame[frame.role.eq('pilot_validation') & frame.budget_seconds.eq('20')]
        if frame.patient_id.duplicated().any():
            raise ValueError('Duplicate displayed cohort visit')
        return frame.sort_values('patient_id').set_index('patient_id')
    a, c = selected(author), selected(corrected)
    if set(a.index) != set(c.index):
        raise ValueError('Variant displayed cohorts differ')
    c = c.loc[a.index]
    if not np.array_equal(a.Murmur, c.Murmur):
        raise ValueError('Variant visit labels differ')
    referral = a.score_referred.map(replay.boolean)
    categories = {'TP': a.Murmur.eq('Present') & referral, 'TN': a.Murmur.eq('Absent') & ~referral,
                  'FP': a.Murmur.eq('Absent') & referral, 'FN': a.Murmur.eq('Present') & ~referral}
    result = []
    for name, mask in categories.items():
        ids = a.index[mask].tolist()
        result.append({'selection': name, 'eligible_count': len(ids), 'patient_id': int(ids[0]) if ids else None,
                       'status': 'selected' if ids else 'empty_category', 'differing_modes': []})
    disagreements = pd.DataFrame({mode: a[f'{mode}_referred'].map(replay.boolean).ne(
        c[f'{mode}_referred'].map(replay.boolean)) for mode in MODES}, index=a.index)
    ids = disagreements.index[disagreements.any(axis=1)].tolist()
    patient = int(ids[0]) if ids else None
    result.append({'selection': 'decoder_referral_disagreement', 'eligible_count': len(ids),
                   'patient_id': patient, 'status': 'selected' if ids else 'empty_category',
                   'differing_modes': [mode for mode in MODES if disagreements.loc[patient, mode]] if ids else []})
    return result


def load_pair_array(row, directory, hashes):
    name = row['array']
    if not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('Invalid posterior array path')
    path = directory / 'posteriors' / name
    actual = digest(path)
    if actual != row['array_file_sha256']:
        raise ValueError('Saved posterior/path archive hash changed')
    hashes[str(path.resolve())] = actual
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    posterior = arrays['posterior']
    raw = hashlib.sha256(np.ascontiguousarray(posterior).tobytes()).hexdigest()
    if raw != row['posterior_raw_sha256'] or list(posterior.shape) != row['posterior_shape']:
        raise ValueError('Saved posterior bytes or shape changed')
    return arrays


def compare_summary(pair, original, tolerance):
    for key in ['failed', 'murmur_model']:
        if pair[key] != original[key]:
            raise ValueError(f'Pair/original {key} differs')
    if pair['reason'] != original['feature_guards']['reason']:
        raise ValueError('Pair/original failure reason differs')
    for key in ['score', 'quality', 'healthy_confidence', 'murmur_confidence']:
        a, b = pair[key], original[key]
        if a is None or b is None:
            if a is not b:
                raise ValueError('Pair/original missing confidence differs')
        elif not math.isclose(float(a), float(b), rel_tol=0., abs_tol=tolerance):
            raise ValueError('Pair/original numerical output differs')


def plot_case(case, variants, output):
    """Both variants, full acquired winner prefix; no cherry-picked time crop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(17, 12), sharex='col',
                             gridspec_kw={'height_ratios': [1.25, 1, 1, 1]})
    for col, variant in enumerate(VARIANTS):
        item = variants[variant]
        site = item['accounting']['winning_site']
        axes[0, col].set_title(f"{variant.title()} traceback | winning site: {site or 'none'}", fontsize=12)
        if item['frames'] is None:
            for ax in axes[:, col]:
                ax.text(.5, .5, 'No successful site: no path-score display', ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
        else:
            f, p = item['frames'], item['posterior']
            t = f.center_seconds.to_numpy()
            axes[0, col].imshow(p.T, aspect='auto', origin='lower', vmin=0, vmax=1, cmap='viridis',
                                extent=[t[0] - .01, t[-1] + .01, -.5, 4.5], interpolation='nearest')
            axes[0, col].set_yticks(range(5), CLASS_NAMES)
            axes[0, col].set_ylabel('Posterior (0–1)')
            axes[1, col].step(t, f.healthy_original_class, where='mid', color='#202020', lw=1.1, label='Healthy path')
            axes[1, col].step(t, f.murmur_original_class, where='mid', color='#d55e00', lw=1, alpha=.8,
                              label=f"Selected {item['summary']['murmur_model']} path")
            axes[1, col].set_yticks(range(5), CLASS_NAMES)
            axes[1, col].set_ylim(-.4, 4.6)
            axes[1, col].legend(loc='upper right', fontsize=7)
            d = f.signed_contribution_float64.to_numpy()
            axes[2, col].axhline(0, color='black', lw=.6)
            axes[2, col].fill_between(t, 0, d, where=d >= 0, color='#d55e00', alpha=.55)
            axes[2, col].fill_between(t, 0, d, where=d < 0, color='#0072b2', alpha=.55)
            axes[2, col].plot(t, d, color='#444444', lw=.45)
            axes[2, col].set_ylim(-1, 1)
            axes[2, col].set_ylabel('Signed contribution dₜ')
            axes[3, col].plot(t, np.cumsum(d) / len(d), color='#222222', label='Cumulative Σ dₜ / T')
            axes[3, col].axhline(item['stats']['reconstructed_stored_score'], color='#d55e00', ls='--', lw=1,
                                label='Saved shared-posterior score')
            axes[3, col].set_ylabel('Score accounting')
            axes[3, col].set_xlabel('Frame center within acquired audio (seconds)')
            axes[3, col].legend(loc='best', fontsize=7)
            axes[3, col].set_xlim(0, item['acquired_seconds'])
        threshold = item['thresholds']['score']
        score = item['accounting']['winning_score']
        text = [f"Original-run max score={score if score is not None else 'unscorable'}; score threshold={threshold:.8g}"]
        for mode in MODES:
            dec = item['accounting']['decisions'][mode]
            cut = 'score>0 / quality<0.65' if mode == 'published' else f"score≥{item['thresholds'][mode]:.8g}" + (' / quality<0.65' if mode == 'score_quality' else '')
            text.append(f"{mode} ({cut}): {'REFER' if dec['referred'] else 'NO REFERRAL'}; {dec['reason']}")
        sites = []
        for s in SITES:
            row = item['site_rows'].get(s)
            sites.append(f"{s}: missing" if row is None else (f"{s}: failed ({row.get('reason')})" if row['failed']
                         else f"{s}: score={row['score']:.6f}, q={row['quality']:.4f}"))
        text.append(' | '.join(sites[:2])); text.append(' | '.join(sites[2:]))
        if item['frames'] is not None:
            text.append(f"Saved score − mean(dₜ float64)={item['stats']['rounding_residual']:.3g}; "
                        f"selected topology={item['summary']['murmur_model']}")
            text.append('Cumulative trace uses final prefix paths and full T; not an online decision.')
        # Separate fixed-width text columns; long witness strings wrap deterministically.
        import textwrap
        text = '\n'.join('\n'.join(textwrap.wrap(line, width=88)) for line in text)
        fig.text(.055 + col * .495, .025, text, ha='left', va='bottom', fontsize=8.5, family='monospace')
    selections = ', '.join(case['selections'])
    modes = ', '.join(case['differing_modes']) or 'none'
    fig.suptitle(f"Visit {case['patient_id']} | recorded visit label: {case['Murmur']} | selection: {selections}\n"
                 f"20-second-per-site policy; differing referral modes: {modes}\n"
                 'Fixed-input score accounting; paths are model outputs, not expert annotations', fontsize=12)
    fig.subplots_adjust(left=.075, right=.98, top=.88, bottom=.27, hspace=.22, wspace=.21)
    for extension in ['png', 'pdf']:
        fig.savefig(output.with_suffix('.' + extension), dpi=180,
                    metadata={'Creator': 'Frozen HSMM score accounting audit'})
    plt.close(fig)


def audit(author_dir, corrected_dir, output, root=ROOT):
    if output.exists():
        raise ValueError('Refusing to overwrite score-accounting results')
    hashes = verify_sources(root)
    source_paths = [Path(__file__), root / 'HSMM_SCORE_ACCOUNTING_PROTOCOL.md', root / 'tests/test_hsmm_score_accounting.py']
    hashes.update({str(p.resolve()): digest(p) for p in source_paths})
    loaded, records, predictions = {}, {}, {}
    for variant, folder in [('author', author_dir), ('corrected', corrected_dir)]:
        loaded[variant] = replay.load_inputs(folder, root)
        hashes.update(loaded[variant][-1])
        rows = json.loads((folder / 'record_predictions.json').read_text())
        records[variant] = {(r['record'], replay.budget_key(r['budget_seconds'])): r for r in rows}
        p = pd.read_csv(folder / 'predictions.csv', float_precision='round_trip')
        p['budget_seconds'] = p.budget_seconds.map(replay.budget_key)
        predictions[variant] = p
    av, inv, _, _, _, _ = loaded['author']
    cv, ci, _, _, _, _ = loaded['corrected']
    if not av.equals(cv) or not inv.equals(ci):
        raise ValueError('Decoder metadata cohorts/inventories differ')
    run = json.loads((corrected_dir / 'run.json').read_text())
    for path, expected in [(author_dir / 'run.json', run['source_run_sha256']),
                           (author_dir / 'record_predictions.json', run['author_record_predictions_sha256'])]:
        if digest(path) != expected:
            raise ValueError('Original run does not match sensitivity provenance')
    for name, expected in run['checkpoint_sha256'].items():
        path = author_dir / 'models' / name
        if digest(path) != expected:
            raise ValueError('Checkpoint hash changed')
        hashes[str(path.resolve())] = expected
    paired_path = corrected_dir / 'paired_decoder_records.json'
    pairs = json.loads(paired_path.read_text())
    hashes[str(paired_path.resolve())] = digest(paired_path)
    pairs_by_key = {(r['record'], replay.budget_key(r['budget_seconds'])): r for r in pairs}
    if len(pairs_by_key) != len(pairs) or set(pairs_by_key) != set(records['author']):
        raise ValueError('Paired decoder record/budget denominator differs')
    reconstruction = []
    used_ordinals = []
    for key, pair in pairs_by_key.items():
        arrays = None
        if pair['array'] is not None:
            used_ordinals.append(pair['ordinal'])
            arrays = load_pair_array(pair, corrected_dir, hashes)
        for variant in VARIANTS:
            original = records[variant][key]
            if pair['role'] != original['role'] or pair['acquired_seconds'] != original['acquired_seconds']:
                raise ValueError('Pair/original role or audio cost differs')
            summary = pair[variant]
            compare_summary(summary, original, AUTHOR_REPLAY_ATOL if variant == 'author' else CONFIDENCE_ATOL)
            row = {'record': key[0], 'budget_seconds': key[1], 'variant': variant, 'role': pair['role'],
                   'failed': summary['failed'], 'reason': summary['reason'], 'murmur_model': summary['murmur_model']}
            if not summary['failed']:
                if arrays is None:
                    raise ValueError('Successful decoded record lacks saved posterior')
                stats, _ = account(arrays['posterior'], arrays[variant + '_healthy_states'],
                                   arrays[variant + '_murmur_states'], summary)
                row.update(stats)
            reconstruction.append(row)
    if sorted(used_ordinals) != list(range(run['posterior_arrays'])):
        raise ValueError('Saved decoder event ordinals are incomplete or duplicated')
    # No need to retain all posterior arrays in memory; reload only displayed winners.
    visit_rows, details = [], {}
    metadata = pd.read_csv(root / 'results/metadata/split_manifest.csv', usecols=['Patient ID', 'Murmur']).set_index('Patient ID')
    for variant in VARIANTS:
        thresholds = loaded[variant][4]
        for row in predictions[variant].to_dict('records'):
            pid, budget = int(row['patient_id']), row['budget_seconds']
            if metadata.loc[pid, 'Murmur'] != row['Murmur']:
                raise ValueError('Prediction label differs from metadata')
            selected = inv[inv.patient_id.eq(pid)]
            sites = {}
            for r in selected.itertuples():
                source = records[variant][(r.record, budget)]
                sites[r.site] = {**source, 'reason': source['feature_guards']['reason']}
            cuts = {m: thresholds[(budget, m)] for m in ['score', 'score_quality']}
            summary = visit_account(sites, cuts)
            score = summary['winning_score']
            if (score is None and not pd.isna(row['score'])) or (score is not None and not math.isclose(
                    score, float(row['score']), rel_tol=0., abs_tol=CONFIDENCE_ATOL)):
                raise ValueError('Max site score differs from saved visit score')
            for mode in MODES:
                if summary['decisions'][mode]['referred'] != replay.boolean(row[mode + '_referred']):
                    raise ValueError('Visit referral rule does not reconstruct')
            details[(variant, pid, budget)] = {'accounting': summary, 'thresholds': cuts, 'site_rows': sites}
            visit_rows.append({'variant': variant, 'patient_id': pid, 'role': row['role'], 'budget_seconds': budget,
                               'winning_site': summary['winning_site'], 'max_score': score,
                               **{m + '_reason': summary['decisions'][m]['reason'] for m in MODES}})
    selection = select_cases(predictions['author'], predictions['corrected'])
    output.mkdir(parents=True)
    figures = output / 'figures'; figures.mkdir()
    pd.DataFrame(reconstruction).to_csv(output / 'record_reconstruction.csv', index=False)
    pd.DataFrame(visit_rows).to_csv(output / 'visit_witnesses.csv', index=False)
    write_json(output / 'case_selection.json', selection)
    cases = []
    for pid in sorted({r['patient_id'] for r in selection if r['patient_id'] is not None}):
        case = {'patient_id': pid, 'Murmur': metadata.loc[pid, 'Murmur'],
                'selections': [r['selection'] for r in selection if r['patient_id'] == pid],
                'differing_modes': next((r['differing_modes'] for r in selection
                                         if r['patient_id'] == pid and r['selection'] == 'decoder_referral_disagreement'), [])}
        displays, serial = {}, {}
        for variant in VARIANTS:
            item = dict(details[(variant, pid, '20')])
            site = item['accounting']['winning_site']
            item.update(frames=None, posterior=None)
            if site is not None:
                record = item['site_rows'][site]['record']
                pair = pairs_by_key[(record, '20')]
                arrays = load_pair_array(pair, corrected_dir, hashes)
                item['summary'] = pair[variant]
                item['stats'], item['frames'] = account(arrays['posterior'], arrays[variant + '_healthy_states'],
                                                        arrays[variant + '_murmur_states'], pair[variant])
                item['posterior'] = arrays['posterior']
                item['acquired_seconds'] = pair['acquired_seconds']
                item['frames'].to_csv(output / f'case_{pid}_{variant}_winning_site_frames.csv', index=False)
            displays[variant] = item
            serial[variant] = json_finite({k: v for k, v in item.items() if k not in ['frames', 'posterior']})
        plot_case(case, displays, figures / f'case_{pid}')
        cases.append({**case, 'variants': serial})
    write_json(output / 'displayed_case_accounting.json', cases)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Input changed during audit')
    successful = [r for r in reconstruction if not r['failed']]
    report = {'status': 'pass', 'record_budget_variant_rows': len(reconstruction),
              'successful_reconstructions': len(successful), 'failed_rows_retained': len(reconstruction) - len(successful),
              'max_absolute_rounding_residual': max((abs(r['rounding_residual']) for r in successful), default=0.),
              'confidence_absolute_tolerance': CONFIDENCE_ATOL, 'signed_mean_rounding_absolute_tolerance': ROUNDING_ATOL,
              'original_author_replay_absolute_tolerance': AUTHOR_REPLAY_ATOL,
              'visit_budget_variant_rows': len(visit_rows), 'unique_displayed_visits': len(cases),
              'display_selection': selection, 'waveforms_read': 0, 'expert_annotations_read': 0,
              'reserved_posteriors_read': 0, 'GPU_used': False, 'model_or_decoder_calls': 0,
              'limits': 'Technical fixed-input score accounting only; not causal attribution, global decoder optimality, clinical localization, or methodological novelty.',
              'input_sha256': hashes,
              'output_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output / 'audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['input_sha256', 'output_sha256']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--author-directory', type=Path, default=ROOT / 'results/hsmm_baseline_v1')
    parser.add_argument('--corrected-directory', type=Path, default=ROOT / 'results/hsmm_traceback_sensitivity_v1')
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'results/hsmm_score_accounting_v1')
    args = parser.parse_args()
    if args.audit:
        audit(args.author_directory, args.corrected_directory, args.output_directory)
    else:
        print(json.dumps({'status': 'frozen_sources_verified', 'clinical_outputs_read': False,
                          'sources_sha256': verify_sources()}, indent=2))


if __name__ == '__main__':
    main()
