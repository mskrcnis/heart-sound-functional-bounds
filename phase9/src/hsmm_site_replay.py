"""Label-free, complete-site OR early-referral replay of completed HSMM outputs.

No model, waveform, annotation, clinical-label or GPU interface is imported.
The original/corrected HSMM driver and its frozen decisions remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES = ('AV', 'PV', 'TV', 'MV')
ORDERS = tuple(permutations(SITES))
MODES = ('published', 'score', 'score_quality')
BUDGETS = ('20', 'full')
ROLES = ('calibration', 'pilot_validation')
OVERHEADS = (0, 2, 5)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in ('True', 'true'):
        return True
    if value in ('False', 'false'):
        return False
    raise ValueError('Expected an explicit Boolean')


def budget_key(value):
    if str(value) == 'full':
        return 'full'
    if float(value) == 20:
        return '20'
    raise ValueError('Only frozen 20-second/full budgets are permitted')


def validate_site(row):
    failed = boolean(row['failed'])
    seconds = float(row['acquired_seconds'])
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid acquired duration')
    if not failed and not all(math.isfinite(float(row[k])) for k in ['score', 'quality']):
        raise ValueError('A scored site must have finite score and quality')
    return failed, seconds


def site_witness(row, mode, threshold=None, quality_threshold=.65):
    """A completed-site binary referral witness; accepts no future-site data."""
    failed, _ = validate_site(row)
    if mode not in MODES or quality_threshold != .65:
        raise ValueError('Unfrozen referral rule')
    if mode != 'published' and (threshold is None or math.isnan(float(threshold))):
        raise ValueError('A calibrated rule requires its frozen non-NaN threshold')
    if failed:
        return 'failed_site'
    if mode == 'published' and float(row['score']) > 0:
        return 'positive_score'
    if mode != 'published' and float(row['score']) >= threshold:
        return 'score_threshold'
    if mode in ('published', 'score_quality') and float(row['quality']) < quality_threshold:
        return 'low_quality'
    return None


def replay_visit(site_rows, order, mode, threshold=None, quality_threshold=.65):
    """Acquire whole available sites until a witness; otherwise finish every site."""
    if tuple(order) not in ORDERS or not set(site_rows) <= set(SITES):
        raise ValueError('Invalid site inventory or permutation')
    if mode not in MODES or quality_threshold != .65:
        raise ValueError('Unfrozen referral rule')
    if mode != 'published' and (threshold is None or math.isnan(float(threshold))):
        raise ValueError('Invalid frozen threshold')
    if not site_rows:
        return {'referred': True, 'acquired_seconds': 0., 'sites_acquired': 0,
                'switches': 0, 'visited_sites': '', 'stop_reason': 'empty_inventory'}
    visited, durations, reason = [], [], None
    for site in order:
        if site not in site_rows:
            continue
        row = site_rows[site]
        _, seconds = validate_site(row)
        visited.append(site); durations.append(seconds)
        reason = site_witness(row, mode, threshold, quality_threshold)
        if reason is not None:
            break
    return {'referred': reason is not None, 'acquired_seconds': math.fsum(durations),
            'sites_acquired': len(visited), 'switches': max(0, len(visited)-1),
            'visited_sites': '|'.join(visited), 'stop_reason': reason or 'all_sites_nonreferral'}


def all_site_reference(site_rows, mode, threshold=None):
    for row in site_rows.values():
        validate_site(row)
    return {'referred': not site_rows or any(site_witness(r, mode, threshold) is not None
                                            for r in site_rows.values()),
            'acquired_seconds': math.fsum(float(r['acquired_seconds']) for r in site_rows.values()),
            'sites_acquired': len(site_rows), 'switches': max(0, len(site_rows)-1)}


def load_inputs(source, root=ROOT):
    source, root = Path(source), Path(root)
    cfg_path = root/'configs/hsmm_baseline.json'
    cfg = json.loads(cfg_path.read_text())
    if cfg['sites'] != list(SITES) or cfg['budgets_seconds'] != [20, 'full'] or cfg['quality_threshold'] != .65:
        raise ValueError('HSMM site/budget/quality configuration changed')
    run_path = source/'run.json'
    run = json.loads(run_path.read_text())
    if run['status'] != 'complete' or run['config'] != cfg or run.get('reserved_predictions') != 0:
        raise ValueError('Complete matching source run with no reserved predictions required')
    paths = [run_path, source/'predictions.csv', source/'record_predictions.json', cfg_path,
             root/'HSMM_SITE_REPLAY_PROTOCOL.md', Path(__file__).resolve()]
    if 'author_record_predictions_sha256' in run:
        verification = source/'author_replay_verification.json'
        if json.loads(verification.read_text())['status'] != 'passed':
            raise ValueError('Corrected decoder source failed author replay verification')
        paths.append(verification)
    folder = root/'results/metadata'
    manifest_path = folder/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest(root/'PILOT_PROTOCOL.md') != manifest['protocol_sha256']:
        raise ValueError('Locked metadata protocol changed')
    paths += [manifest_path, root/'PILOT_PROTOCOL.md']
    for rel, expected in manifest['input_sha256'].items():
        path = root/'data/circor_metadata'/rel
        if digest(path) != expected:
            raise ValueError('Locked metadata input changed')
        paths.append(path)
    for rel, expected in manifest['output_sha256'].items():
        path = folder/rel
        if digest(path) != expected:
            raise ValueError('Locked metadata output changed')
        paths.append(path)
    # Explicit usecols excludes Murmur, Outcome, timing and every other label.
    visits = pd.read_csv(folder/'split_manifest.csv', usecols=['Patient ID', 'group_id', 'role']).rename(columns={'Patient ID': 'patient_id'})
    if visits.patient_id.duplicated().any() or visits.isna().any().any() or visits.groupby('group_id').role.nunique().max() != 1:
        raise ValueError('Invalid visit/group role identity')
    if not set(visits.role) <= {'train', 'calibration', 'pilot_validation', 'reserved_test'}:
        raise ValueError('Unknown metadata role')
    visits = visits[visits.role.isin(ROLES)].reset_index(drop=True)
    if set(visits.role) != set(ROLES):
        raise ValueError('Both development evaluation roles are required')
    inv = pd.read_csv(folder/'record_inventory.csv', usecols=['patient_id', 'site', 'record', 'selected', 'group_id', 'role'])
    inv = inv[inv.selected.map(boolean) & inv.role.isin(ROLES)].copy()
    if inv.record.duplicated().any() or inv.duplicated(['patient_id', 'site']).any() or not set(inv.site) <= set(SITES):
        raise ValueError('Canonical inventory is not unique or has an invalid site')
    people = visits.set_index('patient_id')
    for row in inv.itertuples():
        if row.patient_id not in people.index or row.role != people.loc[row.patient_id, 'role'] or row.group_id != people.loc[row.patient_id, 'group_id']:
            raise ValueError('Inventory identity differs from locked visit metadata')
    cols = ['patient_id', 'group_id', 'role', 'budget_seconds', 'acquired_seconds']
    cols += [f'{mode}_referred' for mode in MODES]
    cols += ['score_threshold', 'score_quality_threshold']
    # Preserve exact saved score-threshold equality across CSV/JSON serialization.
    pred = pd.read_csv(source/'predictions.csv', usecols=cols, float_precision='round_trip')
    pred['budget_seconds'] = pred.budget_seconds.map(budget_key)
    expected = {(p, b) for p in visits.patient_id for b in BUDGETS}
    if pred.duplicated(['patient_id', 'budget_seconds']).any() or set(zip(pred.patient_id, pred.budget_seconds)) != expected:
        raise ValueError('Visit source predictions differ from complete evaluation inventory')
    thresholds = {}
    for budget in BUDGETS:
        rows = pred[pred.budget_seconds.eq(budget)]
        for mode in ('score', 'score_quality'):
            values = rows[f'{mode}_threshold'].to_numpy(dtype=float)
            if np.isnan(values).any() or not np.all(values == values[0]):
                raise ValueError('Frozen threshold must be constant across visits/roles')
            thresholds[(budget, mode)] = float(values[0])
    for row in pred.itertuples():
        if row.role != people.loc[row.patient_id, 'role'] or row.group_id != people.loc[row.patient_id, 'group_id']:
            raise ValueError('Source prediction identity differs from locked metadata')
    records = json.loads((source/'record_predictions.json').read_text())
    record_index = {}
    inventory_index = inv.set_index('record')
    for row in records:
        key = (row['record'], budget_key(row['budget_seconds']))
        if key in record_index or key[0] not in inventory_index.index:
            raise ValueError('Duplicate/extra source record')
        if row['role'] != inventory_index.loc[key[0], 'role']:
            raise ValueError('Source record role differs from inventory')
        validate_site(row)
        record_index[key] = {k: row[k] for k in ['score', 'quality', 'failed', 'acquired_seconds']}
    if set(record_index) != {(r, b) for r in inv.record for b in BUDGETS}:
        raise ValueError('Missing source record/budget')
    for record in inv.record:
        capped = float(record_index[(record, '20')]['acquired_seconds'])
        full = float(record_index[(record, 'full')]['acquired_seconds'])
        if not math.isclose(capped, min(20., full), rel_tol=0., abs_tol=1e-9):
            raise ValueError('20-second and full source costs are inconsistent')
    hashes = {str(p.resolve()): digest(p) for p in dict.fromkeys(paths)}
    return visits, inv, pred, record_index, thresholds, hashes


def replay_frame(visits, inv, pred, records, thresholds):
    source = pred.set_index(['patient_id', 'budget_seconds'])
    inventory = {p: dict(zip(d.site, d.record)) for p, d in inv.groupby('patient_id')}
    traces = []
    for person in visits.itertuples():
        for budget in BUDGETS:
            rows = {s: records[(r, budget)] for s, r in inventory.get(person.patient_id, {}).items()}
            saved = source.loc[(person.patient_id, budget)]
            for mode in MODES:
                threshold = thresholds.get((budget, mode))
                full = all_site_reference(rows, mode, threshold)
                if full['referred'] != boolean(saved[f'{mode}_referred']):
                    raise ValueError('Reconstructed all-site referral differs from saved source')
                if not math.isclose(full['acquired_seconds'], float(saved.acquired_seconds), rel_tol=0, abs_tol=1e-8):
                    raise ValueError('Reconstructed all-site audio differs from saved source')
                for rank, order in enumerate(ORDERS):
                    result = replay_visit(rows, order, mode, threshold)
                    if result['referred'] != full['referred'] or result['acquired_seconds'] > full['acquired_seconds']+1e-9:
                        raise ValueError('Replay violated decision preservation or acquisition cost')
                    traces.append({'patient_id': person.patient_id, 'group_id': person.group_id,
                        'role': person.role, 'budget_seconds': budget, 'decision': mode,
                        'order': '|'.join(order), 'order_rank': rank, **result,
                        'all_site_referred': full['referred'], 'all_site_audio_seconds': full['acquired_seconds'],
                        'all_site_sites': full['sites_acquired'], 'all_site_switches': full['switches']})
    return pd.DataFrame(traces)


def select_orders(traces):
    """Accepts all traces but selects solely from calibration audio costs."""
    cal = traces[traces.role.eq('calibration')]
    if cal.empty:
        raise ValueError('Calibration traces required')
    result = []
    for budget in BUDGETS:
        for mode in MODES:
            subset = cal[cal.budget_seconds.eq(budget) & cal.decision.eq(mode)]
            candidates = []
            for rank, order in enumerate(ORDERS):
                rows = subset[subset.order_rank.eq(rank)]
                if rows.empty or rows.patient_id.duplicated().any():
                    raise ValueError('Invalid calibration order traces')
                mean = math.fsum(rows.acquired_seconds)/len(rows)
                candidates.append({'order': '|'.join(order), 'order_rank': rank,
                    'calibration_visits': len(rows), 'mean_audio_seconds': mean,
                    'selection_key_rounded_mean': round(mean, 9)})
            chosen = min(candidates, key=lambda r: (r['selection_key_rounded_mean'], r['order_rank']))
            result.append({'budget_seconds': budget, 'decision': mode, **chosen,
                           'selection_population': 'calibration_only', 'all_orders': candidates})
    return result


def cost_summaries(traces, selections):
    primary = {(s['budget_seconds'], s['decision']): s['order_rank'] for s in selections}
    rows = []
    for (budget, mode, role, rank), d in traces.groupby(['budget_seconds', 'decision', 'role', 'order_rank'], sort=False):
        for overhead in OVERHEADS:
            replay = d.acquired_seconds.to_numpy()+overhead*d.switches.to_numpy()
            full = d.all_site_audio_seconds.to_numpy()+overhead*d.all_site_switches.to_numpy()
            rows.append({'budget_seconds': budget, 'decision': mode, 'role': role,
                'order': d.order.iloc[0], 'order_rank': int(rank),
                'calibration_selected_primary': rank == primary[(budget, mode)],
                'switch_overhead_seconds': overhead, 'n_visits': len(d),
                'referrals': int(d.referred.sum()), 'decision_disagreements': int((d.referred != d.all_site_referred).sum()),
                'mean_audio_seconds': float(d.acquired_seconds.mean()),
                'all_site_mean_audio_seconds': float(d.all_site_audio_seconds.mean()),
                'mean_audio_saved_seconds': float((d.all_site_audio_seconds-d.acquired_seconds).mean()),
                'mean_sites_acquired': float(d.sites_acquired.mean()),
                'mean_switches': float(d.switches.mean()),
                'all_site_mean_switches': float(d.all_site_switches.mean()),
                'mean_total_scenario_seconds': float(replay.mean()),
                'median_total_scenario_seconds': float(np.median(replay)),
                'p90_total_scenario_seconds': float(np.quantile(replay, .9)),
                'all_site_mean_total_scenario_seconds': float(full.mean()),
                'mean_total_saved_seconds': float((full-replay).mean()),
                'total_fraction_saved': float((full.sum()-replay.sum())/full.sum()) if full.sum() else 0.})
    return pd.DataFrame(rows)


def evaluate(source, output, root=ROOT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite a replay directory')
    visits, inv, pred, records, thresholds, hashes = load_inputs(source, root)
    output.mkdir(parents=True)
    started = time.time()
    run = {'status': 'running', 'source_directory': str(source), 'input_sha256': hashes,
           'fitting_performed': False, 'clinical_labels_used': False, 'annotations_read': 0,
           'waveforms_read': 0, 'reserved_predictions': 0,
           'decision_scope': 'binary referral only', 'site_orders': 24,
           'switch_overheads_seconds': list(OVERHEADS)}
    write_json(output/'run.json', run)
    try:
        traces = replay_frame(visits, inv, pred, records, thresholds)
        selected = select_orders(traces)
        write_json(output/'calibration_order_selection.json', selected)
        summary = cost_summaries(traces, selected)
        traces.to_csv(output/'visit_traces.csv', index=False)
        summary.to_csv(output/'all_order_costs.csv', index=False)
        summary[summary.calibration_selected_primary].to_csv(output/'primary_order_costs.csv', index=False)
        write_json(output/'agreement.json', {'status': 'passed', 'visits': len(visits),
            'visit_budget_rule_order_checks': len(traces), 'binary_decision_disagreements': 0,
            'reconstructed_source_audio_costs_match': True, 'source_binary_decisions_match': True})
        if any(digest(path) != expected for path, expected in hashes.items()):
            raise ValueError('Input changed during replay')
        run.update(status='complete', elapsed_seconds=time.time()-started,
            output_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'run.json'})
    except Exception as exc:
        run.update(status='failed', error=repr(exc), elapsed_seconds=time.time()-started)
        raise
    finally:
        write_json(output/'run.json', run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.source_directory, args.output_directory)
    print(json.dumps({'status': result['status'], 'output_directory': str(args.output_directory),
                      'binary_decision_preservation_required': True}))


if __name__ == '__main__':
    main()
