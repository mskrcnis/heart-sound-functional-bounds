"""Frozen physiological emission compression study; default is verification only.

Research derivative integrating Cambridge 2022 code. Modified 2026-11-07;
GAP Available Source License v1.0, academic noncommercial use, no warranty.
See ../HSMM_LICENSE.md. No upstream source or existing result is modified.
This is ordinary same-fit KD, not OOF distillation or a new XAI method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import hsmm_baseline as baseline
import hsmm_traceback_sensitivity as traceback
import hsmm_score_accounting as accounting

ROOT = baseline.ROOT
ARCHITECTURE = {'input_channels': 40, 'width': 24, 'classes': 5, 'kernel_size': 3,
                'dilations': [1, 2, 4, 8, 16], 'activation': 'GELU',
                'normalization': 'per_time_channel_LayerNorm', 'dropout': 0.0}
TRAINING = {'batch_size': 32, 'validation_batch_size': 128, 'crop_frames': 300,
            'learning_rate': .001, 'max_epochs': 100, 'patience': 10,
            'temperature': 2., 'optimizer': 'Adam', 'amp': False,
            'checkpoint_selection': 'held_fold_weighted_hard_CE'}
SEEDS = [20261107, 20261108, 20261109]
ALPHAS = [0., .5, 1.]
MODES = ('published', 'score', 'score_quality')


def require_config(cfg):
    if (cfg['architecture'] != ARCHITECTURE or cfg['training'] != TRAINING
            or cfg['seeds'] != SEEDS or cfg['alphas'] != ALPHAS
            or cfg['inner_folds'] != 5 or cfg['budgets_seconds'] != [20, 'full']
            or cfg['decoder'] != 'corrected_traceback_only'
            or cfg['reserved_test_access'] != 'forbidden'):
        raise ValueError('Configuration differs from the frozen bounded study')


class InputSeal:
    """Snapshot before reads and reject modifications before outputs are sealed."""
    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        actual = baseline.digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Input hash differs: {path}')
        if str(path) in self.hashes and self.hashes[str(path)] != actual:
            raise ValueError(f'Input changed during study: {path}')
        self.hashes[str(path)] = actual
        return path

    def read_json(self, path, expected=None):
        path = self.add(path, expected)
        value = json.loads(path.read_text())
        self.add(path)
        return value

    def read_csv(self, path, expected=None):
        path = self.add(path, expected)
        value = pd.read_csv(path, float_precision='round_trip')
        self.add(path)
        return value

    def check(self):
        for path, expected in self.hashes.items():
            if baseline.digest(path) != expected:
                raise ValueError(f'Input changed during study: {path}')


def within(directory, relative):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('Path escapes the declared input directory')
    return path


def validate_membership(records, folds, visits):
    if (records.empty or records.record.duplicated().any()
            or not records.role.eq('train').all()
            or not records.visit_label.isin(['Present', 'Absent']).all()
            or set(records.inner_fold) != set(range(5))
            or records.groupby('group_id').inner_fold.nunique().max() != 1
            or folds.patient_id.duplicated().any() or not folds.role.eq('train').all()
            or folds.groupby('group_id').inner_fold.nunique().max() != 1):
        raise ValueError('Training cache violates original group/role boundaries')
    train = visits[visits.role.eq('train')].set_index('patient_id')
    if set(folds.patient_id) != set(train.index):
        raise ValueError('Original fold inventory differs from outer training visits')
    original = folds.set_index('patient_id')
    for row in folds.itertuples():
        v = train.loc[row.patient_id]
        if row.group_id != v.group_id or row.Murmur != v.Murmur:
            raise ValueError('Original fold metadata differs')
    for row in records.itertuples():
        if row.patient_id not in original.index:
            raise ValueError('Record is outside original training inventory')
        f = original.loc[row.patient_id]
        if (row.group_id != f.group_id or row.inner_fold != f.inner_fold
                or row.visit_label != f.Murmur):
            raise ValueError('Record group/fold/label does not match original assignment')


def prepare(cfg, config_path):
    """Verify frozen sources and existing training arrays; no decoder/model loading."""
    require_config(cfg)
    seal = InputSeal()
    if seal.read_json(config_path) != cfg:
        raise ValueError('Config changed before verification')
    for rel, sha in cfg['source_sha256'].items():
        seal.add(within(ROOT, rel), sha)
    base_cfg = seal.read_json(ROOT / 'configs/hsmm_baseline.json')
    teacher_dir = within(ROOT, cfg['teacher_directory'])
    teacher_run = seal.read_json(teacher_dir / 'run.json')
    if (teacher_run['status'] != 'complete' or teacher_run['config'] != base_cfg
            or teacher_run['reserved_predictions'] != 0):
        raise ValueError('Teacher must be the completed frozen original training run')
    for rel, sha in teacher_run['input_sha256'].items():
        seal.add(within(ROOT, rel), sha)
    for name, sha in cfg['teacher_checkpoint_sha256'].items():
        seal.add(within(teacher_dir / 'models', name), sha)
    if set(cfg['teacher_checkpoint_sha256']) != {f'fold_{k}.pt' for k in range(5)}:
        raise ValueError('Exactly five original teachers are required')
    metadata = seal.read_json(ROOT / 'results/metadata/manifest.json')
    for rel, sha in metadata['input_sha256'].items():
        seal.add(within(ROOT / 'data/circor_metadata', rel), sha)
    for rel, sha in metadata['output_sha256'].items():
        seal.add(within(ROOT / 'results/metadata', rel), sha)
    visits, _, _ = baseline.read_metadata()
    cache = within(ROOT, base_cfg['training_cache_directory'])
    manifest = seal.read_json(cache / 'manifest.json')
    expected_cache = {'frontend': base_cfg['frontend'], 'training': base_cfg['training'],
                      'seed': base_cfg['seed'], 'inner_folds': 5,
                      'adapter_sha256': seal.hashes[str((ROOT / 'src/hsmm_baseline.py').resolve())],
                      'metadata_sha256': seal.hashes[str((ROOT / 'results/metadata/manifest.json').resolve())],
                      'official_checksums_sha256': baseline.digest(ROOT / 'data/circor_metadata/SHA256SUMS.txt')}
    if manifest['source'] != expected_cache or manifest['calibration_validation_reserved_annotations_read'] != 0:
        raise ValueError('Training cache provenance differs')
    records = seal.read_csv(cache / 'records.csv', manifest['files_sha256']['records.csv'])
    folds = seal.read_csv(cache / 'inner_folds.csv', manifest['files_sha256']['inner_folds.csv'])
    validate_membership(records, folds, visits)
    if len(records) != manifest['records']:
        raise ValueError('Training cache record denominator differs')
    for row in records.itertuples():
        if not row.array.startswith('arrays/') or not row.array.endswith('.npz'):
            raise ValueError('Unexpected cached array path')
        seal.add(within(cache, row.array), manifest['files_sha256'][row.array])
    # Do not reopen training annotations, archive members, or reserved waveforms.
    seal.check()
    return base_cfg, cache, records, seal


def length_mask(x, lengths):
    if (x.ndim != 3 or len(lengths) != len(x) or lengths.ndim != 1
            or lengths.dtype != torch.long or (lengths <= 0).any() or (lengths > x.shape[-1]).any()):
        raise ValueError('Invalid feature batch or valid lengths')
    return (torch.arange(x.shape[-1], device=x.device)[None, None, :]
            < lengths.to(x.device)[:, None, None]).to(x.dtype)


class ChannelLayerNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MaskedResidual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = ChannelLayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, x, mask):
        value = self.norm(x) * mask
        value = self.depthwise(value) * mask
        value = F.gelu(value) * mask
        value = self.pointwise(value) * mask
        return (x + value) * mask


class PhysiologicalStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(40, 24, 1)
        self.stem_norm = ChannelLayerNorm(24)
        self.blocks = nn.ModuleList([MaskedResidual(24, d) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Conv1d(24, 5, 1)

    def forward(self, features, lengths):
        if features.shape[1] != 40:
            raise ValueError('Expected 40 original frontend channels')
        mask = length_mask(features, lengths)
        # Mask BEFORE the first biased layer and after every biased/time operation.
        x = self.stem(features.masked_fill(~mask.bool(), 0.)) * mask
        x = F.gelu(self.stem_norm(x) * mask) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.head(x) * mask


def deterministic_epoch(records, arrays, seed, fold, epoch, crop_frames, batch_size):
    """Independent RNG makes order/crops identical across alpha and model RNG use.

    Preserve the author's crop-start support: randint(0,max(T-crop,1)), whose
    exclusive upper bound omits the last possible start when T > crop.
    """
    generator = torch.Generator().manual_seed(seed + fold * 1000003 + epoch * 1000000007)
    names = list(records)
    order = torch.randperm(len(names), generator=generator).tolist()
    batch, schedule = [], []
    for index in order:
        name = names[index]
        x, y = arrays[name]
        start = int(torch.randint(0, max(x.shape[-1] - crop_frames, 1), (1,), generator=generator))
        batch.append((x[:, start:start + crop_frames], y[start:start + crop_frames]))
        schedule.append((name, start, len(batch[-1][1])))
        if len(batch) == batch_size:
            yield baseline.collate_frames(batch), schedule
            batch, schedule = [], []
    if batch:
        yield baseline.collate_frames(batch), schedule


def emission_loss(student_logits, target, lengths, weights, alpha, temperature=2., teacher_logits=None):
    if alpha not in ALPHAS or temperature != 2.:
        raise ValueError('Loss is outside the frozen study')
    valid_length = length_mask(student_logits, lengths)[:, 0].bool()
    if target.shape != valid_length.shape:
        raise ValueError('Target shape differs from logits/lengths')
    target = target.masked_fill(~valid_length, -1)
    mask = valid_length & target.ge(0)
    if not mask.any() or (target[mask] > 4).any():
        raise ValueError('No valid five-state training targets')
    hard = F.cross_entropy(student_logits, target, weight=weights, ignore_index=-1)
    if alpha == 0.:
        return hard, {'hard': hard.detach(), 'soft': None}
    if teacher_logits is None or teacher_logits.shape != student_logits.shape:
        raise ValueError('Teacher must see exactly the sampled student batch')
    teacher_probability = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_log_probability = F.log_softmax(student_logits / temperature, dim=1)
    per_frame = F.kl_div(student_log_probability, teacher_probability, reduction='none').sum(dim=1)
    soft = temperature ** 2 * per_frame[mask].mean()
    return (1. - alpha) * hard + alpha * soft, {'hard': hard.detach(), 'soft': soft.detach()}


def batch_objective(student, teacher, x, target, lengths, weights, alpha):
    student_logits = student(x, lengths)
    teacher_logits = None
    if alpha:
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise ValueError('KD teacher must be frozen and in evaluation mode')
        with torch.no_grad():
            # Identical x object and CPU lengths, no full-record teacher cache.
            teacher_logits = teacher(x, lengths)
    return emission_loss(student_logits, target, lengths, weights, alpha, 2., teacher_logits)


def held_hard_loss(model, loader, weights, device):
    """Same full-record weighted CE checkpoint criterion for all three alpha values."""
    model.eval()
    total, count = 0., 0
    with torch.no_grad():
        for x, y, lengths in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x, lengths), y, weight=weights, ignore_index=-1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite held-fold hard CE')
            total += float(loss) * len(x)
            count += len(x)
    if count == 0:
        raise ValueError('Empty held fold')
    return total / count


def recipe_name(seed, alpha):
    return f'seed_{seed}_alpha_{str(alpha).replace(".", "p")}'


def fit_one(records, arrays, teacher, seed, alpha, fold, cfg, directory):
    directory.mkdir(parents=True, exist_ok=False)
    baseline.reset_seed(seed + fold)
    fit_rows, held_rows = records[records.inner_fold.ne(fold)], records[records.inner_fold.eq(fold)]
    if set(fit_rows.group_id) & set(held_rows.group_id):
        raise ValueError('Fit and held groups overlap')
    device = next(teacher.parameters()).device
    teacher.eval().requires_grad_(False)
    model = PhysiologicalStudent().to(device)
    weights, counts = baseline.fit_class_weights(fit_rows.record, arrays)
    weights = weights.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    held_loader = torch.utils.data.DataLoader(
        baseline.TrainingDataset(held_rows.record, arrays), batch_size=cfg['validation_batch_size'],
        shuffle=False, collate_fn=baseline.collate_frames, num_workers=0)
    best, stale, history = float('inf'), 0, []
    checkpoint = directory / 'best.pt'
    for epoch in range(1, cfg['max_epochs'] + 1):
        started = time.perf_counter()
        model.train()
        total, hard_total, soft_total, count = 0., 0., 0., 0
        schedule_hash = hashlib.sha256()
        for (x, y, lengths), schedule in deterministic_epoch(
                fit_rows.record, arrays, seed, fold, epoch, cfg['crop_frames'], cfg['batch_size']):
            schedule_hash.update(json.dumps(schedule, separators=(',', ':')).encode())
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = batch_objective(model, teacher, x, y, lengths, weights, alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite student training loss')
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
            hard_total += float(terms['hard']) * len(x)
            soft_total += 0. if terms['soft'] is None else float(terms['soft']) * len(x)
            count += len(x)
        validation = held_hard_loss(model, held_loader, weights, device)
        improved = validation < best
        if improved:
            best, stale = validation, 0
            torch.save({'model_state_dict': model.state_dict(), 'seed': seed, 'alpha': alpha,
                        'fold': fold, 'epoch': epoch, 'validation_loss': best,
                        'architecture': ARCHITECTURE, 'class_weights': weights.cpu(), 'class_counts': counts,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'fit_group_ids': sorted(set(fit_rows.group_id)),
                        'inner_validation_group_ids': sorted(set(held_rows.group_id)),
                        'teacher_training': 'same_fit_partition_not_OOF',
                        'checkpoint_selection': TRAINING['checkpoint_selection']}, checkpoint)
        else:
            stale += 1
        row = {'seed': seed, 'alpha': alpha, 'fold': fold, 'epoch': epoch,
               'train_objective': total / count, 'train_hard_CE': hard_total / count,
               'train_soft_KL_T2': None if alpha == 0 else soft_total / count,
               'held_weighted_hard_CE': validation, 'improved': improved,
               'nonimproving_epochs': stale, 'crop_schedule_sha256': schedule_hash.hexdigest(),
               'epoch_seconds': time.perf_counter() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(directory / 'epoch_losses.csv', index=False)
        print(json.dumps(row, allow_nan=False), flush=True)
        if stale > cfg['patience']:
            break
    final = {'status': 'complete', 'seed': seed, 'alpha': alpha, 'fold': fold,
             'epochs': len(history), 'best_held_hard_CE': best,
             'checkpoint_sha256': baseline.digest(checkpoint),
             'epoch_losses_sha256': baseline.digest(directory / 'epoch_losses.csv')}
    baseline.write_json(directory / 'fit.json', final)
    return final


class SingleCorrectedProxy:
    """One corrected decode plus exact accounting, with no author decoder call."""
    def __init__(self, segmenter, output):
        self.segmenter, self.output, self.events = segmenter, Path(output), []
        self.output.mkdir(parents=True, exist_ok=False)

    def double_duration_viterbi(self, posterior, fs, **kwargs):
        array = np.asarray(posterior, dtype=np.float32)
        before = traceback.posterior_digest(array)
        validation_failed = False
        try:
            array = traceback.validate_posterior(array)
        except ValueError as exc:
            decoded, error, validation_failed = None, exc, True
            summary = {'failed': True, 'score': None, 'quality': None, 'healthy_confidence': None,
                       'murmur_confidence': None, 'murmur_model': None, 'reason': 'hsmm_failure:ValueError'}
        else:
            decoded, summary, error = traceback.decode_summary(self.segmenter, array, fs, kwargs)
        if traceback.posterior_digest(array) != before:
            raise RuntimeError('Corrected decoder mutated the posterior')
        arrays, stats = {'posterior': array}, None
        if decoded is not None:
            arrays.update(healthy_states=np.asarray(decoded[0]), murmur_states=np.asarray(decoded[1]))
            try:
                stats, _ = accounting.account(array, decoded[0], decoded[1], summary)
            except Exception as exc:
                # Must escape baseline.infer_record numerical-failure fallback.
                raise RuntimeError('Student score accounting failed') from exc
        path = self.output / f'{len(self.events):05d}.npz'
        np.savez_compressed(path, **arrays)
        event = {'ordinal': len(self.events), 'array': path.name,
                 'array_file_sha256': baseline.digest(path), 'posterior_raw_sha256': before,
                 'posterior_shape': list(array.shape), 'corrected': summary,
                 'accounting': stats, 'posterior_validation_failed': validation_failed}
        baseline.write_json(path.with_suffix('.json'), event)
        self.events.append(event)
        if error is not None:
            raise error
        return decoded


def associate_audit(events, rows):
    associated, ordinal = [], 0
    for row in rows:
        reason = row['feature_guards']['reason']
        value = {k: row[k] for k in ['record', 'role', 'budget_seconds', 'acquired_seconds', 'failed']}
        if reason is None or reason.startswith('hsmm_failure:'):
            if ordinal >= len(events):
                raise RuntimeError('Missing decoder audit event')
            event = events[ordinal]
            for key in ['failed', 'score', 'quality', 'healthy_confidence', 'murmur_confidence', 'murmur_model']:
                if event['corrected'][key] != row[key]:
                    raise RuntimeError('Decoder event/result association differs')
            if event['corrected']['reason'] != reason:
                raise RuntimeError('Decoder failure reason differs')
            value.update(event)
            ordinal += 1
        else:
            value.update(ordinal=None, array=None, accounting=None, frontend_failure=reason)
        associated.append(value)
    if ordinal != len(events):
        raise RuntimeError('Unassociated decoder audit event')
    return associated


def visit_witnesses(predictions, rows, inventory):
    locations = {r.record: (int(r.patient_id), r.site) for r in inventory.itertuples()}
    grouped = {}
    for row in rows:
        patient_id, site = locations[row['record']]
        key = (patient_id, str(row['budget_seconds']))
        by_site = grouped.setdefault(key, {})
        if site in by_site:
            raise RuntimeError('Duplicate evaluated site')
        by_site[site] = {**row, 'reason': row['feature_guards']['reason']}
    out = []
    for person in predictions.itertuples():
        key = (int(person.patient_id), str(person.budget_seconds))
        explanation = accounting.visit_account(grouped.get(key, {}),
                         {'score': person.score_threshold, 'score_quality': person.score_quality_threshold})
        for mode in MODES:
            if explanation['decisions'][mode]['referred'] != bool(getattr(person, mode + '_referred')):
                raise RuntimeError('Visit witness does not reproduce referral')
        out.append({'patient_id': int(person.patient_id), 'role': person.role,
                    'budget_seconds': str(person.budget_seconds), **explanation})
    return out


def load_student_ensemble(directory, seed, alpha, records, ledger, device):
    models, hashes = [], {}
    for fold in range(5):
        path = directory / f'fold_{fold}' / 'best.pt'
        sha = baseline.digest(path)
        if sha != ledger[(seed, alpha, fold)]['checkpoint_sha256']:
            raise ValueError('Student checkpoint changed after training')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state['fold'] != fold or state['seed'] != seed or state['alpha'] != alpha
                or state['architecture'] != ARCHITECTURE
                or set(state['fit_group_ids']) != set(records.loc[records.inner_fold.ne(fold), 'group_id'])
                or set(state['inner_validation_group_ids']) != set(records.loc[records.inner_fold.eq(fold), 'group_id'])):
            raise ValueError('Student checkpoint fold/group provenance differs')
        model = PhysiologicalStudent()
        model.load_state_dict(state['model_state_dict'], strict=True)
        models.append(model.to(device).eval())
        hashes[str(path)] = sha
    return models, hashes


def pareto_components(candidate, comparator):
    """Observed development comparison only; no equivalence/noninferiority test."""
    same_denominators = all(int(candidate[k]) == int(comparator[k]) for k in
                            ['n_visits', 'n_known', 'n_positive', 'n_negative', 'n_known_scored', 'n_unknown'])
    comparable_auc = (same_denominators and np.isfinite(candidate['auroc'])
                      and np.isfinite(comparator['auroc']))
    components = {'same_denominators': same_denominators,
                  'no_lower_AUROC': bool(comparable_auc and candidate['auroc'] >= comparator['auroc']),
                  'no_additional_observed_FN': int(candidate['false_negatives']) <= int(comparator['false_negatives']),
                  'no_additional_observed_referrals': int(candidate['referrals']) <= int(comparator['referrals'])}
    return {**components, 'favorable_observed_metrics': bool(all(components.values())),
            'strictly_higher_AUROC': bool(comparable_auc and candidate['auroc'] > comparator['auroc']),
            'strictly_fewer_referrals': int(candidate['referrals']) < int(comparator['referrals'])}


def paired_support(candidate, comparator):
    """Equal counts do not guarantee the same patients were successfully scored."""
    columns = ['patient_id', 'group_id', 'Murmur']
    if candidate.patient_id.duplicated().any() or comparator.patient_id.duplicated().any():
        raise ValueError('Duplicate visit in a development comparison')
    def inventory(table):
        return [[int(r.patient_id), int(r.group_id), str(r.Murmur)]
                for r in table[columns].sort_values('patient_id').itertuples()]
    def scored(table):
        use = table.Murmur.isin(['Present', 'Absent']) & np.isfinite(table.score)
        return sorted(int(x) for x in table.loc[use, 'patient_id'])
    left, right = inventory(candidate), inventory(comparator)
    left_scored, right_scored = scored(candidate), scored(comparator)
    return {'same_patient_group_label_inventory': left == right,
            'same_scored_known_patient_ids': left_scored == right_scored,
            'candidate_scored_known_patient_ids': left_scored,
            'comparator_scored_known_patient_ids': right_scored,
            'candidate_inventory_sha256': hashlib.sha256(json.dumps(left).encode()).hexdigest(),
            'comparator_inventory_sha256': hashlib.sha256(json.dumps(right).encode()).hexdigest()}


def development_screen(cfg, out, seal):
    """Retain all comparisons; mixed alpha=.5 primary, pure alpha=1 secondary."""
    teacher = seal.read_csv(within(ROOT, cfg['corrected_comparator_directory']) / 'metrics.csv')
    teacher = teacher[teacher.role.eq('pilot_validation')]
    teacher_predictions = seal.read_csv(within(ROOT, cfg['corrected_comparator_directory']) / 'predictions.csv')
    teacher_predictions = teacher_predictions[teacher_predictions.role.eq('pilot_validation')]
    tables, predictions = {}, {}
    for seed in SEEDS:
        for alpha in ALPHAS:
            table = pd.read_csv(out / recipe_name(seed, alpha) / 'evaluation/metrics.csv',
                                float_precision='round_trip')
            tables[(seed, alpha)] = table[table.role.eq('pilot_validation')]
            table = pd.read_csv(out / recipe_name(seed, alpha) / 'evaluation/predictions.csv',
                                float_precision='round_trip')
            predictions[(seed, alpha)] = table[table.role.eq('pilot_validation')]
    def row(table, budget, mode):
        subset = table[table.budget_seconds.astype(str).eq(str(budget)) & table.decision.eq(mode)]
        if len(subset) != 1:
            raise ValueError('Development metric inventory differs')
        return subset.iloc[0].to_dict()
    contrasts = []
    for seed in SEEDS:
        for alpha in ALPHAS:
            for budget in [20, 'full']:
                for mode in MODES:
                    candidate = row(tables[(seed, alpha)], budget, mode)
                    candidate_predictions = predictions[(seed, alpha)]
                    candidate_predictions = candidate_predictions[
                        candidate_predictions.budget_seconds.astype(str).eq(str(budget))]
                    comparators = [('corrected_five_GRU', row(teacher, budget, mode), teacher_predictions)]
                    if alpha:
                        comparators.append(('same_seed_hard_student', row(tables[(seed, 0.)], budget, mode),
                                            predictions[(seed, 0.)]))
                    for name, reference, reference_predictions in comparators:
                        support = paired_support(candidate_predictions, reference_predictions[
                            reference_predictions.budget_seconds.astype(str).eq(str(budget))])
                        components = pareto_components(candidate, reference)
                        comparable = support['same_patient_group_label_inventory'] and support['same_scored_known_patient_ids']
                        components['favorable_observed_metrics'] &= comparable
                        components['no_lower_AUROC'] &= comparable
                        components['strictly_higher_AUROC'] &= comparable
                        contrasts.append({'seed': seed, 'alpha': alpha, 'budget_seconds': str(budget),
                                          'decision': mode, 'comparator': name,
                                          'candidate_metrics': candidate, 'comparator_metrics': reference,
                                          **components, **support,
                                          'fewer_parameters': name == 'corrected_five_GRU'})
    gates = []
    for alpha in [.5, 1.]:
        for budget in [20, 'full']:
            selected = [r for r in contrasts if r['alpha'] == alpha
                        and r['budget_seconds'] == str(budget) and r['decision'] == 'score']
            if len(selected) != 6:
                raise ValueError('Primary screen comparison inventory differs')
            gates.append({'alpha': alpha, 'contrast_role': 'primary' if alpha == .5 else 'secondary',
                          'budget_seconds': str(budget), 'decision': 'score',
                          'all_three_seeds_no_worse_than_hard_and_GRU': all(
                              r['favorable_observed_metrics'] for r in selected),
                          'all_three_seeds_strict_AUROC_or_referral_gain_vs_hard': all(
                              r['strictly_higher_AUROC'] or r['strictly_fewer_referrals']
                              for r in selected if r['comparator'] == 'same_seed_hard_student'),
                          'lower_parameters_than_GRU': 24385 < 888325,
                          'measured_student_runtime_available': False})
    result = {'interpretation': 'Adaptive development feasibility screen, not clinical noninferiority or superiority.',
              'seed_dependence': 'Three optimization seeds share the five original teachers and the same patients.',
              'contrasts': contrasts, 'fixed_score_policy_gates': gates,
              'all_variants_reported': True, 'threshold_tuning_performed': False}
    baseline.write_json(out / 'development_screen.json', accounting.json_finite(result))
    return gates


def train_and_evaluate(cfg, config_path, device):
    base_cfg, cache, records, seal = prepare(cfg, config_path)
    out = within(ROOT, cfg['output_directory'])
    if out.exists():
        raise ValueError('Refusing to overwrite an existing study, including incomplete fits')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(cfg['torch_threads'])
    teachers, _, teacher_hashes = traceback.load_completed_models(within(ROOT, cfg['teacher_directory']), base_cfg, device)
    if teacher_hashes != cfg['teacher_checkpoint_sha256']:
        raise ValueError('Teacher checkpoint hashes differ')
    for teacher in teachers:
        teacher.eval().requires_grad_(False)
    corrected, binary = traceback.load_corrected_segmenter(base_cfg)
    seal.add(binary)
    arrays = {}
    for row in records.itertuples():
        path = within(cache, row.array)
        seal.add(path)
        with np.load(path, allow_pickle=False) as data:
            x, y = data['features'].copy(), data['target'].copy()
        seal.add(path)
        if (x.dtype != np.float32 or x.shape != (40, int(row.frames))
                or y.shape != (int(row.frames),) or not np.issubdtype(y.dtype, np.integer)
                or not np.isfinite(x).all() or (y < 0).any() or (y > 4).any()):
            raise ValueError('Invalid existing training array')
        arrays[row.record] = (torch.from_numpy(x), torch.from_numpy(y).long())
    out.mkdir(parents=True)
    run = {'status': 'running', 'started_unix': time.time(), 'config': cfg,
           'input_sha256': seal.hashes.copy(), 'device': str(device),
           'platform': platform.platform(), 'python': platform.python_version(),
           'torch': torch.__version__, 'numpy': np.__version__, 'torch_threads': torch.get_num_threads(),
           'training_records': len(records), 'training_visits': records.patient_id.nunique(),
           'training_groups': records.group_id.nunique(), 'reserved_predictions': 0,
           'heldout_annotations_read': 0, 'decoder': 'corrected_traceback_only',
           'parameters_per_student': 4877, 'parameters_five_student_ensemble': 24385,
           'parameters_original_five_GRU_ensemble': 888325,
           'KD_targets': 'online corresponding frozen teacher, same sampled crop, same fit groups; not OOF',
           'timing_scope': 'Research execution including audits and I/O; not deployment latency',
           'other_GPU_training_may_be_concurrent': True,
           'research_elapsed_time_supports_speed_claim': False,
           'completed_fits': [], 'completed_ensembles': []}
    baseline.write_json(out / 'run.json', run)
    try:
        # Finish all 45 fits before opening any new clinical evaluation output.
        for seed in cfg['seeds']:
            for alpha in cfg['alphas']:
                directory = out / recipe_name(seed, alpha)
                for fold in range(5):
                    result = fit_one(records, arrays, teachers[fold], seed, alpha, fold,
                                     cfg['training'], directory / f'fold_{fold}')
                    run['completed_fits'].append(result)
                    baseline.write_json(out / 'run.json', run)
        if len(run['completed_fits']) != 45:
            raise RuntimeError('Incomplete mandatory fit inventory')
        ledger = {(r['seed'], r['alpha'], r['fold']): r for r in run['completed_fits']}
        # Audit identical schedules for every common epoch across alpha variants.
        schedule = {}
        for seed in cfg['seeds']:
            for alpha in cfg['alphas']:
                for fold in range(5):
                    history = pd.read_csv(out / recipe_name(seed, alpha) / f'fold_{fold}/epoch_losses.csv')
                    for row in history.itertuples():
                        key = (seed, fold, row.epoch)
                        if key in schedule and schedule[key] != row.crop_schedule_sha256:
                            raise RuntimeError('Matched-alpha crop schedule differs')
                        schedule[key] = row.crop_schedule_sha256
        seal.check()
        del arrays, teachers
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Snapshot only authorized calibration/validation WAVs before evaluation.
        visits, inventory, checks = baseline.read_metadata()
        evaluation_ids = set(visits.loc[visits.role.isin(['calibration', 'pilot_validation']), 'patient_id'])
        for row in inventory[inventory.patient_id.isin(evaluation_ids)].itertuples():
            seal.add(ROOT / 'data/development_wav' / (Path(row.record).name + '.wav'), checks[row.record + '.wav'])
        run['input_sha256'] = seal.hashes.copy()
        baseline.write_json(out / 'run.json', run)
        for seed in cfg['seeds']:
            for alpha in cfg['alphas']:
                directory = out / recipe_name(seed, alpha)
                models, hashes = load_student_ensemble(directory, seed, alpha, records, ledger, device)
                evaluation = directory / 'evaluation'
                evaluation.mkdir()
                proxy = SingleCorrectedProxy(corrected, evaluation / 'posteriors')
                baseline.evaluate(models, proxy, base_cfg, evaluation, device)
                rows = json.loads((evaluation / 'record_predictions.json').read_text())
                associated = associate_audit(proxy.events, rows)
                baseline.write_json(evaluation / 'decoder_records.json', associated)
                predictions = pd.read_csv(evaluation / 'predictions.csv', float_precision='round_trip')
                _, inventory, _ = baseline.read_metadata()
                witnesses = visit_witnesses(predictions, rows, inventory)
                baseline.write_json(evaluation / 'visit_witnesses.json', accounting.json_finite(witnesses))
                if sum(not r['failed'] for r in rows) != sum(e['accounting'] is not None for e in associated):
                    raise RuntimeError('Successful-record accounting denominator differs')
                for path, expected in hashes.items():
                    if baseline.digest(path) != expected:
                        raise ValueError('Student checkpoint changed during evaluation')
                summary = {'seed': seed, 'alpha': alpha, 'directory': str(evaluation.relative_to(out)),
                           'checkpoints_sha256': hashes, 'record_rows': len(rows),
                           'successful_record_accounts': sum(e['accounting'] is not None for e in associated),
                           'failed_record_rows': sum(r['failed'] for r in rows),
                           'visit_witness_rows': len(witnesses)}
                baseline.write_json(evaluation / 'audit.json', summary)
                run['completed_ensembles'].append(summary)
                baseline.write_json(out / 'run.json', run)
                del models
        run['development_screen'] = development_screen(cfg, out, seal)
        seal.check()
        run['input_sha256'] = seal.hashes.copy()
        run.update(status='complete', elapsed_seconds=time.time() - run['started_unix'],
                   output_sha256={str(p.relative_to(out)): baseline.digest(p)
                                  for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'run.json'})
    except Exception as exc:
        run.update(status='failed', error=repr(exc), elapsed_seconds=time.time() - run['started_unix'])
        raise
    finally:
        baseline.write_json(out / 'run.json', run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/physiological_distill.json')
    parser.add_argument('--train', action='store_true', help='Run all 45 fits, then all nine fixed evaluations')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.train:
        train_and_evaluate(cfg, args.config, torch.device(args.device))
    else:
        _, _, records, seal = prepare(cfg, args.config)
        print(json.dumps({'status': 'verified_only', 'training_performed': False, 'evaluation_performed': False,
                          'records': len(records), 'input_files': len(seal.hashes),
                          'mandatory_fits': 45, 'mandatory_ensembles': 9}, indent=2))


if __name__ == '__main__':
    main()
