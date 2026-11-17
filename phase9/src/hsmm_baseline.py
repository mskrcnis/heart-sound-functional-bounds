"""Split-safe research adaptation of the Cambridge parallel-HSMM baseline.

Derived training/feature/annotation procedures: University of Cambridge, 2022.
Adapter modifications, 2026-11-07: linked-person folds; train-only annotations;
ignored unspecified frames; numerical guards; raw-prefix inference; audit outputs.
Distributed under the GAP Available Source License v1.0 for academic noncommercial
research, without warranty. See phase9/HSMM_LICENSE.md. Upstream files are unchanged.

Default invocation performs CPU smoke checks only. --prepare-only builds a
training-only cache; --train explicitly starts fitting after protocol review.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import platform
import random
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/parallel-hsmm-murmur'
_AUTHOR = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_author_modules(cfg):
    """Compile into our build directory; never write bytecode into upstream."""
    global _AUTHOR
    if _AUTHOR is not None:
        return _AUTHOR
    commit = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip()
    if commit != cfg['upstream_commit'] or dirty:
        raise ValueError('Upstream commit or clean-checkout requirement failed')
    build = (ROOT / cfg['cython_build_directory']).resolve()
    build.mkdir(parents=True, exist_ok=True)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(UPSTREAM))
    try:
        import pyximport
        pyximport.install(build_dir=str(build), language_level=3)
        import viterbi_hmm
        if not Path(viterbi_hmm.__file__).resolve().is_relative_to(build):
            raise ValueError('HSMM extension was loaded from an unexpected build')
        modules = []
        for name, relative in [('hsmm_author_neural', 'src/neural_networks.py'),
                               ('hsmm_author_segmenter', 'src/segmenter.py')]:
            spec = importlib.util.spec_from_file_location(name, UPSTREAM / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules.append(module)
        _AUTHOR = (modules[0], modules[1], str(viterbi_hmm.__file__))
    finally:
        sys.path.remove(str(UPSTREAM))
        sys.dont_write_bytecode = old_bytecode
    return _AUTHOR


def calculate_features(recording, fs, cfg):
    """Normalize only the waveform actually supplied; no file/label access."""
    front = cfg['frontend']
    if fs != front['sample_rate']:
        raise ValueError('Expected the original mono 4000-Hz waveform')
    x = torch.as_tensor(np.asarray(recording).copy(), dtype=torch.float32)
    if x.ndim != 1:
        raise ValueError('Expected mono audio')
    detail = {'power_floor_cells': 0, 'std_floor_rows': 0, 'reason': None}
    if len(x) < int(fs * front['minimum_prefix_seconds']):
        detail['reason'] = 'too_short'; return None, detail
    if not torch.isfinite(x).all():
        detail['reason'] = 'nonfinite_waveform'; return None, detail
    x -= x.mean()
    peak = x.abs().max()
    if not torch.isfinite(peak) or peak <= 0:
        detail['reason'] = 'zero_peak'; return None, detail
    x /= peak
    spectrum = torch.stft(x, n_fft=front['window_samples'], hop_length=front['hop_samples'],
                          window=torch.hann_window(front['window_samples']), center=False,
                          return_complex=True)
    # Equivalent arithmetic to the author's return_complex=False implementation.
    power = torch.view_as_real(spectrum).pow(2).sum(-1)[:front['frequency_bins']]
    detail['power_floor_cells'] = int((power < front['power_floor']).sum())
    features = torch.log(power.clamp_min(front['power_floor']))
    std = features.std(dim=-1, keepdim=True)
    detail['std_floor_rows'] = int((std < front['minimum_std']).sum())
    features = (features - features.mean(dim=-1, keepdim=True)) / std.clamp_min(front['minimum_std'])
    if not torch.isfinite(features).all():
        detail['reason'] = 'nonfinite_features'; return None, detail
    return features.contiguous(), detail


def segmentation_targets(tsv_text, n_frames, recording_label, timing):
    """Training annotation parser. Unspecified/noise frames always stay ignored."""
    table = pd.read_csv(io.StringIO(tsv_text), sep='\t', header=None)
    if table.shape[1] != 3 or table.empty:
        raise ValueError('Malformed segmentation columns')
    table.columns = ['start', 'end', 'state']
    values = table.to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not table.state.isin([0, 1, 2, 3, 4]).all()
            or (table.start < 0).any() or (table.end < table.start).any()):
        raise ValueError('Malformed segmentation values')
    target = torch.full((n_frames,), -1, dtype=torch.long)
    explicit = torch.zeros(n_frames, dtype=torch.bool)
    portions = {'Early-systolic': (0, .5), 'Holosystolic': (0, 1),
                'Mid-systolic': (.25, .75), 'Late-systolic': (.5, 1)}
    for row in table.itertuples():
        start, end = int(float(row.start) * 50), int(float(row.end) * 50)
        explicit[start:end] = True
        target[start:end] = {-1: -1, 0: -1, 1: 0, 2: 1, 3: 2, 4: 3}[int(row.state)]
        if int(row.state) == 2 and recording_label == 'Present':
            lo, hi = portions.get(timing, (0, 0))
            duration = end - start
            target[int(start + lo * duration):int(np.ceil(start + hi * duration))] = 4
    return target, {'uncovered_frames': int((~explicit).sum()),
                    'ignored_frames': int((target == -1).sum()),
                    'missing_systolic_timing': bool(recording_label == 'Present' and timing not in portions)}


def inner_folds(train_visits, cfg):
    if not train_visits.role.eq('train').all():
        raise ValueError('Inner folds accept only the outer training role')
    data = train_visits.copy().reset_index(drop=True)
    data['inner_fold'] = -1
    splitter = StratifiedGroupKFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=cfg['seed'])
    for fold, (_, valid) in enumerate(splitter.split(data, data.Murmur, data.group_id)):
        data.loc[valid, 'inner_fold'] = fold
    if (data.inner_fold < 0).any() or data.groupby('group_id').inner_fold.nunique().max() != 1:
        raise ValueError('Inner group leakage')
    return data


def read_metadata():
    manifest = json.loads((ROOT / 'results/metadata/manifest.json').read_text())
    for name, expected in manifest['output_sha256'].items():
        if digest(ROOT / 'results/metadata' / name) != expected:
            raise ValueError(f'Metadata changed: {name}')
    for name, expected in manifest['input_sha256'].items():
        if digest(ROOT / 'data/circor_metadata' / name) != expected:
            raise ValueError(f'Official input changed: {name}')
    visits = pd.read_csv(ROOT / 'results/metadata/split_manifest.csv').rename(columns={'Patient ID': 'patient_id'})
    if visits.groupby('group_id').role.nunique().max() != 1:
        raise ValueError('Linked-person overlap')
    visits = visits[visits.role.ne('reserved_test')].reset_index(drop=True)
    inventory = pd.read_csv(ROOT / 'results/metadata/record_inventory.csv')
    inventory = inventory[inventory.selected & inventory.role.ne('reserved_test')].copy()
    checks = dict((f.lstrip('*'), h) for h, f in (line.split(maxsplit=1) for line in
                  (ROOT / 'data/circor_metadata/SHA256SUMS.txt').read_text().splitlines()))
    return visits, inventory, checks


def prepare_training(cfg):
    visits, inventory, checks = read_metadata()
    folds = inner_folds(visits[visits.role.eq('train')], cfg)
    source = {'frontend': cfg['frontend'], 'training': cfg['training'], 'seed': cfg['seed'],
              'inner_folds': cfg['inner_folds'], 'adapter_sha256': digest(__file__),
              'metadata_sha256': digest(ROOT / 'results/metadata/manifest.json'),
              'official_checksums_sha256': digest(ROOT / 'data/circor_metadata/SHA256SUMS.txt')}
    out = ROOT / cfg['training_cache_directory']
    if (out / 'manifest.json').exists():
        manifest = json.loads((out / 'manifest.json').read_text())
        if manifest['source'] != source:
            raise ValueError('Training cache source differs; use a new cache directory')
        for name, sha in manifest['files_sha256'].items():
            if digest(out / name) != sha:
                raise ValueError(f'Training cache changed: {name}')
        return out, manifest
    if out.exists():
        raise ValueError('Incomplete cache exists; inspect before creating a new version')
    out.mkdir(parents=True); (out / 'arrays').mkdir(); (out / 'annotations').mkdir()
    folds.to_csv(out / 'inner_folds.csv', index=False)
    train = inventory[inventory.role.eq('train') & inventory.Murmur.isin(['Present', 'Absent'])].copy()
    train = train.merge(folds[['patient_id', 'inner_fold']], on='patient_id', validate='many_to_one')
    labels = pd.read_csv(ROOT / 'data/circor_metadata/training_data.csv')
    labels = labels[labels['Patient ID'].isin(folds.patient_id)].set_index('Patient ID')
    records, skipped = [], []
    started = time.time()
    with zipfile.ZipFile(ROOT / 'data/circor-heart-sound-1.0.3.zip') as archive:
        names = {n[n.index('training_data/'):]: n for n in archive.namelist() if 'training_data/' in n}
        for i, row in enumerate(train.itertuples()):
            # Only this filtered TRAINING row can select an annotation member.
            tsv_name = row.record + '.tsv'
            content = archive.read(names[tsv_name])
            if hashlib.sha256(content).hexdigest() != checks[tsv_name]:
                raise ValueError(f'Annotation checksum mismatch: {tsv_name}')
            annotation = out / 'annotations' / (Path(row.record).name + '.tsv')
            annotation.write_bytes(content)
            wav = ROOT / 'data/development_wav' / (Path(row.record).name + '.wav')
            if digest(wav) != checks[row.record + '.wav']:
                raise ValueError(f'WAV checksum mismatch: {row.record}')
            fs, raw = wavfile.read(wav)
            features, guard = calculate_features(raw, fs, cfg)
            base = {'record': row.record, 'patient_id': int(row.patient_id), 'group_id': int(row.group_id),
                    'inner_fold': int(row.inner_fold), 'role': 'train', 'visit_label': row.Murmur,
                    'annotation_sha256': checks[tsv_name], 'wav_sha256': checks[row.record + '.wav']}
            if features is None:
                skipped.append({**base, 'reason': guard['reason']}); continue
            person = labels.loc[row.patient_id]
            locations = set(str(person['Murmur locations']).split('+'))
            rec_label = 'Present' if row.Murmur == 'Present' and row.site in locations else 'Absent'
            timing = str(person['Systolic murmur timing'])
            try:
                target, annotation_stats = segmentation_targets(content.decode(), features.shape[-1], rec_label, timing)
            except ValueError as exc:
                skipped.append({**base, 'reason': str(exc)}); continue
            keep = target != -1
            if not keep.any():
                skipped.append({**base, 'reason': 'no_supervised_frames'}); continue
            path = out / 'arrays' / (Path(row.record).name + '.npz')
            np.savez_compressed(path, features=features[:, keep].numpy(), target=target[keep].numpy())
            records.append({**base, 'array': str(path.relative_to(out)), 'recording_label': rec_label,
                            'frames': int(keep.sum()), **annotation_stats, **{k: v for k, v in guard.items() if k != 'reason'}})
            if i % 250 == 0:
                print(json.dumps({'prepared_training_records': i + 1, 'elapsed_seconds': time.time() - started}), flush=True)
    pd.DataFrame(records).to_csv(out / 'records.csv', index=False)
    write_json(out / 'skipped.json', skipped)
    files = [p for p in out.rglob('*') if p.is_file()]
    manifest = {'source': source, 'records': len(records), 'skipped_records': len(skipped),
                'calibration_validation_reserved_annotations_read': 0,
                'elapsed_seconds': time.time() - started,
                'files_sha256': {str(p.relative_to(out)): digest(p) for p in files}}
    write_json(out / 'manifest.json', manifest)
    return out, manifest


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, records, arrays, crop_frames=None):
        self.records = list(records)
        self.arrays = arrays
        self.crop_frames = crop_frames

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        features, target = self.arrays[self.records[index]]
        if self.crop_frames is not None:
            start = torch.randint(0, max(features.shape[-1] - self.crop_frames, 1), (1,)).item()
            features = features[:, start:start + self.crop_frames]
            target = target[start:start + self.crop_frames]
        return features, target


def collate_frames(items):
    lengths = torch.tensor([len(y) for _, y in items], dtype=torch.long)
    maximum = int(lengths.max())
    features = torch.zeros(len(items), items[0][0].shape[0], maximum)
    targets = torch.full((len(items), maximum), -1, dtype=torch.long)
    for i, (x, y) in enumerate(items):
        features[i, :, :len(y)] = x; targets[i, :len(y)] = y
    return features, targets, lengths


def fit_class_weights(record_names, arrays):
    counts = torch.zeros(5, dtype=torch.long)
    for name in record_names:
        target = arrays[name][1]
        counts += torch.bincount(target[target >= 0], minlength=5)
    if (counts == 0).any():
        raise ValueError(f'Fit partition lacks frame classes: {counts.tolist()}')
    return counts.sum() / counts.float(), counts


def reset_seed(seed, use_cuda=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def infer_record(recording, fs, models, segmenter, cfg, budget='full', device='cpu'):
    """Annotation-free raw-prefix API; never accepts files, labels or TSVs."""
    cap = len(recording) if budget == 'full' else min(len(recording), int(float(budget) * fs))
    acquired = np.asarray(recording)[:cap]
    features, detail = calculate_features(acquired, fs, cfg)
    result = {'acquired_seconds': len(acquired) / fs, 'score': None, 'quality': None,
              'healthy_confidence': None, 'murmur_confidence': None,
              'murmur_model': None, 'failed': features is None, 'feature_guards': detail}
    if features is None:
        return result
    lengths = torch.tensor([features.shape[-1]], dtype=torch.long)
    x = features.unsqueeze(0).to(device)
    probabilities = []
    with torch.no_grad():
        for model in models:
            model.eval()
            probabilities.append(torch.softmax(model(x, lengths)[0], dim=0).cpu())
    posterior = torch.stack(probabilities).mean(0).T.contiguous().numpy().astype(np.float32, copy=False)
    if not np.isfinite(posterior).all():
        result.update(failed=True); detail['reason'] = 'nonfinite_posteriors'; return result
    try:
        decoded = segmenter.double_duration_viterbi(posterior, 50, **cfg['hsmm'])
        healthy, murmur, timing = float(decoded[2]), float(decoded[3]), decoded[4]
        if not np.isfinite([healthy, murmur]).all():
            raise FloatingPointError('Nonfinite HSMM confidence')
    except (ValueError, FloatingPointError, ZeroDivisionError, IndexError) as exc:
        result.update(failed=True); detail['reason'] = f'hsmm_failure:{type(exc).__name__}'; return result
    result.update(score=murmur - healthy, quality=max(healthy, murmur),
                  healthy_confidence=healthy, murmur_confidence=murmur, murmur_model=timing)
    return result


def aggregate_visit(site_results, quality_threshold=.65):
    valid = [r for r in site_results if not r['failed']]
    failed = any(r['failed'] for r in site_results) or not site_results
    score = max((r['score'] for r in valid), default=None)
    quality = min((r['quality'] for r in valid), default=None)
    if score is not None and score > 0:
        label = 'Present'
    elif failed or quality is None or quality < quality_threshold:
        label = 'Unknown'
    else:
        label = 'Absent'
    return {'score': score, 'quality': quality, 'failed': bool(failed),
            'published_label': label, 'published_referred': label != 'Absent',
            'acquired_seconds': sum(r['acquired_seconds'] for r in site_results)}


def calibration_threshold(labels, scores, force_referral, target=.95):
    labels, scores, force_referral = np.asarray(labels), np.asarray(scores), np.asarray(force_referral, bool)
    positive = labels == 1
    n_positive = int(positive.sum())
    if not n_positive:
        raise ValueError('No known positive calibration visits')
    unforced = scores[positive & ~force_referral]
    if not np.isfinite(unforced).all():
        raise ValueError('Unscored patients require explicit forced referral')
    allowed = math.floor((1 - target + 1e-10) * n_positive)
    return float('inf') if len(unforced) <= allowed else float(np.sort(unforced)[allowed])


def metrics(frame, referred_column):
    known = frame[frame.Murmur.isin(['Present', 'Absent'])]
    positive = known.Murmur.eq('Present').to_numpy()
    referred = known[referred_column].to_numpy(bool)
    scored = known[np.isfinite(known.score)]
    scorable = scored.Murmur.nunique() == 2
    return {'n_visits': len(frame), 'n_known': len(known), 'n_positive': int(positive.sum()),
            'n_negative': int((~positive).sum()), 'n_known_scored': len(scored),
            'auroc': float(roc_auc_score(scored.Murmur.eq('Present'), scored.score)) if scorable else None,
            'average_precision': float(average_precision_score(scored.Murmur.eq('Present'), scored.score)) if scorable else None,
            'false_negatives': int((positive & ~referred).sum()),
            'sensitivity': float(referred[positive].mean()) if positive.any() else None,
            'specificity': float((~referred[~positive]).mean()) if (~positive).any() else None,
            'referrals': int(frame[referred_column].sum()), 'referral_fraction': float(frame[referred_column].mean()),
            'n_unknown': int(frame.Murmur.eq('Unknown').sum()),
            'unknown_referred': int(frame.loc[frame.Murmur.eq('Unknown'), referred_column].sum()),
            'visits_with_failed_record': int(frame.failed.sum()),
            'mean_audio_seconds': float(frame.acquired_seconds.mean())}


def evaluate(models, segmenter, cfg, out, device):
    visits, inventory, checks = read_metadata()
    visits = visits[visits.role.isin(['calibration', 'pilot_validation'])].reset_index(drop=True)
    index = {(row.patient_id, row.site): row.record for row in inventory.itertuples()}
    predictions, record_rows, summaries = [], [], []
    for budget in cfg['budgets_seconds']:
        rows = []
        for person in visits.itertuples():
            values = []
            for site in cfg['sites']:
                record = index.get((person.patient_id, site))
                if record is None:
                    continue
                path = ROOT / 'data/development_wav' / (Path(record).name + '.wav')
                if digest(path) != checks[record + '.wav']:
                    raise ValueError('Evaluation WAV checksum mismatch')
                fs, raw = wavfile.read(path)
                start = time.perf_counter()
                result = infer_record(raw, fs, models, segmenter, cfg, budget, device)
                elapsed = time.perf_counter() - start
                values.append(result)
                record_rows.append({'record': record, 'role': person.role, 'budget_seconds': budget,
                                    'inference_seconds': elapsed, **result})
            rows.append({**person._asdict(), **aggregate_visit(values, cfg['quality_threshold'])})
        data = pd.DataFrame(rows)
        data['score'] = pd.to_numeric(data.score)
        data['quality'] = pd.to_numeric(data.quality)
        data['budget_seconds'] = budget
        cal = data.role.eq('calibration') & data.Murmur.isin(['Present', 'Absent'])
        for mode in ['score', 'score_quality']:
            forced = data.failed | data.score.isna()
            if mode == 'score_quality':
                forced |= data.quality.lt(cfg['quality_threshold']) | data.quality.isna()
            threshold = calibration_threshold(data.loc[cal, 'Murmur'].eq('Present').astype(int),
                                              data.loc[cal, 'score'], forced[cal], cfg['calibration_target_sensitivity'])
            data[f'{mode}_threshold'] = threshold
            data[f'{mode}_referred'] = forced | data.score.ge(threshold)
        predictions.append(data)
        for role in ['calibration', 'pilot_validation']:
            for mode in ['published', 'score', 'score_quality']:
                result = {'role': role, 'budget_seconds': budget, 'decision': mode,
                          **metrics(data[data.role.eq(role)], f'{mode}_referred')}
                summaries.append(result); print(json.dumps(result, allow_nan=False), flush=True)
        pd.concat(predictions).to_csv(out / 'predictions.csv', index=False)
        pd.DataFrame(summaries).to_csv(out / 'metrics.csv', index=False)
        write_json(out / 'record_predictions.json', record_rows)


def fit_all(cfg, cache_dir, device, out):
    neural, segmenter, _ = load_author_modules(cfg)
    records = pd.read_csv(cache_dir / 'records.csv')
    if not records.role.eq('train').all() or records.groupby('group_id').inner_fold.nunique().max() != 1:
        raise ValueError('Training cache violates the locked group boundary')
    arrays = {}
    for row in records.itertuples():
        with np.load(cache_dir / row.array, allow_pickle=False) as data:
            arrays[row.record] = (torch.from_numpy(data['features'].copy()), torch.from_numpy(data['target'].copy()))
    models, history = [], []
    train_cfg = cfg['training']
    for fold in range(cfg['inner_folds']):
        reset_seed(cfg['seed'] + fold)
        fit_rows, val_rows = records[records.inner_fold.ne(fold)], records[records.inner_fold.eq(fold)]
        if set(fit_rows.group_id) & set(val_rows.group_id):
            raise ValueError('Inner-fold group intersection')
        train_data = TrainingDataset(fit_rows.record, arrays, train_cfg['crop_frames'])
        val_data = TrainingDataset(val_rows.record, arrays)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_cfg['batch_size'],
                                                  shuffle=True, collate_fn=collate_frames)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=train_cfg['validation_batch_size'],
                                                shuffle=False, collate_fn=collate_frames)
        weights, counts = fit_class_weights(fit_rows.record, arrays)
        model = neural.instantiate_model(cfg['architecture']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=weights.to(device))
        best, stale, started = float('inf'), 0, time.perf_counter()
        checkpoint = out / 'models' / f'fold_{fold}.pt'
        for epoch in range(1, train_cfg['max_epochs'] + 1):
            model.train(); loss_sum, count = 0., 0
            for x, target, lengths in train_loader:
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x, lengths), target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward(); optimizer.step()
                loss_sum += float(loss.detach()) * len(x); count += len(x)
            model.eval(); val_sum, val_count = 0., 0
            with torch.no_grad():
                for x, target, lengths in val_loader:
                    x, target = x.to(device), target.to(device)
                    loss = criterion(model(x, lengths), target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite inner validation loss')
                    val_sum += float(loss) * len(x); val_count += len(x)
            validation_loss = val_sum / val_count
            improved = validation_loss < best
            if improved:
                best, stale = validation_loss, 0
                torch.save({'model_state_dict': model.state_dict(), 'fold': fold, 'epoch': epoch,
                            'validation_loss': best, 'architecture': cfg['architecture'],
                            'class_weights': weights, 'class_counts': counts,
                            'parameters': sum(p.numel() for p in model.parameters()),
                            'fit_group_ids': sorted(set(fit_rows.group_id)),
                            'inner_validation_group_ids': sorted(set(val_rows.group_id))}, checkpoint)
            else:
                stale += 1
            row = {'fold': fold, 'epoch': epoch, 'train_loss': loss_sum / count,
                   'inner_validation_loss': validation_loss, 'improved': improved,
                   'nonimproving_epochs': stale, 'elapsed_seconds': time.perf_counter() - started}
            history.append(row); pd.DataFrame(history).to_csv(out / 'epoch_losses.csv', index=False)
            print(json.dumps(row), flush=True)
            if stale > train_cfg['patience']:
                break
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict']); model.eval(); models.append(model)
        del optimizer
    evaluate(models, segmenter, cfg, out, device)


def smoke(cfg):
    neural, segmenter, binary = load_author_modules(cfg)
    torch.set_num_threads(1); reset_seed(cfg['seed'], use_cuda=False)
    raw = np.random.default_rng(3).integers(-20000, 20000, 8000, dtype=np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        original, _ = neural.calculate_features(torch.as_tensor(raw.copy()), 4000)
    ours, guards = calculate_features(raw, 4000, cfg)
    error = float((ours - original).abs().max())
    if not torch.allclose(ours, original, atol=1e-6, rtol=1e-6):
        raise AssertionError(f'Author feature equivalence failed: {error}')
    model = neural.instantiate_model(cfg['architecture']).eval()
    result = infer_record(raw, 4000, [model], segmenter, cfg, 2, 'cpu')
    altered = np.r_[raw, np.random.default_rng(5).integers(-32000, 32000, 8000, dtype=np.int16)]
    repeated = infer_record(altered, 4000, [model], segmenter, cfg, 2, 'cpu')
    if result['failed'] or result != repeated:
        raise AssertionError('Raw-prefix inference invariance failed')
    silence = infer_record(np.zeros(8000, np.int16), 4000, [model], segmenter, cfg, 2, 'cpu')
    if not silence['failed'] or aggregate_visit([silence])['published_label'] != 'Unknown':
        raise AssertionError('Silence fallback failed')
    value = {'status': 'passed', 'parameters': sum(p.numel() for p in model.parameters()),
             'feature_max_abs_error': error, 'ordinary_guards': guards,
             'fixed_prefix_result': result, 'silence_result': silence,
             'cython': importlib.metadata.version('Cython'), 'extension_path': binary,
             'extension_sha256': digest(binary), 'upstream_commit': cfg['upstream_commit'],
             'training_performed': False, 'patient_inference_performed': False}
    write_json(ROOT / 'reviews/auscultation/hsmm_adapter_smoke.json', value)
    print(json.dumps(value, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/hsmm_baseline.json')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--train', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    torch.set_num_threads(cfg['torch_threads'])
    if not args.prepare_only and not args.train:
        smoke(cfg); return
    cache_dir, cache_manifest = prepare_training(cfg)
    if args.prepare_only:
        print(json.dumps({'status': 'training_cache_ready', 'records': cache_manifest['records'],
                          'skipped': cache_manifest['skipped_records'], 'fitting_performed': False}), flush=True)
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    out = ROOT / cfg['output_directory']
    if out.exists():
        raise ValueError('Refusing to overwrite an existing experiment')
    out.mkdir(parents=True); (out / 'models').mkdir()
    _, _, binary = load_author_modules(cfg)
    run = {'status': 'running', 'started_unix': time.time(), 'config': cfg,
           'python': sys.version, 'platform': platform.platform(), 'device': str(device),
           'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
           'packages': {n: importlib.metadata.version(n) for n in ['torch', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'Cython']},
           'input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                            [args.config.resolve(), Path(__file__).resolve(), ROOT / 'HSMM_PROTOCOL.md',
                             ROOT / 'HSMM_LICENSE.md', cache_dir / 'manifest.json']},
           'upstream_commit': cfg['upstream_commit'], 'extension_sha256': digest(binary),
           'reserved_predictions': 0, 'calibration_validation_reserved_annotations_read': 0}
    write_json(out / 'run.json', run)
    try:
        fit_all(cfg, cache_dir, device, out)
        run.update(status='complete', elapsed_seconds=time.time() - run['started_unix'])
    except Exception as exc:
        run.update(status='failed', error=repr(exc), elapsed_seconds=time.time() - run['started_unix'])
        raise
    finally:
        write_json(out / 'run.json', run)


if __name__ == '__main__':
    main()
