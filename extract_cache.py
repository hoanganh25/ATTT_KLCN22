"""One-time, resumable WAV/FLAC -> numeric log-Mel cache -> leak-safe splits.

Usage: python prepare_dataset.py --dataset "D:/Datasets/LibriSpeech" --out prepared
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import numpy as np
from mel_frontend import MelConfig, preprocess
from mel_frontend.audio_io import read_wave
from mel_frontend.mel_filterbank import make_mel_bank

AUDIO_EXTENSIONS = {'.wav', '.flac'}
SPLIT_DIRS = {'train-clean-100':'train', 'train-clean-360':'train', 'train-other-500':'train',
              'dev-clean':'dev', 'dev-other':'dev', 'test-clean':'test', 'test-other':'test'}


def seed_hash(key):
    return hashlib.sha1(key.encode('utf-8')).hexdigest()


def official_split(p: Path):
    for ancestor in [p.parent, *p.parents]:
        if ancestor.name.lower() in SPLIT_DIRS:
            return SPLIT_DIRS[ancestor.name.lower()]
    return None


def speaker_id(p: Path, root: Path):
    # LibriSpeech: 123-456-0001.flac nested under 123/456/. Another
    # common dataset layout: .../speaker_123/utterance.wav.
    match = re.match(r'^(\d+)-\d+-\d+$', p.stem)
    if match:
        return f'librispeech_{match.group(1)}'
    for part in reversed(p.relative_to(root).parts[:-1]):
        if re.match(r'^(speaker[_-]?\w+|sp\d+|\d{3,})$', part, re.I):
            return part.lower()
    return str(p.parent.relative_to(root)).replace('\\', '/').lower()


def atomic_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as fh:
        for obj in records:
            fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def load_jsonl(path):
    if not path.is_file():
        return []
    with path.open('r', encoding='utf-8') as f:
        return [json.loads(row) for row in f if row.strip()]


def partition(rows, *, seed=42, train_frac=.8, dev_frac=.1, split_unit='speaker'):
    """Use official train/dev/test if all exist; else split by speaker (default)."""
    official = {x.get('official') for x in rows if x.get('official')}
    if {'train','dev','test'} <= official and all(x.get('official') for x in rows):
        out = {s:[r for r in rows if r['official']==s] for s in ('train','dev','test')}
        speakers = {s:{r['speaker'] for r in arr} for s,arr in out.items()}
        if any(speakers[a] & speakers[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
            raise ValueError('Official split contains overlapping speaker IDs! Check dataset layout.')
        return out, 'official'
    if official:
        print('WARNING: Only a subset of official train/dev/test directories found; performing fresh split.')
    if split_unit == 'speaker':
        groups = {}
        for r in rows:
            groups.setdefault(r['speaker'], []).append(r)
    else:
        groups = {r['relpath']:[r] for r in rows}
        print('WARNING: FILE-based split can leak speakers across splits; never claim speaker-independent test.')
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('Need >=3 independent split groups. Add speakers or use --split-unit file for a quick demo only.')
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * train_frac)))
    n_dev = max(1, int(round(n * dev_frac)))
    if n_train + n_dev >= n:
        n_train = n - 2
        n_dev = 1
    sets = {'train':set(keys[:n_train]), 'dev':set(keys[n_train:n_train+n_dev]),
            'test':set(keys[n_train+n_dev:])}
    out = {s:[r for key in sorted(gkeys) for r in groups[key]] for s,gkeys in sets.items()}
    return out, f'{split_unit}-disjoint'


def compute_norm(train_rows, cache_dir: Path):
    """Training-only per-Mel-bin mean/std; stored as lists; no train/val leakage."""
    sum_x = np.zeros(96, dtype=np.float64)
    sum_x2 = np.zeros(96, dtype=np.float64)
    count = 0
    rng = np.random.default_rng(2026)
    reservoir = np.empty(200000, dtype=np.float32)
    used = 0
    # Approximate global 75th quantile of raw log-Mel; reproducible sampling
    # across frames/utterances, capped for memory.
    for row in train_rows:
        a = np.load(cache_dir / row['cache'], mmap_mode='r')
        if a.ndim != 2 or a.shape[0] != 96:
            raise ValueError(f'Bad cache shape for {row["relpath"]}: {a.shape}')
        v = np.asarray(a, dtype=np.float64)
        sum_x += v.sum(axis=1)
        sum_x2 += np.square(v).sum(axis=1)
        count += v.shape[1]
        # reservoir approximation, 64 uniformly sampled values per utterance
        flat = a.ravel()
        take = min(128, flat.size)
        samples = flat[rng.integers(0, flat.size, size=take)].astype(np.float32)
        # streaming reservoir over sampled draws; avoids load-all stats
        for sample in samples:
            used += 1
            if used <= len(reservoir):
                reservoir[used-1] = sample
            else:
                j = int(rng.integers(0, used))
                if j < len(reservoir):
                    reservoir[j] = sample
    mean = sum_x / count
    std = np.sqrt(np.maximum(sum_x2 / count - mean**2, 1e-6))
    threshold = float(np.quantile(reservoir[:min(used, len(reservoir))], .75))
    return {'mean':mean.tolist(), 'std':std.tolist(), 'high_energy_threshold':threshold,
            'train_frames':int(count), 'threshold_note':'75th percentile of sampled TRAIN raw log-Mel cells'}


def run(args):
    root = Path(args.dataset).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Dataset directory does not exist: {root}')
    if out == root or root in out.parents:
        # prevent accidental scanning of generated outputs under input dataset
        print('NOTE: Output path is inside source tree; cache files are .npy and excluded by extension.')
    cfg = MelConfig()
    paths = sorted((p for p in root.rglob('*') if p.suffix.lower() in AUDIO_EXTENSIONS and p.is_file()))
    if args.max_files is not None:
        paths = paths[:args.max_files]
    if not paths:
        raise FileNotFoundError(f'No .wav/.flac found under {root}')
    out.mkdir(parents=True, exist_ok=True)
    meta = {'source_root':str(root), 'dsp':cfg.as_dict(), 'files_selected':[str(p.relative_to(root)) for p in paths],
            'split_seed':args.seed, 'split_unit':args.split_unit, 'train_fraction':args.train_frac,
            'dev_fraction':args.dev_frac}
    settings = out / 'settings.json'
    if settings.exists():
        before = json.loads(settings.read_text(encoding='utf8'))
        if before != meta:
            raise RuntimeError(f'Existing {out} was prepared with DIFFERENT inputs/config. Use a NEW --out folder (recommended).')
    settings.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    existing = {r['relpath']:r for r in load_jsonl(out/'manifest.jsonl')}
    cache = out / 'logmel'
    cache.mkdir(exist_ok=True)
    bank = make_mel_bank(cfg)
    records = []
    failures = []
    for idx,p in enumerate(paths,1):
        rel = p.relative_to(root).as_posix()
        cache_rel = f'logmel/{seed_hash(rel)}.npy'
        target = out / cache_rel
        stat = p.stat()
        saved = existing.get(rel)
        row = {'relpath':rel, 'cache':cache_rel, 'speaker':speaker_id(p, root),
               'official':official_split(p), 'source_size':stat.st_size,
               'source_mtime_ns':stat.st_mtime_ns}
        valid_cache = (target.exists() and saved is not None and
                       saved.get('source_size') == stat.st_size and
                       saved.get('source_mtime_ns') == stat.st_mtime_ns)
        try:
            if valid_cache:
                row.update({k:saved[k] for k in ('frames','duration_sec')})
            else:
                wave = read_wave(p, cfg)
                logmel = preprocess(wave, cfg, bank)
                with tempfile.NamedTemporaryFile(mode='wb', dir=cache, suffix='.npy', delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                    np.save(tmp, logmel, allow_pickle=False)
                os.replace(tmp_path, target)
                row['frames'] = int(logmel.shape[1])
                row['duration_sec'] = float(wave.size/cfg.sr)
            records.append(row)
        except Exception as exc:
            failures.append({'file':rel, 'error':str(exc)})
            if not args.skip_errors:
                atomic_jsonl(out/'failed.jsonl', failures)
                raise RuntimeError(f'Failed at {rel}: {exc}') from exc
        if idx % args.progress_every == 0 or idx == len(paths):
            print(f'[{idx}/{len(paths)}] extracted/cached={len(records)}, failed={len(failures)}', flush=True)
        if idx % 50 == 0:
            atomic_jsonl(out/'manifest.jsonl', records)
    atomic_jsonl(out/'manifest.jsonl', records)
    if failures:
        atomic_jsonl(out/'failed.jsonl', failures)
        print(f'WARNING: {len(failures)} failures recorded in failed.jsonl')
    if not records:
        raise RuntimeError('No valid features were extracted')
    splits, method = partition(records, seed=args.seed, train_frac=args.train_frac,
                               dev_frac=args.dev_frac, split_unit=args.split_unit)
    for split,arr in splits.items():
        atomic_jsonl(out/f'{split}.jsonl', arr)
    norm = compute_norm(splits['train'], out)
    (out/'normalization.json').write_text(json.dumps(norm, indent=2), encoding='utf-8')
    summary = {'files':len(records), 'split_method':method, 'split_counts':{s:len(v) for s,v in splits.items()},
               'speakers':{s:len(set(r['speaker'] for r in v)) for s,v in splits.items()},
               'duration_hours':{s:sum(r['duration_sec'] for r in v)/3600 for s,v in splits.items()},
               'dsp':cfg.as_dict(), 'cache_dtype':'float32', 'matrix_shape':'[96,T]',
               'model_input_shape':'[B,1,96,96]', 'original_wav_not_modified':True}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'OK. Prepared data: {out}')
    return summary


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='Root containing LibriSpeech .flac or .wav')
    p.add_argument('--out', default='prepared')
    p.add_argument('--max-files', type=int, default=None, help='Only for smoke tests; do not use for thesis results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-frac', type=float, default=.8)
    p.add_argument('--dev-frac', type=float, default=.1)
    p.add_argument('--split-unit', choices=['speaker','file'], default='speaker')
    p.add_argument('--skip-errors', action='store_true')
    p.add_argument('--progress-every', type=int, default=20)
    return p

if __name__ == '__main__':
    run(parser().parse_args())
