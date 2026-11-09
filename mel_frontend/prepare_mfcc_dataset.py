"""One-time V0.1 raw log-Mel cache -> MFCC cache -> fixed MFCC train/val/test NPY.

DOES NOT read the audio again. DOES NOT resplit speakers/files.
DOES NOT modify existing Mel caches, Mel splits or checkpoints.

python -m mel_frontend.prepare_mfcc_dataset --mel-cache workspace/prepared \
     --mel-splits Data/splits --out Data/mfcc_splits --n-mfcc 40
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np

from .mfcc import logmel_to_mfcc

PATCH = 96


def read_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path, entries):
    p = Path(path)
    tmp = p.with_name(p.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, p)


def positions(t):
    """Same patch positions as original prepare_dataset.positions()."""
    if t <= PATCH:
        return [0]
    starts = list(range(0, t - PATCH + 1, PATCH))
    if starts[-1] != t - PATCH:
        starts.append(t - PATCH)
    return starts


def atomic_npy(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', suffix='.npy', dir=path.parent, delete=False) as f:
        temp = Path(f.name)
        np.save(f, array, allow_pickle=False)
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def validate_origin(mel_cache, mel_splits):
    cache = Path(mel_cache).expanduser().resolve()
    mel = Path(mel_splits).expanduser().resolve()
    manifest_p = mel / 'manifest.json'
    if not manifest_p.is_file():
        raise FileNotFoundError(f'Original Mel split manifest missing: {manifest_p}')
    old = json.loads(manifest_p.read_text(encoding='utf-8'))
    original_source = Path(old.get('source_cache_dir', '')).expanduser().resolve()
    if old.get('source_cache_dir') and original_source != cache:
        raise ValueError(f'Mel split was generated from {original_source}, not {cache}. Pass matching --mel-cache.')
    if old.get('shape') != [96, 96]:
        raise ValueError('This V0.1 MFCC preparation expects original 96x96 Mel patches')
    groups = {}
    for dest, source in (('train', 'train'), ('val', 'dev'), ('test', 'test')):
        path = cache / f'{source}.jsonl'
        if not path.is_file():
            raise FileNotFoundError(f'Missing saved {source} assignments: {path}')
        rows = read_jsonl(path)
        n = sum(len(positions(int(row['frames']))) for row in rows)
        if n != old['sizes'][dest]:
            raise ValueError(f'Original {dest} has {old["sizes"][dest]} patches but manifest rows imply {n}. Refusing to change split.')
        groups[dest] = rows
    digest = hashlib.sha256(manifest_p.read_bytes()).hexdigest()
    return cache, mel, old, digest, groups


def build_cache(mel_cache, mfcc_cache, groups, n_mfcc, signature):
    mfcc_cache = Path(mfcc_cache).expanduser().resolve()
    mfcc_cache.mkdir(parents=True, exist_ok=True)
    settings = {'mel_cache': str(mel_cache), 'mel_split_sha256': signature,
                'n_mfcc': n_mfcc, 'transform': 'DCT-II norm=ortho on RAW natural-log Slaney Mel power, including c0'}
    settings_p = mfcc_cache / 'settings.json'
    if settings_p.exists() and json.loads(settings_p.read_text(encoding='utf-8')) != settings:
        raise RuntimeError(f'MFCC cache has incompatible settings: {settings_p}. Use another --mfcc-cache.')
    settings_p.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    index_p = mfcc_cache / 'cache_manifest.jsonl'
    prior = {r['mel_rel']: r for r in read_jsonl(index_p)} if index_p.exists() else {}
    unique = {}
    for rows in groups.values():
        for row in rows:
            unique[row['cache']] = row
    new_index = []
    for i, (mel_rel, row) in enumerate(unique.items(), 1):
        src = mel_cache / mel_rel
        if not src.is_file():
            raise FileNotFoundError(f'Missing existing raw log-Mel: {src}')
        s = src.stat()
        rel = f'mfcc/{Path(mel_rel).name}'
        dest = mfcc_cache / rel
        prev = prior.get(mel_rel)
        valid = (dest.is_file() and prev and prev['source_bytes'] == s.st_size
                 and prev['source_mtime_ns'] == s.st_mtime_ns and prev['frames'] == int(row['frames']))
        if not valid:
            mel = np.load(src, mmap_mode='r', allow_pickle=False)
            if mel.shape != (PATCH, int(row['frames'])):
                raise ValueError(f'Source shape mismatch: {src} {mel.shape}')
            c = logmel_to_mfcc(mel, n_mfcc)
            atomic_npy(dest, c)
        new_index.append({'mel_rel': mel_rel, 'mfcc_rel': rel, 'source_bytes': s.st_size,
                          'source_mtime_ns': s.st_mtime_ns, 'frames': int(row['frames'])})
        if i % 200 == 0:
            write_jsonl_atomic(index_p, new_index)
            print(f'[MFCC CACHE] {i:,}/{len(unique):,}', flush=True)
    write_jsonl_atomic(index_p, new_index)
    print(f'[MFCC CACHE] {len(unique):,} utterances ready (no WAV/FLAC read).')
    return {row['mel_rel']: row['mfcc_rel'] for row in new_index}, mfcc_cache


def train_stats(train_rows, mapping, cache, n_mfcc):
    sum_x = np.zeros(n_mfcc, dtype=np.float64)
    sum_x2 = np.zeros(n_mfcc, dtype=np.float64)
    total = 0
    for i, row in enumerate(train_rows, 1):
        a = np.load(cache / mapping[row['cache']], mmap_mode='r', allow_pickle=False)
        block = np.asarray(a, dtype=np.float64)
        sum_x += block.sum(axis=1)
        sum_x2 += np.square(block).sum(axis=1)
        total += block.shape[1]
        if i % 3000 == 0:
            print(f'[MFCC NORM] processed {i:,} train utterances', flush=True)
    if not total:
        raise ValueError('Empty train split')
    mean = sum_x / total
    std = np.sqrt(np.maximum(sum_x2 / total - mean ** 2, 1e-6))
    return {'mean': mean.tolist(), 'std': std.tolist(), 'n_mfcc': n_mfcc,
            'train_frames': total, 'method': 'per-coefficient z-score, fitted from TRAIN raw cached MFCC ONLY',
            'dct_type': 2, 'dct_norm': 'ortho', 'log_input': 'natural log power Mel, not dB'}


def prepare(mel_cache='workspace/prepared', mel_splits='Data/splits', out='Data/mfcc_splits',
            mfcc_cache='workspace/mfcc_cache', n_mfcc=40):
    if not 1 <= n_mfcc <= PATCH:
        raise ValueError('Require 1 <= n_mfcc <= 96')
    mel_cache, mel_splits, old, signature, groups = validate_origin(mel_cache, mel_splits)
    out = Path(out).expanduser().resolve()
    mfcc_cache = Path(mfcc_cache).expanduser().resolve()
    if out == mel_splits or mel_splits in out.parents or mel_cache in out.parents:
        raise ValueError('MFCC split output cannot overwrite/lie inside original Mel split or source cache.')
    if mfcc_cache == mel_cache or mel_cache in mfcc_cache.parents:
        raise ValueError('MFCC cache must be distinct from original Mel cache.')
    settings = {'source_mel_split_sha256': signature, 'source_mel_cache': str(mel_cache),
                'n_mfcc': n_mfcc, 'shape': [96, 96]}
    if out.exists():
        manifest = out / 'manifest.json'
        if not manifest.exists():
            raise RuntimeError(f'Output exists without completed manifest: {out}. Choose new --out.')
        prev = json.loads(manifest.read_text(encoding='utf-8'))
        if any(prev.get(k) != v for k, v in settings.items()):
            raise RuntimeError(f'Existing output {out} uses other data or n_mfcc. Choose new --out.')
        for key in ('train', 'val', 'test'):
            a = np.load(out / f'{key}.npy', mmap_mode='r', allow_pickle=False)
            if a.shape != (old['sizes'][key], 96, 96):
                raise RuntimeError(f'Split {key} changed. Refusing reuse.')
        print(f'[MFCC SPLITS] Already prepared at {out}; no preprocessing or splitting performed.')
        return prev
    if not all(groups.values()):
        raise ValueError('Need nonempty train/val/test to prepare MFCC')
    mapping, mfcc_cache = build_cache(mel_cache, mfcc_cache, groups, n_mfcc, signature)
    stats = train_stats(groups['train'], mapping, mfcc_cache, n_mfcc)
    mean = np.asarray(stats['mean'], dtype=np.float32)[:, None]
    std = np.asarray(stats['std'], dtype=np.float32)[:, None]
    staged = out.with_name(out.name + '_building')
    if staged.exists():
        raise RuntimeError(f'Incomplete staging dir exists: {staged}. Inspect or remove after verifying it is safe.')
    staged.mkdir(parents=True, exist_ok=False)
    try:
        for split, rows in groups.items():
            total = old['sizes'][split]
            mm = np.lib.format.open_memmap(staged / f'{split}.npy', mode='w+', dtype=np.float32,
                                            shape=(total, PATCH, PATCH))
            k = 0
            for row in rows:
                a = np.load(mfcc_cache / mapping[row['cache']], mmap_mode='r', allow_pickle=False)
                for at in positions(int(row['frames'])):
                    block = np.asarray(a[:, at:at + PATCH], dtype=np.float32)
                    mm[k].fill(0.0)
                    mm[k, :n_mfcc, :block.shape[1]] = (block - mean) / std
                    k += 1
                    if k % 2000 == 0:
                        print(f'[MFCC SPLITS] {split}: {k:,}/{total:,}', flush=True)
            if k != total:
                raise RuntimeError(f'{split}: wrote {k} patches instead of {total}')
            mm.flush()
            del mm
            print(f'[MFCC SPLITS] {split}: {total:,} patches [96,96] (active={n_mfcc})')
        (staged / 'normalization.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
        manifest_out = {**settings, 'dtype': 'float32', 'sizes': old['sizes'],
                        'num_samples': old['num_samples'], 'seed': old.get('seed', 42), 'source_mel_splits': str(mel_splits),
                        'mfcc_cache': str(mfcc_cache), 'normalization': 'train-only zscore',
                        'dct': 'type 2, ortho, applied to cached RAW natural-log Mel',
                        'padded_coefficients': 96 - n_mfcc,
                        'padding': 'zero normalized values; loss/evaluation MUST ignore rows n_mfcc:96',
                        'split_provenance': 'EXACT saved train/dev/test assignment and patch positions, no re-split',
                        'note': 'Fresh MFCC model only; NO load of Mel checkpoint.'}
        (staged / 'manifest.json').write_text(json.dumps(manifest_out, indent=2), encoding='utf-8')
        staged.rename(out)
        print(f'[DONE] MFCC data saved: {out}', flush=True)
        return manifest_out
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mel-cache', default='workspace/prepared')
    p.add_argument('--mel-splits', default='Data/splits')
    p.add_argument('--mfcc-cache', default='workspace/mfcc_cache')
    p.add_argument('--out', default='Data/mfcc_splits')
    p.add_argument('--n-mfcc', type=int, default=40, help='40 for compact MFCC; 96 for full 96-bin DCT comparison')
    args = p.parse_args()
    prepare(args.mel_cache, args.mel_splits, args.out, args.mfcc_cache, args.n_mfcc)


if __name__ == '__main__':
    main()
