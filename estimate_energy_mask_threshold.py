"""Estimate a fixed raw natural-log Mel POWER energy-mask threshold using TRAIN;
validate its positive-cell rate on DEV. No models, training, or TEST access.

Usage:
 python -B estimate_energy_mask_threshold.py --splits Data/splits --out reports/energy_mask_threshold_v1
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def locate_cache(splits: Path, manifest: dict, cache_arg: str | None):
    if cache_arg:
        path=Path(cache_arg).expanduser().resolve()
        if not path.is_dir(): raise FileNotFoundError(path)
        return path
    candidates=[]
    raw=manifest.get('source_cache_dir')
    if raw:
        candidates.append(Path(raw))
    candidates.extend([splits.parent.parent/'prepared',splits.parent/'prepared',Path('prepared')])
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / f'{split}.jsonl').exists() for split in ('train','dev')):
            return candidate.resolve()
    raise FileNotFoundError('Could not locate cache with train.jsonl and dev.jsonl. Pass --cache prepared')


def collect(split, root, per_file, seed, max_files=0):
    # Draw a fixed number of uniformly sampled raw log-Mel cells per utterance.
    # Sampling is with replacement, reproducible, memory-bounded and independent
    # across TRAIN and DEV. Does not use test.jsonl or test patches.
    rng=np.random.default_rng(seed)
    samples=[]
    n_files=0
    n_cells=0
    for item in read_jsonl(root/f'{split}.jsonl'):
        path=root/item['cache']
        mat=np.load(path, mmap_mode='r', allow_pickle=False)
        if mat.ndim!=2 or mat.shape[0]!=96 or mat.size==0:
            raise ValueError(f'Invalid [96,T] log-Mel cache: {path} {mat.shape}')
        if not np.isfinite(mat).all():
            raise ValueError(f'Non-finite log-Mel values: {path}')
        rows=rng.integers(0,96,size=per_file)
        cols=rng.integers(0,mat.shape[1],size=per_file)
        samples.append(np.asarray(mat[rows,cols],dtype=np.float32))
        n_files+=1
        n_cells+=int(mat.size)
        if max_files and n_files>=max_files:
            break
    if not samples:
        raise ValueError(f'No {split} cache entries')
    return np.concatenate(samples),n_files,n_cells


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',default='Data/splits')
    p.add_argument('--cache',default=None,help='Cache root containing train.jsonl and logmel/*.npy')
    p.add_argument('--out',default='reports/energy_mask_threshold_v1')
    p.add_argument('--samples-per-utterance',type=int,default=128)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--percentiles',default='65,70,75,80,85,90')
    p.add_argument('--choose-percentile',type=float,default=75.0)
    p.add_argument('--max-train-files',type=int,default=0,help='0=all; >0 for smoke tests only')
    p.add_argument('--max-dev-files',type=int,default=0,help='0=all; >0 for smoke tests only')
    a=p.parse_args()
    if a.samples_per_utterance<1 or a.max_train_files<0 or a.max_dev_files<0:
        p.error('Sample and file limits must be valid')
    percentiles=sorted(set([float(x.strip()) for x in a.percentiles.split(',') if x.strip()]+[a.choose_percentile]))
    if any(not np.isfinite(q) or not 0<q<100 for q in percentiles):
        p.error('Percentiles must lie strictly between 0 and 100')
    splits=Path(a.splits).expanduser().resolve()
    manifest_path=splits/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    root=locate_cache(splits,manifest,a.cache)
    train,n_train,c_train=collect('train',root,a.samples_per_utterance,a.seed,a.max_train_files)
    dev,n_dev,c_dev=collect('dev',root,a.samples_per_utterance,a.seed+100003,a.max_dev_files)
    out=Path(a.out).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new/empty --out directory: {out}')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for q in percentiles:
        t=float(np.percentile(train,q))
        rows.append({'train_percentile':q,'threshold_raw_logmel':t,
                     'train_positive_fraction':float(np.mean(train>t)),
                     'dev_positive_fraction':float(np.mean(dev>t)),
                     'train_sampled_cells':len(train),'dev_sampled_cells':len(dev)})
    with (out/'threshold_candidates.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    chosen=next(r for r in rows if r['train_percentile']==a.choose_percentile)
    metadata={'status':'candidate_threshold_not_F1_optimized',
              'train_files':n_train,'dev_files':n_dev,'train_full_cells':c_train,'dev_full_cells':c_dev,
              'train_sampled_cells':int(train.size),'dev_sampled_cells':int(dev.size),
              'seed':a.seed,'samples_per_utterance':a.samples_per_utterance,
              'selected_train_percentile':a.choose_percentile,
              'energy_mask_threshold_raw_logmel':chosen['threshold_raw_logmel'],
              'train_positive_fraction':chosen['train_positive_fraction'],
              'dev_positive_fraction':chosen['dev_positive_fraction'],
              'splits_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'source_cache':str(root),
              'limited_run':bool(a.max_train_files or a.max_dev_files),
              'warning':'Threshold defines energy-mask detection only. Precision/Recall/F1 need predictions from a checkpoint; no TEST was read.'}
    (out/'selected_threshold.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[DATA] TRAIN: {n_train} utterances / {len(train)} sampled cells; DEV: {n_dev} utterances / {len(dev)} sampled cells')
    for r in rows:
        print(f"[P{r['train_percentile']:g}] threshold={r['threshold_raw_logmel']:.6f} TRAIN_positive={r['train_positive_fraction']:.3%} DEV_positive={r['dev_positive_fraction']:.3%}")
    print(f"[CHOSEN] percentile={a.choose_percentile:g}; threshold_raw_logmel={chosen['threshold_raw_logmel']:.8f}")
    print(f'[SAVED] {out}')

if __name__=='__main__':
    main()
