"""STANDALONE (no training, no dataset preparation) statistics for V0.1.

  python dataset_statistics.py --splits Data/splits --out reports/dataset_stats

Reads the saved split metadata, cached utterance manifests and memory-mapped
[N,96,96] NPY split arrays. Does NOT touch model/checkpoint/train/evaluate.
Default patch distribution statistics are REPRODUCIBLE SAMPLES, not population.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np


def read_json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def manifest_stats(p: Path):
    if not p.exists():
        return None
    n, duration, frame_count = 0, 0.0, 0
    speakers, files = set(), set()
    ext = Counter()
    seconds = []
    with p.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            n += 1
            seconds.append(float(item.get('duration_sec', 0.0)))
            duration += float(item.get('duration_sec', 0.0))
            frame_count += int(item.get('frames', 0))
            speakers.add(str(item.get('speaker','unknown')))
            name = str(item.get('relpath',''))
            files.add(name)
            ext[Path(name).suffix.lower()] += 1
    return {
        'utterance_count':n, 'speaker_count':len(speakers),
        'duration_seconds':round(duration, 3),
        'duration_hours':round(duration / 3600, 5),
        'total_logmel_frames':frame_count,
        'duration_mean_sec':float(np.mean(seconds)) if seconds else None,
        'duration_median_sec':float(np.median(seconds)) if seconds else None,
        'duration_min_sec':float(min(seconds)) if seconds else None,
        'duration_max_sec':float(max(seconds)) if seconds else None,
        'audio_extensions':dict(ext),
        '_speakers':speakers, '_files':files,
    }


def sampled_array_stats(array: np.ndarray, sample_patches: int, seed: int):
    n = len(array)
    if n == 0 or sample_patches <= 0:
        return {'n_patches_sampled':0}
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=min(n,sample_patches),replace=False))
    # Do not materialize 250k*96*96. Each sampled patch ~36KB.
    total, sqsum, lo, hi, zero, count = 0., 0., np.inf, -np.inf, 0, 0
    for i in indices:
        a = np.asarray(array[int(i)], dtype=np.float64)
        if not np.isfinite(a).all():
            raise ValueError(f'NaN/Inf in sampled patch {i}')
        total += float(a.sum())
        sqsum += float(np.square(a).sum())
        lo = min(lo,float(a.min()));hi = max(hi,float(a.max()))
        zero += int(np.count_nonzero(a == 0.0))
        count += int(a.size)
    avg = total / count
    return {'n_patches_sampled':len(indices),
            'mean_normalized_value_sample':avg,
            'std_normalized_value_sample':float(np.sqrt(max(0,sqsum/count-avg**2))),
            'min_sample':lo, 'max_sample':hi,
            'fraction_exact_zeros_sample':zero/count,
            'sample_notice':'Patch cells include center/padded frames; these are SAMPLE estimates, not full-dataset statistics'}


def main(args):
    splits = Path(args.splits).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not (splits/'manifest.json').is_file():
        raise FileNotFoundError(f'Missing fixed prepared split manifest {splits}/manifest.json')
    meta = read_json(splits/'manifest.json')
    cache = Path(args.cache).expanduser().resolve() if args.cache else Path(meta.get('source_cache_dir','')).expanduser().resolve()
    output = {'dataset_root':meta.get('source_dataset_root'),
              'split_folder':str(splits),'cache_folder':str(cache),
              'split_strategy':meta.get('split_strategy'),
              'normalization':meta.get('normalization'),
              'dsp':meta.get('dsp'),
              'sampling_seed':args.seed,'sample_patches_per_split':args.sample_patches,
              'split_stats':{},'speaker_overlap':{},'file_overlap':{},
              'description':'Read-only report of previously preprocessed V0.1 dataset'}
    if (splits/'normalization.json').exists():
        norm = read_json(splits/'normalization.json')
        output['train_normalization'] = {
            'mel_bins':len(norm.get('mean',[])),
            'mean_across_mel_bins':float(np.mean(norm['mean'])),
            'std_across_mel_bins':float(np.mean(norm['std'])),
            'threshold':norm.get('high_energy_threshold'),
            'fitted_train_frames':norm.get('train_frames'),
            'note':'These are TRAIN-only normalization stats, not computed from test'}
    speaker_sets, file_sets = {}, {}
    for j, (split, cache_split) in enumerate([('train','train'),('val','dev'),('test','test')]):
        path = splits/f'{split}.npy'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}')
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if array.ndim != 3 or array.shape[1:] != (96,96):
            raise ValueError(f'Wrong shape {path}: {array.shape}')
        if array.dtype != np.float32:
            raise ValueError(f'Wrong dtype {path}: {array.dtype}')
        stats = {'patch_count':int(array.shape[0]),'patch_shape':[96,96],
                 'dtype':str(array.dtype),
                 'npy_file_size_GiB':round(path.stat().st_size/1024**3,4),
                 'patch_sample_stats':sampled_array_stats(array,args.sample_patches,args.seed+j)}
        if meta.get('sizes',{}).get(split) != stats['patch_count']:
            raise ValueError(f'Patch count in {path} disagrees with saved manifest')
        row_stats = manifest_stats(cache/f'{cache_split}.jsonl')
        if row_stats:
            speaker_sets[split] = row_stats.pop('_speakers')
            file_sets[split] = row_stats.pop('_files')
            stats['source_audio_stats'] = row_stats
            # Confirm count from exact original patching policy (no NPY materialization).
            # Exact 96-frame non-overlapping patch policy of prepare_dataset,
            # reproduced locally to keep this statistics script standalone.
            def patch_count_for_frames(T):
                if T <= 96: return 1
                return (T - 96) // 96 + 1 + ((T - 96) % 96 != 0)
            patch_check = 0
            with (cache/f'{cache_split}.jsonl').open(encoding='utf-8') as fh:
                for line in fh:
                    if line.strip():
                        patch_check += patch_count_for_frames(int(json.loads(line)['frames']))
            stats['expected_patch_count_from_audio'] = patch_check
            stats['patch_count_verified'] = (patch_check == stats['patch_count'])
            if not stats['patch_count_verified']:
                raise RuntimeError(f'Patch count mismatch for {split}')
        else:
            stats['source_audio_stats'] = None
            print(f'[WARN] No cache manifest {cache}/{cache_split}.jsonl; only NPY stats available')
        output['split_stats'][split] = stats
        print(f'[{split.upper()}] {stats["patch_count"]:,} patches | {stats.get("source_audio_stats",{}).get("utterance_count","unknown") if stats.get("source_audio_stats") else "unknown"} utterances',flush=True)
    for a,b in (('train','val'),('train','test'),('val','test')):
        key=f'{a}_vs_{b}'
        if a in speaker_sets and b in speaker_sets:
            output['speaker_overlap'][key]=len(speaker_sets[a]&speaker_sets[b])
            output['file_overlap'][key]=len(file_sets[a]&file_sets[b])
    with (out/'dataset_statistics.json').open('w',encoding='utf-8') as fh:
        json.dump(output,fh,ensure_ascii=False,indent=2,allow_nan=False)
    columns=['split','patch_count','npy_file_size_GiB','utterance_count','speaker_count',
             'duration_hours','duration_mean_sec','duration_median_sec','mel_frames',
             'sampled_patches','sample_mean_normalized','sample_std_normalized','sample_padding_zero_fraction']
    with (out/'dataset_statistics.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        wr=csv.DictWriter(fh,fieldnames=columns);wr.writeheader()
        for name,stat in output['split_stats'].items():
            aud = stat.get('source_audio_stats') or {}
            sample = stat['patch_sample_stats']
            wr.writerow(dict(split=name,patch_count=stat['patch_count'],
                npy_file_size_GiB=stat['npy_file_size_GiB'],
                utterance_count=aud.get('utterance_count'),speaker_count=aud.get('speaker_count'),
                duration_hours=aud.get('duration_hours'),duration_mean_sec=aud.get('duration_mean_sec'),
                duration_median_sec=aud.get('duration_median_sec'),mel_frames=aud.get('total_logmel_frames'),
                sampled_patches=sample['n_patches_sampled'],
                sample_mean_normalized=sample.get('mean_normalized_value_sample'),
                sample_std_normalized=sample.get('std_normalized_value_sample'),
                sample_padding_zero_fraction=sample.get('fraction_exact_zeros_sample')))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        splits_list = ['train','val','test']
        vals = [output['split_stats'][s]['patch_count'] for s in splits_list]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.bar(splits_list,vals)
        ax.set(ylabel='Patches (96x96)',title='V0.1 fixed dataset split sizes')
        for i,v in enumerate(vals):ax.text(i,v,f'{v:,}',ha='center',va='bottom',fontsize=9)
        fig.tight_layout();fig.savefig(out/'dataset_split_counts.png',dpi=150);plt.close(fig)
    except ImportError:
        print('[WARN] matplotlib unavailable: skipping optional split chart')
    print('[DONE] Read-only dataset statistics:',out,flush=True)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',default='Data/splits')
    p.add_argument('--cache',default=None,help='Optional override; auto-detect from splits/manifest.json')
    p.add_argument('--out',default='reports/dataset_stats')
    p.add_argument('--sample-patches',type=int,default=512,
                   help='Per-split random patch sample size for distribution stats; 0 disables')
    p.add_argument('--seed',type=int,default=42)
    main(p.parse_args())
