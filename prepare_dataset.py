"""ONE-TIME WAV/FLAC -> log-Mel cache -> fixed [N,96,96] V0.1 NPY splits.

Creates Data/splits/{train,val,test}.npy compatible with original Data.Dataset,
Encoder, Decode, Train, Evaluate and Evaluate_advanced_V01 unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from extract_cache import run as extract_run, load_jsonl

PATCH = 96


def positions(T: int) -> list[int]:
    """Cover every frame; avoid heavily padded tail for T>=PATCH."""
    if T <= PATCH:
        return [0]
    offsets = list(range(0, T - PATCH + 1, PATCH))
    if offsets[-1] != T - PATCH:
        offsets.append(T - PATCH)
    return offsets


def patch_index(rows):
    return [(r, at) for r in rows for at in positions(int(r['frames']))]


def prepare(dataset: str, cache: str='workspace/prepared', out: str='Data/splits',
            max_files=None, split_unit='speaker', seed=42, rebuild=False,
            train_frac=.8, dev_frac=.1):
    root=Path(dataset).expanduser().resolve()
    cache=Path(cache).expanduser().resolve()
    out=Path(out).expanduser().resolve()
    manifest=out/'manifest.json'
    if manifest.exists() and not rebuild:
        info=json.loads(manifest.read_text(encoding='utf-8'))
        if info.get('source_dataset_root') != str(root):
            raise ValueError(f'Splits at {out} belong to a DIFFERENT dataset. Use another --out/--work folder.')
        if (info.get('max_files')!=max_files or info.get('split_unit')!=split_unit or
            info.get('seed')!=seed or info.get('train_frac')!=train_frac or info.get('dev_frac')!=dev_frac):
            raise ValueError('Existing prepared split uses different settings. Choose a new output folder.')
        for split in ('train','val','test'):
            arr=np.load(out/f'{split}.npy', mmap_mode='r', allow_pickle=False)
            if arr.shape != (info['sizes'][split],PATCH,PATCH):
                raise ValueError(f'{split}.npy does not match manifest')
        print(f'[CACHE] Fixed split files already exist: {out}. Nothing extracted or resplit.')
        return info
    if out.exists():
        raise RuntimeError(f'{out} already exists without a complete manifest (or --rebuild used). Use a new --out path; do not overwrite a prepared experiment.')
    args=argparse.Namespace(dataset=str(root),out=str(cache),max_files=max_files,seed=seed,
                            split_unit=split_unit,train_frac=train_frac,dev_frac=dev_frac,
                            skip_errors=False,progress_every=50)
    extract_run(args)
    norm=json.loads((cache/'normalization.json').read_text(encoding='utf-8'))
    mean=np.asarray(norm['mean'],dtype=np.float32)[:,None]
    std=np.asarray(norm['std'],dtype=np.float32)[:,None]
    if mean.shape != (PATCH,1) or std.shape != (PATCH,1):
        raise RuntimeError(f'Frontend must output native {PATCH} Mel bins to preserve the V0.1 network, got {mean.shape[0]}')
    rows_by_split={n:load_jsonl(cache/f'{name}.jsonl') for n,name in
                   (('train','train'),('val','dev'),('test','test'))}
    if any(not values for values in rows_by_split.values()):
        raise RuntimeError('All train/dev/test splits must be nonempty')
    index_by_split={k:patch_index(v) for k,v in rows_by_split.items()}
    staging=out.with_name(out.name+'_building')
    if staging.exists():
        raise RuntimeError(f'Incomplete preparation exists at {staging}; inspect it before retrying')
    staging.mkdir(parents=True)
    try:
        for split, index in index_by_split.items():
            mm=np.lib.format.open_memmap(staging/f'{split}.npy','w+',dtype=np.float32,
                                        shape=(len(index),PATCH,PATCH))
            previous=None
            raw=None
            for i,(row,start) in enumerate(index):
                if row['cache'] != previous:
                    previous=row['cache']
                    raw=np.load(cache/row['cache'],mmap_mode='r',allow_pickle=False)
                    if raw.ndim != 2 or raw.shape[0] != PATCH:
                        raise ValueError(f'Bad Mel shape: {row["relpath"]} {raw.shape}')
                sample=np.asarray(raw[:,start:start+PATCH],dtype=np.float32)
                mm[i].fill(0.0) # normalized mean for any trailing padded frames
                mm[i,:,:sample.shape[1]] = (sample-mean)/std
                if not np.isfinite(mm[i]).all():
                    raise RuntimeError('NaN or Inf in prepared Mel')
                if (i+1)%2000 == 0: print(f'[{split}] written {i+1}/{len(index)} patches',flush=True)
            mm.flush()
            del mm,raw
            print(f'[{split}] saved {len(index):,} patches of 96x96')
        shutil.copy2(cache/'normalization.json',staging/'normalization.json')
        shutil.copy2(cache/'summary.json',staging/'source_summary.json')
        info={
            'shape':[PATCH,PATCH], 'dtype':'float32',
            'sizes':{k:len(v) for k,v in index_by_split.items()},
            'num_samples':sum(map(len,index_by_split.values())),
            'seed':seed,'max_files':max_files,'split_unit':split_unit,'train_frac':train_frac,'dev_frac':dev_frac,
            'source_dataset_root':str(root),
            'source_cache_dir':str(cache),'split_strategy':json.loads((cache/'summary.json').read_text())['split_method'],
            'dataset':'LibriSpeech WAV/FLAC audio-derived numerical log-Mel',
            'dsp':json.loads((cache/'summary.json').read_text())['dsp'],
            'normalization':'per-Mel-bin z-score fitted from TRAIN ONLY',
            'patching':'96 time frames per example; trailing coverage from last 96 frames, short utterances zero-padded after normalization',
            'note':'V0.1 Encoder/Decoder kept byte-for-byte unchanged. Trained from scratch; not compatible with legacy COALA checkpoint.'
        }
        (staging/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        staging.rename(out)
        return info
    except Exception:
        shutil.rmtree(staging,ignore_errors=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,help='Root directory holding .wav and/or LibriSpeech .flac')
    parser.add_argument('--cache',default='workspace/prepared',help='Persistent 96xT log-Mel cache')
    parser.add_argument('--out',default='Data/splits',help='Fixed 96x96 NPY train/val/test files')
    parser.add_argument('--max-files',type=int,default=None)
    parser.add_argument('--split-unit',choices=('speaker','file'),default='speaker')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-frac',type=float,default=.8)
    parser.add_argument('--dev-frac',type=float,default=.1)
    args=parser.parse_args()
    result=prepare(args.dataset,args.cache,args.out,args.max_files,args.split_unit,args.seed,
                   train_frac=args.train_frac,dev_frac=args.dev_frac)
    print('Dataset READY:',result['sizes'])


if __name__=='__main__':main()
