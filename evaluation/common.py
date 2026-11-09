"""Shared TEST-only data/model plumbing; never updates checkpoint weights."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from model.CNN_Baseline.Encoder import Encoder
from model.CNN_Baseline.Decode import Decoder


def load_model(checkpoint, split_dir, device):
    p=Path(checkpoint)
    obj=torch.load(p,map_location='cpu',weights_only=True)
    manifest=Path(split_dir)/'manifest.json'
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    sig=obj.get('data_manifest_sha256')
    if sig and sig!=digest:
        raise RuntimeError(f'Dataset/checkpoint SHA mismatch: {p}')
    if obj.get('feature','logmel')!='logmel':
        raise ValueError('The main evaluator expects log-Mel checkpoint, not MFCC')
    enc=Encoder().to(device).eval();dec=Decoder().to(device).eval()
    enc.load_state_dict(obj['encoder']);dec.load_state_dict(obj['decoder'])
    return enc,dec,obj


def read_metadata(splits):
    folder=Path(splits)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    norm=json.loads((folder/'normalization.json').read_text(encoding='utf-8'))
    return manifest,norm


def set_seed(seed,device):
    torch.manual_seed(int(seed))
    if device.type=='cuda':torch.cuda.manual_seed_all(int(seed))


def as_device(cpu=False):
    return torch.device('cpu' if cpu or not torch.cuda.is_available() else 'cuda')


def write_csv(path,rows):
    import csv
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def choose_utterances(rows,maximum,seed):
    if maximum < 0:raise ValueError('max-utterances must be nonnegative')
    if not maximum or maximum>=len(rows):return rows
    rng=np.random.default_rng(seed)
    indices=sorted(rng.choice(len(rows),size=maximum,replace=False).tolist())
    return [rows[i] for i in indices]


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]
