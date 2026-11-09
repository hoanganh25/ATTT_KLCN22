"""Isolated DeepSC-S SE-ResNet log-Mel training. Does not touch train.py / V0.1.

Each run is a new CNN pair, trained end-to-end over the exact V0.1 channel.
Train ALL patches by default (250,194); validate on official DEV, never TEST.

Examples in README_SE_RESNET_VI.md. First run --probe from the project root.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn

# The user-facing folder is literally "model/SE-ResNet".  A hyphen cannot
# appear in normal Python `from ... import ...` syntax; importlib supports it.
from importlib import import_module
_se_encoder = import_module("model.SE-ResNet.Encoder")
_se_decoder = import_module("model.SE-ResNet.Decode")
SEConfig = _se_encoder.SEConfig

def build_pair(config=None):
    cfg = config or SEConfig()
    return _se_encoder.Encoder(cfg), _se_decoder.Decoder(cfg)

def count_parameters(enc, dec):
    e = sum(p.numel() for p in enc.parameters())
    d = sum(p.numel() for p in dec.parameters())
    return {'encoder':e, 'decoder':d, 'total':e+d}
from channels.factory import make_channel
from training.multi_snr import MultiSNRSampler, parse_snr_grid


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='Data/splits')
    p.add_argument('--channel', choices=['awgn','rayleigh'], default='awgn')
    p.add_argument('--profile', choices=['repo','paper'], default='repo',
                   help='repo=official TF source structure; paper=smaller published description')
    p.add_argument('--latent-channels', type=int, default=8,
                   help='8 preserves V0.1 transmitted latent [B,8,24,24]')
    p.add_argument('--blocks', type=int, default=0, help='0=profile default (repo=5, paper=6)')
    p.add_argument('--checkpoint-blocks', action='store_true',
                   help='EXPERIMENTAL activation checkpoint; BN running stats may differ')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2, help='Microbatch; try 2 on RTX 3050 4GB')
    p.add_argument('--accum-steps', type=int, default=8,
                   help='Effective batch= batch_size * accum_steps (except last batch, BN microbatch)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--snr-min', type=float, default=0.)
    p.add_argument('--snr-max', type=float, default=20.)
    p.add_argument('--snr-sampling', choices=['uniform','grid'], default='uniform')
    p.add_argument('--snr-grid', default='0,5,10,15,20')
    p.add_argument('--val-snrs', default='0,5,10,15,20')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--out', default='runs/se_resnet_deepscs/awgn')
    p.add_argument('--resume', default=None, help='Path to last_training.pth (resume at epoch boundary)')
    p.add_argument('--probe', action='store_true', help='ONE synthetic optimizer step, no data/train artifacts')
    p.add_argument('--max-train-batches', type=int, default=0, help='SMOKE ONLY; 0=all TRAIN patches')
    p.add_argument('--max-val-batches', type=int, default=0, help='SMOKE ONLY; 0=all DEV patches per SNR')
    return p.parse_args()


def config_from_args(a):
    return SEConfig(profile=a.profile, latent_channels=a.latent_channels,
                    blocks=a.blocks, checkpoint_blocks=a.checkpoint_blocks)


def device_of(a):
    if not a.cpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def create_optimizer(enc, dec, a):
    return torch.optim.Adam([*enc.parameters(),*dec.parameters()], lr=a.lr)


def get_scaler(device):
    return torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))


def channel_forward(channel, z, snr):
    return channel(z, snr_db=snr)


def probe(a, enc, dec, device):
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f'[GPU] {props.name}; VRAM={props.total_memory / 1024**3:.2f} GiB')
        torch.cuda.reset_peak_memory_stats(device)
    else:
        print('[WARN] No CUDA active. This ONLY tests CPU; cannot prove GPU fit.')
    channel=make_channel(a.channel,10.0).to(device)
    opt=create_optimizer(enc,dec,a)
    scaler=get_scaler(device)
    # One real forward+backward+optimizer step, includes Adam allocation & AMP.
    x=torch.randn(a.batch_size,1,96,96,device=device)
    enc.train();dec.train()
    try:
        start=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            z=enc(x);output=dec(channel_forward(channel,z,10.0))
            loss=torch.nn.functional.mse_loss(output,x)
        scaler.scale(loss).backward()
        scaler.step(opt);scaler.update()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        print(f'[PROBE OK] B={a.batch_size} input={tuple(x.shape)} '
              f'latent={tuple(z.shape)} output={tuple(output.shape)} '
              f'loss={float(loss.detach()):.5f} elapsed={elapsed:.3f}s')
        if device.type=='cuda':
            print(f'[GPU] max_allocated={torch.cuda.max_memory_allocated() / 1024**3:.3f}GiB '
                  f'max_reserved={torch.cuda.max_memory_reserved() / 1024**3:.3f}GiB')
        print('[NOTE] One synthetic step is a memory feasibility test, NOT a speed benchmark or training.')
        return True
    except torch.cuda.OutOfMemoryError:
        print('[GPU OOM] Try --batch-size 1, keep --profile repo. Then consider paper profile.')
        return False


def validate(val_loader, enc, dec, channel, device, snrs, seed, max_batches):
    enc.eval(); dec.eval()
    result={}
    for i,snr in enumerate(snrs):
        devices=[torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed+100000+i*1009)
            if device.type=='cuda':torch.cuda.manual_seed_all(seed+100000+i*1009)
            sum_sq=0.;cells=0
            with torch.inference_mode():
                for j,x in enumerate(val_loader):
                    if max_batches and j>=max_batches:break
                    x=x.to(device,non_blocking=device.type=='cuda')
                    with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                        y=dec(channel_forward(channel,enc(x),snr))
                    if y.shape!=x.shape:raise RuntimeError(f'Bad shape {y.shape}')
                    delta=y.float()-x.float()
                    sum_sq+=float(torch.square(delta).sum())
                    cells+=x.numel()
            if not cells:raise RuntimeError('DEV empty')
            result[float(snr)]=sum_sq/cells
    return sum(result.values())/len(result),result


def train_epoch(loader,enc,dec,channel,optimizer,scaler,device,a,epoch,grid):
    enc.train();dec.train()
    snr_sample=MultiSNRSampler(min_db=a.snr_min,max_db=a.snr_max,
        mode=a.snr_sampling,grid=grid,seed=a.seed+epoch*1009)
    opt_steps=0;total_sq=0.;total_cells=0;micro=0
    optimizer.zero_grad(set_to_none=True)
    max_batches=(a.max_train_batches or len(loader))
    n_batches=min(len(loader),max_batches)
    for i,x in enumerate(loader):
        if i>=n_batches:break
        x=x.to(device,non_blocking=device.type=='cuda')
        db=snr_sample.sample()
        # Compensate the possibly shorter final accumulation group.
        group_size=min(a.accum_steps, n_batches-(i//a.accum_steps)*a.accum_steps)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            y=dec(channel_forward(channel,enc(x),db))
            if y.shape!=x.shape:raise RuntimeError(f'Unexpected output {tuple(y.shape)}')
            loss=torch.nn.functional.mse_loss(y.float(),x.float())
        scaler.scale(loss/group_size).backward()
        total_sq+=float(loss.detach())*x.numel()
        total_cells+=x.numel()
        micro+=1
        if (i+1)%a.accum_steps==0 or i==n_batches-1:
            scaler.step(optimizer);scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_steps+=1
        if (i+1)%250==0:
            print(f'[TRAIN] epoch={epoch} micro={i+1}/{n_batches} '
                  f'opt_steps={opt_steps} batch_MSE={float(loss):.6f} '
                  f'SNR={db:.2f}',flush=True)
    if not micro:raise RuntimeError('No TRAIN batches')
    return total_sq/total_cells,opt_steps,snr_sample.summary(),micro


def save_atomic(state,path):
    tmp=Path(str(path)+'.tmp')
    torch.save(state,tmp)
    os.replace(tmp,path)


def write_history(path,history):
    tmp=Path(str(path)+'.tmp')
    keys=list(history[0])
    with tmp.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(history)
    os.replace(tmp,path)


def main(a):
    if a.epochs<1 or a.batch_size<1 or a.accum_steps<1 or a.lr<=0 or a.threads<1:
        raise ValueError('epochs,batch-size,accum-steps,threads,lr must be positive')
    if a.max_train_batches<0 or a.max_val_batches<0 or a.num_workers<0:
        raise ValueError('batch limits/workers must be nonnegative')
    grid=parse_snr_grid(a.snr_grid);val_snrs=parse_snr_grid(a.val_snrs)
    torch.set_num_threads(a.threads)
    device=device_of(a)
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    if device.type=='cuda':torch.cuda.manual_seed_all(a.seed)
    cfg=config_from_args(a)
    enc,dec=build_pair(cfg)
    enc.to(device);dec.to(device)
    counts=count_parameters(enc,dec)
    print(f'[MODEL] profile={a.profile} blocks={cfg.resolve()[3]} '
          f'latent={a.latent_channels} parameters={counts} device={device}',flush=True)
    if a.probe:
        if not probe(a,enc,dec,device):raise SystemExit(2)
        return
    splits=Path(a.splits).expanduser().resolve()
    manifest=splits/'manifest.json'
    if not manifest.is_file():raise FileNotFoundError(manifest)
    metadata=json.loads(manifest.read_text(encoding='utf-8'))
    if metadata.get('shape')!=[96,96]:raise RuntimeError('Requires original V0.1 [96,96] splits')
    signature=hashlib.sha256(manifest.read_bytes()).hexdigest()
    out=Path(a.out).expanduser().resolve()
    if a.resume:
        if not out.is_dir() or not (out/'config.json').exists():
            raise ValueError('Resume requires an existing --out with config.json')
    elif out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output nonempty: {out}. Use a new --out or --resume.')
    from Data.Dataset import load_saved_loaders
    train_dl,val_dl,_=load_saved_loaders(
        split_dir=str(splits),batch_size=a.batch_size,seed=a.seed,
        num_workers=a.num_workers,pin_memory=device.type=='cuda')
    # Recreate loader explicitly per epoch with seeded generator for exact
    # epoch-boundary resumption independent of earlier shuffling state.
    train_dataset=train_dl.dataset
    opt=create_optimizer(enc,dec,a)
    scaler=get_scaler(device)
    channel=make_channel(a.channel,10.0).to(device)
    config={**vars(a),'resume':None, 'architecture':'DeepSC-S-repo-SE-adapted-logMel',
        'model_config':as_dict(cfg),'model_parameters':counts,
        'source_repo':'https://github.com/Zhenzi-Weng/DeepSC-S',
        'source_discrepancy':'paper 6x(2-branch16,transition32) vs public repo 5x(4-branch128,transition128)',
        'train_strategy':'multi','train_snr_sampling_unit':'one SNR per microbatch',
        'val_metric':'mean normalized DEV MSE across fixed validation SNR grid',
        'latent_shape':[a.latent_channels,24,24],
        'manifest_sha256':signature,'torch':torch.__version__,
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'limited_run':bool(a.max_train_batches or a.max_val_batches),
        'train_examples_per_full_epoch':len(train_dataset),
        'notes':'No TEST data accessed. Full-epoch checkpoint; model is trained from scratch.'}
    if not a.resume:
        out.mkdir(parents=True,exist_ok=True)
        (out/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    else:
        old=json.loads((out/'config.json').read_text(encoding='utf-8'))
        core=('model_config','channel','batch_size','accum_steps','snr_min','snr_max',
              'snr_sampling','snr_grid','val_snrs','manifest_sha256','lr','seed',
              'max_train_batches','max_val_batches')
        if any(old.get(k)!=config.get(k) for k in core):
            raise RuntimeError('Resume configuration mismatch. See config.json.')
    best=float('inf');start_epoch=1;history=[]
    if a.resume:
        state=torch.load(a.resume,map_location='cpu',weights_only=False)
        if state['manifest_sha256']!=signature or state['model_config']!=as_dict(cfg):
            raise RuntimeError('Resume checkpoint wrong model/dataset')
        enc.load_state_dict(state['encoder'],strict=True)
        dec.load_state_dict(state['decoder'],strict=True)
        opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=int(state['epoch'])+1
        best=float(state['best_val_mse']);history=list(state['history'])
        print(f'[RESUME] Starting at epoch {start_epoch}; best dev={best:.6f}',flush=True)
    if start_epoch>a.epochs:
        print('[DONE] Epoch target already reached');return
    from torch.utils.data import DataLoader
    for epoch in range(start_epoch,a.epochs+1):
        # Fixed RNG/loader seed per epoch. Resume at next epoch only.
        torch.manual_seed(a.seed+epoch*117)
        if device.type=='cuda':torch.cuda.manual_seed_all(a.seed+epoch*117)
        generator=torch.Generator().manual_seed(a.seed+epoch*1237)
        epoch_loader=DataLoader(train_dataset,batch_size=a.batch_size,shuffle=True,
            generator=generator,num_workers=a.num_workers,pin_memory=device.type=='cuda',
            drop_last=False)
        t0=time.perf_counter()
        tr,opt_steps,ss,micro=train_epoch(epoch_loader,enc,dec,channel,opt,scaler,
                                        device,a,epoch,grid)
        va,by=validate(val_dl,enc,dec,channel,device,val_snrs,a.seed,a.max_val_batches)
        if device.type=='cuda':torch.cuda.synchronize()
        duration=time.perf_counter()-t0
        row={'epoch':epoch,'train_normalized_mse':tr,'val_normalized_mse':va,
             'duration_s':duration,'micro_steps':micro,'optimizer_steps':opt_steps,
             **ss,**{f'val_mse_{snr:g}db':by[snr] for snr in val_snrs}}
        history.append(row)
        state={'epoch':epoch,'encoder':enc.state_dict(),'decoder':dec.state_dict(),
               'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
               'history':history,'best_val_mse':min(best,va),
               'val_mse':va,'val_mse_by_snr':by,'model_config':as_dict(cfg),
               'architecture':'DeepSC-S-repo-SE-adapted-logMel',
               'manifest_sha256':signature,'channel':a.channel,'train_strategy':'multi',
               'train_snr_range_db':[a.snr_min,a.snr_max],
               'validation_snr_db':list(val_snrs),
               'limited_run':bool(a.max_train_batches or a.max_val_batches)}
        if va<best:
            best=va
            best_state={k:v for k,v in state.items()
                        if k not in ('optimizer','scaler','history')}
            save_atomic(best_state,out/'best_model.pth')
        save_atomic(state,out/'last_training.pth')
        write_history(out/'history.csv',history)
        print(f'[EPOCH] {epoch}/{a.epochs} TRAIN={tr:.7f} '
              f'DEV={va:.7f} best={best:.7f} '
              f'time={duration/60:.1f}min samples={len(train_dataset)} '
              f'micro_steps={micro} opt_steps={opt_steps}',flush=True)
    print('[DONE] best:',out/'best_model.pth',flush=True)


def as_dict(cfg):
    return {'profile':cfg.profile,'latent_channels':cfg.latent_channels,
            'blocks':cfg.blocks,'checkpoint_blocks':cfg.checkpoint_blocks}


if __name__=='__main__':
    main(cli())
