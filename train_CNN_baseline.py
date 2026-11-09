"""Train original V0.1 Encoder/Decoder from scratch, on AWGN or Rayleigh.

Strategies:
  fixed: one nominal SNR for all training batches (legacy behavior).
  multi: independently sample nominal SNR in [--snr-min, --snr-max] per batch.

No mixed channels, robust/burst/impulse, or previous checkpoint loading.
"""
import argparse
import gc
import time
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn

from model.CNN_Baseline.Encoder import Encoder
from model.CNN_Baseline.Decode import Decoder
from Data.Dataset import load_saved_loaders
from channels.factory import make_channel
from training.core import run_epoch
from training.multi_snr import MultiSNRSampler, parse_snr_grid



def probe_hardware(args):
    """Test REAL CNN+channel+Adam training steps on synthetic [B,1,96,96].

    This does NOT load data, write checkpoints, or update any existing run.
    Memory estimates are valid only for the current machine/GPU workload.
    """
    if args.probe_batch_sizes:
        try:
            batches = [int(s.strip()) for s in args.probe_batch_sizes.split(',')]
        except ValueError as exc:
            raise ValueError('--probe-batch-sizes must be comma-separated positive integers') from exc
    else:
        batches = [args.batch_size]
    if not batches or any(b < 1 for b in batches) or len(set(batches)) != len(batches):
        raise ValueError('Probe batch sizes must be unique positive integers')
    if args.probe_steps < 1:
        raise ValueError('--probe-steps must be positive')
    if args.lr <= 0 or not math.isfinite(args.lr) or not math.isfinite(args.snr):
        raise ValueError('Probe learning rate must be positive and finite; SNR finite')

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        print(f'[GPU] {props.name} total_vram={props.total_memory/1024**3:.3f} GiB', flush=True)
    else:
        print('[WARNING] CUDA unavailable/disabled: CPU probe cannot establish VRAM fit.', flush=True)

    safe = []
    for b in batches:
        enc = dec = channel = opt = scaler = x = latent = output = loss = None
        if device.type == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()
            free_start, total = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            free_start = total = None
        try:
            torch.manual_seed(args.seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(args.seed)
            enc, dec = Encoder().to(device), Decoder().to(device)
            channel = make_channel(args.channel, args.snr).to(device)
            opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)
            scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
            enc.train(); dec.train()
            start = time.perf_counter()
            for step in range(args.probe_steps):
                x = torch.randn(b, 1, 96, 96, device=device)
                opt.zero_grad(set_to_none=True)
                db = args.snr if args.train_strategy == 'fixed' else args.snr_min
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    latent = enc(x)
                    output = dec(channel(latent, snr_db=db))
                    if output.shape != x.shape:
                        raise RuntimeError(f'Output shape mismatch: {output.shape} vs {x.shape}')
                    loss = nn.functional.mse_loss(output, x)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                # Release references to each step's computation graph.
                x = latent = output = loss = None
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            if device.type == 'cuda':
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                # This is a deliberately conservative heuristic, NOT a guarantee.
                tentative = reserved <= 0.80 * free_start
                tag = 'PASS (headroom heuristic)' if tentative else 'PASS, LOW HEADROOM'
                print(f'[PROBE {tag}] batch_size={b} steps={args.probe_steps} '
                      f'peak_alloc={allocated/1024**3:.3f}GiB '
                      f'peak_reserved={reserved/1024**3:.3f}GiB '
                      f'free_before={free_start/1024**3:.3f}GiB '
                      f'time={elapsed:.2f}s', flush=True)
                if tentative:
                    safe.append(b)
            else:
                print(f'[PROBE CPU PASS] batch_size={b} steps={args.probe_steps} '
                      f'time={elapsed:.2f}s (NOT a GPU memory test)', flush=True)
        except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
            print(f'[PROBE OOM] batch_size={b}: {type(exc).__name__}', flush=True)
            # A failed CUDA allocation can contaminate subsequent measurements;
            # stop after the first OOM, then retry in a fresh process if needed.
            break
        finally:
            del enc, dec, channel, opt, scaler, x, latent, output, loss
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    if device.type == 'cuda':
        if safe:
            print(f'[SUGGESTED] Highest tested batch with >=20% free-memory '
                  f'headroom (heuristic): {max(safe)}', flush=True)
        else:
            print('[SUGGESTED] No batch passed the 20% headroom check; '
                  'try smaller batches or close GPU applications.', flush=True)
    print('[NOTE] Probe tests synthetic training, not data I/O or long-run stability; '
          'keep an additional memory margin. No checkpoint was written.', flush=True)


def _validated_values(args):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError('epochs and batch-size must be positive')
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError('--max-train-batches must be positive')
    if args.max_val_batches is not None and args.max_val_batches < 1:
        raise ValueError('--max-val-batches must be positive')
    if args.lr <= 0 or not math.isfinite(args.lr) or not math.isfinite(args.snr):
        raise ValueError('Learning rate and --snr must be finite; LR > 0')
    grid = parse_snr_grid(args.snr_grid)
    if args.train_strategy == 'multi':
        # Validate before creating any run artifacts.
        MultiSNRSampler(min_db=args.snr_min, max_db=args.snr_max,
                        seed=args.seed, mode=args.snr_sampling, grid=grid)
    if args.val_snrs:
        val_snrs = parse_snr_grid(args.val_snrs)
    elif args.train_strategy == 'multi':
        # Mean over the validation SNR grid: checkpoint chosen for broad-range
        # performance, NOT test-set tuning. Expensive but appropriate for multi.
        val_snrs = grid
    else:
        val_snrs = (float(args.snr),)
    return grid, val_snrs


def _validate_at_snrs(val_dl, enc, dec, channel, loss, device, args, snrs):
    """Compare epochs fairly using exactly the same channel RNG realization."""
    values = {}
    for i, snr in enumerate(snrs):
        # fork_rng restores torch RNG state so validation does not influence
        # next epoch's training noise or weights. CUDA is included if enabled.
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.seed + 100000 + i * 1009)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(args.seed + 100000 + i * 1009)
            values[float(snr)] = run_epoch(
                val_dl, enc, dec, channel, loss, device,
                max_batches=args.max_val_batches, snr_db_override=float(snr))
    return sum(values.values()) / len(values), values


def train(args):
    # Compatible with run_pipeline.py and standalone CLI.
    grid, val_snrs = _validated_values(args)
    split_dir = Path(args.splits).resolve()
    manifest_file = split_dir / 'manifest.json'
    if not manifest_file.exists():
        raise FileNotFoundError(f'Missing prepared split: {manifest_file}')
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    if manifest.get('shape') != [96, 96]:
        raise ValueError('Only V0.1 [96,96] log-Mel splits supported')

    default_run = (f'runs/{args.channel}' if args.train_strategy == 'fixed'
                   else f'runs/multi_snr/{args.channel}')
    run_dir = Path(args.run or default_run).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f'Run directory is not empty: {run_dir}. Choose a new --run (never overwrite).')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    train_dl, val_dl, _ = load_saved_loaders(
        split_dir=split_dir, batch_size=args.batch_size, seed=args.seed,
        num_workers=0, pin_memory=(device.type == 'cuda'))

    enc = Encoder().to(device)
    dec = Decoder().to(device)
    channel = make_channel(args.channel, args.snr).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    criterion = nn.MSELoss()
    signature = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    config = dict(
        channel=args.channel, train_strategy=args.train_strategy,
        train_fixed_snr_db=float(args.snr) if args.train_strategy == 'fixed' else None,
        train_snr_min_db=float(args.snr_min) if args.train_strategy == 'multi' else None,
        train_snr_max_db=float(args.snr_max) if args.train_strategy == 'multi' else None,
        train_snr_sampling=args.snr_sampling if args.train_strategy == 'multi' else None,
        train_snr_grid_db=list(grid) if args.train_strategy == 'multi' and args.snr_sampling == 'grid' else None,
        snr_sampling_unit='one SNR per training batch, independent channel noise per sample',
        validation_snr_db=list(val_snrs), validation_selection='arithmetic mean of dev normalized MSE across validation SNRs',
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed,
        feature='logmel', dataset_manifest_sha256=signature,
        architecture='V0.1 original CNN (model/CNN_Baseline/Encoder.py, model/CNN_Baseline/Decode.py)',
        limited_run=(args.max_train_batches is not None or args.max_val_batches is not None),
        train_patches=len(train_dl.dataset), dev_patches=len(val_dl.dataset),
        train_batches_per_epoch=min(len(train_dl), args.max_train_batches) if args.max_train_batches is not None else len(train_dl),
        dev_batches_per_snr=min(len(val_dl), args.max_val_batches) if args.max_val_batches is not None else len(val_dl),
        rayleigh_assumptions='complex I/Q pairing, block flat fading, perfect CSI, MMSE equalizer',
        python=platform.python_version(), torch=torch.__version__, device=str(device))

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    history = []
    best = float('inf')
    print(f'[DATA] TRAIN={len(train_dl.dataset):,} patches | batches/epoch='
          f'{min(len(train_dl), args.max_train_batches) if args.max_train_batches is not None else len(train_dl):,} | '
          f'DEV={len(val_dl.dataset):,} patches | batches/SNR='
          f'{min(len(val_dl), args.max_val_batches) if args.max_val_batches is not None else len(val_dl):,}', flush=True)
    print(f'[TRAIN] channel={args.channel} strategy={args.train_strategy} '
          f'epochs={args.epochs} device={device} validation_snrs={val_snrs}', flush=True)
    for epoch in range(1, args.epochs + 1):
        sampler = None
        if args.train_strategy == 'multi':
            sampler = MultiSNRSampler(min_db=args.snr_min, max_db=args.snr_max,
                                      mode=args.snr_sampling, grid=grid,
                                      seed=args.seed + epoch * 1009)
        train_mse = run_epoch(train_dl, enc, dec, channel, criterion, device,
                              optimizer=opt, scaler=scaler,
                              max_batches=args.max_train_batches, snr_sampler=sampler)
        snr_stats = (sampler.summary() if sampler is not None else dict(
            train_snr_batches=len(train_dl) if args.max_train_batches is None else min(len(train_dl), args.max_train_batches),
            train_snr_mean_db=float(args.snr), train_snr_min_db=float(args.snr), train_snr_max_db=float(args.snr)))
        val_mse, val_per_snr = _validate_at_snrs(
            val_dl, enc, dec, channel, criterion, device, args, val_snrs)
        row = dict(epoch=epoch, train_normalized_mse=train_mse,
                   val_normalized_mse=val_mse, **snr_stats)
        row.update({f'val_mse_{snr:g}db':val_per_snr.get(float(snr), '') for snr in val_snrs})
        history.append(row)
        print(f'[TRAIN] {args.channel} {args.train_strategy} {epoch}/{args.epochs} '
              f'train={train_mse:.7f} val_mean={val_mse:.7f} '
              f'train_snr_mean={snr_stats["train_snr_mean_db"]:.2f} dB', flush=True)
        if val_mse < best:
            best = val_mse
            torch.save(dict(
                encoder=enc.state_dict(), decoder=dec.state_dict(), epoch=epoch,
                val_mse=val_mse, val_mse_by_snr=val_per_snr,
                channel=args.channel, snr_db=float(args.snr) if args.train_strategy=='fixed' else None,
                train_strategy=args.train_strategy,
                train_snr_range_db=[args.snr_min, args.snr_max] if args.train_strategy=='multi' else None,
                train_snr_sampling=args.snr_sampling if args.train_strategy=='multi' else None,
                validation_snr_db=list(val_snrs), train_snr_summary=snr_stats,
                seed=args.seed, feature='logmel', data_manifest_sha256=signature,
                rayleigh_csi='perfect-MMSE' if args.channel=='rayleigh' else None),
                run_dir / 'best_model.pth')
        # Persist history every epoch so interruption does not discard completed epochs.
        with (run_dir / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    return run_dir / 'best_model.pth'


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits', default='Data/splits')
    p.add_argument('--channel', choices=['awgn','rayleigh'], required=True)
    p.add_argument('--run', default=None)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--snr', type=float, default=10., help='Fixed training SNR; also factory default for channel')
    p.add_argument('--train-strategy', choices=['fixed','multi'], default='fixed')
    p.add_argument('--snr-min', type=float, default=0.)
    p.add_argument('--snr-max', type=float, default=20.)
    p.add_argument('--snr-sampling', choices=['uniform','grid'], default='uniform')
    p.add_argument('--snr-grid', default='0,5,10,15,20')
    p.add_argument('--val-snrs', default=None,
                   help='Optional comma-separated validation grid; default fixed=--snr, multi=--snr-grid')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--probe', action='store_true',
                   help='Test synthetic forward/backward/Adam without loading data or writing checkpoints')
    p.add_argument('--probe-batch-sizes', default=None,
                   help='Comma-separated batch sizes for VRAM probe, e.g. 16,32,64,128')
    p.add_argument('--probe-steps', type=int, default=2,
                   help='Training steps per candidate batch size during probe (default 2)')
    p.add_argument('--max-train-batches', type=int, default=None, help='Smoke tests ONLY')
    p.add_argument('--max-val-batches', type=int, default=None, help='Smoke tests ONLY')
    return p.parse_args()


if __name__ == '__main__':
    args = cli()
    if args.probe:
        probe_hardware(args)
    else:
        train(args)
