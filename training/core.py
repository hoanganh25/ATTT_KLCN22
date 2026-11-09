"""One epoch: unchanged V0.1 network and MSE; optional per-batch SNR policy."""
import torch

SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-3


def run_epoch(loader, encoder, decoder, channel, criterion, device, optimizer=None,
              scaler=None, max_batches=None, snr_sampler=None, snr_db_override=None):
    """Fixed SNR by default; sample SNR per batch only when snr_sampler is set.

    snr_db_override is for validation at a specific SNR. Only AWGN/Rayleigh
    are passed here, both supporting ``channel(z, snr_db=value)``.
    """
    if snr_sampler is not None and snr_db_override is not None:
        raise ValueError('Choose either snr_sampler or snr_db_override')
    training = optimizer is not None
    encoder.train(training)
    decoder.train(training)
    amp = device.type == 'cuda'
    error_sum = 0.
    elements = 0
    with torch.set_grad_enabled(training):
        for idx, mel in enumerate(loader):
            if max_batches is not None and idx >= max_batches:
                break
            mel = mel.to(device, non_blocking=amp)
            if training:
                optimizer.zero_grad(set_to_none=True)
            # Only this decision is new: it does not change Encoder/Decoder or loss.
            db = snr_sampler.sample() if snr_sampler is not None else snr_db_override
            with torch.autocast(device_type=device.type, enabled=amp):
                latent = encoder(mel)
                received = channel(latent) if db is None else channel(latent, snr_db=db)
                recovered = decoder(received)
                if recovered.shape != mel.shape:
                    raise ValueError(f'Decoder output {recovered.shape} != target {mel.shape}')
                loss = criterion(recovered, mel)
            if training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            error_sum += float(loss.detach()) * mel.numel()
            elements += mel.numel()
    if not elements:
        raise ValueError('No batches processed')
    return error_sum / elements
