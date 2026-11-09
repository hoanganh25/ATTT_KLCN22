"""MSE ONLY: normalized/raw log-Mel MSE over fixed full TEST patch split."""
import torch


@torch.inference_mode()
def evaluate_mse(test_loader, encoder, decoder, channel, std, device, max_batches=0):
    norm_sse=raw_sse=0.0
    num_cells=0
    for i, batch in enumerate(test_loader):
        if max_batches and i>=max_batches:break
        x=batch.to(device)
        pred=decoder(channel(encoder(x)))
        if pred.shape!=x.shape:raise RuntimeError('Model output mismatches target dimensions')
        err=pred.float()-x.float()
        norm_sse+=float(err.square().double().sum().item())
        raw_sse+=float((err*std).square().double().sum().item())
        num_cells+=err.numel()
    if num_cells<=0:raise RuntimeError('Empty test loader')
    return {'norm_mse':norm_sse/num_cells,'raw_logmel_mse':raw_sse/num_cells,'num_cells':num_cells}
