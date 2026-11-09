"""Visual SSIM ONLY, z-score clipped to [-3,3] on fixed TEST patch subset."""
import numpy as np
import torch


@torch.inference_mode()
def evaluate_ssim(loader,encoder,decoder,channel,device,max_patches=40):
    from skimage.metrics import structural_similarity
    if max_patches<1:raise ValueError('ssim_samples must be >= 1')
    values=[]
    for batch in loader:
        x=batch.to(device)
        y=decoder(channel(encoder(x)))
        for a,b in zip(x[:,0].detach().cpu().numpy(),y[:,0].detach().cpu().numpy()):
            values.append(float(structural_similarity(np.clip(a,-3,3),np.clip(b,-3,3),data_range=6)))
            if len(values)>=max_patches:break
        if len(values)>=max_patches:break
    return {'ssim_clipped_zscore':float(np.mean(values)) if values else None,'ssim_num_patches':len(values)}
