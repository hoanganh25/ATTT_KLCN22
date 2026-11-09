"""Single optional energy-mask metric family (Precision/Recall/F1) on log-Mel cells.
NOTE: not ASR accuracy or attack detection; normalized threshold=0 includes padding.
"""
import torch


@torch.inference_mode()
def evaluate_energy_mask(loader,encoder,decoder,channel,device,max_batches=0):
    tp=fp=fn=tn=0
    for i,x in enumerate(loader):
        if max_batches and i>=max_batches:break
        x=x.to(device)
        pred=decoder(channel(encoder(x)))
        gt=x>=0;hat=pred>=0
        tp+=int((gt&hat).sum());tn+=int((~gt&~hat).sum())
        fp+=int((~gt&hat).sum());fn+=int((gt&~hat).sum())
    precision=tp/max(1,tp+fp)
    recall=tp/max(1,tp+fn)
    return {'energy_mask_precision':precision,'energy_mask_recall':recall,
            'energy_mask_f1':2*precision*recall/max(1e-12,precision+recall),
            'energy_mask_accuracy':(tp+tn)/max(1,tp+tn+fp+fn)}
