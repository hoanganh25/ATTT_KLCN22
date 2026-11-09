"""PESQ-WB ONLY: original vs reconstructed mono 16kHz waveform."""
import numpy as np


def score_pesq(reference, estimate, sr=16000):
    from pesq import pesq
    if sr!=16000:raise ValueError('PESQ-WB requires 16 kHz')
    a=np.asarray(reference,dtype=np.float32).ravel()
    b=np.asarray(estimate,dtype=np.float32).ravel()
    if a.shape!=b.shape or len(a)<sr//2:
        raise ValueError('PESQ requires aligned equal-length waveforms >=0.5 s')
    return float(pesq(sr,a,b,'wb'))
