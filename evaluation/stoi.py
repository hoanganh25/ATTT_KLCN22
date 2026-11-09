"""STOI ONLY: input must be 16-kHz aligned reference and synthesized waveform."""
import numpy as np


def score_stoi(reference, estimate, sr=16000):
    from pystoi import stoi
    a=np.asarray(reference,dtype=np.float64).ravel()
    b=np.asarray(estimate,dtype=np.float64).ravel()
    if a.shape!=b.shape or len(a)<sr//2:
        raise ValueError('STOI requires aligned equal-length waveforms of at least 0.5 s')
    return float(stoi(a,b,sr,extended=False))
