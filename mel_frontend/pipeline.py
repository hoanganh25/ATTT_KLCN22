"""Same waveform-to-log-Mel convention as mel_audio_v01; NO waveform inverse."""
import numpy as np
from .config import MelConfig
from .framing import frame_wave
from .windowing import apply_hann
from .stft import complex_stft, power_spectrogram
from .mel_filterbank import make_mel_bank, to_mel_power
from .log_scale import mel_to_logmel


def preprocess(waveform: np.ndarray, cfg: MelConfig, bank: np.ndarray | None = None) -> np.ndarray:
    """Return float32 log-Mel [96, T]: ln(SlaneyBank @ abs(STFT)**2 + 1e-8)."""
    frames = frame_wave(np.asarray(waveform), cfg.n_fft, cfg.hop_length)
    power = power_spectrogram(complex_stft(apply_hann(frames)))
    mel = to_mel_power(power, make_mel_bank(cfg) if bank is None else bank)
    output = mel_to_logmel(mel, cfg.eps).astype(np.float32)
    if output.shape[0] != cfg.n_mels or not np.isfinite(output).all():
        raise ValueError(f'Invalid log-Mel output {output.shape}')
    return output
