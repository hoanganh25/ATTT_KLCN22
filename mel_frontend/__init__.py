"""Forward-only log-Mel DSP derived from the prior mel_audio_v01 project."""
from .config import MelConfig
from .pipeline import preprocess
__all__ = ['MelConfig', 'preprocess']
