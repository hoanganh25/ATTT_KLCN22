"""Build one channel per run. No robust/burst/impulse training modes."""
from .awgn import AWGNChannel
from .rayleigh import RayleighChannel
from .no_noise import NoNoiseChannel


def make_channel(name: str, snr_db: float = 10.0):
    key = name.lower().strip()
    if key == 'awgn':
        return AWGNChannel(snr_db=snr_db)
    if key == 'rayleigh':
        return RayleighChannel(snr_db=snr_db)
    if key == 'no_noise':
        return NoNoiseChannel()
    raise ValueError(f'Unknown channel: {name}. Choose awgn, rayleigh, no_noise.')
