"""
Efeitos sonoros curtos de edição, sintetizados na hora (sem arquivo de
áudio de terceiros, sem licença pra se preocupar):
  - "whoosh": ruído com filtro passa-faixa varrendo do grave pro agudo e
    envelope de subida/descida — o som clássico de transição que marca a
    entrada do título-gancho no começo do clipe.
"""
import numpy as np

from . import config
from .jumpcut import _read_wav_mono, _write_wav_mono


def _whoosh(sr: int, seconds: float = 0.55) -> np.ndarray:
    n = int(seconds * sr)
    rng = np.random.default_rng(7)  # sempre o mesmo som
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # filtro por blocos: a frequência central sobe de ~300Hz a ~5kHz
    out = np.zeros(n, dtype=np.float32)
    blocks = 24
    edges = np.linspace(0, n, blocks + 1).astype(int)
    for i in range(blocks):
        center = 300.0 * (5000.0 / 300.0) ** (i / (blocks - 1))
        band = np.exp(-0.5 * ((freqs - center) / (center * 0.6)) ** 2)
        filtered = np.fft.irfft(spec * band, n)
        out[edges[i]:edges[i + 1]] = filtered[edges[i]:edges[i + 1]]
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    env = np.where(t < 0.55, (t / 0.55) ** 2, ((1 - t) / 0.45) ** 1.5)
    out *= env
    peak = float(np.abs(out).max()) or 1.0
    return out / peak


def add_whoosh(wav_path: str, at: float = 0.0) -> str:
    """Soma um whoosh em `wav_path` (sobrescreve) no instante `at`."""
    if not getattr(config, "SFX_ENABLED", True):
        return wav_path
    pcm, sr = _read_wav_mono(wav_path)
    fx = _whoosh(sr) * (10 ** (getattr(config, "SFX_WHOOSH_DB", -18.0) / 20.0)) * 32767.0
    start = max(int(at * sr), 0)
    end = min(start + len(fx), len(pcm))
    if end <= start:
        return wav_path
    mixed = pcm.astype(np.float32)
    mixed[start:end] += fx[:end - start]
    _write_wav_mono(wav_path, np.clip(mixed, -32768, 32767), sr)
    return wav_path
