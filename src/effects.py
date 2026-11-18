"""
Detecção de picos de energia de áudio para aplicar "zoom punches" (pequenos
zooms de ênfase, como os usados por editores humanos nos momentos de maior
intensidade da fala).
"""
from typing import List
import numpy as np

from . import config


def find_energy_peaks(energies: np.ndarray, hop: float = 0.5, offset: float = 0.0,
                       threshold: float = 0.75) -> List[float]:
    """Retorna timestamps (em segundos, relativos ao início do áudio analisado)
    dos picos de energia acima do threshold, respeitando um intervalo mínimo
    entre picos consecutivos para não deixar o efeito cansativo."""
    peaks = []
    last_peak_t = -999.0
    for i in range(1, len(energies) - 1):
        if (energies[i] > threshold
                and energies[i] >= energies[i - 1]
                and energies[i] >= energies[i + 1]):
            t = offset + i * hop
            if t - last_peak_t >= config.ZOOM_PUNCH_MIN_GAP:
                peaks.append(t)
                last_peak_t = t
    return peaks
