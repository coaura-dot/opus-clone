"""
Utilidades de análise de áudio: extrai um contorno de energia (RMS) do
sinal de voz, usado tanto para pontuar trechos "virais" quanto para
detectar picos de ênfase (zoom punches).
"""
import subprocess
import numpy as np


def audio_energy(audio_path: str, duration: float, hop: float = 0.5) -> np.ndarray:
    """Retorna um array de energia RMS normalizada (0..1), uma amostra a cada `hop` segundos."""
    sr = 16000
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_path),
        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(max(int(duration / hop), 1))

    pcm = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop_samples = max(int(hop * sr), 1)
    n_hops = max(len(pcm) // hop_samples, 1)
    energies = np.zeros(n_hops)
    for i in range(n_hops):
        chunk = pcm[i * hop_samples:(i + 1) * hop_samples]
        energies[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) else 0.0

    peak = energies.max()
    if peak > 0:
        energies = energies / peak
    return energies
