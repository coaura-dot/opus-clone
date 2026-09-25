"""
Corte de "ar morto" (jump cuts): remove as pausas de dentro de cada clipe —
silêncio entre frases, respiro antes de responder, a deixa entre um falante
e outro — que numa conversa de podcast deixam o ritmo lento demais pra
Shorts/Reels/TikTok. É o corte mais básico de qualquer editor de corte
viral.

Como funciona:
  1. As pausas candidatas vêm dos timestamps de palavra da transcrição
     (vão entre o fim de uma palavra e o início da próxima maior que
     JUMPCUT_MIN_GAP_SECONDS).
  2. Cada pausa só é cortada se o ÁUDIO ali estiver de fato em silêncio —
     risada, "hmm", reação e música não viram palavra no Whisper e ficariam
     picotados se o corte confiasse só no texto.
  3. Sobra uma folga natural antes e depois de cada corte (JUMPCUT_PAD_*),
     pra não comer o fim/início das palavras.
  4. Os limites de todos os trechos mantidos são alinhados à grade de
     quadros do vídeo — o vídeo só consegue cortar em quadro inteiro, e se o
     áudio cortasse em qualquer outro ponto, cada corte somaria até meio
     quadro de dessincronia (dez cortes = boca fora do tempo).

Os trechos mantidos são devolvidos em segundos RELATIVOS ao início do
clipe; `remap_time`/`remap_words` convertem qualquer instante do clipe
original pra linha do tempo já sem as pausas.
"""
import wave
from typing import List, Tuple

import numpy as np

from . import config
from .transcriber import Word

Segment = Tuple[float, float]


def _read_wav_mono(path: str):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"esperado PCM 16-bit, veio {width * 8}-bit: {path}")
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        pcm = pcm.reshape(-1, n_ch).mean(axis=1).astype(np.int16)
    return pcm, sr


def _write_wav_mono(path: str, pcm: np.ndarray, sr: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.astype(np.int16).tobytes())


def _rms_track(pcm: np.ndarray, sr: int, hop: float = 0.02) -> np.ndarray:
    n = max(int(hop * sr), 1)
    usable = len(pcm) // n * n
    if usable == 0:
        return np.zeros(1)
    frames = pcm[:usable].astype(np.float32).reshape(-1, n) / 32768.0
    return np.sqrt((frames ** 2).mean(axis=1))


def compute_keep_segments(words: List[Word], clip_start: float, clip_end: float,
                          voice_wav: str, fps: float) -> List[Segment]:
    """Trechos (relativos ao início do clipe) que ficam no corte final.
    Sem palavras, ou com o recurso desligado, devolve o clipe inteiro."""
    duration = clip_end - clip_start
    whole = [(0.0, duration)]
    if not getattr(config, "JUMPCUT_ENABLED", True):
        return whole

    local = sorted(
        (Word(w.start - clip_start, w.end - clip_start, w.text) for w in words
         if w.end > clip_start and w.start < clip_end and w.text.strip()),
        key=lambda w: w.start,
    )
    if len(local) < 2:
        return whole

    min_gap = getattr(config, "JUMPCUT_MIN_GAP_SECONDS", 0.4)
    pad_after = getattr(config, "JUMPCUT_PAD_AFTER_SECONDS", 0.10)
    pad_before = getattr(config, "JUMPCUT_PAD_BEFORE_SECONDS", 0.06)
    min_cut = getattr(config, "JUMPCUT_MIN_CUT_SECONDS", 0.15)
    silence_ratio = getattr(config, "JUMPCUT_SILENCE_RATIO", 0.35)

    pcm, sr = _read_wav_mono(voice_wav)
    hop = 0.02
    rms = _rms_track(pcm, sr, hop)
    speech_rms = [rms[int(w.start / hop):max(int(w.end / hop), int(w.start / hop) + 1)]
                  for w in local]
    speech_level = float(np.median(np.concatenate(speech_rms))) if speech_rms else 0.0
    silence_level = speech_level * silence_ratio

    def is_silent(a: float, b: float) -> bool:
        seg = rms[int(a / hop):int(b / hop)]
        # percentil 80, não a média: um "hmm"/risada curtinho no meio da
        # pausa já basta pra ela NÃO ser silêncio
        return len(seg) > 0 and float(np.percentile(seg, 80)) < silence_level

    cuts: List[Segment] = []
    # silêncio antes da primeira palavra e depois da última
    lead_end = local[0].start - pad_before
    if lead_end > min_cut and is_silent(0.0, lead_end):
        cuts.append((0.0, lead_end))
    for prev, nxt in zip(local, local[1:]):
        if nxt.start - prev.end < min_gap:
            continue
        a, b = prev.end + pad_after, nxt.start - pad_before
        if b - a >= min_cut and is_silent(a, b):
            cuts.append((a, b))
    tail_start = local[-1].end + getattr(config, "JUMPCUT_TAIL_KEEP_SECONDS", 0.35)
    if duration - tail_start > min_cut and is_silent(tail_start, duration):
        cuts.append((tail_start, duration))

    if not cuts:
        return whole

    # alinha na grade de quadros e monta a lista do que FICA
    def snap(t: float) -> float:
        return round(t * fps) / fps

    keep: List[Segment] = []
    cursor = 0.0
    for a, b in cuts:
        a, b = snap(a), snap(b)
        if a > cursor:
            keep.append((cursor, a))
        cursor = max(cursor, b)
    end = snap(duration)
    if end > cursor:
        keep.append((cursor, end))
    keep = [(a, b) for a, b in keep if b - a >= 1.0 / fps]
    return keep or whole


def kept_duration(segments: List[Segment]) -> float:
    return sum(b - a for a, b in segments)


def remap_time(t: float, segments: List[Segment]) -> float:
    """Instante relativo ao clipe original -> instante na linha do tempo sem
    as pausas. Um instante dentro de uma pausa cortada vai pro ponto do
    corte."""
    out = 0.0
    for a, b in segments:
        if t < a:
            return out
        if t <= b:
            return out + (t - a)
        out += b - a
    return out


def remap_words(words: List[Word], clip_start: float, segments: List[Segment]) -> List[Word]:
    """Palavras com timestamp ABSOLUTO do vídeo de origem -> palavras com
    timestamp na linha do tempo do clipe já cortado (relativo, começa em 0)."""
    total = kept_duration(segments)
    out = []
    for w in words:
        s = remap_time(w.start - clip_start, segments)
        e = remap_time(w.end - clip_start, segments)
        if e <= 0 or s >= total or not w.text.strip():
            continue
        out.append(Word(start=max(s, 0.0), end=min(max(e, s + 0.05), total), text=w.text))
    return out


def tighten_audio(in_wav: str, out_wav: str, segments: List[Segment]) -> str:
    """Monta o áudio só com os trechos mantidos, com micro-fades (8ms) em
    cada emenda pra não estalar."""
    pcm, sr = _read_wav_mono(in_wav)
    fade = max(int(0.008 * sr), 1)
    pieces = []
    for a, b in segments:
        chunk = pcm[int(round(a * sr)):int(round(b * sr))].astype(np.float32)
        if len(chunk) > 2 * fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            chunk[:fade] *= ramp
            chunk[-fade:] *= ramp[::-1]
        pieces.append(chunk)
    joined = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    _write_wav_mono(out_wav, np.clip(joined, -32768, 32767), sr)
    return out_wav
