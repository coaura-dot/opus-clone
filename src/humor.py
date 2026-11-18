"""
Momentos engraçados: onde o povo RI (ou aplaude) no áudio.

Pedido do usuário: "pode permitir fazer vídeos engraçados também, com
momentos engraçados em podcasts, programas de TV...". Piada boa se mede
pela reação: risada da mesa/plateia logo depois da fala.

O Whisper não escreve risada (testado nas transcrições reais do projeto:
nenhum "haha"/"kkk"/"risos" em ~6 mil palavras), então a risada aparece
como um BURACO na transcrição: som alto, perto do volume da fala, sem
palavra nenhuma. Isso é risada, aplauso ou grito de reação -- tudo sinal
de momento que funcionou. Pra não confundir com música (vinheta, trilha),
o trecho não pode ter grave (baixo/bumbo): medido em 15 risadas reais e 23
faixas, a energia abaixo de 150 Hz é ~0,1% numa risada e ~50% na música; e
risada não dura mais que uns 8 s.

Conferido com risadas reais (ESC-50, 40 gravações) coladas no meio de fala
de podcast real: ver o teste em README ("Momentos engraçados").
"""
import re
import subprocess
from typing import List, Optional, Tuple

import numpy as np

SR = 16000
_FRAME = 0.01  # 10 ms

LAUGH_TOKEN_RE = re.compile(r"(?i)\b(?:(?:ha){2,}h?|(?:he){2,}|(?:hu?a){2,}|k{3,}|(?:rs){2,}|risos?|"
                            r"(?:ri)?sadas?|laughs?|laughter|applause|aplausos?)\b")
FUNNY_RE = re.compile(r"(?i)\b(engra[çc]ad\w*|piada|rachei|morri de rir|chorei de rir|que isso|"
                      r"n[ãa]o acredito|mentira\b|para[,!]|meu deus|t[ôo] passando mal|socorro|"
                      r"funny|hilarious|joke|lmao|bro what)\b")


def load_audio(path: str) -> Optional[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return x if len(x) > SR else None


def _env_db(x: np.ndarray) -> np.ndarray:
    hop = int(SR * _FRAME)
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop)
    return 10 * np.log10((fr ** 2).mean(1) + 1e-10)


def _bass_ratio(x: np.ndarray) -> float:
    """Fração da energia abaixo de 150 Hz: música (baixo, bumbo) ~0,5;
    risada ~0,001 (medido: 15 risadas reais x 23 faixas)."""
    if len(x) < 256:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    tot = spec[f > 40].sum() + 1e-12
    return float(spec[(f > 40) & (f < 150)].sum() / tot)


def reactions(x: np.ndarray, words, min_len: float = 0.7) -> List[Tuple[float, float]]:
    """Trechos (início, fim em s) de risada/aplauso/reação: alto, sem
    palavra transcrita, sem cara de música. `words`: objetos com
    .start/.end/.text no MESMO tempo do áudio `x`."""
    db = _env_db(x)
    n = len(db)
    if n < 100:
        return []
    covered = np.zeros(n, bool)
    laugh_words = []
    for w in words:
        a, b = int(max(w.start - 0.12, 0) / _FRAME), int((w.end + 0.12) / _FRAME)
        if LAUGH_TOKEN_RE.search(w.text or ""):
            laugh_words.append((w.start, w.end))  # o Whisper às vezes escreve "Hahaha"
            continue
        covered[a:b] = True
    if covered.sum() < 50:
        return []
    speech_level = np.percentile(db[covered], 80)
    loud = (db > speech_level - 12.0) & ~covered
    out = list(laugh_words)
    i = 0
    while i < n:
        if not loud[i]:
            i += 1
            continue
        j = i
        gap = 0
        while j + 1 < n and (loud[j + 1] or gap < 25):  # tolera 250 ms de respiro entre um "ha" e outro
            j += 1
            gap = 0 if loud[j] else gap + 1
        j -= gap
        dur = (j - i + 1) * _FRAME
        if min_len <= dur <= 8.0 and not covered[i:j + 1].any() and _bass_ratio(
                x[int(i * _FRAME * SR):int((j + 1) * _FRAME * SR)]) < 0.2:
            out.append((i * _FRAME, (j + 1) * _FRAME))
        i = j + 1
    # risada vem DEPOIS de alguém falar, no meio de uma conversa: trecho
    # longo sem transcrição nenhuma em volta é outra coisa (vídeo passando na
    # tela, música de fundo, falação cruzada -- achado no teste com o Flow)
    starts = np.array([w.start for w in words]) if words else np.array([])
    ends = np.array([w.end for w in words]) if words else np.array([])
    keep = []
    for a, b in sorted(out):
        said_before = bool(((ends <= a + 0.2) & (ends >= a - 3.0)).any())
        lo, hi = int(max(a - 10, 0) / _FRAME), int((b + 10) / _FRAME)
        talk_around = covered[lo:hi].mean() if hi > lo else 0.0
        if said_before and talk_around >= 0.35:
            if keep and a - keep[-1][1] < 0.6:
                keep[-1] = (keep[-1][0], b)  # "ha... ha ha" = uma risada só
            else:
                keep.append((a, b))
    return keep


def stats(reacts: List[Tuple[float, float]], start: float, end: float, text: str = "") -> dict:
    """Reações dentro de [start, end]: por minuto e se o trecho TERMINA numa."""
    dur = max(end - start, 1.0)
    inside = [(a, b) for a, b in reacts if a >= start - 0.2 and b <= end + 2.5]
    n = len(inside)  # "Hahaha" transcrito já entra em `reacts`
    funny_words = len(FUNNY_RE.findall(text or ""))
    return {"laughs": n, "laughs_per_min": round(n / (dur / 60.0), 2),
            "funny_words": funny_words,
            "ends_on_laugh": any(abs(a - end) <= 2.0 or a <= end <= b for a, b in inside)}
