"""
Nota de VIRALIDADE (0-100) de um clipe pronto -- análise mais pesada que a
nota de qualidade (src/quality.py), feita pelo piloto automático no tempo em
que ele está parado esperando a hora de postar. A fila posta sempre o clipe
com a maior nota combinada (viralidade + qualidade).

O que conta (cada item vale de 0 a 1 e entra com um peso):
  gancho      os 3 primeiros segundos: fala começando logo (sem silêncio pra
              rolar o feed), frase forte/pergunta, energia da voz
  picos       momentos de pico de voz (risada, grito, ênfase) por minuto
  ritmo       fala densa e variada (sem trecho morto, sem monotonia)
  conteúdo    emoção, números/dinheiro, conflito/curiosidade, história
  fechamento  termina numa frase completa (não corta no meio da ideia)
  imagem      rosto em quadro e movimento na medida (nem parado, nem caos)
  duração     25-75 s rendem mais até o fim
  título      tamanho bom pro feed, com pergunta ou palavra forte

Os pesos são uma estimativa (regras de editor de cortes), não aprendidos de
dados de views -- dá pra ajustar em VIRAL_WEIGHTS (config.py).
"""
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import config

_EMOTION = re.compile(r"\b(absurd|bizarr|chocad|chorei|chorando|chorou|loucura|louco|insano|medo|raiva|"
                      r"vergonha|odeio|amo|incr[ií]vel|surreal|assustador|nojo|p[âa]nico|desesper|"
                      r"chocante|revolta|traumat|perigo|morri de rir|rachei)\w*", re.I)
_MONEY = re.compile(r"\b(\d[\d.,]*\s*(mil|milh(ão|ões|oes)|bilh(ão|ões|oes)|reais|d[óo]lares|%|por cento)|"
                    r"dinheiro|grana|sal[áa]rio|fal[êe]ncia|rico|pobre|milion[áa]rio|bilion[áa]rio)", re.I)
_CONFLICT = re.compile(r"\b(mentira|verdade|pol[ée]mic|briga|treta|processo|pres[oa]|cadeia|pol[íi]cia|"
                       r"crime|golpe|traição|traiu|traido|demitid|cancelad|ningu[ée]m (te )?(conta|fala)|"
                       r"segredo|nunca|jamais|errado|absurdo|proibid)\w*", re.I)
_STORY = re.compile(r"\b(a[ií] (ele|ela|eu|o cara)|do nada|de repente|quando eu|um dia|teve uma vez|"
                    r"sabe o que (aconteceu|ele fez)|nisso|foi a[ií] que)\b", re.I)
_STRONG_TITLE = re.compile(r"\?|\b(nunca|ningu[ée]m|verdade|segredo|mentira|dinheiro|pol[íi]cia|"
                           r"milh|bilh|pior|melhor|imposs[íi]vel|absurd|louc|medo)\w*", re.I)

DEFAULT_WEIGHTS = {"gancho": 22, "picos": 12, "ritmo": 12, "conteudo": 16, "fechamento": 8,
                   "imagem": 12, "duracao": 10, "titulo": 8}


def _audio(video: str) -> Optional[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=180).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return x if len(x) > 16000 else None


def _frames(video: str, fps: float = 1.0):
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={fps},scale=180:320",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=240).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    n = len(raw) // (180 * 320 * 3)
    return np.frombuffer(raw[:n * 180 * 320 * 3], np.uint8).reshape(n, 320, 180, 3) if n >= 3 else None


def _db_curve(x: np.ndarray, hop: int = 1600) -> np.ndarray:  # 100 ms
    n = len(x) // hop
    return 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)


def analyze(video: str, meta: dict) -> Tuple[int, Dict[str, float]]:
    """(nota 0-100, notas de cada item 0-1)."""
    text = meta.get("text") or meta.get("description", "") or ""
    title = meta.get("title", "") or ""
    hook = meta.get("hook") or title
    start, end = meta.get("start"), meta.get("end")
    parts: Dict[str, float] = {}

    x = _audio(video)
    dur = len(x) / 16000.0 if x is not None else ((end - start) if start is not None and end is not None else 60.0)
    minutes = max(dur / 60.0, 0.25)

    # --- áudio: fala, picos, ritmo
    if x is not None:
        db = _db_curve(x)
        floor, loud = np.percentile(db, 10), np.percentile(db, 95)
        speech = db > max(floor + 0.35 * (loud - floor), floor + 6)
        first3 = speech[:30]
        speech_start = (np.argmax(first3) / 10.0) if first3.any() else 3.0
        hook_energy = float(np.clip((db[:30].mean() - db.mean()) / (db.std() + 1e-6) * 0.5 + 0.5, 0, 1))
        hook_audio = float(np.clip(1.0 - speech_start / 1.5, 0, 1)) * 0.6 + hook_energy * 0.4
        # picos: subidas fortes acima do normal da própria fala (risada, grito, ênfase)
        sp = db[speech] if speech.any() else db
        thr = np.percentile(sp, 90) + 3.0
        peaks = np.nonzero((db[1:] > thr) & (db[:-1] <= thr))[0]
        parts["picos"] = float(np.clip(len(peaks) / minutes / 6.0, 0, 1))
        speech_frac = float(speech.mean())
        variety = float(np.clip(sp.std() / 8.0, 0, 1))  # voz monótona -> baixo
        parts["ritmo"] = float(np.clip((speech_frac - 0.55) / 0.35, 0, 1)) * 0.6 + variety * 0.4
    else:
        hook_audio = 0.5
        parts["picos"] = parts["ritmo"] = 0.4

    # --- gancho (3 primeiros segundos)
    hook_words = len(hook.split())
    hook_text = 1.0 if _STRONG_TITLE.search(hook) else 0.4
    if hook_words > 18:
        hook_text -= 0.3
    parts["gancho"] = float(np.clip(0.55 * hook_audio + 0.45 * hook_text, 0, 1))

    # --- conteúdo (por minuto)
    emo = len(_EMOTION.findall(text)) / minutes
    money = len(_MONEY.findall(text)) / minutes
    conflict = len(_CONFLICT.findall(text)) / minutes
    story = len(_STORY.findall(text)) / minutes
    parts["conteudo"] = float(np.clip(min(emo, 3) / 3 * 0.3 + min(money, 2) / 2 * 0.25 +
                                      min(conflict, 3) / 3 * 0.3 + min(story, 2) / 2 * 0.15, 0, 1))

    # --- fechamento: última frase completa
    tail = text.strip()[-80:]
    parts["fechamento"] = 1.0 if re.search(r"[.!?]\s*$", tail) and not tail.endswith("...") else 0.3

    # --- imagem
    fr = _frames(video)
    if fr is not None:
        g = fr.mean(axis=3).astype(np.float32)
        motion = float(np.mean(np.abs(np.diff(g, axis=0))))
        motion_ok = float(np.exp(-((motion - 12.0) / 10.0) ** 2))  # nem parado, nem caos
        face = 0.5
        try:
            from .reframer import _new_yunet, _detect_yunet
            det = _new_yunet()
            if det is not None:
                idx = np.linspace(0, len(fr) - 1, min(len(fr), 20)).astype(int)
                face = sum(1 for i in idx if _detect_yunet(det, fr[i], 180, 0.6)) / len(idx)
        except Exception:
            pass
        parts["imagem"] = float(np.clip(0.6 * face + 0.4 * motion_ok, 0, 1))
    else:
        parts["imagem"] = 0.4

    # --- duração
    if 25 <= dur <= 75:
        parts["duracao"] = 1.0
    elif dur < 25:
        parts["duracao"] = 0.5
    else:
        parts["duracao"] = float(np.clip(1.0 - (dur - 75) / 105.0, 0.1, 1.0))

    # --- título
    tl = len(title)
    parts["titulo"] = (0.5 if 25 <= tl <= 75 else 0.2) + (0.5 if _STRONG_TITLE.search(title) else 0.0)

    weights = dict(DEFAULT_WEIGHTS)
    weights.update(getattr(config, "VIRAL_WEIGHTS", {}) or {})
    total = sum(weights.values()) or 1
    score = sum(weights[k] * parts.get(k, 0.0) for k in weights) / total * 100
    return int(round(score)), {k: round(v, 2) for k, v in parts.items()}


def rank_key(item: dict) -> float:
    """Nota usada pra escolher o que postar: viralidade + qualidade."""
    q = item.get("quality", 50)
    v = item.get("viral")
    if v is None:
        return q - 5.0  # ainda sem análise: um pouco atrás dos já analisados
    return 0.6 * v + 0.4 * q


def summary(parts: Dict[str, float]) -> str:
    names = {"gancho": "gancho", "picos": "picos de voz", "ritmo": "ritmo", "conteudo": "conteúdo",
             "fechamento": "fechamento", "imagem": "imagem", "duracao": "duração", "titulo": "título"}
    good = [names[k] for k, v in sorted(parts.items(), key=lambda kv: -kv[1]) if v >= 0.7][:3]
    bad = [names[k] for k, v in sorted(parts.items(), key=lambda kv: kv[1]) if v < 0.35][:2]
    out = []
    if good:
        out.append("forte: " + ", ".join(good))
    if bad:
        out.append("fraco: " + ", ".join(bad))
    return "; ".join(out)
