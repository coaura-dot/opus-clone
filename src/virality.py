"""
Nota de VIRALIDADE (0-100) de um clipe pronto -- análise mais pesada que a
nota de qualidade (src/quality.py), feita pelo piloto automático no tempo em
que ele está parado esperando a hora de postar. A fila posta sempre o clipe
com a maior nota combinada (viralidade + qualidade).

O que conta (cada item vale de 0 a 1 e entra com um peso):
  gancho      o que é DITO nos primeiros ~4 s (antes: a frase-gancho, que
              podia estar no meio do clipe): pergunta, afirmação forte,
              número, nome, "você"; fala começando logo; energia da voz
  contexto    o começo se sustenta sozinho (sem "ele/isso..." de quem não
              foi apresentado, sem "Mas..." de ideia anterior)
  picos       momentos de pico de voz (risada, grito, ênfase) por minuto
  ritmo       fala densa e variada (sem trecho morto, sem monotonia)
  conteúdo    emoção, dinheiro/números, conflito, história, curiosidade e
              substância (explicação, dado, assunto de gente pensando)
  fechamento  termina numa frase completa, de preferência com conclusão --
              não numa pergunta sem resposta nem no meio da ideia
  imagem      rosto em quadro e movimento na medida (nem parado, nem caos)
  duração     25-75 s rendem mais até o fim
  título      tamanho bom pro feed, com pergunta ou palavra forte

Trava: trecho que é recado do canal, abertura ou ENCERRAMENTO do episódio
("quer deixar um recado final?" -- achado real, foi postado) fica com no
máximo 25, seja qual for o resto.

Com o juiz de IA ligado, a nota dele entra na média (ele entende o assunto,
as regras não). E os pesos se ajustam sozinhos com as views reais dos
vídeos postados (src/feedback.py); dá pra fixar à mão em VIRAL_WEIGHTS.
"""
import re
import subprocess
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

_CURIOSITY = re.compile(r"(voc[êe] sabia|o que ningu[ée]m|a verdade (sobre|[ée])|ningu[ée]m (fala|conta|sabe)|"
                        r"por que (que )?|como ([ée] que|funciona)|o motivo|descobri|a hist[óo]ria d|"
                        r"o (maior|pior|melhor) |nunca (vi|imaginei)|imagina (se|s[óo])|sabe o que)", re.I)
_YOU = re.compile(r"\b(voc[êe]|vc|tu|seu|sua|you|your)\b", re.I)

DEFAULT_WEIGHTS = {"gancho": 22, "contexto": 10, "picos": 8, "ritmo": 10, "conteudo": 18,
                   "fechamento": 10, "imagem": 10, "duracao": 7, "titulo": 5}


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


def _sentences(text: str):
    return [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]


def _opening(text: str, n_words: int = 12) -> str:
    """O que é dito nos primeiros ~4 s (fala normal: ~3 palavras/s)."""
    return " ".join(text.split()[:n_words])


def _text_parts(text: str, title: str, minutes: float) -> Dict[str, float]:
    """Os itens que saem só do texto (servem pra testar sem o vídeo)."""
    from .clip_selector import (_SUBSTANCE_RE, _ends_on_open_question, _ends_on_topic_conclusion,
                                _starts_mid_thought, _text_score)
    parts: Dict[str, float] = {}
    sents = _sentences(text)
    first = sents[0] if sents else ""
    second = sents[1] if len(sents) > 1 else ""

    # gancho (texto): o começo de verdade do clipe
    op = _opening(text)
    g = 0.25
    g += min(_text_score(op), 6.0) / 6.0 * 0.35
    if "?" in first[:120]:
        g += 0.15
    if _YOU.search(op):
        g += 0.1                                  # fala COM quem assiste
    if re.search(r"\d", op) or re.search(r"(?<!^)\b[A-ZÀ-Ý][a-zà-ÿ]{2,}", op):
        g += 0.1                                  # concreto: número, nome
    if _CURIOSITY.search(op):
        g += 0.15
    if _starts_mid_thought(first):
        g -= 0.3                                  # "Mas...", "Só que..."
    parts["_gancho_texto"] = float(np.clip(g, 0, 1))

    # contexto: o começo se sustenta sozinho?
    try:
        from .opener import opening_depends_on_context
        dependent = opening_depends_on_context(first, second)
    except Exception:
        dependent = False
    parts["contexto"] = 0.25 if dependent else (0.6 if _starts_mid_thought(first) else 1.0)

    # conteúdo (por minuto)
    emo = len(_EMOTION.findall(text)) / minutes
    money = len(_MONEY.findall(text)) / minutes
    conflict = len(_CONFLICT.findall(text)) / minutes
    story = len(_STORY.findall(text)) / minutes
    curious = len(_CURIOSITY.findall(text)) / minutes
    substance = len(_SUBSTANCE_RE.findall(text)) / minutes
    parts["conteudo"] = float(np.clip(min(emo, 3) / 3 * 0.2 + min(money, 2) / 2 * 0.15 +
                                      min(conflict, 3) / 3 * 0.2 + min(story, 2) / 2 * 0.15 +
                                      min(curious, 2) / 2 * 0.1 + min(substance, 4) / 4 * 0.2, 0, 1))

    # fechamento: termina numa frase completa, com conclusão
    last = sents[-1] if sents else ""
    tail = text.strip()[-80:]
    fe = 0.6 if re.search(r"[.!]\s*$", tail) and not tail.endswith("...") else 0.25
    if any(_ends_on_topic_conclusion(x) for x in sents[-2:]):
        fe += 0.4
    if _ends_on_open_question(last):
        fe -= 0.3                                 # pergunta sem resposta no fim
    parts["fechamento"] = float(np.clip(fe, 0, 1))

    # título
    tl = len(title)
    parts["titulo"] = (0.5 if 25 <= tl <= 75 else 0.2) + (0.5 if _STRONG_TITLE.search(title) else 0.0)
    return parts


def _capped(text: str, title: str, hook: str) -> bool:
    """Recado do canal / abertura / encerramento do episódio no começo."""
    from .quality import _CLOSING, _OPENING, _PROMO, _hits
    head = " ".join(text.split()[:45])
    return bool(_hits(_CLOSING, head) or _hits(_CLOSING, hook) or _hits(_OPENING, head)
                or len(_hits(_PROMO, head)) >= 1 and len(_hits(_PROMO, text)) >= 2)


def analyze(video: str, meta: dict, weights: Optional[Dict[str, float]] = None) -> Tuple[int, Dict[str, float]]:
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

    # --- texto: gancho (o começo de verdade), contexto, conteúdo, fechamento, título
    tp = _text_parts(text, title, minutes)
    parts["gancho"] = float(np.clip(0.4 * hook_audio + 0.6 * tp.pop("_gancho_texto"), 0, 1))
    parts.update(tp)

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
            if det is not None and not meta.get("react"):  # react: rosto pequeno é o normal
                idx = np.linspace(0, len(fr) - 1, min(len(fr), 20)).astype(int)
                face = sum(1 for i in idx if _detect_yunet(det, fr[i], 180, 0.6)) / len(idx)
            elif meta.get("react"):
                face = 0.7
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

    parts = {k: round(v, 2) for k, v in parts.items()}
    if _capped(text, title, hook):
        parts["_trava"] = 1.0
    return score_from_parts(parts, meta.get("ai_score"), weights), parts


def current_weights(learned: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Pesos: padrão < aprendidos com as views reais (src/feedback.py) <
    fixados à mão em VIRAL_WEIGHTS."""
    w = dict(DEFAULT_WEIGHTS)
    w.update(learned or {})
    w.update(getattr(config, "VIRAL_WEIGHTS", {}) or {})
    return w


def score_from_parts(parts: Dict[str, float], ai_score=None,
                     weights: Optional[Dict[str, float]] = None) -> int:
    w = weights or current_weights()
    keys = [k for k in w if k in parts]
    total = sum(w[k] for k in keys) or 1
    score = sum(w[k] * parts[k] for k in keys) / total * 100
    if isinstance(ai_score, (int, float)):
        score = 0.5 * score + 0.5 * ai_score  # a IA entende o assunto; as regras, a forma
    if parts.get("_trava"):
        score = min(score, 25.0)
    return int(round(score))


def rank_key(item: dict, weights: Optional[Dict[str, float]] = None) -> float:
    """Nota usada pra escolher o que postar: viralidade + qualidade (+ a
    comparação da IA entre os finalistas, quando ligada)."""
    q = item.get("quality", 50)
    v = item.get("viral")
    if v is None:
        return q - 5.0  # ainda sem análise: um pouco atrás dos já analisados
    if weights and item.get("viral_parts"):
        v = score_from_parts(item["viral_parts"], item.get("ai_score"), weights)
    key = 0.6 * v + 0.4 * q
    if isinstance(item.get("ai_compare"), (int, float)):
        key = 0.5 * key + 0.5 * item["ai_compare"]
    if (item.get("viral_parts") or {}).get("_trava"):
        key = min(key, 30.0)  # recado/encerramento: nem a IA tira daqui
    return key


def summary(parts: Dict[str, float]) -> str:
    names = {"gancho": "gancho", "contexto": "começo se sustenta", "picos": "picos de voz", "ritmo": "ritmo",
             "conteudo": "conteúdo", "fechamento": "fechamento", "imagem": "imagem", "duracao": "duração",
             "titulo": "título"}
    items = [(k, v) for k, v in parts.items() if k in names]
    good = [names[k] for k, v in sorted(items, key=lambda kv: -kv[1]) if v >= 0.7][:3]
    bad = [names[k] for k, v in sorted(items, key=lambda kv: kv[1]) if v < 0.35][:2]
    out = ["TRAVADO: recado/abertura/encerramento do episódio"] if parts.get("_trava") else []
    if good:
        out.append("forte: " + ", ".join(good))
    if bad:
        out.append("fraco: " + ", ".join(bad))
    return "; ".join(out)
