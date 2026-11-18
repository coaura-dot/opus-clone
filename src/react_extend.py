"""
React: o corte tem que pegar o conteúdo reagido INTEIRO.

Achado real (corte do orochidois, "influenciadores aos prantos com o fim
das bets", Short lBqOLQdGLZo): o clipe acabava no streamer apresentando o
vídeo ("ó esse aqui") e o depoimento do Magal -- que era o vídeo que ele ia
reagir -- ficava de fora. A escolha do trecho só olha a fala (assunto,
pausa, gancho); não sabe que, num react, a fala de OUTRA pessoa logo
depois é o próprio conteúdo.

O que faz, sem IA pesada:
  1. o fim do clipe tem o streamer APRESENTANDO algo ("ó esse aqui", "olha
     o que ele falou", "vamos ver", "escuta isso", "dá o play", "look at
     this"...) -- só olha a transcrição, é instantâneo;
  2. logo depois (até 20 s) entra uma voz DIFERENTE da voz principal do
     clipe (o vídeo reagido) -- "impressão digital" da voz, src/voices.py;
  3. aí o fim vai até essa outra voz terminar (o streamer pode comentar
     por cima uns segundos no meio) + a frase de reação dele logo depois.
O mesmo no começo: clipe que começa no meio do vídeo reagido, com o
streamer apresentando logo antes, volta pra apresentação.

Sem a voz NOVA logo depois, nada muda: num podcast, "olha isso" seguido da
mesma pessoa (ou de quem já vinha conversando) não estende nada; e a voz
nova tem que dominar o trecho (vídeo tocando, o streamer mais assiste do que
fala). Testado em 4 podcasts reais (76 candidatos): nenhuma extensão. Respeita
o tamanho máximo (até 175 s -- Shorts aceita 3 min --, puxando o começo pra
frente se precisar) e não invade outro clipe do mesmo vídeo.
"""
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import config

LOOKAHEAD = 150.0     # até onde procura o fim do conteúdo depois do clipe (s)
SETUP_WINDOW = 15.0   # a apresentação tem que estar nos últimos N s do clipe
FIRST_GAP = 20.0      # o conteúdo tem que começar até N s depois da apresentação (o Whisper às
                      # vezes pula o começo do vídeo reagido -- vinheta, música: achado no teste)
MAX_INTERJECTION = 6.0  # streamer falando por cima do vídeo: até N s seguidos
MAX_SILENCE = 8.0     # silêncio maior que isso acaba o conteúdo
MIN_CONTENT = 5.0     # pelo menos N s de fala da outra voz
DOMINANCE = 1.5       # ...e N vezes mais fala dela que do streamer no trecho: vídeo reagido
                      # tocando, o streamer mais assiste do que fala; numa CONVERSA as vozes
                      # se alternam (achado: podcast do Bam Margera, live do Impera)

_SETUP_RE = re.compile(
    r"\b(?:"
    # "olha" sozinho é muleta ("olha, eu acho que..."): só apontando algo
    r"(?:ó|olha|olhem|olhe|veja|vejam|v[êe]|repara|reparem|escuta|escutem|ouve|ou[çc]am|ou[çc]a|assiste|assistam)"
    r"\s+(?:isso|isto|esse|essa|esses|essas|aqui|a[íi]|s[óo]|o que|como|ele|ela|o\s+\w+\s+falando)|"
    r"presta(?:m)? aten[çc][ãa]o|"
    r"(?:vamos|vamo|bora|deixa eu|vou)\s+(?:ver|assistir|escutar|ouvir|mostrar|botar|colocar|soltar|dar\s+(?:o\s+)?play)|"
    r"d[áa]\s+(?:o\s+)?play|solta\s+(?:o|a|esse|essa)|"
    r"(?:esse|essa|esses|essas)\s+(?:aqui|a[íi]|v[íi]deo|tiktok|post|tweet|depoimento|story|stories|reels|clipe|"
    r"trecho|coment[áa]rio|not[íi]cia|print|mat[ée]ria|entrevista|fala)|"
    # ("ele falou", "ela disse" ficam de fora: é narração de história --
    #  achado no Maicon Küster lendo relatos)
    r"look at (?:this|that|him|her)|watch this|listen to (?:this|that|him|her)|check (?:this|it) out|"
    r"let'?s (?:watch|hear|take a look)|"
    r"play it|roll (?:it|the clip)"
    r")\b", re.IGNORECASE)


def _w(w):
    return float(w.start), float(w.end), (w.text or "").strip()


def _phrases(words, t0: float, t1: float) -> List[List[float]]:
    """Trechos de fala pra comparar a voz: cortados em pontuação, pausa e no
    máximo 4 s (igual ao modo gameplay). [início, fim, nº de palavras]"""
    out = []
    end_punct = False
    for w in sorted((w for w in words if w.end > t0 and w.start < t1), key=lambda w: w.start):
        a, b, txt = _w(w)
        if out and a - out[-1][1] <= 0.3 and b - out[-1][0] <= 4.0 and not end_punct:
            out[-1][1], out[-1][2] = b, out[-1][2] + 1
        else:
            out.append([a, b, 1])
        end_punct = txt[-1:] in ".?!"
    return out


def _setup_in(words, t0: float, t1: float) -> Optional[Tuple[float, float]]:
    """Última apresentação ("ó esse aqui", "vamos ver"...) entre t0 e t1:
    (início, fim) em s, ou None."""
    ws = [w for w in words if t0 <= w.start and w.end <= t1 + 0.3]
    if not ws:
        return None
    text, spans = "", []
    for w in ws:
        a, b, txt = _w(w)
        s0 = len(text) + (1 if text else 0)
        text = (text + " " + txt) if text else txt
        spans.append((s0, len(text), a, b))
    last = None
    for m in _SETUP_RE.finditer(text):
        last = m
    if last is None:
        return None
    a = next(sa for s0, s1, sa, sb in spans if s1 > last.start())
    b = next(sb for s0, s1, sa, sb in reversed(spans) if s0 < last.end())
    return a, b


def _ensure_words(words: list, t1: float, source_path: str, duration: float, work_dir: str, log) -> list:
    """Transcreve o pedaço que falta depois do fim (vídeo longo é transcrito
    em blocos de 20 min: a fala depois do fim do bloco ainda não existe)."""
    have = max((float(w.end) for w in words), default=0.0)
    t1 = min(t1, duration)
    if have >= t1 - 5.0:
        return words
    from .video_editor import extract_audio_segment
    from .transcriber import transcribe
    wav = Path(work_dir) / f"react_extra_{int(have)}.wav"
    try:
        extract_audio_segment(source_path, have, t1, str(wav))
        tr = transcribe(str(wav), model_size=config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE,
                        compute_type=config.WHISPER_COMPUTE_TYPE)
    except Exception as e:  # sem o pedaço extra: decide com o que tem
        log(f"    [react] não consegui transcrever o trecho depois do clipe ({e.__class__.__name__}).")
        return words
    finally:
        try:
            wav.unlink()
        except OSError:
            pass
    from .transcriber import Word
    extra = [Word(w.start + have, w.end + have, w.text) for w in tr.words if w.start + have > have + 0.05]
    return list(words) + extra


def _voices(words, t0: float, t1: float, ref: Tuple[float, float], source_path: str, log):
    """[(início, fim, nº palavras, "main"|"other"|None)] das falas entre t0 e
    t1. "other" = voz NOVA: diferente de TODAS as falas do trecho `ref`.
    Medido: pessoa diferente fica com 0,10-0,24 de semelhança com qualquer
    fala do clipe; o próprio streamer, mesmo lendo uma história com outra
    entonação (achado no Maicon Küster: "aí ela disse..." virava "outra
    voz" comparando só com a voz principal), 0,45-0,80."""
    import numpy as np
    from . import voices
    ph = _phrases(words, t0, t1)
    if not ph:
        return []
    audio = voices.load_audio(source_path, t0, t1)
    if audio is None:
        return None
    embs = voices.embed_phrases(audio, [(a - t0, b - t0) for a, b, _ in ph], log)
    if embs is None:
        return None
    R = [e for (a, b, _), e in zip(ph, embs) if e is not None and ref[0] <= a and b <= ref[1] + 0.3]
    if len(R) < 2:
        return None
    R = np.stack(R)
    new_max = getattr(config, "REACT_EXTEND_NEW_VOICE_MAX_SIM", 0.35)
    out = []
    for (a, b, n), e in zip(ph, embs):
        if e is None:
            out.append((a, b, n, None))
            continue
        sims = R @ e
        if ref[0] <= a and b <= ref[1] + 0.3:
            sims = sims[sims < 0.999]  # ela mesma
        m = float(sims.max()) if len(sims) else 1.0
        out.append((a, b, n, "other" if m < new_max and n >= 2 else "main" if m >= 0.45 else None))
    return out


def _content_run(vs, after: float, limit: float):
    """O conteúdo reagido que começa depois de `after`: (início, fim, s de
    fala da outra voz) ou None."""
    first = next((v for v in vs if v[0] >= after - 0.5 and v[3] == "other"), None)
    if first is None or first[0] - after > FIRST_GAP:
        return None
    start, last_end, talk, main_talk = first[0], first[1], 0.0, 0.0
    main_run = pending_main = 0.0
    for a, b, n, who in vs:
        if a < first[0]:
            continue
        if a - last_end > MAX_SILENCE or a > limit:
            break
        if who == "other":
            talk += b - a
            last_end = b
            main_talk += pending_main  # comentário no meio do vídeo: conta
            main_run = pending_main = 0.0
        else:
            main_run += b - a
            pending_main += b - a
            if main_run > MAX_INTERJECTION:
                break
    if talk < MIN_CONTENT or talk < DOMINANCE * main_talk:
        return None
    return start, last_end, talk


def _sentence_end_after(words, t: float, within: float = 12.0) -> float:
    """Fim da primeira frase completa depois de t (a reação do streamer)."""
    for w in sorted((w for w in words if t < w.end <= t + within), key=lambda w: w.end):
        if (w.text or "").strip()[-1:] in ".?!":
            return float(w.end)
    nxt = [float(w.end) for w in words if t < w.end <= t + 4.0]
    return max(nxt) if nxt else t


def _sentence_starts(words) -> List[float]:
    ws = sorted(words, key=lambda w: w.start)
    out = [float(ws[0].start)] if ws else []
    for p, w in zip(ws, ws[1:]):
        if (p.text or "").strip()[-1:] in ".?!" or w.start - p.end > 0.7:
            out.append(float(w.start))
    return out


def extend(cand, words: list, source_path: str, duration: float, work_dir: str,
           others: Sequence = (), log=print) -> list:
    """Estende o candidato pra pegar o conteúdo reagido inteiro. Devolve a
    lista de palavras (pode ganhar o trecho transcrito a mais)."""
    if not getattr(config, "REACT_EXTEND_ENABLED", True) or not words:
        return words
    start, end = float(cand.start), float(cand.end)
    # Shorts aceita até 3 min: com folga pra não passar
    max_total = min(config.MAX_CLIP_DURATION + getattr(config, "TOPIC_BOUNDARY_GRACE_SECONDS", 0.0), 175.0)
    setup_end = _setup_in(words, end - SETUP_WINDOW, end)
    setup_start = _setup_in(words, max(start - 25.0, 0.0), start + 2.0)
    if setup_end is None and setup_start is None:
        return words
    t0 = max(start - (25.0 if setup_start else 0.0), 0.0)
    t1 = min(end + LOOKAHEAD, duration)
    if setup_end is not None:
        words = _ensure_words(words, t1, source_path, duration, work_dir, log)
    try:
        vs = _voices(words, t0, t1, (start, end), source_path, log) if setup_end is not None else []
    except Exception as e:  # nunca derruba o clipe
        log(f"    [react] não consegui comparar as vozes ({e.__class__.__name__}: {e}).")
        return words
    if vs is None:
        return words
    # outro clipe do mesmo vídeo logo depois: não invade
    nxt = min((float(o.start) for o in others if o is not cand and float(o.start) >= end - 1.0), default=duration)
    prv = max((float(o.end) for o in others if o is not cand and float(o.end) <= start + 1.0), default=0.0)
    new_start, new_end = start, end

    # --- fim: apresentou e o vídeo começou -> vai até o vídeo acabar
    if setup_end is not None:
        run = _content_run(vs, setup_end[1], min(t1, nxt))
        if run and run[1] > end + 1.0:
            react_end = _sentence_end_after(words, run[1])
            for cand_end in (react_end + 0.3, run[1] + 0.3):
                cand_end = min(cand_end, nxt - 0.2, duration)
                if cand_end < run[1]:
                    continue
                s = new_start
                if cand_end - s > max_total:
                    # puxa o começo pra frente (numa frase), sem perder a apresentação
                    ok = [x for x in _sentence_starts([w for w in words if start <= w.start <= setup_end[0]])
                          if x >= cand_end - max_total and x <= setup_end[0] - 2.0]
                    if not ok:
                        continue
                    s = min(ok)
                new_start, new_end = s, cand_end
                log(f"    [react] o clipe acabava no streamer apresentando o vídeo; o conteúdo reagido "
                    f"(outra voz, {run[2]:.0f} s de fala) vai até {run[1] - start:.0f} s: fim "
                    f"{end - start:.0f} s -> {new_end - start:.0f} s"
                    + (f", começo puxado {s - start:.0f} s pra caber" if s > start else ""))
                break

    # --- começo: abriu no meio do vídeo reagido -> volta pra apresentação
    if setup_start is not None and new_start == start:
        # referência: os 45 s até a apresentação (o próprio clipe começa no
        # vídeo reagido). Voz que já falava antes é conversa, não vídeo
        # reagido (achado: podcast do Bam Margera, "Look at that finger" e a
        # apresentadora respondendo)
        try:
            ref0 = max(setup_start[0] - 45.0, 0.0)
            vs = _voices(words, ref0, min(start + 30.0, t1), (ref0, setup_start[1]), source_path, log) or []
        except Exception:
            vs = []
        head = [v for v in vs if start - 0.3 <= v[0] < start + 6.0 and v[3] is not None]
        if head and head[0][3] == "other":
            run = _content_run(vs, setup_start[1], start + 2.0)
            if run and run[0] <= start + 2.0:
                s = max([x for x in _sentence_starts([w for w in words if w.start <= setup_start[0] + 0.01
                                                      and w.start >= setup_start[0] - 8.0])] or [setup_start[0]])
                s = max(s, prv + 0.2)
                if s < start and new_end - s <= max_total:
                    log(f"    [react] o clipe começava no meio do vídeo reagido; volta {start - s:.0f} s "
                        "pro streamer apresentando.")
                    new_start = s

    if (new_start, new_end) != (start, end):
        cand.start, cand.end = new_start, new_end
        cand.text = " ".join(_w(w)[2] for w in sorted(words, key=lambda w: w.start)
                             if w.start >= new_start - 0.05 and w.end <= new_end + 0.05)
    return words


def extend_all(cands: list, words: Optional[list], source_path: str, duration: float, work_dir: str,
               log=print) -> None:
    """Todos os candidatos de um vídeo (`words` None = cada um tem as suas,
    vídeo longo)."""
    for c in cands:
        try:
            ws = words if words is not None else c.words
            new = extend(c, ws, source_path, duration, work_dir, others=cands, log=log)
            if words is None and new is not ws:
                c.words = new
        except Exception as e:  # nunca derruba o vídeo
            log(f"    [react] extensão do clipe falhou ({e.__class__.__name__}: {e}); segue como estava.")
