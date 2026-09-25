"""Modo REACT (RELATORIO_PROXIMOS_PASSOS.txt, item 14).

Duas responsabilidades bem separadas:

1. `detect_react_video`: varre o vídeo INTEIRO em intervalos grosseiros
   (a cada REACT_CHECK_INTERVAL_SECONDS, padrão 10s) procurando uma região
   de "tela dentro da tela" (ver reframer._detect_screen_region) de forma
   RECORRENTE ao longo do tempo -- decide se o vídeo como um todo é um
   REACT (pessoa comentando/reagindo a conteúdo de tela) ou um vídeo comum
   de pessoa(s) falando. Roda UMA VEZ por vídeo, antes de gerar qualquer
   clipe, e o resultado liga (ou não) o pipeline de rastreamento de tela já
   existente (_ScreenTracker/_compose_screen_frame em reframer.py) para
   TODOS os clipes deste vídeo.

   Isso é deliberadamente separado do rastreamento fino DENTRO de cada
   clipe (que já roda a cada poucos frames via _ScreenTracker) -- aqui só
   precisamos de um veredito grosso "este vídeo tem conteúdo de tela
   reagido, vale a pena tentar" antes de pagar o custo/risco desse
   pipeline em vídeos que não são react nenhum.

2. `find_reference_times`: varre o TEXTO transcrito por frases que
   indicam o comentador apontando pra algo visualmente na tela (ex.:
   "olha a cor da camisa dele", "repara ali", "vê aquilo") -- quando uma
   bate, devolve o timestamp (relativo ao início do clipe) pra forçar um
   zoom breve na tela reagida mesmo que ela esteja pausada/parada naquele
   momento (ver REFERENCE_ZOOM_HOLD_SECONDS em config.py e o uso em
   render_vertical_clip).

HONESTIDADE (mesma de sempre neste projeto): nenhuma das duas heurísticas
foi validada contra um vídeo de reação real (não tenho um disponível neste
ambiente). A geometria de detecção de tela já reusada aqui (Canny +
contorno de 4 vértices) é a mesma do item 3a, que também nunca foi
validada. Teste com um clipe real antes de confiar cegamente no resultado.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2

from . import config
from .reframer import _new_cascades, _detect_all_faces, _detect_screen_region


@dataclass
class ReactVideoReport:
    is_react: bool
    screen_fraction: float          # fração das checagens com uma tela plausível detectada
    face_fraction: float            # fração das checagens com um rosto detectado
    n_checks: int
    samples: List[tuple] = field(default_factory=list)  # (t, has_screen, has_face)


def detect_react_video(source_path: str, duration: float,
                        check_interval: Optional[float] = None) -> ReactVideoReport:
    """Amostra o vídeo inteiro a cada `check_interval` segundos (padrão
    config.REACT_CHECK_INTERVAL_SECONDS) e decide se é um vídeo REACT."""
    check_interval = check_interval or getattr(config, "REACT_CHECK_INTERVAL_SECONDS", 10.0)
    check_interval = max(check_interval, 1.0)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        return ReactVideoReport(is_react=False, screen_fraction=0.0, face_fraction=0.0, n_checks=0)

    cascades = _new_cascades()
    n_checks = max(int(duration // check_interval), 1)
    samples = []
    screen_hits = face_hits = 0

    for i in range(n_checks):
        t = min(i * check_interval + check_interval / 2.0, max(duration - 0.1, 0.0))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        has_screen = _detect_screen_region(frame) is not None
        has_face = len(_detect_all_faces(cascades, frame)) > 0
        samples.append((t, has_screen, has_face))
        screen_hits += int(has_screen)
        face_hits += int(has_face)

    cap.release()

    n = max(len(samples), 1)
    screen_fraction = screen_hits / n
    face_fraction = face_hits / n
    min_fraction = getattr(config, "REACT_MIN_SCREEN_FRACTION", 0.25)
    is_react = screen_fraction >= min_fraction

    return ReactVideoReport(is_react=is_react, screen_fraction=screen_fraction,
                             face_fraction=face_fraction, n_checks=len(samples), samples=samples)


# --- Referências visuais ("olha a camisa dele") ---
# Heurístico e deliberadamente conservador: verbo de atenção visual seguido,
# a poucas palavras de distância, de uma referência apontada (substantivo
# concreto, "isso"/"aquilo"/"ali"/"aqui"). Não tenta entender SEMÂNTICA (não
# sabe se "aquilo" é mesmo algo na tela reagida) -- só reconhece o PADRÃO de
# fala de quem está apontando pra algo que está vendo.
_REFERENCE_PATTERNS = [
    r"\bolh[ae]m?\s+(a|o|pra|para|ali|aqui|isso|aquilo)\b",
    r"\bvê\s+(a|o|aquilo|isso)\b",
    r"\brepar[ae]\s+(na|no|ali|nisso|naquilo)\b",
    r"\bpercebe(ram)?\s+(a|o|isso|aquilo)\b",
    r"\bnot[ae]\s+(a|o|isso|aquilo)\b",
    r"\bdá\s+uma\s+olhada\b",
    r"\baponta(ndo)?\s+pra\b",
    r"\bpresta\s+atenção\s+(n[eio]sso|nisso|naquilo)\b",
]
_REFERENCE_RE = re.compile("|".join(_REFERENCE_PATTERNS), re.IGNORECASE)


def find_reference_times(words, clip_start: float, clip_end: float,
                          window_words: int = 6) -> List[float]:
    """Varre `words` (lista de objetos com .start/.end/.text, na mesma
    estrutura usada pelo captioner) dentro de [clip_start, clip_end]
    procurando frases de referência visual. Devolve os timestamps
    (RELATIVOS ao início do clipe) de cada palavra-gatilho encontrada, já
    deduplicados (evita dois disparos muito próximos um do outro).

    `window_words` é mantido só por compatibilidade de assinatura -- não é
    mais usado (ver bug corrigido abaixo).

    BUG ENCONTRADO E CORRIGIDO (achado testando com dados sintéticos, sem
    nenhum vídeo real): a versão anterior testava uma janela de
    `window_words` palavras OLHANDO PRA FRENTE a partir de CADA palavra do
    clipe. Isso faz uma frase-gatilho perto do FIM de uma janela "vazar"
    pro timestamp de uma palavra bem mais cedo (a que abre aquela janela)
    -- ex.: com "...depois disso a gente segue, repara nisso aqui", a
    frase real ("repara nisso") acabava também disparando um hit fantasma
    ancorado em "depois", vários segundos ANTES de onde a referência
    visual de fato começa, forçando um zoom na tela reagida num momento
    sem relação nenhuma com o que está sendo dito. Corrigido construindo o
    texto do clipe inteiro UMA vez (com o offset de caractere de cada
    palavra) e buscando o regex nele uma única vez via `finditer` --
    cada match é então mapeado de volta pra palavra exata (não uma janela)
    em que ele começa."""
    clip_words = [w for w in words if clip_start <= w.start < clip_end]
    if not clip_words:
        return []

    text_parts: List[str] = []
    offsets: List[int] = []  # offsets[i] = posição do 1º caractere de clip_words[i] no texto unido
    pos = 0
    for w in clip_words:
        offsets.append(pos)
        text_parts.append(w.text.lower())
        pos += len(w.text) + 1  # +1 pelo espaço separador usado no join abaixo
    full_text = " " + " ".join(text_parts) + " "  # bordas com espaço, pro \b das regexes
    # offsets foram calculados pro texto SEM o espaço inicial -- desloca em 1
    offsets = [o + 1 for o in offsets]

    hits: List[float] = []
    min_gap = getattr(config, "REFERENCE_ZOOM_HOLD_SECONDS", 2.5)
    word_idx = 0
    n_words = len(clip_words)
    for m in _REFERENCE_RE.finditer(full_text):
        match_start = m.start()
        while word_idx + 1 < n_words and offsets[word_idx + 1] <= match_start:
            word_idx += 1
        t_rel = clip_words[word_idx].start - clip_start
        if not hits or t_rel - hits[-1] >= min_gap:
            hits.append(t_rel)
    return hits
