"""Modo REACT: referências visuais na fala ("olha a camisa dele").

`find_reference_times` varre o TEXTO transcrito por frases que indicam o
comentador apontando pra algo visualmente na tela (ex.: "olha a cor da
camisa dele", "repara ali", "vê aquilo") e devolve o timestamp (relativo ao
início do clipe) de cada uma. (A detecção de que o vídeo é um react e o
layout em tela dividida ficam em src/react_layout.py.)
"""

import re
from typing import List

from . import config


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
