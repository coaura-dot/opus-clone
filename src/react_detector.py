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


# --- Apontando pra uma IMAGEM na tela ("essa moça aqui que botou a foto") ---
# Usado pela tela dividida (src/shot_plan.py, "split"): uma borda reta e fixa
# de cima a baixo pode ser só um móvel (lateral de estante, batente); com a
# fala apontando pra imagem, é a foto/print que a edição pôs do lado.
_IMAGE_REF_PATTERNS = _REFERENCE_PATTERNS + [
    r"\b(ess[ae]s?|est[ae]s?)\s+(foto|fotos|imagem|imagens|print|prints|perfil|perfis|post|postagem|"
    r"mensagem|mensagens|tweet|bio|descri[çc][ãa]o|v[íi]deo|manchete|not[íi]cia|meme|desenho|placa|"
    r"conversa|coment[áa]rio|story|stories)\b",
    # "essa moça aqui", "esse perfil aqui" -- pessoa ou imagem; objeto ("esse
    # óculos aqui", achado num react real: ele apontava pros próprios óculos) não
    r"\b(ess[ae]|est[ae]|aquel[ae])\s+(mo[çc]a|menina|mulher|mina|garota|gata|cara|rapaz|menino|homem|"
    r"maluco|mano|velho|velha|senhor|senhora|casal|gente|fam[íi]lia|crian[çc]a|beb[êe]|noiva|noivo|"
    r"foto|imagem|print|perfil|post|frase|bio|mensagem)\s+(\w+\s+)?aqui\b",
    r"\bt[aá]\s+vendo\b",
    r"\bvoc[êe]s?\s+(t[aã]o\s+)?vendo\b",
    r"\bolh[ae]m?\s+(s[oó]|ess[ae]s?|est[ae]s?|que|como)\b",
    r"\bn[ao]\s+(foto|imagem|tela|bio|print|perfil)\b",
    # canais em inglês (Theo Von, Lex Fridman, DrDonut)
    r"\blook\s+at\s+(this|that|these|him|her|his|the)\b",
    r"\b(this|that)\s+(picture|photo|pic|image|post|tweet|profile|screenshot|meme|headline|bio)\b",
    r"\b(this|that)\s+(guy|girl|woman|man|dude|lady|chick|couple|kid|baby)\s+(right\s+)?here\b",
    r"\bcheck\s+(this|that)\s+out\b",
    r"\byou\s+see\s+(this|that|the|him|her|his)\b",
]
_IMAGE_REF_RE = re.compile("|".join(_IMAGE_REF_PATTERNS), re.IGNORECASE)


def points_at_image(words, t0: float, t1: float) -> bool:
    """A fala entre t0 e t1 (mesmo tempo das palavras) aponta pra uma
    imagem na tela?"""
    text = " " + " ".join((w.text or "").strip().lower() for w in words if t0 <= w.start < t1) + " "
    return bool(_IMAGE_REF_RE.search(text))
