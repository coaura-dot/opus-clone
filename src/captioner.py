"""
Geração de legendas estilo "karaokê" (efeito Opus Clip / CapCut / Submagic):
mostra poucas palavras por vez na tela, destacando com uma cor diferente
exatamente a palavra que está sendo falada naquele instante.

Visual de corte viral:
  - grupos curtos (CAPTION_WORDS_PER_GROUP) em CAIXA ALTA, sem vírgula/ponto
    solto no fim das palavras (só "?" e "!" ficam — mudam o sentido);
  - cada grupo entra com um "pop" (escala 70% -> 100% em ~80ms);
  - a palavra falada acende na cor de destaque e cresce um pouco;
  - palavras de impacto (números, dinheiro, palavras emocionais) ficam na
    cor de ênfase o tempo todo — o olho bate nelas primeiro;
  - contorno grosso + sombra suave, legível em qualquer fundo;
  - opcional: título-gancho num balão no topo nos primeiros segundos.
"""
import re
from pathlib import Path
from typing import List, Optional

from .transcriber import Word
from . import config
from .utils import fmt_time


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{fontsize},&H00{primary},&H00{primary},&H00{outline},&H{shadow_alpha}000000,1,0,0,0,100,100,{spacing},0,1,{outline_w},{shadow},2,{marginh},{marginh},{marginv},1
Style: Hook,{hook_font},{hook_fontsize},&H00{hook_text},&H00{hook_text},&H00{hook_box},&H00{hook_box},1,0,0,0,100,100,0,0,3,{hook_pad},0,8,{hook_marginh},{hook_marginh},{hook_marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# palavras que ganham a cor de ênfase além de números/valores (curadoria
# PT-BR/EN de termos que "seguram o olho" em corte de podcast)
# (palavras comuns demais -- "tudo", "todo", "melhor", "problema" --
# ficam de fora: se meia legenda fica verde, o destaque perde o efeito)
_EMPHASIS_WORDS = {
    "dinheiro", "milhão", "milhões", "bilhão", "bilhões", "reais",
    "dólar", "dólares", "grana", "rico", "ricos", "pobre", "pobres", "salário",
    "nunca", "ninguém", "jamais", "segredo", "mentira", "mentiu", "proibido",
    "obrigatório", "grátis", "morte", "morreu", "matar", "matou", "guerra",
    "crime", "preso", "prisão", "polícia", "golpe", "roubo", "roubou",
    "deus", "diabo", "ódio", "medo", "absurdo", "mentiroso", "bandido",
    "money", "never", "nobody", "secret", "lie", "free", "death", "war",
}


def _emphasis_set() -> set:
    from .clip_selector import EMOTION_WORDS
    return _EMPHASIS_WORDS | {w.lower() for w in EMOTION_WORDS}


def _bare(word: str) -> str:
    return re.sub(r"[^\wà-öø-ÿ%$]", "", word.lower())


def _is_emphasis(word: str, emphasis: set) -> bool:
    bare = _bare(word)
    if not bare:
        return False
    return bool(re.search(r"\d", bare)) or "%" in word or "$" in word or bare in emphasis


def _display(word: str) -> str:
    """Texto que aparece na tela: sem chaves (quebrariam o ASS), sem
    pontuação solta no fim (vírgula/ponto poluem a legenda de corte), em
    caixa alta se configurado."""
    w = word.replace("{", "").replace("}", "").strip()
    w = re.sub(r"[,.;:…]+$", "", w)
    if getattr(config, "CAPTION_UPPERCASE", True):
        w = w.upper()
    return w


def _group_words(words: List[Word], group_size: int) -> List[List[Word]]:
    """Agrupa palavras consecutivas em pequenos blocos exibidos juntos na tela,
    quebrando também ao fim de frases para respeitar a pontuação — e numa
    pausa longa, pra um grupo não ficar na tela "esperando" a próxima frase."""
    groups = []
    cur: List[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        long_pause = nxt is not None and nxt.start - w.end > 0.6
        if (len(cur) >= group_size or long_pause
                or w.text.strip().endswith((".", "!", "?", "…", ","))):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _line_with_highlight(group: List[Word], active_idx: int, emphasis: set) -> str:
    """Monta o texto ASS com todas as palavras do grupo: a palavra
    `active_idx` acende na cor de destaque e cresce com uma animação curta
    (a linha começa no instante em que ela passa a ser falada, então o \\t
    dispara junto); palavras de impacto ficam na cor de ênfase."""
    primary = config.CAPTION_PRIMARY_BGR
    highlight = config.CAPTION_HIGHLIGHT_BGR
    emph_color = getattr(config, "CAPTION_EMPHASIS_BGR", highlight)
    hl_scale = getattr(config, "CAPTION_HIGHLIGHT_SCALE", 115)
    parts = []
    for i, w in enumerate(group):
        clean = _display(w.text)
        if not clean:
            continue
        if i == active_idx:
            parts.append(
                f"{{\\c&H{highlight}&\\fscx100\\fscy100\\t(0,70,\\fscx{hl_scale}\\fscy{hl_scale})}}"
                f"{clean}{{\\c&H{primary}&\\fscx100\\fscy100}}"
            )
        elif _is_emphasis(w.text, emphasis):
            parts.append(f"{{\\c&H{emph_color}&}}{clean}{{\\c&H{primary}&}}")
        else:
            parts.append(clean)
    return " ".join(parts)  # texto em uma linha só dentro do grupo


def _wrap_hook(text: str, max_chars: int) -> str:
    """Quebra o título em linhas de até ~max_chars (o balão fica compacto
    e centralizado em vez de uma linha comprida de ponta a ponta)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return r"\N".join(lines)


def generate_ass(words: List[Word], clip_offset: float, output_path: str,
                  clip_duration: float = None,
                  video_width: int = None, video_height: int = None,
                  hook_text: Optional[str] = None) -> str:
    """Gera um arquivo .ass com legendas estilo karaokê, com timestamps
    relativos ao início do clipe (clip_offset = tempo de início no vídeo
    original). Se `clip_duration` for informado, descarta/corta palavras
    que ultrapassem o fim do clipe (evita legendas "vazando" de além do
    trecho selecionado). `hook_text`: título mostrado num balão no topo nos
    primeiros HOOK_SECONDS (None/vazio = sem título)."""
    video_width = video_width or config.TARGET_WIDTH
    video_height = video_height or config.TARGET_HEIGHT

    # normaliza timestamps das palavras para o tempo local do clipe
    local_words = [
        Word(start=w.start - clip_offset, end=w.end - clip_offset, text=w.text)
        for w in words
        if w.end > clip_offset and (clip_duration is None or w.start - clip_offset < clip_duration)
    ]
    if local_words and local_words[0].start < 0:
        local_words[0] = Word(start=0.0, end=local_words[0].end, text=local_words[0].text)
    if local_words and clip_duration is not None and local_words[-1].end > clip_duration:
        last = local_words[-1]
        local_words[-1] = Word(start=last.start, end=clip_duration, text=last.text)

    groups = _group_words(local_words, config.CAPTION_WORDS_PER_GROUP)

    # todo dimensionamento da legenda escala com a largura do vídeo em relação
    # à referência de 1080px, para continuar proporcional em qualquer resolução
    scale = video_width / 1080
    fontsize = max(int(config.CAPTION_FONT_SIZE * scale), 10)
    outline_w = max(int(round(getattr(config, "CAPTION_OUTLINE_WIDTH", 6) * scale)), 1)
    shadow = max(int(round(getattr(config, "CAPTION_SHADOW", 4) * scale)), 0)
    marginv = max(int(round(config.CAPTION_MARGIN_V * scale)), 0)
    marginh = max(int(round(getattr(config, "CAPTION_MARGIN_H", 50) * scale)), 0)

    header = ASS_HEADER.format(
        width=video_width, height=video_height,
        font=config.CAPTION_FONT, fontsize=fontsize,
        primary=config.CAPTION_PRIMARY_BGR, outline=config.CAPTION_OUTLINE_BGR,
        shadow_alpha="80",  # sombra preta 50% transparente
        spacing=int(round(getattr(config, "CAPTION_LETTER_SPACING", 1) * scale)),
        outline_w=outline_w, shadow=shadow, marginv=marginv, marginh=marginh,
        hook_font=getattr(config, "HOOK_FONT", config.CAPTION_FONT),
        hook_fontsize=max(int(getattr(config, "HOOK_FONT_SIZE", 64) * scale), 10),
        hook_text=getattr(config, "HOOK_TEXT_BGR", "000000"),
        hook_box=getattr(config, "HOOK_BOX_BGR", "FFFFFF"),
        hook_pad=max(int(round(getattr(config, "HOOK_BOX_PADDING", 16) * scale)), 1),
        hook_marginh=max(int(round(getattr(config, "HOOK_MARGIN_H", 110) * scale)), 0),
        hook_marginv=max(int(round(getattr(config, "HOOK_MARGIN_V", 250) * scale)), 0),
    )

    lines = []
    hook_seconds = getattr(config, "HOOK_SECONDS", 3.2)
    if hook_text and getattr(config, "HOOK_ENABLED", True) and hook_seconds > 0:
        end = hook_seconds if clip_duration is None else min(hook_seconds, clip_duration)
        text = _wrap_hook(hook_text.replace("{", "").replace("}", ""),
                          getattr(config, "HOOK_MAX_LINE_CHARS", 24))
        # balão entra com um pop rápido e some com fade
        anim = r"{\fad(80,220)\fscx80\fscy80\t(0,140,\fscx100\fscy100)}"
        lines.append(f"Dialogue: 1,{fmt_time(0)},{fmt_time(end)},Hook,,0,0,0,,{anim}{text}")

    emphasis = _emphasis_set()
    pop = getattr(config, "CAPTION_POP_START_SCALE", 70)
    for group in groups:
        clean_group = [w for w in group if _display(w.text)]
        if not clean_group:
            continue
        # cada palavra do grupo vira uma linha .ass (pra destacar a palavra
        # certa em cada instante), mas as linhas precisam ser CONTÍGUAS: o
        # fim da linha da palavra i tem que ser exatamente o início da
        # linha da palavra i+1. Antes, cada linha usava w.start/w.end da
        # própria palavra — como o Whisper quase nunca entrega um end[i]
        # que bate exatamente com o start[i+1] (sempre sobra um micro-vão
        # de silêncio entre palavras), a legenda sumia da tela nesse vão
        # dezenas de vezes por segundo (o "piscar" estroboscópico).
        n = len(clean_group)
        cursor = max(clean_group[0].start, 0.0)
        for i, w in enumerate(clean_group):
            start = cursor
            if i < n - 1:
                end = max(clean_group[i + 1].start, start + 0.05)
            else:
                end = max(w.end, start + 0.05)
            text = _line_with_highlight(clean_group, i, emphasis)
            # o grupo entra com "pop" (só na primeira palavra) e sai com um
            # fade curto (só na última) — entre palavras do mesmo grupo não
            # há animação de entrada/saída, senão a legenda "pisca"
            tags = ""
            if i == 0:
                tags += rf"\fscx{pop}\fscy{pop}\t(0,80,\fscx100\fscy100)"
                tags += r"\fad(40,0)" if i != n - 1 else r"\fad(40,60)"
            elif i == n - 1:
                tags += r"\fad(0,60)"
            tag_block = f"{{{tags}}}" if tags else ""
            line = (f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Caption,,0,0,0,,"
                    f"{tag_block}{text}")
            lines.append(line)
            cursor = end

    Path(output_path).write_text(header + "\n".join(lines), encoding="utf-8")
    return output_path
