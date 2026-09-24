"""
Geração de legendas estilo "karaokê" (efeito Opus Clip / CapCut / Submagic):
mostra poucas palavras por vez na tela, destacando com uma cor diferente
exatamente a palavra que está sendo falada naquele instante.
"""
from pathlib import Path
from typing import List

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
Style: Caption,{font},{fontsize},&H00{primary},&H00{primary},&H00{outline},&H00000000,1,0,0,0,100,100,0,0,1,{outline_w},{shadow},2,50,50,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _group_words(words: List[Word], group_size: int) -> List[List[Word]]:
    """Agrupa palavras consecutivas em pequenos blocos exibidos juntos na tela,
    quebrando também ao fim de frases para respeitar a pontuação."""
    groups = []
    cur: List[Word] = []
    for w in words:
        cur.append(w)
        if len(cur) >= group_size or w.text.strip().endswith((".", "!", "?", "…")):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _line_with_highlight(group: List[Word], active_idx: int) -> str:
    """Monta o texto ASS com todas as palavras do grupo, colorindo apenas a
    palavra `active_idx` com a cor de destaque (efeito de leitura em tempo real)."""
    parts = []
    for i, w in enumerate(group):
        clean = w.text.replace("{", "").replace("}", "").strip()
        if not clean:
            continue
        if i == active_idx:
            hl_scale = getattr(config, "CAPTION_HIGHLIGHT_SCALE", 108)
            parts.append(
                f"{{\\c&H{config.CAPTION_HIGHLIGHT_BGR}&\\fscx{hl_scale}\\fscy{hl_scale}}}{clean}"
                f"{{\\c&H{config.CAPTION_PRIMARY_BGR}&\\fscx100\\fscy100}}"
            )
        else:
            parts.append(clean)
    return r"\N".join([" ".join(parts)])  # texto em uma linha só dentro do grupo


def generate_ass(words: List[Word], clip_offset: float, output_path: str,
                  clip_duration: float = None,
                  video_width: int = None, video_height: int = None) -> str:
    """Gera um arquivo .ass com legendas estilo karaokê, com timestamps
    relativos ao início do clipe (clip_offset = tempo de início no vídeo
    original). Se `clip_duration` for informado, descarta/corta palavras
    que ultrapassem o fim do clipe (evita legendas "vazando" de além do
    trecho selecionado)."""
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
    shadow = max(int(round(outline_w / 3)), 0)
    marginv = max(int(round(config.CAPTION_MARGIN_V * scale)), 0)

    header = ASS_HEADER.format(
        width=video_width, height=video_height,
        font=config.CAPTION_FONT, fontsize=fontsize,
        primary=config.CAPTION_PRIMARY_BGR, outline=config.CAPTION_OUTLINE_BGR,
        outline_w=outline_w, shadow=shadow, marginv=marginv,
    )

    lines = []
    for group in groups:
        clean_group = [w for w in group if w.text.strip()]
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
            text = _line_with_highlight(clean_group, i)
            # fade só na entrada do grupo (primeira palavra) e na saída
            # (última palavra) — como as linhas agora são contíguas, não há
            # mais motivo pra fade entre uma palavra e outra dentro do
            # mesmo grupo (isso também contribuía pro piscar).
            fade = ""
            if i == 0:
                fade += r"\fad(60,0)" if i != n - 1 else r"\fad(60,60)"
            elif i == n - 1:
                fade += r"\fad(0,60)"
            fade_tag = f"{{{fade}}}" if fade else ""
            line = (f"Dialogue: 0,{fmt_time(start)},{fmt_time(end)},Caption,,0,0,0,,"
                    f"{fade_tag}{text}")
            lines.append(line)
            cursor = end

    Path(output_path).write_text(header + "\n".join(lines), encoding="utf-8")
    return output_path
