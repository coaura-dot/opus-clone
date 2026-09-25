"""
Kit de postagem por clipe: título, descrição, hashtags por rede (YouTube
Shorts, Instagram Reels, TikTok) e crédito da música, gravados num .txt ao
lado do .mp4 — pronto pra copiar e colar na hora de publicar.

Tudo sai da própria transcrição do clipe, com heurística local (mesma
filosofia do resto do projeto: sem LLM, sem API, sem custo por vídeo):
  - título = a frase do clipe com mais cara de gancho (mesma pontuação de
    `clip_selector._text_score`: perguntas, números, palavras de emoção,
    padrões de gancho), preferindo as do começo — é o que a pessoa vai ouvir
    primeiro, então o título "promete" o que o clipe entrega logo de cara;
  - hashtags de assunto = palavras de conteúdo que mais se repetem no clipe
    (mesmo filtro de stopwords do TextTiling em clip_selector), mais as tags
    fixas de cada rede e as do seu canal (config.POST_EXTRA_HASHTAGS).
"""
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

from . import config
from .clip_selector import _content_words, _text_score

TITLE_MAX_CHARS = 70          # cabe inteiro no Shorts/TikTok sem ser cortado com "..."
DESCRIPTION_MAX_CHARS = 220

# Palavras que passam no filtro de stopwords do TextTiling (servem pra medir
# troca de assunto) mas não dizem nada como hashtag.
_WEAK_TAG_WORDS = {
    "porque", "gente", "coisa", "coisas", "fazer", "feito", "sabe", "acho",
    "tava", "pode", "podia", "tudo", "ainda", "sempre", "nunca", "todo",
    "toda", "todos", "todas", "outro", "outra", "outros", "outras", "cada",
    "pouco", "hoje", "agora", "antes", "verdade", "digamos", "quer", "dizer",
    "falar", "falou", "fala", "olha", "cara", "mano", "tipo", "assim",
    "sobre", "nada", "algum", "alguma", "alguns", "algumas", "fica", "ficou",
    "está", "estava", "tinha", "tenho", "temos", "teve", "vezes", "parte",
    "exemplo", "forma", "maneira", "ponto", "questão", "realmente",
    "because", "people", "thing", "things", "something", "really", "right",
    "know", "think", "going", "want", "said", "says", "actually",
}

PLATFORM_HASHTAGS = {
    "YouTube Shorts": ["#shorts"],
    "Instagram Reels": ["#reels"],
    "TikTok": ["#fyp", "#viral"],
}


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?…])\s+", re.sub(r"\s+", " ", text).strip())
    return [p.strip() for p in parts if p.strip()]


def _trim_at_word(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars + 1].rsplit(" ", 1)[0].rstrip(",;:—-")
    return cut + "…"


def _clean_title(sentence: str) -> str:
    # conectivos soltos no começo ("E", "Então", "Mas"...) só fazem sentido
    # com a frase anterior — num título isolado soam como frase cortada
    s = re.sub(r"^(e|então|mas|aí|daí|porque|tipo|né|and|so|but)\b[,\s]*", "",
               sentence, flags=re.IGNORECASE)
    s = s.strip(" .…,;:")
    if len(s) > TITLE_MAX_CHARS:
        # frase longa: fecha na última vírgula que caiba (oração completa)
        # antes de apelar pro corte seco em palavra com "…"
        clauses = [m.start() for m in re.finditer(r"[,;:—]", s[:TITLE_MAX_CHARS + 1])]
        clauses = [c for c in clauses if c >= 25]
        s = s[:clauses[-1]] if clauses else _trim_at_word(s, TITLE_MAX_CHARS)
    return s[:1].upper() + s[1:] if s else ""


def make_title(clip_text: str) -> str:
    sentences = _sentences(clip_text)
    best, best_score = None, -1.0
    for i, sent in enumerate(sentences[:8]):  # o título tem que vir do começo do clipe
        n_chars = len(sent)
        if n_chars < 20:
            continue
        score = _text_score(sent)
        score += 1.0 / (1 + i)                    # desempate: prefere o que é dito primeiro
        if 30 <= n_chars <= TITLE_MAX_CHARS:
            score += 2.0                          # frase curta e completa = título que cabe inteiro
        elif n_chars > 90:
            score -= 3.0                          # sairia truncada com "…"
        if sent[:1].islower():
            score -= 2.0                          # começa em minúscula = pedaço de frase maior
        if sent.endswith(("...", "…")):
            score -= 3.0                          # fala interrompida, não é frase completa
        if score > best_score:
            best, best_score = sent, score
    title = _clean_title(best or clip_text)
    return title or "Corte"


def make_topic_hashtags(clip_text: str, max_tags: int = 4) -> List[str]:
    counts = Counter(
        w for w in _content_words(clip_text)
        if len(w) >= 4 and not w.isdigit() and w not in _WEAK_TAG_WORDS
    )
    # repetição pesa mais, mas palavra longa (substantivo/nome próprio)
    # ganha de palavra curta com a mesma contagem
    ranked = sorted(counts.items(), key=lambda kv: (kv[1] * (1 + min(len(kv[0]), 12) / 12), len(kv[0])),
                    reverse=True)
    return ["#" + w for w, c in ranked if c >= 2][:max_tags]


def make_description(clip_text: str) -> str:
    sentences = _sentences(clip_text)
    desc = ""
    for sent in sentences:
        nxt = f"{desc} {sent}".strip()
        if len(nxt) > DESCRIPTION_MAX_CHARS:
            break
        desc = nxt
    desc = desc or _trim_at_word(clip_text.strip(), DESCRIPTION_MAX_CHARS)
    return desc[:1].upper() + desc[1:]


def build_post_text(clip_text: str, music_credit: Optional[str] = None,
                    source_title: Optional[str] = None, source_url: Optional[str] = None,
                    title: Optional[str] = None) -> str:
    title = title or make_title(clip_text)
    description = make_description(clip_text)
    topic_tags = make_topic_hashtags(clip_text)
    extra_tags = [t if t.startswith("#") else f"#{t}"
                  for t in getattr(config, "POST_EXTRA_HASHTAGS", [])]

    lines = [
        "TÍTULO (YouTube Shorts / legenda de capa)",
        title,
        "",
        "DESCRIÇÃO",
        description,
        "",
    ]
    for platform, platform_tags in PLATFORM_HASHTAGS.items():
        tags = list(dict.fromkeys(topic_tags + extra_tags + platform_tags))
        lines += [f"HASHTAGS — {platform}", " ".join(tags), ""]
    if music_credit:
        lines += ["MÚSICA (obrigatório colocar na descrição — licença de atribuição)",
                  music_credit, ""]
    if getattr(config, "EXPORT_NO_MUSIC_VERSION", True):
        lines += ["DICA: quer usar um som em alta? Poste o arquivo *_sem_musica.mp4 e",
                  "escolha o som pela biblioteca do app (aí não precisa do crédito acima).", ""]
    if source_title or source_url:
        src = " — ".join(x for x in (source_title, source_url) if x)
        lines += ["FONTE (dê crédito ao vídeo original)", f"Corte de: {src}", ""]
    return "\n".join(lines)


def write_post_kit(path: str, clip_text: str, music_credit: Optional[str] = None,
                   source_title: Optional[str] = None, source_url: Optional[str] = None,
                   title: Optional[str] = None) -> str:
    Path(path).write_text(
        build_post_text(clip_text, music_credit, source_title, source_url, title), encoding="utf-8")
    return path
