"""
Nota de qualidade de um clipe pronto (0-100): decide o que o piloto
automático posta primeiro e o que ele NÃO posta.

Achado real que motivou: o piloto escolheu como corte o trecho
"Amazon music, você pode seguir a gente lá no spotify..." -- o recado de
patrocínio/redes do podcast, que o seletor de cortes viu como fala animada.

O que pesa:
  - propaganda, recado de patrocínio, "se inscreve", abertura/encerramento
    do episódio: descarta ou derruba a nota;
  - gancho do começo (pergunta, frase forte, curta);
  - ritmo da fala (palavras por segundo): trecho arrastado perde nota;
  - duração: Shorts de 30-75s seguram mais gente até o fim;
  - a nota do seletor de cortes (o quanto o trecho se destacou no episódio);
  - imagem: quadro preto/congelado descarta; rosto em quadro ajuda.

Sem rede e sem IA externa: tudo local e rápido (~1-2s por clipe).
"""
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import config

# recado do canal / patrocínio / abertura e encerramento do episódio
_PROMO = [  # só chamada do PRÓPRIO canal; falar "sobre" publi/redes é conteúdo, não propaganda
    r"spotify", r"amazon music", r"deezer", r"apple podcasts?", r"youtube music",
    r"se inscrev", r"inscreva", r"ativ\w* o sininho", r"deix\w* (o |seu )?like",
    r"link na descri", r"link (aqui )?embaixo", r"na descri[çc][ãa]o (do v[ií]deo|do epis)", r"cupom",
    r"c[óo]digo (de desconto|promocional)", r"use o c[óo]digo", r"nosso patroc", r"patroc[ií]nio d[oe] (epis|podcast|canal)",
    r"oferecimento", r"seja membro", r"membros do canal", r"apoia\.?se", r"lojinha", r"nossa loja",
    r"segu\w* a gente (l[áa] )?(no|nas|na|em) (spotify|instagram|insta|tiktok|youtube|redes|canal|twitter|x\b)", r"segue o podcast", r"siga o podcast", r"chave pix", r"super ?chat",
]
_OPENING = [
    r"^(e a[ií],? )?(fala|salve),? (galera|pessoal|rapaziada|família)", r"sejam bem[- ]vindos?",
    r"bem[- ]vindos? a(o)? (mais um|nosso)", r"come[çc]ando (mais um|o) (epis[óo]dio|podcast)",
    r"(hoje|nesse epis[óo]dio) (a gente )?recebe", r"obrigad[oa] por (ter )?vir", r"valeu (galera|pessoal),? at[ée]",
    r"at[ée] (o )?pr[óo]ximo (epis[óo]dio|v[ií]deo)", r"encerrando (o|mais um)",
]
# FIM da conversa (achado real, postado: "Walter, quer deixar um recado final
# antes de te liberar..." -- e ainda veio do trailer no começo do vídeo)
_CLOSING = [
    r"recado final", r"(um|o seu|seu) recado (final|pra galera|pro pessoal|pra quem)",
    r"antes de (te|você|vc) liberar", r"(pra|para) (gente )?(encerrar|finalizar|terminar)",
    r"considera[çc][õo]es finais", r"[úu]ltimas palavras", r"pra fechar (o|a) (papo|conversa|epis)",
    r"onde (é que )?(a galera|as pessoas|o pessoal|o povo|a gente) (pode|podem|consegue|vai) te "
    r"(encontrar|seguir|acompanhar|achar)",
    r"obrigad[oa] (pela|por) (presen[çc]a|conversa|participa[çc][ãa]o|vinda|ter vindo|ter aceitado)",
    r"foi um (prazer|privil[ée]gio|honra) (te receber|ter voc[êe]|estar aqui|conversar)",
    r"valeu (demais )?por (ter )?vir", r"volta (mais )?vezes", r"a casa [ée] sua",
    # achado real: "A gente agradecer a receptividade do Tirando Dúvidas"
    r"agradecer (a|pela|o) (receptividade|presen[çc]a|participa[çc][ãa]o|audi[êe]ncia|carinho|convite)",
    r"(quero|queria|vamos|gostaria de) agradecer", r"obrigad[oa] a (todos|voc[êe]s) que (assistiram|ficaram)",
]
_STRONG_HOOK = [
    r"\?$", r"\bnunca\b", r"\bningu[ée]m\b", r"\bverdade\b", r"\bsegredo\b", r"\bmentira\b", r"\bmorr",
    r"\bpreso\b", r"\bpol[ií]cia\b", r"\bmilh(ão|ões|oes)\b", r"\bbilh", r"\bdinheiro\b", r"\bmedo\b",
    r"\berro\b", r"\bpior\b", r"\bmelhor\b", r"\bimposs[ií]vel\b", r"\bloucura\b", r"\bchocante\b",
]


def _hits(patterns: List[str], text: str) -> List[str]:
    t = (text or "").lower()
    return [p for p in patterns if re.search(p, t)]


def _visual(video: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(brilho médio, mudança entre amostras, fração de amostras com rosto)."""
    try:
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", "fps=1/3,scale=270:480",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, None, None
    n = len(raw) // (270 * 480 * 3)
    if n < 2:
        return None, None, None
    frames = np.frombuffer(raw[:n * 270 * 480 * 3], np.uint8).reshape(n, 480, 270, 3)
    gray = frames.mean(axis=3)
    bright = float(gray.mean())
    change = float(np.mean(np.abs(np.diff(gray.astype(np.float32), axis=0))))
    face_frac = None
    try:
        from .reframer import _new_yunet, _detect_yunet
        det = _new_yunet()
        if det is not None:
            face_frac = sum(1 for f in frames if _detect_yunet(det, f, 270, 0.6)) / n
    except Exception:
        face_frac = None
    return bright, change, face_frac


def assess(video: str, meta: dict) -> Tuple[int, List[str], bool]:
    """(nota 0-100, motivos, descartar?)"""
    text = meta.get("text") or meta.get("description", "")
    hook = meta.get("hook") or meta.get("title", "")
    start, end = meta.get("start"), meta.get("end")
    dur = (end - start) if start is not None and end is not None else None
    reasons, reject = [], False
    score = 50.0

    # 1) propaganda / recado do canal / abertura e encerramento
    head = " ".join(text.split()[:45])
    promo_all, promo_head = _hits(_PROMO, text), _hits(_PROMO, head)
    if len(promo_head) >= 1 and len(promo_all) >= 2 or len(promo_all) >= 3 or _hits(_PROMO, hook):
        reject = True
        reasons.append("propaganda/recado do canal")
    elif promo_all:
        score -= 12
        reasons.append("menciona propaganda")
    if _hits(_OPENING, head) or _hits(_OPENING, hook):
        score -= 25
        reasons.append("abertura/encerramento do episódio")
    if _hits(_CLOSING, head) or _hits(_CLOSING, hook):
        reject = True
        reasons.append("fim da conversa (recado final/agradecimento)")
    elif _hits(_CLOSING, text):
        score -= 10
        reasons.append("termina no encerramento do episódio")

    # 2) gancho
    hook_words = len(hook.split())
    strong = _hits(_STRONG_HOOK, hook.strip().lower())
    if strong:
        score += 10
        reasons.append("gancho forte")
    if hook_words > 18:
        score -= 5
        reasons.append("gancho comprido")

    # 3) ritmo da fala (risada não vira palavra: momento engraçado não é "fala arrastada")
    laughs = float(meta.get("laughs_per_min") or 0.0)
    if laughs >= 1.0:
        score += min(laughs, 4.0) * 2.0 + (3 if meta.get("ends_on_laugh") else 0)
        reasons.append("momento engraçado (risadas)")
    if dur and dur > 0:
        wps = len(text.split()) / dur
        if wps < 1.4 and laughs < 1.0:
            score -= 15
            reasons.append(f"fala arrastada ({wps:.1f} palavras/s)")
        elif wps >= 2.4:
            score += 6
            reasons.append("ritmo bom")

    # 4) duração
    if dur:
        if 25 <= dur <= 75:
            score += 8
        elif dur > 150:
            score -= 10
            reasons.append(f"longo ({dur:.0f}s)")
        elif dur > 100:
            score -= 4

    # 5) nota do seletor de cortes (destaque dentro do episódio)
    sel = float(meta.get("score") or 0.0)
    score += float(np.clip((sel - 12.0) * 2.0, -10, 14))

    # 6) nota do juiz de IA (src/ai_judge.py), quando ligado: é a que mais entende de conteúdo
    ai = meta.get("ai_score")
    if isinstance(ai, (int, float)):
        score += (ai - 50) * 0.6
        if ai < getattr(config, "AI_REJECT_BELOW", 35):
            reject = True
            reasons.append(f"IA: {ai}/100" + (f" ({meta.get('ai_reason')})" if meta.get("ai_reason") else ""))
        elif ai >= 75:
            reasons.append(f"IA: {ai}/100")

    # 7) imagem
    if Path(video).exists():
        bright, change, face_frac = _visual(video)
        if bright is not None and bright < 18:
            reject = True
            reasons.append("vídeo escuro/preto")
        if change is not None and change < 1.0:
            reject = True
            reasons.append("imagem congelada")
        if face_frac is not None and not meta.get("react"):  # react: rosto pequeno é o normal
            if face_frac >= 0.6:
                score += 6
            elif face_frac < 0.25:
                score -= 8
                reasons.append("pouco rosto em quadro")
    else:
        reject = True
        reasons.append("arquivo sumiu")

    return int(round(np.clip(score, 0, 100))), reasons, reject


def assess_file(video: str, meta_path: str) -> Tuple[int, List[str], bool]:
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, ["sem título/descrição"], True
    return assess(video, meta)


def min_quality() -> int:
    return int(getattr(config, "AUTOPILOT_MIN_QUALITY", 45))
