"""
A nota de viralidade acerta? Confere com as VIEWS REAIS dos vídeos postados
e ajusta os pesos sozinha.

Pedido do usuário: "a avaliação tem que ser boa, verifique isso". Regra
escrita à mão é chute calibrado; quem diz se acertou é o público. Então:

  1. cada vídeo postado guarda a nota que teve e cada item dela (gancho,
     conteúdo, fechamento...);
  2. uma vez por dia o piloto busca as views de cada vídeo com 2 a 10 dias
     de vida (API do YouTube: 1 unidade de cota a cada 50 vídeos -- nada
     perto das 1600 de um upload) e guarda a de ~48 h;
  3. com 8+ vídeos medidos, mostra no log o quanto a nota acerta a ordem
     das views (correlação de Spearman: 0 = chute, 1 = acerta tudo);
  4. com 15+ vídeos, ajusta os pesos: item que anda junto com as views ganha
     peso, item que não anda perde. Com poucos vídeos o ajuste é pequeno
     (cresce com a quantidade, até 70%), pra não seguir ruído.

O canal novo tem pouca view e muita variação; o ajuste vai ficando bom com
o tempo. VIRAL_WEIGHTS em config.py continua mandando, se você fixar.
"""
import math
import time
from typing import Dict, List, Optional

from . import config

_MIN_REPORT = 8
_MIN_LEARN = 15


def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0  # empate: posição média
        i = j + 1
    return r


def spearman(a: List[float], b: List[float]) -> Optional[float]:
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if va == 0 or vb == 0:
        return None
    return cov / (va * vb)


def refresh_views(state, service, log) -> int:
    """Busca as views dos vídeos com 2 a 10 dias (1x por dia)."""
    from datetime import datetime
    if service is None:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    fb = state.data.setdefault("feedback", {})
    if fb.get("views_day") == today:
        return 0
    now = time.time()
    todo = [p for p in state.data["posted"]
            if p.get("youtube_id") and "views_48h" not in p and 40 * 3600 <= now - p.get("at", 0) <= 10 * 86400]
    fb["views_day"] = today
    if not todo:
        state.save()
        return 0
    stats = {}
    try:
        ids = [p["youtube_id"] for p in todo]
        for i in range(0, len(ids), 50):
            r = service.videos().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
            for it in r.get("items") or []:
                stats[it["id"]] = int(it.get("statistics", {}).get("viewCount", 0))
    except Exception as e:  # nunca derruba o piloto
        log(f"  [feedback] não consegui buscar as views agora ({e.__class__.__name__}).")
        state.save()
        return 0
    n = 0
    for p in todo:
        if p["youtube_id"] not in stats:
            continue
        age_h = (now - p["at"]) / 3600.0
        # normaliza pra ~48 h (view de Shorts cresce rápido no começo e desacelera)
        p["views_48h"] = round(stats[p["youtube_id"]] * (48.0 / max(age_h, 48.0)) ** 0.5, 1)
        p["views_at"] = now
        n += 1
    state.save()
    if n:
        log(f"  [feedback] views de {n} vídeo(s) postado(s) atualizadas.")
    return n


def calibrate(state, log) -> Optional[Dict[str, float]]:
    """Mede o acerto da nota e devolve os pesos aprendidos (ou None)."""
    from .virality import DEFAULT_WEIGHTS
    rows = [p for p in state.data["posted"]
            if isinstance(p.get("views_48h"), (int, float)) and isinstance(p.get("viral"), (int, float))
            and not p.get("replaces")]
    fb = state.data.setdefault("feedback", {})
    if len(rows) < _MIN_REPORT:
        return fb.get("weights")
    if fb.get("calibrated_n") == len(rows):
        return fb.get("weights")
    views = [math.log1p(p["views_48h"]) for p in rows]
    rho = spearman([p["viral"] for p in rows], views)
    fb["calibrated_n"] = len(rows)
    fb["rho"] = round(rho, 2) if rho is not None else None
    msg = f"  [feedback] nota de viralidade x views reais ({len(rows)} vídeos): correlação "
    msg += f"{rho:+.2f}" if rho is not None else "indefinida"
    if rho is not None:
        msg += (" -- acerta bem a ordem" if rho >= 0.5 else " -- acerta parte" if rho >= 0.2
                else " -- ainda não acerta; ajustando os pesos")
    log(msg)
    with_parts = [p for p in rows if p.get("viral_parts")]
    if len(with_parts) < _MIN_LEARN:
        state.save()
        return fb.get("weights")
    v = [math.log1p(p["views_48h"]) for p in with_parts]
    alpha = min(len(with_parts) / 60.0, 0.7)
    learned, notes = {}, []
    for k, w0 in DEFAULT_WEIGHTS.items():
        xs = [p["viral_parts"].get(k) for p in with_parts]
        pairs = [(x, y) for x, y in zip(xs, v) if isinstance(x, (int, float))]
        r = spearman([x for x, _ in pairs], [y for _, y in pairs]) if len(pairs) >= _MIN_LEARN else None
        if r is None or abs(r) < 0.15:  # abaixo disso é ruído com tão poucos vídeos
            learned[k] = w0
            continue
        learned[k] = round(w0 * (1 + alpha * max(min(1.5 * r, 1.0), -0.8)), 2)
        if abs(r) >= 0.3:
            notes.append(f"{k} {'+' if r > 0 else '-'}")
    fb["weights"] = learned
    state.save()
    if notes:
        log(f"  [feedback] pesos ajustados pelas views (o que mais pesa no seu canal: {', '.join(notes)}).")
    return learned
