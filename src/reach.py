"""
Freio automático de alcance: o YouTube parou de mostrar os Shorts novos?

Achado real (canal DontStopMeNow, 9-10/out): com 1 post a cada 4 h os
Shorts iam de 369 a ~1.300 views cada; no MESMO dia em que o piloto passou
a postar 1 por hora (limite de 20 e depois 24 por dia), todos os seguintes
ficaram com 0 a 6 views -- o YouTube praticamente parou de entregar os
vídeos. Canal pequeno despejando cortes de outros canais em série é o que
os filtros de spam do YouTube seguram.

Aqui o piloto confere sozinho (pela API, 1 unidade de cota a cada 2 h) as
views dos próprios Shorts. Se os 3 últimos com 8 h ou mais de vida tiveram
quase nada (menos de 5% da mediana dos anteriores), liga o FREIO: posta no
máximo 1 a cada AUTOPILOT_REACH_BRAKE_HOURS até um Short novo voltar a
pegar views. A interface mostra o aviso.

Vários canais (src/accounts.py): cada canal tem o próprio freio -- canal
novo, sem histórico, não freia os outros.
"""
import statistics
import time

from . import config
from .accounts import FIRST, posted_by, sub

CHECK_EVERY = 2 * 3600      # confere no máximo a cada 2 h
MATURE = 8 * 3600           # Short com menos que isso ainda está sendo testado
RECENT = 3                  # quantos dos últimos precisam ter caído
MIN_BASE = 100              # canal sem histórico de views: não dá pra comparar


def brake_hours() -> float:
    return float(getattr(config, "AUTOPILOT_REACH_BRAKE_HOURS", 8))


def brake_of(data, acc: str = FIRST):
    """O freio de um canal (ou None). `data`: State ou o dict do state.json."""
    if not getattr(config, "AUTOPILOT_REACH_BRAKE", True) or not data:
        return None
    return sub(data, acc).get("reach_brake")


def braked(data, acc: str = FIRST) -> bool:
    return bool(brake_of(data, acc))


def check(service, state, log, force: bool = False, acc: str = FIRST, name: str = "") -> None:
    """Atualiza as views dos posts recentes do canal e liga/desliga o freio dele."""
    if service is None or not getattr(config, "AUTOPILOT_REACH_BRAKE", True):
        return
    now = time.time()
    me = sub(state, acc)
    if not force and now - me.get("reach_checked_at", 0) < CHECK_EVERY:
        return
    me["reach_checked_at"] = now
    tag = f" [{name}]" if name else ""
    posted = sorted((p for p in posted_by(state, acc)
                     if p.get("youtube_id") and now - p.get("at", 0) < 14 * 86400),
                    key=lambda p: p.get("at", 0))[-50:]
    if len(posted) < RECENT + 3:
        state.save()
        return
    try:
        r = service.videos().list(part="statistics", id=",".join(p["youtube_id"] for p in posted)).execute()
    except Exception as e:  # sem conferir agora: tenta na próxima
        log(f"    [alcance]{tag} não consegui ler as views dos últimos Shorts ({e.__class__.__name__}); "
            "confiro depois.")
        state.save()
        return
    views = {it["id"]: int((it.get("statistics") or {}).get("viewCount", 0) or 0) for it in r.get("items") or []}
    for p in posted:
        if p["youtube_id"] in views:
            p["views"] = views[p["youtube_id"]]
            p["views_at"] = now
    mature = [p for p in posted if p["youtube_id"] in views and now - p.get("at", 0) >= MATURE]
    brake = me.get("reach_brake")
    if brake:
        last = mature[-1] if mature else None
        if last and last.get("at", 0) > brake.get("since", 0) - MATURE \
                and views[last["youtube_id"]] >= max(0.25 * brake.get("base", 0), 50):
            me.pop("reach_brake", None)
            log(f"  [alcance]{tag} o YouTube voltou a entregar: \"{last.get('title', '')[:50]}\" já tem "
                f"{views[last['youtube_id']]} views. Freio desligado -- volta ao ritmo normal.")
    elif len(mature) >= RECENT + 3:
        recent, before = mature[-RECENT:], mature[-RECENT - 10:-RECENT]
        base = statistics.median(views[p["youtube_id"]] for p in before)
        low = max(15.0, 0.05 * base)
        rv = [views[p["youtube_id"]] for p in recent]
        if base >= MIN_BASE and all(v < low for v in rv):
            me["reach_brake"] = {"since": now, "base": base, "recent": rv}
            log(f"  [alcance]{tag} [!] o YouTube parou de entregar os Shorts: os {RECENT} últimos tiveram "
                f"{', '.join(map(str, rv))} views, contra ~{base:.0f} dos anteriores. FREIO LIGADO: "
                f"no máximo 1 post a cada {brake_hours():.0f} h até um Short voltar a pegar views.")
    state.save()
