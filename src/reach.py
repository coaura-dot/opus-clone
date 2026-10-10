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
"""
import statistics
import time

from . import config

CHECK_EVERY = 2 * 3600      # confere no máximo a cada 2 h
MATURE = 8 * 3600           # Short com menos que isso ainda está sendo testado
RECENT = 3                  # quantos dos últimos precisam ter caído
MIN_BASE = 100              # canal sem histórico de views: não dá pra comparar


def brake_hours() -> float:
    return float(getattr(config, "AUTOPILOT_REACH_BRAKE_HOURS", 8))


def braked(data: dict) -> bool:
    return bool(getattr(config, "AUTOPILOT_REACH_BRAKE", True) and (data or {}).get("reach_brake"))


def check(service, state, log, force: bool = False) -> None:
    """Atualiza as views dos posts recentes e liga/desliga o freio."""
    if service is None or not getattr(config, "AUTOPILOT_REACH_BRAKE", True):
        return
    now = time.time()
    if not force and now - state.data.get("reach_checked_at", 0) < CHECK_EVERY:
        return
    state.data["reach_checked_at"] = now
    posted = sorted((p for p in state.data.get("posted", [])
                     if p.get("youtube_id") and now - p.get("at", 0) < 14 * 86400),
                    key=lambda p: p.get("at", 0))[-50:]
    if len(posted) < RECENT + 3:
        state.save()
        return
    try:
        r = service.videos().list(part="statistics", id=",".join(p["youtube_id"] for p in posted)).execute()
    except Exception as e:  # sem conferir agora: tenta na próxima
        log(f"    [alcance] não consegui ler as views dos últimos Shorts ({e.__class__.__name__}); confiro depois.")
        state.save()
        return
    views = {it["id"]: int((it.get("statistics") or {}).get("viewCount", 0) or 0) for it in r.get("items") or []}
    for p in posted:
        if p["youtube_id"] in views:
            p["views"] = views[p["youtube_id"]]
            p["views_at"] = now
    mature = [p for p in posted if p["youtube_id"] in views and now - p.get("at", 0) >= MATURE]
    brake = state.data.get("reach_brake")
    if brake:
        last = mature[-1] if mature else None
        if last and last.get("at", 0) > brake.get("since", 0) - MATURE \
                and views[last["youtube_id"]] >= max(0.25 * brake.get("base", 0), 50):
            state.data.pop("reach_brake", None)
            log(f"  [alcance] o YouTube voltou a entregar: \"{last.get('title', '')[:50]}\" já tem "
                f"{views[last['youtube_id']]} views. Freio desligado -- volta ao ritmo normal.")
    elif len(mature) >= RECENT + 3:
        recent, before = mature[-RECENT:], mature[-RECENT - 10:-RECENT]
        base = statistics.median(views[p["youtube_id"]] for p in before)
        low = max(15.0, 0.05 * base)
        rv = [views[p["youtube_id"]] for p in recent]
        if base >= MIN_BASE and all(v < low for v in rv):
            state.data["reach_brake"] = {"since": now, "base": base, "recent": rv}
            log(f"  [alcance] [!] o YouTube parou de entregar os Shorts: os {RECENT} últimos tiveram "
                f"{', '.join(map(str, rv))} views, contra ~{base:.0f} dos anteriores. FREIO LIGADO: "
                f"no máximo 1 post a cada {brake_hours():.0f} h até um Short voltar a pegar views.")
    state.save()
