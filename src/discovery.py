"""
Modo automático: acha o próximo vídeo pra cortar.

De onde vêm os candidatos (config.py, seção "Postagem automática"):
  - AUTOPILOT_CHANNELS: podcasts/youtubers famosos que você acompanha (os
    últimos AUTOPILOT_VIDEOS_PER_CHANNEL vídeos de cada um).
  - AUTOPILOT_SEARCHES: buscas extras no YouTube ("esta semana, mais
    vistos").

Como escolhe: o vídeo com MAIS VIEWS (não precisa ser lançamento: um
podcast de 2 meses atrás com 2 milhões de views rende ótimos cortes), com
uma leve preferência pelos mais recentes do canal, variando o canal, com
duração boa pra render cortes, que não seja live e que ainda não foi usado.

Detalhe que importa: a listagem é pedida em inglês. Em português o YouTube
escreve "91 mil visualizações" e o número saía 91 (achado real: nada
passava no filtro de views). O título em português vem depois, só do
vídeo escolhido.

Tudo pelo yt-dlp, sem chave de API e sem gastar a cota do YouTube.
"""
import re
import time
import urllib.parse
from typing import List, Optional

from . import config

# filtro da busca do YouTube: ordenar por visualizações + enviados nesta semana + só vídeos
_SEARCH_FILTER = "CAMSBAgDEAE"
_HEADERS = {"Accept-Language": "pt-BR,pt;q=0.9"}

_PT_WORDS = {"de", "do", "da", "dos", "das", "que", "não", "nao", "com", "para", "pra", "um",
             "uma", "os", "as", "o", "a", "e", "no", "na", "em", "é", "eu", "você", "voce", "ele",
             "ela", "mais", "meu", "minha", "sobre", "como", "por", "quem", "isso", "seu", "sua",
             "qual", "quando", "porque", "foi", "tem", "ser", "ao", "vai", "podcast", "cortes"}
_ES_MARKERS = re.compile(r"[ñ¿¡]|\b(el|los|las|con|una|del|muy|pero|cómo|qué)\b")


def _ydl(flat: bool, lang: Optional[str] = None):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "ignore_no_formats_error": True, "socket_timeout": 30}
    if lang:
        opts["extractor_args"] = {"youtube": {"lang": [lang]}}
        opts["http_headers"] = _HEADERS
    if flat:
        opts["extract_flat"] = "in_playlist"
    cookies = getattr(config, "YTDLP_COOKIES_FROM_BROWSER", None)
    if cookies:
        opts["cookiesfrombrowser"] = (cookies,)
    return yt_dlp.YoutubeDL(opts)


def looks_portuguese(text: str) -> bool:
    t = (text or "").lower()
    if re.search(r"[ãõ]", t):
        return True
    if _ES_MARKERS.search(t):
        return False
    words = re.findall(r"[a-zà-úç]+", t)
    return sum(w in _PT_WORDS for w in words) >= 2


def _channel_videos_url(channel: str) -> str:
    c = channel.strip().rstrip("/")
    if not c.startswith("http"):
        c = "https://www.youtube.com/" + (c if c.startswith("@") else "@" + c)
    if not re.search(r"/(videos|streams|shorts)$", c):
        c += "/videos"
    return c


def list_channel(channel: str, n: int = 10) -> List[dict]:
    with _ydl(flat=True) as y:
        y.params["playlistend"] = n
        info = y.extract_info(_channel_videos_url(channel), download=False)
    out = []
    for rank, e in enumerate((info or {}).get("entries") or []):
        if e and e.get("id"):
            out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                        "view_count": e.get("view_count"), "channel": info.get("channel") or channel,
                        "origin": "canal", "rank": rank})
    return out


def search(query: str, n: int = 15) -> List[dict]:
    url = ("https://www.youtube.com/results?" +
           urllib.parse.urlencode({"search_query": query, "sp": _SEARCH_FILTER, "hl": "pt-BR", "gl": "BR"}))
    with _ydl(flat=True) as y:
        y.params["playlistend"] = n
        info = y.extract_info(url, download=False)
    out = []
    for e in (info or {}).get("entries") or []:
        if e and e.get("id") and len(e["id"]) == 11:
            out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                        "view_count": e.get("view_count"), "channel": e.get("channel") or "",
                        "origin": f"busca '{query}'"})
    return out


_INFO_CACHE: dict = {}
_INFO_TTL = 3 * 3600


def full_info(video_id: str) -> Optional[dict]:
    """Data de publicação, views, live... (uma consulta por vídeo; guarda
    por 3h pra não consultar de novo o mesmo vídeo a cada rodada)."""
    hit = _INFO_CACHE.get(video_id)
    if hit and time.time() - hit[0] < _INFO_TTL:
        return hit[1]
    info = _full_info(video_id)
    _INFO_CACHE[video_id] = (time.time(), info)
    return info


def _full_info(video_id: str) -> Optional[dict]:
    try:
        with _ydl(flat=False, lang="pt") as y:
            v = y.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception:
        return None
    if not v:
        return None
    ts = v.get("timestamp") or v.get("release_timestamp")
    if not ts and v.get("upload_date"):
        try:
            ts = time.mktime(time.strptime(v["upload_date"], "%Y%m%d")) + 12 * 3600
        except ValueError:
            ts = None
    return {"timestamp": ts, "view_count": v.get("view_count"), "duration": v.get("duration"),
            "live_status": v.get("live_status"), "availability": v.get("availability"),
            "title": v.get("title"), "channel": v.get("channel"), "language": v.get("language"),
            "age_limit": v.get("age_limit") or 0}


def _duration_ok(d) -> bool:
    if not d:
        return True  # desconhecida na listagem rápida: confere depois
    lo = getattr(config, "AUTOPILOT_MIN_SOURCE_MINUTES", 8) * 60
    hi = getattr(config, "AUTOPILOT_MAX_SOURCE_MINUTES", 240) * 60
    return lo <= d <= hi


def pick_source(seen: set, recent_channels: Optional[dict] = None, log=print) -> Optional[dict]:
    """O melhor vídeo pra cortar agora (ou None se não achou nada novo).
    `seen`: IDs já usados/descartados. `recent_channels`: canal -> quantas
    vezes foi usado nas últimas 24h (pra variar a fonte)."""
    recent_channels = recent_channels or {}
    cands, fails = [], 0
    for ch in getattr(config, "AUTOPILOT_CHANNELS", []):
        try:
            cands += list_channel(ch, getattr(config, "AUTOPILOT_VIDEOS_PER_CHANNEL", 30))
        except Exception as e:
            fails += 1
            log(f"    [busca] canal {ch}: {str(e).splitlines()[0][:120]}")
    for q in getattr(config, "AUTOPILOT_SEARCHES", []):
        try:
            cands += [dict(c, rank=i) for i, c in enumerate(search(q))]
        except Exception as e:
            fails += 1
            log(f"    [busca] '{q}': {str(e).splitlines()[0][:120]}")

    min_views = getattr(config, "AUTOPILOT_MIN_VIEWS", 100000)
    decay = getattr(config, "AUTOPILOT_RECENCY_DECAY", 0.97)
    uniq = {}
    for c in cands:
        views = c.get("view_count")
        if c["id"] in seen or c["id"] in uniq or not views or views < min_views:
            continue  # sem número de views = estreia/membros/live: pula
        if not _duration_ok(c.get("duration")):
            continue
        # muitas views primeiro; leve preferência pelos mais novos do canal;
        # canal já usado nas últimas 24h perde prioridade (varia a fonte)
        c["score"] = views * (decay ** c.get("rank", 0)) * (0.6 ** recent_channels.get(c.get("channel"), 0))
        uniq[c["id"]] = c
    if not uniq:
        if not fails or cands:
            log(f"    [busca] nenhum vídeo ainda não usado com {min_views:,} views ou mais "
                f"nos canais configurados.".replace(",", "."))
        return None

    max_age_d = getattr(config, "AUTOPILOT_MAX_AGE_DAYS", 0) or 0
    for c in sorted(uniq.values(), key=lambda c: -c["score"])[:getattr(config, "AUTOPILOT_CHECK_TOP", 6)]:
        info = full_info(c["id"])
        if info:
            if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                continue
            if info.get("availability") not in (None, "public") or info.get("age_limit", 0) >= 18:
                continue
            if info.get("duration") and not _duration_ok(info["duration"]):
                continue
            ts = info.get("timestamp")
            if max_age_d and ts and time.time() - ts > max_age_d * 86400:
                continue
            c.update(title=info.get("title") or c["title"], channel=info.get("channel") or c["channel"],
                     timestamp=ts, view_count=info.get("view_count") or c["view_count"])
        c["url"] = f"https://www.youtube.com/watch?v={c['id']}"
        age = f", publicado há {(time.time() - c['timestamp']) / 86400:.0f} dias" if c.get("timestamp") else ""
        log(f"    [busca] escolhido: \"{c['title']}\" ({c['channel']}) -- "
            f"{c['view_count']:,} views{age}".replace(",", "."))
        return c
    log("    [busca] os candidatos com mais views não passaram nos filtros (live/idade/duração); "
        "tento de novo depois.")
    return None
