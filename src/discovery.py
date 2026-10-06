"""
Modo automático: acha o próximo vídeo pra cortar.

De onde vêm os candidatos (config.py, seção "Postagem automática"):
  - AUTOPILOT_CHANNELS: canais que você acompanha (os últimos vídeos de
    cada um). É a fonte principal: use canais que LIBERAM cortes.
  - AUTOPILOT_SEARCHES: buscas no YouTube filtradas por "esta semana,
    mais vistos" -- acha vídeos bombando fora da sua lista.

Como escolhe: o vídeo que está ganhando views MAIS RÁPIDO agora
(views por hora desde a publicação), entre os publicados há no máximo
AUTOPILOT_MAX_AGE_DAYS dias, com duração boa pra render cortes, que não
seja live acontecendo, e que ainda não foi usado.

Tudo pelo yt-dlp, sem chave de API e sem gastar a cota do YouTube (que
fica toda pros uploads).
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


def _ydl(flat: bool):
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extractor_args": {"youtube": {"lang": ["pt"]}},
            "ignore_no_formats_error": True, "http_headers": _HEADERS,
            "socket_timeout": 30}
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
    for e in (info or {}).get("entries") or []:
        if e and e.get("id"):
            out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                        "view_count": e.get("view_count"), "channel": info.get("channel") or channel,
                        "origin": "canal"})
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
        with _ydl(flat=False) as y:
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
            cands += list_channel(ch, getattr(config, "AUTOPILOT_VIDEOS_PER_CHANNEL", 10))
        except Exception as e:
            fails += 1
            log(f"    [busca] canal {ch}: {str(e).splitlines()[0][:120]}")
    for q in getattr(config, "AUTOPILOT_SEARCHES", []):
        try:
            cands += [c for c in search(q) if looks_portuguese(c["title"])]
        except Exception as e:
            fails += 1
            log(f"    [busca] '{q}': {str(e).splitlines()[0][:120]}")

    uniq = {}
    for c in cands:
        if c["id"] in seen or c["id"] in uniq or not _duration_ok(c.get("duration")):
            continue
        uniq[c["id"]] = c
    if not uniq:
        if cands or not fails:
            log("    [busca] nenhum vídeo novo nos canais/buscas configurados.")
        return None

    # pré-seleção pelas views (a listagem rápida não traz a data) e depois
    # confere a data de cada um pra medir a velocidade de views
    pre = sorted(uniq.values(), key=lambda c: c.get("view_count") or 0, reverse=True)
    max_age_h = getattr(config, "AUTOPILOT_MAX_AGE_DAYS", 7) * 24
    min_views = getattr(config, "AUTOPILOT_MIN_VIEWS", 5000)
    best, best_score = None, -1.0
    for c in pre[:getattr(config, "AUTOPILOT_CHECK_TOP", 8)]:
        info = full_info(c["id"])
        if not info:
            continue
        if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
            continue
        if info.get("availability") not in (None, "public"):
            continue
        if info.get("age_limit", 0) >= 18:
            continue
        dur = info.get("duration") or c.get("duration")
        if dur and not _duration_ok(dur):
            continue
        views = info.get("view_count") or c.get("view_count") or 0
        ts = info.get("timestamp")
        age_h = max((time.time() - ts) / 3600.0, 1.0) if ts else None
        if age_h is None or age_h > max_age_h or views < min_views:
            continue
        lang = (info.get("language") or "")[:2]
        if lang and lang != "pt" and c["origin"] != "canal":
            continue
        velocity = views / max(age_h, 2.0)
        channel = info.get("channel") or c.get("channel") or ""
        # varia a fonte: canal já usado nas últimas 24h perde prioridade
        score = velocity * (0.6 ** recent_channels.get(channel, 0))
        c.update(info, title=info.get("title") or c["title"], channel=channel,
                 view_count=views, age_hours=round(age_h, 1), velocity=round(velocity),
                 score=score, duration=dur)
        if score > best_score:
            best, best_score = c, score
    if best:
        best["url"] = f"https://www.youtube.com/watch?v={best['id']}"
        n = lambda v: f"{v:,}".replace(",", ".")
        log(f"    [busca] escolhido: \"{best['title']}\" ({best['channel']}) -- "
            f"{n(best['view_count'])} views em {best['age_hours']:.0f}h "
            f"(~{n(best['velocity'])}/h, via {best['origin']})")
    else:
        log("    [busca] nenhum candidato passou nos filtros (recente/views/duração).")
    return best
