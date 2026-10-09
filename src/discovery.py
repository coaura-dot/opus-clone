"""
Modo automático: acha o próximo vídeo pra cortar.

De onde vêm os candidatos (config.py, seção "Postagem automática"):
  - AUTOPILOT_CHANNELS: podcasts/youtubers famosos que você acompanha (os
    últimos AUTOPILOT_VIDEOS_PER_CHANNEL vídeos de cada um).
  - AUTOPILOT_SEARCHES: buscas extras no YouTube ("esta semana, mais
    vistos").

Como escolhe: só vídeo RECENTE (publicado nos últimos
AUTOPILOT_MAX_AGE_DAYS dias -- pedido do usuário: "tá escolhendo vídeo muito
velho"), e entre eles o que está ganhando views mais rápido (views por dia),
variando o canal, com duração boa pra render cortes, que não seja live, que
não tenha palavra bloqueada (AUTOPILOT_BLOCK_WORDS: programa de zoeira) e
que ainda não foi usado.

Canal em outro idioma: "link_do_canal|en" na lista -- o vídeo escolhido leva
o idioma junto, pra transcrição e legenda saírem no idioma certo.

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
    # sleep_interval_requests: uma pausa entre as consultas -- o YouTube passou
    # a bloquear o IP (HTTP 429 / "not a bot") de quem consulta em rajada
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "ignore_no_formats_error": True, "socket_timeout": 30,
            "sleep_interval_requests": getattr(config, "YTDLP_SLEEP_REQUESTS", 1.0)}
    if lang:
        opts["extractor_args"] = {"youtube": {"lang": [lang]}}
        opts["http_headers"] = _HEADERS
    if flat:
        opts["extract_flat"] = "in_playlist"
    from .downloader import cookies_file
    cookie_txt = cookies_file()
    cookies = getattr(config, "YTDLP_COOKIES_FROM_BROWSER", None)
    if cookie_txt:
        opts["cookiefile"] = str(cookie_txt)
    elif cookies:
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


def _split_spec(spec: str):
    """"@canal" ou "link|en" -> (canal, idioma ou None)."""
    if "|" in spec:
        ch, lang = spec.rsplit("|", 1)
        return ch.strip(), (lang.strip() or None)
    return spec.strip(), None


def blocked(*texts) -> Optional[str]:
    words = [w.lower() for w in getattr(config, "AUTOPILOT_BLOCK_WORDS", []) if w]
    joined = " ".join(t or "" for t in texts).lower()
    return next((w for w in words if w in joined), None)


_LIST_CACHE: dict = {}
_LIST_TTL = 90 * 60


def list_channel(channel: str, n: int = 10) -> List[dict]:
    """Últimos `n` vídeos do canal (guarda por 90 min: canal não lança vídeo
    a cada meia hora, e cada consulta a menos é menos chance de bloqueio)."""
    hit = _LIST_CACHE.get((channel, n))
    if hit and time.time() - hit[0] < _LIST_TTL:
        return [dict(c) for c in hit[1]]
    out = _list_channel(channel, n)
    _LIST_CACHE[(channel, n)] = (time.time(), out)
    return [dict(c) for c in out]


def _list_channel(channel: str, n: int = 10) -> List[dict]:
    channel, lang = _split_spec(channel)
    with _ydl(flat=True) as y:
        y.params["playlistend"] = n
        info = y.extract_info(_channel_videos_url(channel), download=False)
    out = []
    for rank, e in enumerate((info or {}).get("entries") or []):
        if e and e.get("id"):
            out.append({"id": e["id"], "title": e.get("title") or "", "duration": e.get("duration"),
                        "view_count": e.get("view_count"), "channel": info.get("channel") or channel,
                        "origin": "canal", "rank": rank, "lang": lang})
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


# convidado com profissão/autoridade no título ("DIRETOR DE MERCADOS GLOBAIS
# DA HOTEIS.COM - Juan Pasquel", "NEUROCIENTISTA explica...") rende corte com
# conteúdo; título de zoeira/desafio, não (pedido do usuário: podcasts com
# gente inteligente, convidados interessantes)
_EXPERT_RE = re.compile(
    r"\b(cientista|neurocientista|f[íi]sic[oa]|qu[íi]mic[oa]|bi[óo]log[oa]|astr[ôo]nom[oa]|astronauta|"
    r"m[ée]dic[oa]|psiquiatra|psic[óo]log[oa]|neurologista|cardiologista|cirurgi[ãa]o|nutr[óo]log[oa]|"
    r"economista|historiador[a]?|fil[óo]sof[oa]|soci[óo]log[oa]|antrop[óo]log[oa]|professor[a]?|"
    r"doutor[a]?|phd|pesquisador[a]?|engenheir[oa]|matem[áa]tic[oa]|escritor[a]?|jornalista|"
    r"juiz[a]?|delegad[oa]|promotor[a]?|advogad[oa]|perit[oa]|militar|coronel|general|ex-?agente|"
    r"piloto|diplomata|ministr[oa]|ex-?presidente|governador[a]?|senador[a]?|deputad[oa]|"
    r"ceo|fundador[a]?|empres[áa]ri[oa]|investidor[a]?|diretor[a]?|executiv[oa]|bilion[áa]ri[oa]|"
    r"milion[áa]ri[oa]|campe[ãa]o|medalhista|ol[íi]mpic[oa]|especialista|expert|"
    r"scientist|doctor|professor|founder|ceo|economist|historian|astronaut|author)\b", re.IGNORECASE)
_WEAK_RE = re.compile(
    r"\b(desafio|challenge|respondendo (coment[áa]rios|seguidores)|q ?& ?a|unboxing)\b", re.IGNORECASE)
# (humor entra: "tente não rir", zoeira, treta não perdem mais -- pedido do usuário;
# pegadinha/trollagem continuam fora pela lista de bloqueio)
# (react e compilado NÃO entram: os reacts bons -- Orochinho, Dr Donut -- usam esses nomes)


def topic_factor(title: str) -> float:
    """Peso do assunto/convidado pelo título (1 = neutro)."""
    t = title or ""
    f = 1.0
    if _EXPERT_RE.search(t):
        f *= getattr(config, "AUTOPILOT_EXPERT_BONUS", 1.3)
    if _WEAK_RE.search(t):
        f *= getattr(config, "AUTOPILOT_WEAK_TOPIC_FACTOR", 0.7)
    return f


def length_factor(duration) -> float:
    """Episódio inteiro rende mais (mais assunto pra escolher o melhor
    trecho); vídeo curto costuma já ser um corte -- achado real: um vídeo
    de 17 min (corte do Market Makers) com trailer no começo virou 2 posts
    fracos. Reacts longos também ficam com peso cheio."""
    if not duration:
        return 1.0
    pref = getattr(config, "AUTOPILOT_PREFER_EPISODE_MINUTES", 35) * 60
    if duration >= pref:
        return 1.0
    return 0.6 + 0.4 * max(duration - 8 * 60, 0) / max(pref - 8 * 60, 1)


def _duration_ok(d) -> bool:
    if not d:
        return True  # desconhecida na listagem rápida: confere depois
    lo = getattr(config, "AUTOPILOT_MIN_SOURCE_MINUTES", 8) * 60
    hi = getattr(config, "AUTOPILOT_MAX_SOURCE_MINUTES", 240) * 60
    return lo <= d <= hi


def pick_source(seen: set, recent_channels: Optional[dict] = None, log=print,
                avoid_channels: Optional[set] = None) -> Optional[dict]:
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

    min_views = getattr(config, "AUTOPILOT_MIN_VIEWS", 20000)
    est_days = getattr(config, "AUTOPILOT_EST_DAYS_PER_VIDEO", 2.0)
    max_age_d = getattr(config, "AUTOPILOT_MAX_AGE_DAYS", 30) or 0

    def velocity(views, age_d):
        return views / (age_d + 2.0) ** 0.6  # views por dia, suavizado

    # o "normal" de cada canal: mediana das views/dia dos vídeos recentes da
    # listagem (idade estimada pela posição -- a lista vem do mais novo pro
    # mais velho). Achado na revisão: comparando views absolutas, canal
    # gigante de entretenimento (milhões) ganhava sempre e os podcasts de
    # conteúdo nunca nem eram conferidos.
    per_channel = {}
    for c in cands:
        if c.get("view_count"):
            v = velocity(c["view_count"], (c.get("rank", 0) + 1) * est_days)
            per_channel.setdefault(c.get("channel"), []).append(v)
    channel_norm = {ch: float(sorted(vs)[len(vs) // 2]) for ch, vs in per_channel.items() if vs}

    def score(c, age_d):
        v = velocity(c["view_count"], age_d)
        rel = v / max(channel_norm.get(c.get("channel"), v), 1.0)  # >1 = acima do normal do canal
        # tamanho (v: views por dia) pesa mais que o "bombando pro canal dele"
        # (rel) -- AUTOPILOT_POPULARITY_WEIGHT; canal usado nas últimas 24h
        # perde prioridade (varia a fonte)
        pw = getattr(config, "AUTOPILOT_POPULARITY_WEIGHT", 0.65)
        foreign = getattr(config, "AUTOPILOT_FOREIGN_FACTOR", 0.5) if c.get("lang") else 1.0
        return ((rel ** (1 - pw)) * (v ** pw) * foreign * (0.6 ** recent_channels.get(c.get("channel"), 0))
                * topic_factor(c.get("title")) * length_factor(c.get("duration")))

    uniq = {}
    for c in cands:
        views = c.get("view_count")
        if c["id"] in seen or c["id"] in uniq or not views or views < min_views:
            continue  # sem número de views = estreia/membros/live: pula
        if not _duration_ok(c.get("duration")):
            continue
        if blocked(c.get("title"), c.get("channel")):
            continue
        if avoid_channels and c.get("channel") in avoid_channels:
            continue  # canal que já deu bloqueio de direitos autorais (src/rights.py)
        est_age = (c.get("rank", 0) + 1) * est_days
        if max_age_d and est_age > max_age_d * 1.5:
            continue  # bem fundo na lista do canal: com certeza velho
        c["score"] = score(c, est_age)
        uniq[c["id"]] = c
    if not uniq:
        if not fails or cands:
            log(f"    [busca] nenhum vídeo ainda não usado com {min_views:,} views ou mais "
                f"nos canais configurados.".replace(",", "."))
        return None

    # os melhores pra conferir a fundo, no máximo 2 por canal (senão um canal
    # grande ocupava todas as vagas)
    shortlist, per = [], {}
    for c in sorted(uniq.values(), key=lambda c: -c["score"]):
        if per.get(c.get("channel"), 0) >= 2:
            continue
        per[c.get("channel")] = per.get(c.get("channel"), 0) + 1
        shortlist.append(c)
        if len(shortlist) >= getattr(config, "AUTOPILOT_CHECK_TOP", 10):
            break

    checked = []
    for c in shortlist:
        info = full_info(c["id"])
        if info:
            if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                continue
            if info.get("availability") not in (None, "public") or info.get("age_limit", 0) >= 18:
                continue
            if info.get("duration") and not _duration_ok(info["duration"]):
                continue
            if blocked(info.get("title"), info.get("channel")):
                continue
            if avoid_channels and info.get("channel") in avoid_channels:
                continue
            ts = info.get("timestamp")
            if max_age_d and ts and time.time() - ts > max_age_d * 86400:
                continue
            c.update(title=info.get("title") or c["title"], channel=info.get("channel") or c["channel"],
                     timestamp=ts, view_count=info.get("view_count") or c["view_count"],
                     duration=info.get("duration") or c.get("duration"))
        # idade: a data real; sem ela (consulta falhou/bloqueada), estima pela
        # posição na lista do canal (vem do mais novo pro mais velho) -- antes,
        # vídeo sem data escapava do limite de idade e ainda ganhava na nota
        if c.get("timestamp"):
            age_d = max((time.time() - c["timestamp"]) / 86400.0, 0.0)
        else:
            age_d = (c.get("rank", 0) + 1) * est_days
            c["age_estimated"] = True
            if max_age_d and age_d > max_age_d:
                continue
        c["age_days"] = age_d
        c["final"] = score(c, age_d)  # agora com a idade real
        checked.append(c)
    if not checked:
        log(f"    [busca] nenhum vídeo dos últimos {max_age_d} dias passou nos filtros "
            "(live/idade/duração/bloqueio); tento de novo depois.")
        return None
    c = max(checked, key=lambda c: c["final"])
    c["url"] = f"https://www.youtube.com/watch?v={c['id']}"
    age = (f", publicado há ~{c['age_days']:.0f} dias (estimado)" if c.get("age_estimated")
           else f", publicado há {c['age_days']:.0f} dias")
    lang = f" [{c['lang']}]" if c.get("lang") else ""
    extra = []
    if c.get("duration"):
        extra.append(f"{c['duration'] / 60:.0f} min")
    if _EXPERT_RE.search(c.get("title") or ""):
        extra.append("convidado especialista")
    log(f"    [busca] escolhido: \"{c['title']}\" ({c['channel']}){lang} -- "
        f"{c['view_count']:,} views{age}".replace(",", ".") + (f" ({', '.join(extra)})" if extra else ""))
    return c
