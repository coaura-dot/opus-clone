"""
Antes de postar: esse vídeo já está no canal?

Pedido do usuário (modo só postar): "ele sempre verifica no canal se já
postou o vídeo em questão". Vale pra toda postagem, em qualquer modo.

Como confere, antes de CADA upload:
  1. o histórico do próprio piloto (state.json): mesmo título já postado;
  2. o CANAL de verdade, pela API do YouTube: os ~200 vídeos mais recentes
     da lista de uploads (inclui os privados e os postados à mão pelo
     YouTube Studio/celular). Custa 1 unidade de cota por página de 50 --
     nada perto das 1.600 de um upload.
Mesmo título (sem diferença de maiúscula, acento ou pontuação; título
cortado no limite de 100 letras também conta) = já postado: o clipe sai da
fila e o piloto passa pro próximo.

Se o YouTube não responder a conferência, o piloto NÃO posta às cegas:
espera 10 min e confere de novo.
"""
import re
import time
import unicodedata
from typing import Optional

_PAGES = 4  # 4 x 50 = os 200 uploads mais recentes


class CheckFailed(Exception):
    """Não deu pra conferir o canal agora."""


def norm(title: str) -> str:
    t = unicodedata.normalize("NFKD", title or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def same_title(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = sorted((na, nb), key=len)
    # o YouTube corta título com mais de 100 letras
    return len(short) >= 60 and long_.startswith(short)


def _uploads_playlist(service, state) -> str:
    pid = state.data.get("uploads_playlist")
    if pid:
        return pid
    r = service.channels().list(part="contentDetails", mine=True).execute()
    items = r.get("items") or []
    if not items:
        raise CheckFailed("o login do YouTube não tem canal")
    pid = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    state.data["uploads_playlist"] = pid
    state.save()
    return pid


def channel_videos(service, state, pages: int = _PAGES) -> list:
    """[(título, id do vídeo, publicado em)] dos uploads mais recentes do canal."""
    try:
        pid = _uploads_playlist(service, state)
        out, token = [], None
        for _ in range(pages):
            kw = {"part": "snippet", "playlistId": pid, "maxResults": 50}
            if token:
                kw["pageToken"] = token
            r = service.playlistItems().list(**kw).execute()
            for it in r.get("items") or []:
                sn = it.get("snippet") or {}
                vid = (sn.get("resourceId") or {}).get("videoId")
                if vid:
                    out.append((sn.get("title") or "", vid, sn.get("publishedAt") or ""))
            token = r.get("nextPageToken")
            if not token:
                break
        return out
    except CheckFailed:
        raise
    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower():  # cota do dia acabou: o upload também não passaria
            raise CheckFailed("quotaExceeded: a cota da API do YouTube de hoje acabou")
        raise CheckFailed(f"{e.__class__.__name__}: {msg[:120]}")


def already_posted(service, state, title: str) -> Optional[dict]:
    """{"id", "at", "where"} do vídeo com esse título que já está no canal,
    ou None. CheckFailed se não deu pra conferir (aí não posta agora)."""
    for p in reversed(state.data.get("posted", [])):
        if p.get("youtube_id") and same_title(p.get("title", ""), title):
            return {"id": p["youtube_id"], "at": p.get("at") or time.time(), "where": "histórico do piloto"}
    for t, vid, iso in channel_videos(service, state):
        if same_title(t, title):
            return {"id": vid, "at": _published_ts(iso) or time.time(), "where": "canal"}
    return None


def _published_ts(iso: str) -> Optional[float]:
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return None
