"""
O vídeo postado está VISÍVEL ou foi bloqueado por direitos autorais?

Achado real: um corte do The Noite (SBT) subiu às 19:32, o piloto marcou
como postado, mas o vídeo não aparecia no canal ("This video is not
available" pra quem tentava abrir) -- bloqueio do Content ID da emissora.
O usuário só percebeu horas depois ("o último vídeo foi postado faz 5h").

Agora, de 20 min a 24 h depois de cada postagem (a cada 15 min), o piloto
pergunta pra API do YouTube (1 unidade de cota a cada 50 vídeos):
  - o upload foi rejeitado (direitos autorais, reivindicação, duplicado...)?
  - o vídeo está bloqueado no Brasil (ou em quase todo lugar)?
Se sim:
  - avisa no log;
  - o canal de origem vai pra lista de canais que dão bloqueio: o piloto
    não pega mais vídeo dele e tira da fila os clipes dele que já estavam
    prontos;
  - libera a próxima postagem na hora (o horário foi desperdiçado num
    vídeo que ninguém vê).
O vídeo bloqueado não é apagado sozinho: dá pra apagar no YouTube Studio.
"""
import time
from typing import Optional

_CHECK_EVERY = 15 * 60
_REJECT_STATUS = ("rejected", "failed", "deleted")


def _blocked_reason(item: dict) -> Optional[str]:
    st = item.get("status", {}) or {}
    up = st.get("uploadStatus")
    if up in _REJECT_STATUS:
        why = st.get("rejectionReason") or st.get("failureReason") or up
        return {"claim": "reivindicação de direitos autorais (Content ID)",
                "copyright": "direitos autorais", "duplicate": "vídeo duplicado",
                "inappropriate": "conteúdo impróprio", "termsOfUse": "termos de uso",
                "trademark": "marca registrada", "legal": "questão legal"}.get(why, why)
    rr = (item.get("contentDetails", {}) or {}).get("regionRestriction") or {}
    blocked = rr.get("blocked") or []
    allowed = rr.get("allowed")
    if "BR" in blocked or (allowed is not None and "BR" not in allowed) or len(blocked) >= 50:
        return "bloqueado no Brasil (direitos autorais)"
    return None


def check_recent_posts(state, service, log) -> int:
    """Confere os vídeos postados nas últimas 24 h. Devolve quantos estão bloqueados."""
    if service is None:
        return 0
    now = time.time()
    if now - state.data.get("rights_checked_at", 0) < _CHECK_EVERY:
        return 0
    # reivindicação do Content ID pode chegar horas depois: confere de novo
    # a cada 15 min durante 24 h
    todo = [p for p in state.data["posted"]
            if p.get("youtube_id") and not p.get("blocked")
            and 20 * 60 <= now - p.get("at", 0) <= 24 * 3600]
    state.data["rights_checked_at"] = now
    if not todo:
        state.save()
        return 0
    try:
        got = {}
        ids = [p["youtube_id"] for p in todo]
        for i in range(0, len(ids), 50):
            r = service.videos().list(part="status,contentDetails", id=",".join(ids[i:i + 50])).execute()
            for it in r.get("items") or []:
                got[it["id"]] = it
    except Exception as e:  # conferência nunca derruba o piloto
        log(f"  [direitos] não consegui conferir os vídeos postados agora ({e.__class__.__name__}).")
        state.save()
        return 0
    bad = 0
    for p in todo:
        it = got.get(p["youtube_id"])
        if it is None:
            if now - p["at"] > 2 * 3600:
                p["blocked"] = "vídeo sumiu do canal (removido)"
            else:
                continue
        else:
            reason = _blocked_reason(it)
            if reason is None:
                continue
            p["blocked"] = reason
        bad += 1
        ch = p.get("channel") or "?"
        bc = state.data.setdefault("bad_channels", {})
        bc[ch] = bc.get(ch, 0) + 1
        gone = [c for c in state.data["queue"] if _queue_channel(state, c) == ch]
        state.data["queue"] = [c for c in state.data["queue"] if c not in gone]
        replace_now = now - p["at"] < 6 * 3600
        if replace_now:
            state.data["last_post_at"] = 0  # o horário foi perdido: posta outro já
        log(f"  [!] O YouTube bloqueou o vídeo \"{p.get('title', '')}\" (https://youtube.com/shorts/"
            f"{p['youtube_id']}): {p['blocked']}.")
        log(f"      Canal de origem \"{ch}\" vai pra lista de canais que dão bloqueio: não pego mais vídeo "
            f"dele" + (f" e tirei {len(gone)} clipe(s) dele da fila" if gone else "")
            + "." + (" Posto outro no lugar agora." if replace_now else "")
            + " (O bloqueado pode ser apagado no YouTube Studio.)")
    state.save()
    return bad


def _queue_channel(state, item: dict) -> str:
    src = state.data["sources"].get(item.get("source_id") or "", {})
    return item.get("channel") or src.get("channel") or "?"
