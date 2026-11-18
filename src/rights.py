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
  - o canal de origem fica em PAUSA: 7 dias no 1º bloqueio, 30 dias se for
    o 2º em 30 dias. Sai da fila o que é do mesmo vídeo (no 2º bloqueio,
    tudo do canal);
  - libera a próxima postagem na hora (o horário foi desperdiçado num
    vídeo que ninguém vê).
Antes (até a v42) um bloqueio só banía o canal PRA SEMPRE -- e Short que o
próprio usuário apagou no Studio ("sumiu do canal") também contava. Pedido
do usuário: "permitir clipagem de todos estes" (Flow, Inteligência Ltda e
os outros da lista). Agora vídeo que sumiu só vai pro log, e os banimentos
antigos foram liberados (migrate()).
O vídeo bloqueado não é apagado sozinho: dá pra apagar no YouTube Studio.
"""
import time
from typing import Optional

_CHECK_EVERY = 15 * 60
_REJECT_STATUS = ("rejected", "failed", "deleted")
_PAUSE_FIRST = 7 * 86400      # 1º bloqueio: o canal descansa 7 dias
_PAUSE_REPEAT = 30 * 86400    # 2º bloqueio em 30 dias: 30 dias
_REPEAT_WINDOW = 30 * 86400


def migrate(state, log) -> None:
    """Banimentos permanentes antigos (contagem por canal) viram passado:
    todos os canais da lista voltam a valer."""
    bc = state.data.get("bad_channels") or {}
    old = sorted(ch for ch, v in bc.items() if not isinstance(v, list))
    if old:
        for ch in old:
            bc[ch] = []
        state.save()
        log(f"  [direitos] canais liberados de novo (antes um bloqueio só banía pra sempre): {', '.join(old)}.")


def avoid_channels(state) -> set:
    """Canais em pausa por bloqueio de direitos agora."""
    now = time.time()
    return {ch for ch, until in (state.data.get("channel_pause") or {}).items() if until > now}


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


def check_recent_posts(state, service, log, acc: str = None) -> int:
    """Confere os vídeos postados nas últimas 24 h no canal `acc` (cada canal
    com o login dele: o motivo do bloqueio só aparece pro dono). Devolve
    quantos estão bloqueados."""
    from .accounts import FIRST, posted_by, sub
    if service is None:
        return 0
    acc = acc or FIRST
    me = sub(state, acc)
    now = time.time()
    if now - me.get("rights_checked_at", 0) < _CHECK_EVERY:
        return 0
    # reivindicação do Content ID pode chegar horas depois: confere de novo
    # a cada 15 min durante 24 h
    todo = [p for p in posted_by(state, acc)
            if p.get("youtube_id") and not p.get("blocked")
            and 20 * 60 <= now - p.get("at", 0) <= 24 * 3600]
    me["rights_checked_at"] = now
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
            # sumiu: pode ter sido o próprio usuário apagando no Studio -- não
            # é bloqueio de direitos, não pausa o canal
            if now - p["at"] > 2 * 3600 and not p.get("gone_logged"):
                p["gone_logged"] = True
                log(f"  [direitos] o vídeo \"{p.get('title', '')}\" não está mais no canal (apagado?).")
            continue
        reason = _blocked_reason(it)
        if reason is None:
            continue
        p["blocked"] = reason
        bad += 1
        ch = p.get("channel") or "?"
        bc = state.data.setdefault("bad_channels", {})
        hist = bc.get(ch) if isinstance(bc.get(ch), list) else []
        hist = [t for t in hist if now - t < _REPEAT_WINDOW] + [now]
        bc[ch] = hist
        repeat = len(hist) >= 2
        pause = _PAUSE_REPEAT if repeat else _PAUSE_FIRST
        state.data.setdefault("channel_pause", {})[ch] = now + pause
        src = p.get("source_id")
        gone = [c for c in state.data["queue"]
                if (repeat and _queue_channel(state, c) == ch) or (src and c.get("source_id") == src)]
        state.data["queue"] = [c for c in state.data["queue"] if c not in gone]
        replace_now = now - p["at"] < 6 * 3600
        if replace_now:  # o horário foi perdido: posta outro já
            me["last_post_at"] = 0
            state.data["last_post_at"] = 0
        log(f"  [!] O YouTube bloqueou o vídeo \"{p.get('title', '')}\" (https://youtube.com/shorts/"
            f"{p['youtube_id']}): {p['blocked']}.")
        log(f"      Canal de origem \"{ch}\" fica {pause // 86400} dias sem vídeo novo"
            + (" (2º bloqueio em 30 dias)" if repeat else "")
            + (f"; tirei {len(gone)} clipe(s) " + ("dele" if repeat else "do mesmo vídeo") + " da fila" if gone else "")
            + "." + (" Posto outro no lugar agora." if replace_now else "")
            + " (O bloqueado pode ser apagado no YouTube Studio.)")
    state.save()
    return bad


def _queue_channel(state, item: dict) -> str:
    src = state.data["sources"].get(item.get("source_id") or "", {})
    return item.get("channel") or src.get("channel") or "?"
