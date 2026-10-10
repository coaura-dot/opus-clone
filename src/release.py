"""
Liberação automática dos vídeos que subiram PRIVADOS.

Enquanto o projeto da API não passa pela auditoria do YouTube, todo vídeo
enviado pela API fica travado como privado. Este módulo:

  1. confere uma vez por dia (logo que o piloto liga e, depois, a partir de
     AUTOPILOT_RELEASE_CHECK_HOUR) se a aprovação já saiu: tenta deixar
     público o vídeo privado mais antigo e confere se mudou de verdade.
     Um vídeo novo que já sobe público também conta como sinal de aprovação.
  2. aprovado, solta os privados DE POUCO EM POUCO: no máximo
     AUTOPILOT_RELEASE_PER_DAY por dia, com AUTOPILOT_RELEASE_MINUTES_BETWEEN
     entre um e outro, e só no horário de postagem.
  3. se um vídeo antigo continuar travado mesmo depois da aprovação (o
     YouTube pode manter travado o que subiu antes dela), reposta o arquivo
     que ficou no PC, já público, e marca o privado antigo como substituído
     (pra você apagar no YouTube Studio quando quiser).

Mudar a visibilidade precisa da permissão "gerenciar sua conta do YouTube",
que o login antigo não tinha: rode o login (menu, opção 3) uma vez.
"""
import time
from datetime import datetime
from pathlib import Path

from . import config

MANAGE_SCOPE = "https://www.googleapis.com/auth/youtube"


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def privacy_of(service, ids: list) -> dict:
    """{id: "public"/"private"/"unlisted"} -- 1 unidade de cota a cada 50."""
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = service.videos().list(part="status", id=",".join(chunk)).execute()
        for it in r.get("items") or []:
            out[it["id"]] = it.get("status", {}).get("privacyStatus")
    return out


def make_public(service, video_id: str) -> bool:
    """Tenta deixar público e CONFERE (o YouTube pode aceitar o pedido e
    manter o vídeo travado como privado)."""
    from googleapiclient.errors import HttpError
    from .youtube_uploader import AuthError, QuotaExceeded, _error_reason
    body = {"id": video_id, "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False,
                                       "embeddable": True}}
    try:
        service.videos().update(part="status", body=body).execute()
    except HttpError as e:
        reason = _error_reason(e)
        if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
            raise QuotaExceeded(f"limite do YouTube atingido ({reason})") from e
        if reason in ("insufficientPermissions", "authError") or getattr(e.resp, "status", 0) == 401:
            raise AuthError("o login não tem permissão pra mudar a visibilidade dos vídeos -- "
                            "faça o login de novo (menu, opção 3)") from e
        if reason in ("videoNotFound", "notFound"):
            return False
        return False  # recusado (vídeo travado/bloqueado): trata como "ainda privado"
    return privacy_of(service, [video_id]).get(video_id) == "public"


def _pending(state) -> list:
    return [p for p in state.data["posted"]
            if p.get("privacy") != "public" and not p.get("replaced") and not p.get("gave_up")]


def _rstate(state) -> dict:
    r = state.data.setdefault("release", {})
    r.setdefault("approved", False)
    r.setdefault("last_probe_day", "")
    r.setdefault("day", "")
    r.setdefault("count", 0)
    r.setdefault("last_at", 0)
    r.setdefault("warned_day", "")
    if r["day"] != _today():
        r.update(day=_today(), count=0)
    return r


def refresh_unknown(state, service) -> None:
    """Vídeos postados antes deste módulo não têm a visibilidade anotada."""
    unknown = [p for p in state.data["posted"] if "privacy" not in p and p.get("youtube_id")]
    if not unknown:
        return
    st = privacy_of(service, [p["youtube_id"] for p in unknown])
    for p in unknown:
        p["privacy"] = st.get(p["youtube_id"], "removido")
        if p["privacy"] == "removido":
            p["gave_up"] = True
    state.save()


def release_step(state, service, log, startup: bool = False, force: bool = False) -> bool:
    """Uma rodada de liberação. True se fez alguma coisa no YouTube."""
    from .autopilot import _in_post_window
    from .youtube_uploader import AuthError
    if not getattr(config, "AUTOPILOT_RELEASE_ENABLED", True) or service is None:
        return False
    if not state.data["posted"]:
        return False
    r = _rstate(state)
    refresh_unknown(state, service)
    pend = _pending(state)
    if not pend:
        return False
    if r["count"] >= getattr(config, "AUTOPILOT_RELEASE_PER_DAY", 3) and not force:
        return False
    if not getattr(service, "autoclipper_can_manage", True):
        if r["warned_day"] != _today():
            r["warned_day"] = _today()
            state.save()
            log(f"  [!] {len(pend)} vídeo(s) privado(s) esperando liberação, mas o login não tem a "
                "permissão de mudar visibilidade -- faça o login de novo (menu, opção 3).")
        return False

    pend.sort(key=lambda p: p.get("at", 0))
    oldest = pend[0]
    now = time.time()
    # vídeo antigo com o título-gancho no começo (src/hook_check.py): fica privado
    from .autopilot import _old_hook
    if oldest.get("video") and Path(oldest["video"]).exists() and _old_hook(oldest):
        oldest["gave_up"] = oldest["old_hook"] = True
        state.save()
        log(f"    o vídeo {oldest['youtube_id']} é antigo, com o título no começo -- fica privado.")
        return True

    # sinal passivo: um vídeo mais novo que os privados já subiu público
    if not r["approved"] and any(p.get("privacy") == "public" and p.get("at", 0) > oldest.get("at", 0)
                                 for p in state.data["posted"]):
        r["approved"] = True
        log("  ✓ Auditoria aprovada (os vídeos novos já sobem públicos). "
            f"Vou liberar os {len(pend)} privado(s) aos poucos.")
        state.save()

    if not r["approved"]:
        hour = getattr(config, "AUTOPILOT_RELEASE_CHECK_HOUR", 10)
        due = r["last_probe_day"] != _today() and (startup or force or datetime.now().hour >= hour)
        if not due:
            return False
        r["last_probe_day"] = _today()
        state.save()
        log(f"  >> Conferindo se a auditoria do YouTube já saiu ({len(pend)} vídeo(s) privado(s))...")
        try:
            ok = make_public(service, oldest["youtube_id"])
        except AuthError as e:
            service.autoclipper_can_manage = False
            log(f"  [!] {e}")
            return False
        if not ok:
            log(f"    ainda não: o vídeo continua privado. Confiro de novo amanhã "
                f"({'ao ligar ou ' if not startup else ''}a partir das {hour}h).")
            return True
        r["approved"] = True
        _mark_public(state, r, oldest, log, first=True)
        return True

    # aprovado: solta de pouco em pouco
    gap = getattr(config, "AUTOPILOT_RELEASE_MINUTES_BETWEEN", 120) * 60
    if not force and (not _in_post_window() or now - r["last_at"] < gap):
        return False
    try:
        ok = make_public(service, oldest["youtube_id"])
    except AuthError as e:
        service.autoclipper_can_manage = False
        log(f"  [!] {e}")
        return False
    if ok:
        _mark_public(state, r, oldest, log)
        return True
    _handle_locked(state, r, oldest, log)
    return True


def _mark_public(state, r, p, log, first: bool = False):
    p["privacy"] = "public"
    p["released_at"] = time.time()
    r["count"] += 1
    r["last_at"] = time.time()
    state.save()
    left = len(_pending(state))
    if first:
        log("  ✓ Auditoria aprovada! Vídeos privados liberando aos poucos.")
    log(f"    agora público: https://youtube.com/shorts/{p['youtube_id']} \"{p.get('title', '')}\" "
        f"-- faltam {left}")


def _handle_locked(state, r, p, log):
    """Aprovado, mas este vídeo antigo ficou travado: reposta o arquivo."""
    video = p.get("video")
    r["last_at"] = time.time()
    if getattr(config, "AUTOPILOT_REUPLOAD_LOCKED", True) and video and Path(video).exists():
        meta = Path(video).with_suffix(".meta.json")
        if meta.exists():
            state.data["queue"].insert(0, {"video": video, "meta": str(meta), "score": 10 ** 5, "priority": 1,
                                           "source_id": p.get("source_id"), "added": time.time(),
                                           "video_sem_musica": p.get("video_sem_musica"),
                                           "replaces": p["youtube_id"]})
            p["replaced"] = True
            state.save()
            log(f"    o vídeo {p['youtube_id']} ficou travado como privado mesmo com a aprovação -- "
                f"vou repostar o arquivo como público (o privado antigo pode ser apagado no YouTube Studio).")
            return
    p["gave_up"] = True
    state.save()
    log(f"    o vídeo {p['youtube_id']} ficou travado como privado e o arquivo não está mais no PC "
        f"-- esse fica privado.")
