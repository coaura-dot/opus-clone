"""
Vários canais do YouTube: o piloto alterna entre eles pra postar.

Pedido do usuário (v46): "postar em 6 contas diferentes, 1 vídeo por hora,
alternando" -- com 4 Shorts por dia em cada canal, 6 canais dão 24 por dia,
1 por hora no total, e cada canal posta 1 a cada 6 h (o ritmo que funcionou
no canal principal; com 1 por hora NUM canal só o YouTube parou de entregar,
ver src/reach.py).

Registro: credentials/contas.json (na pasta do login, fora do git):
  [{"id": "c1", "name": "DontStopMeNow", "channel_id": "UC...",
    "token": "youtube_token.json", "on": true}, ...]
  - c1 é o canal de antes (credentials/youtube_token.json);
  - os novos ganham o id do próprio canal (UC...) e o login em
    youtube_token_<id>.json. Conectar de novo um canal que já está na lista
    só troca o login dele.

Regras que o programa segue (do YouTube, não dele):
  - cada clipe vai pra UM canal só: o mesmo vídeo em vários canais é spam
    pelas regras do YouTube ("conteúdo repetido em um ou mais canais");
  - todos os canais usam o MESMO projeto do Google Cloud (client_secret.json).
    A cota de upload é do projeto (100 uploads por dia hoje); dividir em
    vários projetos pra ganhar cota é proibido pelas regras da API.
"""
import json
import os
import time
from pathlib import Path
from typing import List, Optional

FIRST = "c1"            # o canal de antes (posts antigos sem "account" são dele)
LEGACY_TOKEN = "youtube_token.json"


def _dir() -> Path:
    from .youtube_uploader import credentials_dir
    return credentials_dir()


def path() -> Path:
    return _dir() / "contas.json"


def mtime() -> Optional[float]:
    try:
        return path().stat().st_mtime
    except OSError:
        return None


def load() -> List[dict]:
    """Canais cadastrados (na ordem em que foram adicionados)."""
    p = path()
    if p.exists():
        try:
            accs = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(accs, list):
                return [a for a in accs if isinstance(a, dict) and a.get("id") and a.get("token")]
        except (OSError, ValueError):
            pass
    # antes da v46: um canal só, em youtube_token.json
    if (_dir() / LEGACY_TOKEN).exists():
        return [{"id": FIRST, "name": None, "channel_id": None, "token": LEGACY_TOKEN, "on": True}]
    return []


def save(accs: List[dict]) -> None:
    p = path()
    tmp = p.with_suffix(".tmp")
    text = json.dumps(accs, ensure_ascii=False, indent=1)
    for i in range(40):  # Windows: o arquivo pode estar aberto pela interface/piloto agora
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.25)


def active() -> List[dict]:
    return [a for a in load() if a.get("on", True)]


def get(acc_id: str) -> Optional[dict]:
    return next((a for a in load() if a["id"] == acc_id), None)


def token_file(acc: dict) -> Path:
    return _dir() / acc["token"]


def label(acc) -> str:
    """Nome pra mostrar (dict da conta ou id)."""
    if isinstance(acc, str):
        acc = get(acc) or {"id": acc}
    return acc.get("name") or ("canal principal" if acc.get("id") == FIRST else acc.get("id", "?"))


def set_on(acc_id: str, on: bool) -> None:
    accs = load()
    for a in accs:
        if a["id"] == acc_id:
            a["on"] = bool(on)
    save(accs)


def remove(acc_id: str) -> Optional[dict]:
    """Tira o canal da lista e apaga o login dele deste PC (os vídeos já
    postados continuam no canal)."""
    accs = load()
    gone = next((a for a in accs if a["id"] == acc_id), None)
    if gone is None:
        return None
    save([a for a in accs if a["id"] != acc_id])
    try:
        token_file(gone).unlink()
    except OSError:
        pass
    return gone


def learn(acc_id: str, channel_id: Optional[str], name: Optional[str]) -> Optional[str]:
    """Anota o canal de verdade de um login (o piloto descobre ao conectar).
    Devolve um problema se esse canal já está em OUTRA conta da lista."""
    accs = load()
    me = next((a for a in accs if a["id"] == acc_id), None)
    if me is None:
        return None
    other = next((a for a in accs if a["id"] != acc_id and channel_id and a.get("channel_id") == channel_id), None)
    if other is not None:
        return f"é o mesmo canal da conta \"{label(other)}\" (ficou só uma)"
    if me.get("channel_id") != channel_id or (name and me.get("name") != name):
        me["channel_id"] = channel_id or me.get("channel_id")
        me["name"] = name or me.get("name")
        save(accs)
    return None


# ------------------------------------------------------------ login ---
def login(out=print) -> Optional[dict]:
    """Abre o navegador pra escolher a conta/canal e autorizar. Canal novo
    entra na lista; canal que já estava só troca o login."""
    from . import youtube_uploader as yt
    accs = load()
    # canal antigo (antes da v46) ainda sem o id anotado: descobre agora, pra
    # não cadastrar o mesmo canal duas vezes
    for a in accs:
        if not a.get("channel_id"):
            try:
                svc = yt.get_service(interactive=False, token=token_file(a))
                cid, name, problem = yt.channel_info(svc)
                if not problem:
                    a["channel_id"], a["name"] = cid, name
            except Exception:
                pass
    creds = yt.new_login()
    svc = yt.service_from(creds)
    cid, name, problem = yt.channel_info(svc)
    if problem:
        out(f"\n[!] Login feito, mas: {problem}")
        return None
    acc = next((a for a in accs if a.get("channel_id") == cid), None)
    if acc is None:
        # o canal de antes com o login morto (não deu pra saber qual é): o
        # primeiro login depois disso é ele (era o que "Conectar canal" fazia)
        unknown = [a for a in accs if not a.get("channel_id")]
        acc = unknown[0] if len(unknown) == 1 and len(accs) == 1 else None
    new = acc is None
    if new:
        acc = {"id": cid, "token": f"youtube_token_{cid}.json", "on": True, "added": time.time()}
        accs.append(acc)
    acc.update(channel_id=cid, name=name, on=True)
    (_dir() / acc["token"]).write_text(creds.to_json(), encoding="utf-8")
    save(accs)
    out(f"\nConectado ao canal: {name}")
    n = sum(1 for a in accs if a.get("on", True))
    if new and n > 1:
        out(f"Canal novo na lista: agora são {n} canais. O piloto alterna entre eles pra postar.")
        from . import control
        per = control.posts_per_day()
        if n == 2 and (per is None or per > MULTI_PER_DAY):
            control.set_posts_per_day(MULTI_PER_DAY)
            out(f"Com vários canais: {MULTI_PER_DAY} Shorts por dia em CADA canal (muda no − / + da interface).")
    elif not new:
        out("Esse canal já estava na lista: login atualizado.")
    return acc


MULTI_PER_DAY = 4   # com 6 canais: 24 por dia, 1 por hora no total


# --------------------------------------------- estado de cada canal ---
def acc_of(p: dict) -> str:
    return p.get("account") or FIRST


def sub(data, acc_id: str) -> dict:
    """O pedaço do state.json de UM canal (último post, freio, bloqueio...).
    `data`: State ou o dict do state.json."""
    data = getattr(data, "data", data)
    return data.setdefault("accounts", {}).setdefault(acc_id, {})


def posted_by(data, acc_id: str) -> list:
    data = getattr(data, "data", data)
    return [p for p in data.get("posted", []) if acc_of(p) == acc_id]


_PER_CHANNEL_KEYS = ("uploads_playlist", "reach_brake", "reach_checked_at", "rights_checked_at", "auth_down")


def migrate_state(state) -> bool:
    """state.json de antes da v46 (um canal só): o que era do canal vai pra
    conta c1. Devolve True se mudou."""
    if "accounts" in state.data:
        return False
    s = sub(state, FIRST)
    for k in _PER_CHANNEL_KEYS:
        if k in state.data:
            s[k] = state.data.pop(k)
    s["last_post_at"] = state.data.get("last_post_at", 0)
    state.save()
    return True
