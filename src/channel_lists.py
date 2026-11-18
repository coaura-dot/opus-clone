"""
Lista branca (canais que o piloto corta) e lista negra (canais que ele
NUNCA corta), editáveis na interface (botão "Canais: cortar / não cortar").

Pedido do usuário (v48): "cria um lugar onde eu possa criar uma lista de
canais recomendados (white list) e uma black list de canais que não quero
clip, no programa".

Arquivo: autopilot_data/canais.json (a atualização não mexe nessa pasta):
  {"white": [{"canal": "@flowpodcast", "nome": "Flow Podcast", "idioma": null, "gameplay": false}, ...],
   "black": [{"canal": "Felipe Neto"}, ...]}
  - na primeira vez, a lista branca vem de AUTOPILOT_CHANNELS (config.py);
  - lista branca: o @ ou o link do canal (o piloto lista os vídeos novos dele);
  - lista negra: @, link ou NOME do canal ("Felipe Neto" pega também
    "Felipe Neto Clips"). Vale pra busca, pros canais da lista branca e pra
    fila: clipe já pronto de canal da lista negra sai da fila e não é postado.
O piloto relê o arquivo a cada busca: mudou na interface, vale na hora.
"""
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Optional

from . import config

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "autopilot_data" / "canais.json"

# nomes dos canais que na config são só link (pra lista não mostrar um código)
_KNOWN_NAMES = {
    "UCgSAH94ZjV6-w46hPJ0-ffQ": "orochidois (Orochinho)",
    "UC4aiJNDUviw_vMhdCq5Kq1Q": "Cortes do Casimito",
    "UCldCI5K_HmwS_iQX7xqIlPg": "DrDonut Clips",
}


def _split(spec: str):
    parts = [p.strip() for p in spec.split("|")]
    flags = {p.lower() for p in parts[1:]}
    lang = next((p for p in parts[1:] if p and p.lower() != "game"), None)
    return parts[0], lang, "game" in flags


def _from_config() -> dict:
    white = []
    for spec in getattr(config, "AUTOPILOT_CHANNELS", []):
        ch, lang, game = _split(spec)
        cid = re.search(r"(UC[\w-]{22})", ch)
        white.append({"canal": ch, "nome": _KNOWN_NAMES.get(cid.group(1)) if cid else None,
                      "idioma": lang, "gameplay": game})
    return {"white": white, "black": []}


def mtime() -> Optional[float]:
    try:
        return FILE.stat().st_mtime
    except OSError:
        return None


def load() -> dict:
    if FILE.exists():
        try:
            d = json.loads(FILE.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("white"), list) and isinstance(d.get("black"), list):
                return d
        except (OSError, ValueError):
            pass
    return _from_config()


def save(d: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    text = json.dumps(d, ensure_ascii=False, indent=1)
    for i in range(40):  # Windows: a interface e o piloto podem estar lendo agora
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, FILE)
            return
        except PermissionError:
            if i == 39:
                raise
            time.sleep(0.25)


# ------------------------------------------------------------ chaves ---
def norm(s: Optional[str]) -> str:
    """"https://www.youtube.com/@FlowPodcast/videos", "@flowpodcast",
    "Flow Podcast" -> "flowpodcast"."""
    t = unicodedata.normalize("NFKD", (s or "").strip()).encode("ascii", "ignore").decode().lower()
    t = re.sub(r"^https?://", "", t)
    t = re.sub(r"^(www\.|m\.)?youtube\.com/", "", t)
    t = re.sub(r"^(channel|c|user)/", "", t)
    t = re.sub(r"/(videos|streams|shorts|featured|about)?/?$", "", t)
    t = t.split("|")[0].lstrip("@")
    return re.sub(r"[^a-z0-9_-]+", "", t)


def spec(entry: dict) -> str:
    """Entrada da lista branca -> "canal|idioma|game" (formato da busca)."""
    s = entry["canal"]
    if entry.get("idioma"):
        s += "|" + entry["idioma"]
    if entry.get("gameplay"):
        s += "|game"
    return s


def white_specs() -> List[str]:
    d = load()
    return [spec(e) for e in d["white"] if e.get("canal") and not _black_hit(d, e.get("canal"), e.get("nome"))]


def label(entry: dict) -> str:
    extra = (" · inglês" if (entry.get("idioma") or "").lower() == "en" else
             f" · {entry['idioma']}" if entry.get("idioma") else "") + (" · gameplay" if entry.get("gameplay") else "")
    name = entry.get("nome")
    ch = entry["canal"]
    if name and norm(name) != norm(ch):
        return f"{name}  ({ch if len(ch) <= 40 else ch[:37] + '...'}){extra}"
    return f"{ch}{extra}"


def _black_hit(d: dict, *ids) -> Optional[str]:
    keys = [norm(i) for i in ids if i]
    keys = [k for k in keys if k]
    for b in d["black"]:
        bk = norm(b.get("canal"))
        if not bk:
            continue
        for k in keys:
            # mesmo canal; ou o nome da lista negra dentro do nome do canal
            # ("felipeneto" pega "felipenetoclips") -- só com 5+ letras, pra
            # um nome curto não derrubar canal que não tem nada a ver
            if k == bk or (len(bk) >= 5 and bk in k):
                return b.get("canal")
    return None


def is_black(*ids) -> Optional[str]:
    """O canal (nome, @, link ou id) está na lista negra? Devolve a entrada."""
    return _black_hit(load(), *ids)


# -------------------------------------------------- edição (interface) ---
def looks_like_channel(text: str) -> bool:
    t = (text or "").strip()
    return bool(re.match(r"^@[\w.\-]{2,}$", t) or re.match(r"^(https?://)?(www\.|m\.)?youtube\.com/\S+$", t)
                or re.match(r"^UC[\w-]{22}$", t))


def clean_channel(text: str) -> str:
    t = (text or "").strip().rstrip("/")
    if re.match(r"^UC[\w-]{22}$", t):
        return "https://www.youtube.com/channel/" + t
    t = re.sub(r"/(videos|streams|shorts|featured|about)$", "", t)
    return t


def add_white(canal: str, en: bool = False, game: bool = False) -> dict:
    d = load()
    canal = clean_channel(canal)
    k = norm(canal)
    for e in d["white"]:
        if norm(e["canal"]) == k:
            e.update(idioma="en" if en else None, gameplay=bool(game))
            save(d)
            return e
    e = {"canal": canal, "nome": None, "idioma": "en" if en else None, "gameplay": bool(game)}
    d["white"].append(e)
    # estava na lista negra: sai de lá (a última escolha vale)
    d["black"] = [b for b in d["black"] if norm(b.get("canal")) != k]
    save(d)
    return e


def remove_white(canal: str) -> None:
    d = load()
    d["white"] = [e for e in d["white"] if e["canal"] != canal]
    save(d)


def set_name(canal: str, name: Optional[str]) -> None:
    d = load()
    for e in d["white"]:
        if e["canal"] == canal and name and e.get("nome") != name:
            e["nome"] = name
            save(d)
            return


def add_black(canal: str) -> List[str]:
    """Põe na lista negra; devolve os canais da lista branca que saíram por isso."""
    d = load()
    canal = clean_channel(canal) if looks_like_channel(canal) else canal.strip()
    if not norm(canal):
        return []
    if all(norm(b.get("canal")) != norm(canal) for b in d["black"]):
        d["black"].append({"canal": canal, "desde": time.strftime("%Y-%m-%d")})
    gone = [e for e in d["white"] if _black_hit({"black": [{"canal": canal}]}, e["canal"], e.get("nome"))]
    d["white"] = [e for e in d["white"] if e not in gone]
    save(d)
    return [label(e) for e in gone]


def remove_black(canal: str) -> None:
    d = load()
    d["black"] = [b for b in d["black"] if b.get("canal") != canal]
    save(d)
