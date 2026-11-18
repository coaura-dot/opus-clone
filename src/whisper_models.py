"""
Modelos do whisper.cpp: confere se o arquivo está inteiro e baixa de novo
o que estiver quebrado.

Achado real (log do usuário, uma noite inteira sem postar nada): o
ggml-large-v3-turbo-q5_0.bin tinha parado de baixar no meio (~63 MB de
574 MB). O programa escolhia ele por ser o "melhor modelo da pasta", o
whisper.cpp recusava ("not all tensors loaded from model file - expected
587, got 71") e TODO vídeo falhava na transcrição.

Agora:
  - modelo com tamanho errado (download pela metade) não é escolhido: a
    transcrição usa o próximo que estiver bom (no fim, o WHISPERCPP_MODEL);
  - modelo que o whisper.cpp recusa ao carregar é renomeado pra
    <nome>.incompleto (ou .incompativel, se o tamanho está certo mas a sua
    versão do whisper.cpp não lê) e não é tentado de novo;
  - o piloto baixa de novo, em segundo plano, o modelo quebrado (continua
    de onde parou se a conexão cair; confere tamanho e SHA-256 antes de
    usar). Enquanto isso, os cortes continuam saindo com o modelo que
    funciona.

Tamanhos e SHA-256 conferidos no repositório oficial
(huggingface.co/ggerganov/whisper.cpp).
"""
import hashlib
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{name}"

# nome -> (tamanho em bytes, sha256)
MODELS = {
    "ggml-large-v3-turbo.bin": (1624555275, "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"),
    "ggml-large-v3-turbo-q8_0.bin": (874188075, "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"),
    "ggml-large-v3-turbo-q5_0.bin": (574041195, "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"),
    "ggml-large-v3.bin": (3095033483, "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"),
    "ggml-large-v3-q5_0.bin": (1081140203, "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"),
    "ggml-medium.bin": (1533763059, "6c14d5adee5f86394037b4e4e8b59f1673b6cee10e3cf0b11bbdbee79c156208"),
    "ggml-medium-q5_0.bin": (539212467, "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f"),
    "ggml-small.bin": (487601967, "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b"),
    "ggml-small-q8_0.bin": (264464607, "49c8fb02b65e6049d5fa6c04f81f53b867b5ec9540406812c643f177317f779f"),
    "ggml-small-q5_1.bin": (190085487, "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb"),
    "ggml-base.bin": (147951465, "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"),
}

BROKEN = ".incompleto"        # download pela metade / arquivo estragado: baixa de novo
INCOMPATIBLE = ".incompativel"  # arquivo inteiro, mas este whisper.cpp não lê: não baixa de novo
PART = ".part"

_STATE = {"thread": None}


def expected_size(path) -> Optional[int]:
    info = MODELS.get(Path(path).name)
    return info[0] if info else None


def problem(path) -> Optional[str]:
    """None se o arquivo parece inteiro; senão, o que há de errado."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except OSError:
        return "não encontrado"
    exp = expected_size(p)
    if exp is not None and size != exp:
        return f"incompleto ({size / 1024 ** 2:.0f} MB de {exp / 1024 ** 2:.0f} MB)"
    if exp is None and size < 10 * 1024 ** 2:
        return f"pequeno demais ({size / 1024 ** 2:.0f} MB)"
    return None


def mark_broken(path) -> Optional[Path]:
    """Tira da frente um modelo que o whisper.cpp não conseguiu carregar."""
    p = Path(path)
    suffix = INCOMPATIBLE if problem(p) is None and expected_size(p) else BROKEN
    dst = p.with_name(p.name + suffix)
    try:
        if dst.exists():
            dst.unlink()
        p.rename(dst)
        return dst
    except OSError:
        return None  # em uso / sem permissão: só não usa nesta rodada


def _to_repair(folder: Path, names) -> Optional[str]:
    """O melhor modelo da lista que está quebrado (pela metade, marcado
    como incompleto ou com download interrompido)."""
    for name in names:
        if name not in MODELS:
            continue
        f = folder / name
        if f.exists() and problem(f):
            return name
        if (folder / (name + BROKEN)).exists() or (folder / (name + PART)).exists():
            if not f.exists():
                return name
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(folder: Path, name: str, log: Callable[[str], None], tries: int = 6) -> bool:
    """Baixa `name` pra `folder` (continua um .part interrompido)."""
    size, sha = MODELS[name]
    final, part = folder / name, folder / (name + PART)
    # um arquivo pela metade com o nome certo vira o começo do .part
    for old in (final, folder / (name + BROKEN)):
        if old.exists() and not part.exists() and old.stat().st_size < size:
            try:
                old.rename(part)
            except OSError:
                pass
    last_pct = -1
    for attempt in range(tries):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have < size:
            req = urllib.request.Request(URL.format(name=name), headers={
                "Range": f"bytes={have}-", "User-Agent": "AutoClipper"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:
                        have = 0  # servidor ignorou o "continuar": começa do zero
                    with open(part, "ab" if have else "wb") as out:
                        while True:
                            buf = r.read(1024 * 1024)
                            if not buf:
                                break
                            out.write(buf)
                            have += len(buf)
                            pct = int(have * 100 / size) // 25 * 25
                            if pct != last_pct and pct < 100:
                                last_pct = pct
                                log(f"  [modelo] baixando {name}: {have / 1024 ** 2:.0f} de "
                                    f"{size / 1024 ** 2:.0f} MB")
            except Exception as e:  # conexão caiu: continua de onde parou
                log(f"  [modelo] download de {name} interrompido ({e.__class__.__name__}) -- "
                    f"continuo em 1 min")
                time.sleep(60)
                continue
        if part.stat().st_size != size:
            continue
        if _sha256(part) != sha:
            log(f"  [modelo] {name} baixou corrompido -- baixando de novo do zero")
            part.unlink()
            continue
        os.replace(part, final)
        for junk in (folder / (name + BROKEN),):
            try:
                junk.unlink()
            except OSError:
                pass
        log(f"  [modelo] {name} baixado e conferido -- os próximos cortes já usam ele.")
        return True
    log(f"  [modelo] não consegui baixar {name} agora; tento de novo quando o piloto reiniciar.")
    return False


def start_repair(model_path: str, names, log: Callable[[str], None]) -> Optional[str]:
    """Em segundo plano, baixa de novo o melhor modelo quebrado da pasta do
    modelo configurado. Devolve o nome do que vai baixar (ou None)."""
    if _STATE["thread"] is not None and _STATE["thread"].is_alive():
        return None
    folder = Path(model_path).parent
    if not folder.is_dir():
        return None
    name = _to_repair(folder, names)
    if not name:
        return None
    f = folder / name
    why = problem(f) if f.exists() else "download interrompido"
    log(f"  [modelo] {name} está {why} -- os cortes usam outro modelo enquanto eu baixo ele de "
        f"novo ({MODELS[name][0] / 1024 ** 2:.0f} MB, em segundo plano).")
    t = threading.Thread(target=download, args=(folder, name, log), daemon=True)
    _STATE["thread"] = t
    t.start()
    return name
