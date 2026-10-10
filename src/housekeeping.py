"""
Faxina automática da pasta de cortes do piloto (output/autopiloto) e da
pasta de trabalho (.autoclip_work).

Pedido do usuário: a pasta acumulava pastas vazias (downloads que falharam)
e clipes que não serviam mais -- "não deixar passar de 1 GB e ir se auto
limpando com o tempo".

O que sai, sempre:
  - pastas vazias e o resultado.json de cada edição (só serve na hora);
  - arquivos de clipe que não estão na fila nem esperando liberação no
    YouTube (reprovados pela nota, sobras de edição que falhou);
  - a versão _sem_musica.mp4 (o piloto posta a versão com música; ela só
    serve pra quem posta à mão com som do app) -- AUTOPILOT_KEEP_NO_MUSIC;
  - clipe já postado e PÚBLICO, depois de AUTOPILOT_KEEP_POSTED_DAYS dias;
  - na pasta de trabalho: o vídeo baixado e os arquivos de cada clipe de
    uma edição que já terminou (o vídeo original sozinho passa de 1 GB).

Se mesmo assim passar de AUTOPILOT_OUTPUT_MAX_GB, apaga (mais antigo
primeiro): postados públicos ainda no prazo, depois postados que ainda
estão privados esperando a auditoria (esses perdem a chance de repostagem).
Clipe da FILA (ainda vai ser postado) nunca é apagado: com a pasta cheia, o
piloto só para de produzir até postar e liberar espaço.
"""
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from . import config

_SUFFIXES = ("_sem_musica.mp4", ".meta.json", ".post.txt", ".contexto.txt", ".mp4")


def _base(path: Path) -> Optional[str]:
    """clip_01_xyz.mp4 / clip_01_xyz_sem_musica.mp4 / clip_01_xyz.meta.json
    -> "<pasta>/clip_01_xyz" (todos os arquivos do mesmo clipe)."""
    name = path.name
    for suf in _SUFFIXES:
        if name.endswith(suf):
            return str(path.parent / name[: -len(suf)])
    return None


def _size(paths: Iterable[Path]) -> int:
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def folder_gb(root: Path) -> float:
    return _size(f for f in root.rglob("*") if f.is_file()) / 1024 ** 3


def _rm(path: Path) -> int:
    try:
        n = path.stat().st_size
        path.unlink()
        return n
    except OSError:
        return 0


def _files_of(base: str) -> list:
    b = Path(base)
    return [b.parent / (b.name + suf) for suf in _SUFFIXES if (b.parent / (b.name + suf)).exists()]


def cleanup(state, out_root: Path, log, quiet: bool = False) -> float:
    """Faz a faxina e devolve o tamanho final da pasta (GB)."""
    if not out_root.exists():
        return 0.0
    now = time.time()
    keep_days = getattr(config, "AUTOPILOT_KEEP_POSTED_DAYS", 2)
    keep_no_music = getattr(config, "AUTOPILOT_KEEP_NO_MUSIC", False)
    cap = getattr(config, "AUTOPILOT_OUTPUT_MAX_GB", 1.0) * 1024 ** 3

    queued = {_base(Path(c["video"])) for c in state.data["queue"] if c.get("video")}
    pending, public = [], []  # postados: (data, base, entrada)
    for p in state.data["posted"]:
        v = p.get("video")
        if not v:
            continue
        b = _base(Path(v))
        if p.get("privacy") == "public" or p.get("replaced") or p.get("gave_up"):
            public.append((p.get("at", 0), b, p))
        else:
            pending.append((p.get("at", 0), b, p))
    pending_bases = {b for _, b, _ in pending}
    freed, removed = 0, 0

    # 1) lixo: resultado.json, arquivo de clipe que ninguém mais usa
    for f in list(out_root.rglob("*")):
        if not f.is_file():
            continue
        if f.name == "resultado.json":
            freed += _rm(f)
            removed += 1
            continue
        b = _base(f)
        if b is None:
            continue
        in_use = b in queued or b in pending_bases or any(b == pb for _, pb, _ in public)
        old = now - f.stat().st_mtime > 600  # arquivo recém-escrito pode ser de uma edição em andamento
        if not in_use and old:
            freed += _rm(f)
            removed += 1
        elif f.name.endswith("_sem_musica.mp4") and not keep_no_music and old:
            freed += _rm(f)
            removed += 1

    # 2) postados e públicos: depois do prazo
    for at, b, p in sorted(public, key=lambda x: x[0]):
        if now - at > keep_days * 86400:
            for f in _files_of(b):
                freed += _rm(f)
                removed += 1

    # 3) acima do limite: postados públicos no prazo, depois os privados
    def over() -> bool:
        return folder_gb(out_root) * 1024 ** 3 > cap

    if over():
        for at, b, p in sorted(public, key=lambda x: x[0]) + sorted(pending, key=lambda x: x[0]):
            if not over():
                break
            files = _files_of(b)
            if not files:
                continue
            for f in files:
                freed += _rm(f)
                removed += 1
            if p in [e for _, _, e in pending]:
                p["gave_up"] = True  # sem o arquivo não dá pra repostar depois
        state.save()

    # 4) pastas vazias
    for d in sorted((d for d in out_root.rglob("*") if d.is_dir()), key=lambda d: -len(d.parts)):
        try:
            if not any(d.iterdir()):
                d.rmdir()
        except OSError:
            pass

    size = folder_gb(out_root)
    if removed and not quiet:
        log(f"  Faxina: {removed} arquivo(s) apagado(s), {freed / 1024 ** 2:.0f} MB liberados "
            f"-- pasta de cortes com {size * 1024:.0f} MB "
            f"(limite {getattr(config, 'AUTOPILOT_OUTPUT_MAX_GB', 1.0) * 1024:.0f} MB).")
    return size


def has_room(state, out_root: Path, log) -> bool:
    """Cabe mais um vídeo editado na pasta sem passar do limite?"""
    cap_gb = getattr(config, "AUTOPILOT_OUTPUT_MAX_GB", 1.0)
    size = cleanup(state, out_root, log)
    clips = [Path(c["video"]) for c in state.data["queue"] if c.get("video")]
    sizes = [f.stat().st_size for f in clips if f.exists()]
    per_clip = (sum(sizes) / len(sizes) / 1024 ** 3) if sizes else 0.05
    need = per_clip * getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3)
    if size + need <= cap_gb:
        return True
    if state.data.get("_full_logged") != round(size, 2):
        state.data["_full_logged"] = round(size, 2)
        log(f"  Pasta de cortes cheia ({size * 1024:.0f} MB de {cap_gb * 1024:.0f} MB) com clipes "
            "esperando postagem -- volto a produzir quando liberar espaço.")
    return False


def clean_work_dir(log=None) -> None:
    """Pasta de trabalho (.autoclip_work): o vídeo baixado e os arquivos de
    cada clipe de uma edição que já terminou. Mantém os caches (histórico
    de músicas etc.)."""
    work = Path(getattr(config, "WORK_DIR", ".autoclip_work"))
    if not work.is_absolute():
        work = Path(__file__).resolve().parent.parent / work
    if not work.exists():
        return
    freed = 0
    for f in work.glob("source*"):
        if f.is_file():
            freed += _rm(f)
    for name in ("full_audio.wav",):
        f = work / name
        if f.exists():
            freed += _rm(f)
    for d in list(work.glob("clip_*")) + [work / "chunks"]:
        if d.is_dir():
            freed += _size(f for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
    if log and freed > 50 * 1024 ** 2:
        log(f"  Faxina: pasta de trabalho limpa ({freed / 1024 ** 2:.0f} MB do vídeo baixado e temporários).")
