"""
Pasta "postar à mão": quando o YouTube não aceita mais uploads pela API no
dia, os melhores clipes da fila vão pra cá com título, descrição e hashtags
prontos, pra completar a meta do dia postando pelo celular/YouTube Studio.

Por que existe: o usuário pediu 24 posts por dia. A API do YouTube, na cota
padrão de um projeto (10.000 unidades), aceita ~6 uploads por dia (cada um
custa 1.600). O piloto tenta postar até o próprio YouTube recusar (se a cota
for aumentada, ele já posta mais sem mudar nada); o que passar disso vem pra
esta pasta. Não dá pra contornar a cota de outro jeito (vários projetos,
robô no navegador) sem quebrar as regras do YouTube e arriscar o canal.

  output/postar_a_mao/AAAA-MM-DD/
      01 - <título>.mp4        o clipe (já com legenda e música)
      01 - <título>.txt        título, descrição e hashtags pra copiar
      LEIA-ME.txt              a ordem (do mais viral pro menos)

Os clipes saem da fila automática (senão seriam postados duas vezes) e a
pasta do dia é apagada depois de AUTOPILOT_MANUAL_KEEP_DAYS dias.
"""
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from . import config

ROOT = Path(__file__).resolve().parent.parent


def folder() -> Path:
    return ROOT / "output" / "postar_a_mao"


def _safe(name: str, n: int = 60) -> str:
    name = re.sub(r'[<>:"/\\|?*\n\r\t]+', " ", name or "corte").strip(" .")
    return (name[:n].rstrip(" .") or "corte")


def export(state, n: int, day: str, key: Callable, log) -> int:
    """Move os `n` melhores clipes da fila pra pasta do dia. Uma vez por dia."""
    if not getattr(config, "AUTOPILOT_MANUAL_EXPORT", True) or n <= 0:
        return 0
    if state.data.get("manual_export_day") == day:
        return 0
    import json
    queue = state.data["queue"]
    cands = [c for c in sorted(queue, key=key) if not c.get("priority") and Path(c["video"]).exists()]
    picked = cands[:n]
    if not picked:
        return 0
    out = folder() / day
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"Clipes pra postar à mão hoje ({day}), do mais viral pro menos.",
             "A API do YouTube não aceitou mais uploads hoje; poste estes pelo app ou pelo YouTube Studio.",
             "Dica: espalhe ao longo do dia (1 a cada hora, mais ou menos).", ""]
    done = 0
    for i, c in enumerate(picked, 1):
        try:
            meta = json.loads(Path(c["meta"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        title = meta.get("title") or Path(c["video"]).stem
        base = f"{i:02d} - {_safe(title)}"
        try:
            shutil.move(c["video"], out / f"{base}.mp4")
        except OSError as e:
            log(f"    [!] não consegui mover {Path(c['video']).name} ({e.__class__.__name__})")
            continue
        kit = Path(c["video"]).with_suffix(".post.txt")
        txt = [f"TÍTULO:\n{title}\n", f"DESCRIÇÃO:\n{meta.get('description', '')}\n"]
        if meta.get("tags"):
            txt.append("TAGS:\n" + ", ".join(meta["tags"]) + "\n")
        if kit.exists():
            txt.append("KIT COMPLETO (Shorts / Reels / TikTok):\n" + kit.read_text(encoding="utf-8", errors="ignore"))
        (out / f"{base}.txt").write_text("\n".join(txt), encoding="utf-8")
        lines.append(f"{i:02d}. {title}")
        queue.remove(c)
        state.data.setdefault("manual", []).append({"title": title, "day": day, "file": str(out / f"{base}.mp4"),
                                                    "source_id": c.get("source_id"), "at": time.time()})
        done += 1
    state.data["manual"] = state.data.get("manual", [])[-200:]
    (out / "LEIA-ME.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state.data["manual_export_day"] = day
    state.save()
    if done:
        log(f"  >> {done} clipe(s) pra postar à mão hoje em {out} (título e descrição no .txt de cada um).")
    return done


def cleanup(log: Optional[Callable] = None) -> None:
    keep = getattr(config, "AUTOPILOT_MANUAL_KEEP_DAYS", 3) * 86400
    root = folder()
    if not root.exists():
        return
    now = time.time()
    for d in root.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > keep:
            shutil.rmtree(d, ignore_errors=True)
            if log:
                log(f"  Faxina: pasta \"postar à mão\" de {d.name} apagada.")
