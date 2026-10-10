"""
Clipe antigo com o título-gancho queimado no vídeo?

Até a v12 todo clipe abria com o título num balão branco no topo e um
whoosh. Isso foi desligado (HOOK_ENABLED = False), mas os clipes que já
estavam prontos na fila (e os que subiram privados esperando a auditoria
do YouTube) continuaram com o balão -- relato do usuário: "ainda está
aparecendo essa legenda no começo com o sound effect". O balão e o som
fazem parte do arquivo e não têm como sair sem editar de novo, então o
piloto tira esses clipes da fila e não libera esses vídeos privados.

Como saber:
  - clipe novo: o .meta.json diz ("hook_burned");
  - clipe antigo (sem essa anotação): olha o vídeo. O balão é um retângulo
    branco opaco (texto preto dentro) que fica PARADO nos primeiros
    segundos e some depois de HOOK_SECONDS. A imagem do vídeo muda nesse
    tempo; a legenda é branca mas só traço de letra, com contorno preto.
"""
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from . import config

_W, _H, _FPS = 270, 480, 5


def _frames(video: str, seconds: float) -> Optional[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-t", f"{seconds:.2f}", "-i", str(video),
           "-vf", f"fps={_FPS},scale={_W}:{_H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    n = len(raw) // (_W * _H * 3)
    return np.frombuffer(raw[:n * _W * _H * 3], np.uint8).reshape(n, _H, _W, 3) if n else None


def _white(f: np.ndarray) -> np.ndarray:
    lo, hi = f.min(axis=2), f.max(axis=2)
    return (lo > 205) & (hi - lo < 40)


def has_balloon(video: str) -> Optional[bool]:
    """True = tem o balão do título no começo; None = não deu pra conferir."""
    import cv2
    hook_s = float(getattr(config, "HOOK_SECONDS", 3.2))
    t_in = [0.8, 1.6, 2.4]                      # balão já entrou (pop de 0,2 s) e não começou a sumir
    t_out = [hook_s + 0.6, hook_s + 1.4]        # balão já sumiu
    fr = _frames(video, t_out[-1] + 0.4)
    if fr is None:
        return None
    idx = lambda t: int(round(t * _FPS))
    if idx(t_out[-1]) >= len(fr):
        return None  # clipe curto demais pra comparar
    stay = np.logical_and.reduce([_white(fr[idx(t)]) for t in t_in])
    gone = np.logical_or.reduce([_white(fr[idx(t)]) for t in t_out])
    mask = (stay & ~gone).astype(np.uint8)
    dark_px = fr[idx(t_in[1])].max(axis=2) < 70
    if mask.sum() < 0.006 * _W * _H:
        return False
    # o fundo branco do balão envolve as letras: um bloco só depois de
    # fechar os buracos do texto
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 0.012 * _W * _H or w < 0.15 * _W or h < 0.04 * _H:
            continue
        comp = labels == i
        pts = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        hull = cv2.convexHull(max(pts, key=cv2.contourArea))
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        # uma caixa por linha de texto (larguras diferentes): "escada" de
        # retângulos -- nos clipes reais, 0,64-0,90 do casco convexo
        solid = area / hull_area
        fill = float(mask[comp].mean())                  # fundo branco (0,78-0,87 nos reais)...
        dark = float(dark_px[comp].mean())               # ...com letra preta dentro (0,07-0,13)
        if solid >= 0.55 and 0.6 <= fill <= 0.95 and dark >= 0.03:
            return True
    return False


def burned_hook(video: str, meta_path: Optional[str] = None) -> Optional[bool]:
    """O clipe tem o título-gancho (e o whoosh) no arquivo? None = não deu
    pra saber (arquivo sumiu / vídeo ilegível)."""
    if meta_path:
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            if "hook_burned" in meta:
                return bool(meta["hook_burned"])
        except (OSError, ValueError):
            pass
    if not video or not Path(video).exists():
        return None
    return has_balloon(video)
