"""
Vídeo "amassado": imagem gravada numa proporção e salva em outra, sem a
informação de proporção de pixel no arquivo -- o player mostra todo mundo
largo e achatado. Achado real: pré-live gravada no celular em pé (9:16) e
salva esticada em 1280x720; os clipes saíam com as pessoas deformadas.

Como detectar sem metadado nenhum: o detector de rosto (YuNet) é treinado
com rostos de proporção normal. Num vídeo esticado ele quase não acha rosto;
"desesticando" a imagem na proporção certa, acha em quase todo quadro. Num
vídeo normal é o contrário (desesticar só piora). Entre as proporções
comuns que funcionam, fica a que deixa o rosto com o formato mais natural
(distância entre os olhos / distância olhos-boca).
"""
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config

# fator horizontal (largura real / largura gravada) das proporções comuns
# quando o vídeo foi salvo em 16:9: 4:3, 1:1, 3:4 e 9:16
_CANDIDATES = (1.0, 0.75, 0.5625, 0.421875, 0.31640625)
_NATURAL_FACE_RATIO = 0.88   # olhos/olhos-boca medido em vídeos normais (0.82-0.91)


def detect_squeeze(source_path: str, duration: float, src_w: int, src_h: int,
                   n_samples: int = 16) -> float:
    """Fator horizontal a aplicar (1.0 = proporção do arquivo está certa)."""
    from .react_layout import _analysis_size, _read_frame_at
    from .reframer import _new_yunet
    det = _new_yunet()
    if det is None or duration < 5:
        return 1.0
    W, H = _analysis_size(src_w, src_h)
    times = np.linspace(min(3.0, duration * 0.05), max(duration - 3.0, 1.0), n_samples)
    frames = [f for f in (_read_frame_at(source_path, t, W, H) for t in times) if f is not None]
    if len(frames) < 6:
        return 1.0

    stats = {}
    for k in _CANDIDATES:
        hits, ratios = 0, []
        w = max(int(W * k), 16)
        det.setInputSize((w, H))
        for f in frames:
            img = f if k == 1.0 else cv2.resize(f, (w, H), interpolation=cv2.INTER_AREA)
            _, faces = det.detect(img)
            if faces is None:
                continue
            fc = max(faces, key=lambda x: x[-1])
            if float(fc[-1]) < 0.6:
                continue
            hits += 1
            eye_y, mouth_y = (fc[5] + fc[7]) / 2, (fc[11] + fc[13]) / 2
            if mouth_y - eye_y > 3:
                ratios.append(abs(fc[6] - fc[4]) / (mouth_y - eye_y))
        stats[k] = (hits, float(np.median(ratios)) if ratios else None)

    base = stats[1.0][0]
    best_hits = max(h for h, _ in stats.values())
    # só corrige com evidência forte: na proporção do arquivo quase não há
    # rosto, e desesticando aparece rosto em boa parte dos quadros
    if best_hits < max(6, 0.4 * len(frames)) or base > 0.4 * best_hits:
        return 1.0
    good = [k for k, (h, r) in stats.items() if k < 1.0 and h >= 0.8 * best_hits and r is not None]
    if not good:
        return 1.0
    return min(good, key=lambda k: abs(stats[k][1] - _NATURAL_FACE_RATIO))


def fix_squeezed_source(source_path: str, info: dict, work_dir: str) -> Tuple[str, dict]:
    """Se o vídeo estiver esticado, gera uma cópia na proporção certa
    (uma vez só) e devolve o caminho e as informações novas."""
    if not getattr(config, "ASPECT_FIX_ENABLED", True):
        return source_path, info
    k = detect_squeeze(source_path, info["duration"], info["width"], info["height"])
    if k >= 0.999:
        return source_path, info
    # mantém a quantidade de pixels: estreita a largura e aumenta a altura
    s = k ** 0.5
    new_w = int(round(info["width"] * s / 2)) * 2
    new_h = int(round(info["height"] / s / 2)) * 2
    out = str(Path(work_dir) / "source_fixed_aspect.mp4")
    print(f"    -> Vídeo esticado detectado (pessoas largas/achatadas): corrigindo a "
          f"proporção para {new_w}x{new_h} antes de cortar...")
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", source_path,
           "-vf", f"scale={new_w}:{new_h},setsar=1", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "copy", out]
    if subprocess.run(cmd).returncode != 0:
        print("    [aviso] não consegui corrigir a proporção; seguindo com o vídeo original.")
        return source_path, info
    fixed = dict(info, width=new_w, height=new_h)
    return out, fixed
