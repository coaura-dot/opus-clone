"""
Modo REACT: streamer com facecam sobreposta ao vídeo que ele está reagindo.

Problema real (clipe de react enviado pelo usuário): o reenquadramento comum
via dois rostos -- o do vídeo reagido e o do streamer na facecam -- e ficava
pulando entre eles; o detector antigo de "tela dentro da tela" (contorno de
4 vértices) confundia a moldura da facecam com uma tela e alternava pra um
terceiro enquadramento. Resultado: a câmera piscava entre três layouts.

Aqui o vídeo é tratado como o que ele é: DUAS camadas fixas.
  1. `detect_react_layout` (uma vez por vídeo): amostra o vídeo inteiro e
     procura um rosto que fica SEMPRE no mesmo lugar, pequeno, enquanto o
     resto da imagem muda de forma independente dele (o conteúdo corta de
     cena, a facecam não). Acha a caixa da facecam pelas bordas que não
     mudam ao longo do vídeo (moldura) e pela fronteira entre a região
     parada (quarto do streamer) e a que muda (conteúdo).
  2. `plan_clip` (uma vez por clipe): confere em que trechos a facecam está
     mesmo na tela (o streamer pode trocar de cena), com trechos mínimos de
     alguns segundos -- nada de pisca-pisca -- e escolhe UMA janela fixa do
     conteúdo pro clipe inteiro (onde estão os rostos/a ação do vídeo
     reagido, sem pegar a facecam).
  3. `ReactClipPlan.compose`: monta o quadro vertical em tela dividida --
     conteúdo em cima, streamer embaixo -- o layout padrão de cortes de
     react no TikTok/Shorts.
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .reframer import _new_yunet, _detect_yunet

ANALYSIS_W = 960          # resolução de análise (largura); só pra detectar
_REF_W, _REF_H = 64, 36   # miniatura da facecam usada pra conferir presença


def _analysis_size(src_w: int, src_h: int) -> Tuple[int, int]:
    w = min(ANALYSIS_W, src_w - src_w % 2)
    h = max(int(round(src_h * w / src_w / 2.0)) * 2, 2)
    return w, h


def _read_frame_at(source_path: str, t: float, w: int, h: int) -> Optional[np.ndarray]:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(t, 0.0):.3f}", "-i", source_path,
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-an",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    if len(out) < w * h * 3:
        return None
    return np.frombuffer(out[:w * h * 3], dtype=np.uint8).reshape(h, w, 3)


def _read_frames(source_path: str, start: float, duration: float, rate: float,
                 w: int, h: int) -> List[np.ndarray]:
    """Decodifica o trecho uma vez só, em baixa resolução, a `rate` quadros/s."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", source_path,
           "-t", f"{max(duration, 0.1):.3f}", "-vf", f"fps={rate},scale={w}:{h}", "-an",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            frames.append(np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3))
    finally:
        proc.stdout.close()
        proc.wait()
    return frames


def _faces(detector, img) -> list:
    """(cx, cy, fw, fh) na escala de `img`, SEM o filtro de altura do
    detector principal (facecam costuma ficar no rodapé do quadro)."""
    if detector is None:
        return []
    return _detect_yunet(detector, img, img.shape[1], getattr(config, "YUNET_SCORE_THRESHOLD", 0.6))


@dataclass
class ReactLayout:
    src_w: int
    src_h: int
    cam_box: Tuple[int, int, int, int]          # (x, y, w, h) em pixels da fonte
    face: Tuple[float, float, float, float]     # rosto típico do streamer (cx, cy, fw, fh), fonte
    cam_ref: Optional[np.ndarray]               # miniatura cinza da facecam (mediana)
    cam_mask: Optional[np.ndarray]              # 1 = fundo parado da facecam (sem a pessoa)
    presence: float
    independence: float
    sides: str                                  # como cada lado foi achado (debug)

    def _scale(self, w: int):
        return w / float(self.src_w)

    def box_at(self, w: int) -> Tuple[float, float, float, float]:
        s = self._scale(w)
        x, y, bw, bh = self.cam_box
        return x * s, y * s, bw * s, bh * s

    def cam_matches(self, gray_small: np.ndarray) -> Optional[float]:
        """Diferença média do fundo da facecam neste quadro pra referência
        (None se não há referência confiável)."""
        if self.cam_ref is None or self.cam_mask is None:
            return None
        bx, by, bw, bh = (int(round(v)) for v in self.box_at(gray_small.shape[1]))
        patch = gray_small[by:by + bh, bx:bx + bw]
        if patch.size == 0:
            return None
        patch = cv2.resize(patch, (_REF_W, _REF_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        return float(np.sum(np.abs(patch - self.cam_ref) * self.cam_mask) / max(self.cam_mask.sum(), 1.0))


def _walk(E, V, face, W, H, direction: str, relaxed: bool = False):
    """Anda do rosto pra fora até achar a borda da facecam. Retorna
    (posição, tipo) -- tipo: 'line' (moldura/borda fixa), 'var' (fronteira
    parado|mudando), 'edge' (borda do quadro) ou 'guess'."""
    cx, cy, fw, fh = face
    fl, fr, ft, fb = cx - fw / 2, cx + fw / 2, cy - fh / 2, cy + fh / 2
    max_dist = 7.0 * fw
    if direction in ("left", "right"):
        r0, r1 = int(max(ft - 0.2 * fh, 0)), int(min(fb + 0.2 * fh, H))
        if direction == "left":
            xs = range(int(fl - 0.35 * fw), 0, -1)
        else:
            xs = range(int(fr + 0.35 * fw), W - 1)
        sign = -1 if direction == "left" else 1
        for x in xs:
            if abs(x - cx) > max_dist:
                return None, "guess"
            cov = float(np.mean(E[r0:r1, x] > 0.5)) if r1 > r0 else 0.0
            a0, a1 = sorted((x - sign * 2, x - sign * 8))
            b0, b1 = sorted((x + sign * 2, x + sign * 8))
            if min(a0, b0) < 0 or max(a1, b1) > W:
                continue
            vin = float(np.median(V[r0:r1, a0:a1]))
            vout = float(np.median(V[r0:r1, b0:b1]))
            # linha fixa só é a borda da facecam se do lado de FORA a imagem
            # muda mais (o jogo/vídeo) -- batente de porta ou quadro na parede
            # do quarto do streamer também é linha fixa, mas com quarto parado
            # dos dois lados (achado real: webcam do SMP cortada no meio)
            if (cov >= 0.85 and vout > 1.15 * vin + 1.0) or (cov >= 0.75 and vout > 1.4 * vin + 3.0):
                return float(x), "line"
            if vout > 2.5 * vin + 8.0:
                # gameplay: o streamer mexe a cabeça e a cadeira, e entre a
                # cabeça e a cadeira fica um "vale" parado que imitava a borda
                # (achado real: SMP, borda achada em 200 em vez de 256). Borda
                # de verdade tem o jogo mexendo numa faixa LARGA do lado de fora.
                c0_, c1_ = sorted((x + sign * 2, x + sign * 26))
                if not relaxed or float(np.median(V[r0:r1, max(c0_, 0):min(c1_, W)])) > 2.5 * vin + 8.0:
                    return float(x), "var"
        return (0.0 if direction == "left" else float(W)), "edge"
    c0, c1 = int(max(fl - 0.2 * fw, 0)), int(min(fr + 0.2 * fw, W))
    if direction == "up":
        ys = range(int(ft - 0.3 * fh), 0, -1)
    else:
        ys = range(int(fb + 0.3 * fh), H - 1)
    sign = -1 if direction == "up" else 1
    for y in ys:
        if abs(y - cy) > max_dist:
            return None, "guess"
        cov = float(np.mean(E[y, c0:c1] > 0.5)) if c1 > c0 else 0.0
        a0, a1 = sorted((y - sign * 2, y - sign * 8))
        b0, b1 = sorted((y + sign * 2, y + sign * 8))
        if min(a0, b0) < 0 or max(a1, b1) > H:
            continue
        vin = float(np.median(V[a0:a1, c0:c1]))
        vout = float(np.median(V[b0:b1, c0:c1]))
        # pra baixo fica o corpo de quem está na câmera (mexe tanto quanto o
        # conteúdo): ali só a linha fixa decide; pra cima, a mesma regra dos lados
        line_ok = cov >= 0.85 and (direction == "down" or vout > 1.15 * vin + 1.0)
        if line_ok or (cov >= 0.75 and vout > 1.4 * vin + 3.0):
            return float(y), "line"
        # pra baixo fica o corpo do streamer (mexe): só aceita fronteira
        # de variação bem marcada
        if vout > (3.0 if direction == "down" else 2.5) * vin + 8.0:
            return float(y), "var"
    return (0.0 if direction == "up" else float(H)), "edge"


def _find_cam_box(E, V, face, W, H, relaxed: bool = False):
    cx, cy, fw, fh = face
    found = {d: _walk(E, V, face, W, H, d, relaxed) for d in ("left", "right", "up", "down")}
    left, kl = found["left"]
    right, kr = found["right"]
    top, ku = found["up"]
    bottom, kd = found["down"]
    # lado não achado: chute conservador (caixa menor = rosto maior no
    # painel, nunca vaza conteúdo pra dentro da facecam)
    if left is None:
        left = cx - fw / 2 - (min(right - cx - fw / 2, 1.6 * fw) if right is not None else 1.6 * fw)
    if right is None:
        right = cx + fw / 2 + min(cx - fw / 2 - left, 1.6 * fw)
    if top is None:
        top = cy - fh / 2 - 0.9 * fh
    if bottom is None:
        bottom = cy + fh / 2 + 1.4 * fh
    # base não achada (corpo mexendo até a borda do quadro) mas os lados e o
    # topo sim: webcam é 16:9 ou 4:3 -- a que deixa o rosto inteiro dentro.
    # Gameplay (relaxed): vale também com um dos lados sendo a própria borda
    # do quadro e a webcam cortada no queixo (achado real: SMP do Tommyinnit)
    sides_ok = (kl[0] in "lv" and kr[0] in "lv") or (relaxed and "e" in kl[0] + kr[0]
                                                     and (kl[0] in "lv" or kr[0] in "lv"))
    chin = 0.45 if relaxed else 0.9
    if kd[0] in "eg" and sides_ok and ku[0] in "lve":
        bw_ = right - left
        for ratio in (0.5625, 0.75):
            if cy + chin * fh <= top + bw_ * ratio:
                bottom = min(bottom, top + bw_ * ratio)
                kd = "a"  # pela proporção
                break
    left, right = max(left, 0.0), min(right, float(W))
    top, bottom = max(top, 0.0), min(bottom, float(H))
    return (left, top, right - left, bottom - top), kl[0] + kr[0] + ku[0] + kd[0]


def _sides_covered(E, box, sides, min_cov: float = 0.8) -> bool:
    """As bordas 'line' da caixa cobrem o lado inteiro (de canto a canto)?"""
    bx, by, bw, bh = box
    H, W = E.shape
    x0, y0 = int(np.clip(bx, 0, W - 1)), int(np.clip(by, 0, H - 1))
    x1, y1 = int(np.clip(bx + bw, 0, W - 1)), int(np.clip(by + bh, 0, H - 1))
    segs = {"left": E[y0:y1, max(x0 - 1, 0):x0 + 2].max(axis=1),
            "right": E[y0:y1, max(x1 - 1, 0):x1 + 2].max(axis=1),
            "up": E[max(y0 - 1, 0):y0 + 2, x0:x1].max(axis=0),
            "down": E[max(y1 - 1, 0):y1 + 2, x0:x1].max(axis=0)}
    for kind, side in zip(sides, ("left", "right", "up", "down")):
        if kind == "l":
            seg = segs[side]
            if seg.size == 0 or float(np.mean(seg > 0.5)) < min_cov:
                return False
    return True


def _person_mask(box, face, shape) -> np.ndarray:
    """1 = fundo da facecam, 0 = região da pessoa (cabeça e tronco)."""
    bx, by, bw, bh = box
    cx, cy, fw, fh = face
    m = np.ones(shape, np.float32)
    h, w = shape
    sx, sy = w / max(bw, 1e-3), h / max(bh, 1e-3)
    x0 = int(max((cx - 1.1 * fw - bx) * sx, 0))
    x1 = int(min((cx + 1.1 * fw - bx) * sx, w))
    y0 = int(max((cy - 1.0 * fh - by) * sy, 0))
    m[y0:, x0:x1] = 0.0
    return m


def _liveness(faces: np.ndarray, grays) -> Tuple[float, float]:
    """(quanto o rosto anda, relativo ao tamanho dele; quanto a imagem do
    rosto muda entre uma amostra e outra, em níveis de cinza)."""
    fw, fh = float(np.median(faces[:, 2])), float(np.median(faces[:, 3]))
    mx, my = float(np.median(faces[:, 0])), float(np.median(faces[:, 1]))
    jitter = float(np.median(np.abs(faces[:, 0] - mx)) / max(fw, 1e-3)
                   + np.median(np.abs(faces[:, 1] - my)) / max(fh, 1e-3))
    x0, y0 = int(max(mx - fw / 2, 0)), int(max(my - fh / 2, 0))
    crops = []
    for g in grays:
        cr = g[y0:y0 + int(fh), x0:x0 + int(fw)]
        if cr.size:
            crops.append(cv2.resize(cr, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32))
    change = float(np.median([np.mean(np.abs(a - b)) for a, b in zip(crops, crops[1:])])) if len(crops) > 2 else 0.0
    return jitter, change


def detect_react_layout(source_path: str, duration: float, src_w: int, src_h: int,
                        n_samples: Optional[int] = None, verbose: bool = False,
                        relaxed: bool = False) -> Optional[ReactLayout]:
    """Decide se o vídeo é um react com facecam sobreposta e, se for, onde
    ela está. None = vídeo comum (podcast, vlog, entrevista...).

    `relaxed` (canal marcado como gameplay): aceita a facecam sem a prova de
    "independência" -- jogo calmo (andando, menu, construindo) muda pouco e
    a prova falhava em facecam de verdade (SMP do Tommyinnit) -- desde que o
    rosto vivo fique fora do miolo do quadro, onde facecam fica sempre."""
    n = int(n_samples or getattr(config, "REACT_LAYOUT_SAMPLES", 48))
    if duration < 10:
        return None
    W, H = _analysis_size(src_w, src_h)
    t0, t1 = min(5.0, duration * 0.02), max(duration - 2.0, 1.0)
    times = list(np.linspace(t0, t1, n))
    with ThreadPoolExecutor(max_workers=4) as pool:
        frames = list(pool.map(lambda t: _read_frame_at(source_path, t, W, H), times))
    frames = [f for f in frames if f is not None]
    if len(frames) < 8:
        return None
    n = len(frames)
    det = _new_yunet()
    if det is None:
        return None
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # agrupa rostos que aparecem no MESMO lugar e tamanho em vários quadros
    clusters: list = []  # [cx, cy, fh, [(i, face)]]
    for i, f in enumerate(frames):
        for face in _faces(det, f):
            cx, cy, fw, fh = face
            for c in clusters:
                if abs(cx - c[0]) < 0.035 * W and abs(cy - c[1]) < 0.05 * H and 0.7 < fh / c[2] < 1.4:
                    c[3].append((i, face))
                    k = len(c[3])
                    c[0] += (cx - c[0]) / k
                    c[1] += (cy - c[1]) / k
                    c[2] += (fh - c[2]) / k
                    break
            else:
                clusters.append([cx, cy, fh, [(i, face)]])

    min_presence = getattr(config, "REACT_MIN_PRESENCE", 0.25)
    cands = []
    for c in clusters:
        idx = sorted({i for i, _ in c[3]})
        presence = len(idx) / n
        if presence < min_presence or not (0.035 * H <= c[2] <= 0.30 * H):
            continue
        cands.append((presence, c, idx))
    cands.sort(key=lambda x: -x[0])

    for presence, c, idx in cands[:3]:
        faces = np.array([f for _, f in c[3]])
        face = tuple(float(v) for v in np.median(faces, axis=0))
        # rosto VIVO: quem está na facecam mexe a cabeça e muda de expressão.
        # Achado real (Felipe Neto, estúdio): o "rosto" fixo era um boneco na
        # estante -- parado no mesmo lugar enquanto as pessoas se mexiam em
        # volta, ele passava em todos os testes e a tela dividia mostrando o
        # boneco borrado. Medido: facecam real mexe 0,09-0,17 do tamanho do
        # rosto e a imagem do rosto muda 15-37 (níveis de cinza); boneco,
        # desenho na parede e foto colada: 0,007-0,05 e 1,4-1,8.
        jitter, appearance = _liveness(faces, [grays[i] for i, _ in c[3]])
        # webcam fechada no rosto: o rosto é grande, então o "quanto anda"
        # relativo ao tamanho dele é menor (medido: 0,056) -- aí vale mudar
        # bastante de expressão
        jit_min = getattr(config, "REACT_MIN_FACE_JITTER", 0.06)
        live = (jitter >= jit_min and appearance >= getattr(config, "REACT_MIN_FACE_CHANGE", 6.0)) or \
            (jitter >= 0.66 * jit_min and appearance >= 12.0)
        if not live:
            if verbose:
                print(f"    [react] rosto fixo em ({face[0]:.0f},{face[1]:.0f}) PARADO demais "
                      f"(mexe {jitter:.3f}, muda {appearance:.1f}) -- boneco/quadro/desenho, não facecam")
            continue
        stack = np.stack([grays[i] for i in idx]).astype(np.float32)
        V = cv2.blur(np.std(stack, axis=0), (5, 5))
        E = np.mean([cv2.Canny(grays[i], 60, 150) > 0 for i in idx], axis=0).astype(np.float32)
        E = cv2.dilate(E, np.ones((3, 3), np.uint8))
        box, sides = _find_cam_box(E, V, face, W, H, relaxed=relaxed)
        bx, by, bw, bh = box
        # webcam fechada no rosto (comum em gameplay) é estreita: 1,5x o rosto basta
        if bw < 1.5 * face[2] or bh < 1.4 * face[3] or bw * bh > 0.35 * W * H:
            if verbose:
                print(f"    [react] rosto fixo em ({face[0]:.0f},{face[1]:.0f}) mas caixa implausível {box}")
            continue

        # independência: o conteúdo (fora da caixa) muda sem a facecam mudar
        outside = np.ones((H, W), bool)
        outside[int(by):int(by + bh), int(bx):int(bx + bw)] = False
        ring = np.zeros((H, W), bool)
        ring[int(by):int(by + bh), int(bx):int(bx + bw)] = True
        pm = _person_mask(box, face, (int(by + bh) - int(by), int(bx + bw) - int(bx)))
        ring[int(by):int(by + bh), int(bx):int(bx + bw)] &= pm > 0.5
        pairs = hits = 0
        for a, b in zip(idx, idx[1:]):
            d = np.abs(grays[b].astype(np.float32) - grays[a].astype(np.float32))
            d_out = float(d[outside].mean()) if outside.any() else 0.0
            d_in = float(d[ring].mean()) if ring.any() else d_out
            pairs += 1
            hits += int(d_out > 18.0 and d_in < 0.35 * d_out)
        independence = hits / max(pairs, 1)
        # sem conteúdo variado o bastante pra provar a independência, aceita
        # uma moldura: caixa pequena fechada por bordas fixas (ou pela borda
        # do quadro) em todos os lados, com as linhas cobrindo o lado INTEIRO
        # -- um retângulo desenhado em volta de um rosto parado não acontece
        # numa cena de podcast/entrevista
        n_lines = sides.count("l")
        framed = (n_lines >= 2 and "g" not in sides and "v" not in sides
                  and bw * bh <= 0.15 * W * H and presence >= 0.7
                  and _sides_covered(E, box, sides) and len(sides) == 4)
        off_center = not (0.3 * W < face[0] < 0.7 * W and 0.3 * H < face[1] < 0.7 * H)
        ok = independence >= 0.25 or framed or (relaxed and off_center and presence >= 0.35)
        if verbose:
            print(f"    [react] rosto fixo presença={presence:.2f} caixa={tuple(int(v) for v in box)} "
                  f"lados={sides} independência={independence:.2f} -> {'REACT' if ok else 'não'}")
        if not ok:
            continue

        # tira uma lasquinha da borda (moldura colorida/antialias)
        inset_x, inset_y = max(2.0, 0.015 * bw), max(2.0, 0.015 * bh)
        bx, by, bw, bh = bx + inset_x, by + inset_y, bw - 2 * inset_x, bh - 2 * inset_y
        # referência do fundo da facecam pra conferir presença por clipe
        crops = [cv2.resize(grays[i][int(by):int(by + bh), int(bx):int(bx + bw)], (_REF_W, _REF_H),
                            interpolation=cv2.INTER_AREA).astype(np.float32) for i in idx]
        ref = np.median(np.stack(crops), axis=0)
        mask = _person_mask((bx, by, bw, bh), face, (_REF_H, _REF_W))
        spread = np.std(np.stack(crops), axis=0)
        mask *= (spread < 14.0).astype(np.float32)
        if mask.mean() < 0.15:
            ref = mask = None
        s = src_w / float(W)
        return ReactLayout(
            src_w=src_w, src_h=src_h,
            cam_box=(int(round(bx * s)), int(round(by * s)), int(round(bw * s)), int(round(bh * s))),
            face=tuple(v * s for v in face), cam_ref=ref, cam_mask=mask,
            presence=presence, independence=independence, sides=sides,
        )
    return None


def _smooth_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
    """Tira trechos curtos (pisca-pisca): todo trecho com menos de
    `min_len` amostras vira o valor do vizinho."""
    flags = flags.copy()
    if len(flags) == 0:
        return flags
    for _ in range(3):
        runs = []
        start = 0
        for i in range(1, len(flags) + 1):
            if i == len(flags) or flags[i] != flags[start]:
                runs.append((start, i))
                start = i
        changed = False
        for k, (a, b) in enumerate(runs):
            if b - a < min_len and len(runs) > 1:
                flags[a:b] = flags[runs[k - 1][0]] if k > 0 else flags[runs[k + 1][0]]
                changed = True
        if not changed:
            break
    return flags


class ReactClipPlan:
    """Plano de um clipe em tela dividida: em que trechos a facecam está na
    tela, onde está o rosto do streamer e qual janela do conteúdo mostrar."""

    def __init__(self, layout: ReactLayout, times, present, face_track, content_window):
        self.layout = layout
        self.times = np.asarray(times, dtype=np.float64)
        self.present = np.asarray(present, dtype=bool)
        self.face_track = face_track          # (N, 3): cx, cy, fh na fonte
        self.content_window = content_window  # (x, y, w, h) na fonte

    def active(self, t: float) -> bool:
        if len(self.times) == 0:
            return False
        i = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 1))
        return bool(self.present[i])

    @property
    def active_fraction(self) -> float:
        return float(self.present.mean()) if len(self.present) else 0.0

    def _face_at(self, t: float):
        tr = self.face_track
        return (float(np.interp(t, self.times, tr[:, 0])), float(np.interp(t, self.times, tr[:, 1])),
                float(np.interp(t, self.times, tr[:, 2])))

    def compose(self, frame, t: float, out_w: int, out_h: int, zoom: float = 1.0) -> np.ndarray:
        frac = float(np.clip(getattr(config, "REACT_CONTENT_FRAC", 0.5), 0.3, 0.7))
        top_h = int(round(out_h * frac / 2.0)) * 2
        bot_h = out_h - top_h
        src_h, src_w = frame.shape[:2]

        x, y, w, h = self.content_window
        top = frame[y:y + h, x:x + w]
        top = cv2.resize(top, (out_w, top_h),
                         interpolation=cv2.INTER_AREA if w > out_w else cv2.INTER_LINEAR)

        bx, by, bw, bh = self.layout.cam_box
        fcx, fcy, ffh = self._face_at(t)
        aspect = out_w / float(bot_h)
        face_frac = getattr(config, "REACT_CAM_FACE_FRAC", 0.30)
        crop_h = float(np.clip(ffh / face_frac, ffh * 2.2, bh)) / max(zoom, 1.0)
        crop_w = crop_h * aspect
        if crop_w > bw:
            crop_w = float(bw)
            crop_h = crop_w / aspect
        x0 = int(np.clip(fcx - crop_w / 2, bx, bx + bw - crop_w))
        y0 = int(np.clip(fcy - crop_h * 0.42, by, by + bh - crop_h))
        cw, ch = max(int(crop_w), 2), max(int(crop_h), 2)
        bot = frame[y0:y0 + ch, x0:x0 + cw]
        up = out_w / float(cw)
        bot = cv2.resize(bot, (out_w, bot_h),
                         interpolation=cv2.INTER_CUBIC if up > 1.5 else
                         (cv2.INTER_AREA if up < 1 else cv2.INTER_LINEAR))
        if up > 2.5:
            # facecam pequena ampliada 3-6x fica mole: realce leve de nitidez
            soft = cv2.GaussianBlur(bot, (0, 0), 2.0)
            bot = cv2.addWeighted(bot, 1.45, soft, -0.45, 0)

        out = np.empty((out_h, out_w, 3), np.uint8)
        out[:top_h] = top
        out[top_h:] = bot
        # divisória fina entre os painéis
        sep = max(int(round(out_h * 0.002)), 2)
        out[top_h - sep // 2: top_h + (sep - sep // 2)] = (18, 18, 18)
        return out


def plan_clip(layout: ReactLayout, source_path: str, start: float, end: float,
              out_w: int, out_h: int) -> Optional[ReactClipPlan]:
    """Analisa o trecho do clipe (2 quadros/s, baixa resolução)."""
    W, H = _analysis_size(layout.src_w, layout.src_h)
    rate = 2.0
    frames = _read_frames(source_path, start, end - start, rate, W, H)
    if not frames:
        return None
    det = _new_yunet()
    bx, by, bw, bh = layout.box_at(W)
    s = layout.src_w / float(W)
    thr = getattr(config, "REACT_CAM_MATCH_MAX_DIFF", 22.0)

    present, cam_faces, content_faces = [], [], []
    grays = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        grays.append(gray)
        faces = _faces(det, f)
        in_cam = [fc for fc in faces
                  if bx + 0.03 * bw < fc[0] < bx + 0.97 * bw and by < fc[1] < by + bh
                  and fc[3] < 0.95 * bh]
        out_cam = [fc for fc in faces if fc not in in_cam
                   and not (bx - 0.05 * bw < fc[0] < bx + 1.05 * bw and by - 0.05 * bh < fc[1] < by + 1.05 * bh)]
        diff = layout.cam_matches(gray)
        ring_ok = diff is not None and diff < thr
        present.append(bool(in_cam) or ring_ok)
        cam_faces.append(max(in_cam, key=lambda fc: fc[3]) if in_cam else None)
        content_faces.append(out_cam)

    present = np.array(present)
    min_len = max(int(round(getattr(config, "REACT_MIN_SEGMENT_SECONDS", 3.0) * rate)), 1)
    present = _smooth_runs(present, min_len)
    # tela dividida é do CLIPE INTEIRO ou de nada: ligar e desligar no meio
    # (achado real: "react em tela dividida (48% do clipe)") parecia defeito.
    # Só divide se a facecam está na tela em quase todo o trecho.
    if present.mean() < getattr(config, "REACT_MIN_CLIP_PRESENCE", 0.85):
        return None
    present = np.ones_like(present)

    # rosto do streamer: segura o último visto nas falhas e suaviza (2.5s)
    default = (layout.face[0] / s, layout.face[1] / s, layout.face[3] / s)
    track, last = [], default
    for fc in cam_faces:
        if fc is not None:
            last = (fc[0], fc[1], fc[3])
        track.append(last)
    track = np.array(track, dtype=np.float64)
    k = 5
    if len(track) >= k:
        pad = np.pad(track, ((k // 2, k // 2), (0, 0)), mode="edge")
        kernel = np.ones(k) / k
        track = np.stack([np.convolve(pad[:, j], kernel, mode="valid") for j in range(3)], axis=1)
    track *= s

    # a janela do conteúdo só olha os trechos em tela dividida (nos outros o
    # streamer pode estar em tela cheia, e o rosto dele não é "conteúdo")
    on = [i for i in range(len(frames)) if present[i]]
    window = _content_window(layout, [grays[i] for i in on], [content_faces[i] for i in on],
                             W, H, out_w, out_h)
    times = np.arange(len(frames)) / rate
    return ReactClipPlan(layout, times, present, track, window)


def _content_window(layout: ReactLayout, grays, content_faces, W, H, out_w, out_h):
    """Uma janela FIXA do conteúdo pro clipe inteiro: cobre os rostos/ação
    do vídeo reagido e evita a facecam."""
    frac = float(np.clip(getattr(config, "REACT_CONTENT_FRAC", 0.5), 0.3, 0.7))
    aspect = out_w / (out_h * frac)
    bx, by, bw, bh = layout.box_at(W)
    # atividade = o quanto cada ponto muda ao longo do clipe, medida numa
    # versão BORRADA da imagem: o vídeo reagido muda em áreas grandes; texto
    # do chat rolando e contadores da live mudam em detalhe fino e somem no
    # borrão. Interface do navegador e barras da live não mudam.
    if len(grays) > 1:
        blurred = [cv2.GaussianBlur(g, (0, 0), 6.0).astype(np.float32) for g in grays]
        act = np.std(np.stack(blurred), axis=0)
    else:
        act = np.zeros((H, W), np.float32)
    pad_x, pad_y = 0.04 * bw, 0.04 * bh
    cx0, cy0 = int(max(bx - pad_x, 0)), int(max(by - pad_y, 0))
    cx1, cy1 = int(min(bx + bw + pad_x, W)), int(min(by + bh + pad_y, H))
    act[cy0:cy1, cx0:cx1] = 0.0
    integ = cv2.integral(act)
    total = float(integ[-1, -1])
    face_frames = [fl for fl in content_faces if fl]
    cam_area = max((cx1 - cx0) * (cy1 - cy0), 1)
    # referência de "região ativa": média dos 30% de pontos que mais mudam
    hot = float(np.mean(act[act >= np.percentile(act, 70)])) if total > 1 else 1.0

    best, best_score = None, -1e9
    for sc in (1.0, 0.85, 0.72, 0.6):
        h = H * sc
        w = h * aspect
        if w > W:
            w = float(W)
            h = w / aspect
        nx = 25
        ny = 1 if h >= H - 1 else 5
        for x in np.linspace(0, W - w, nx):
            for y in np.linspace(0, H - h, ny):
                x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
                ix = max(0, min(x1, cx1) - max(x0, cx0))
                iy = max(0, min(y1, cy1) - max(y0, cy0))
                cam_ov = ix * iy / cam_area
                mass = integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0]
                a = mass / total if total > 1 else 0.0
                # pureza: a janela é quase toda conteúdo que se mexe (não
                # interface parada, barra da live, bordas pretas)?
                purity = min(mass / max((x1 - x0) * (y1 - y0), 1) / max(hot, 1e-3), 1.0)
                if face_frames:
                    inside = sum(any(x0 + 0.5 * fc[2] <= fc[0] <= x1 - 0.5 * fc[2]
                                     and y0 + 0.5 * fc[3] <= fc[1] <= y1 - 0.3 * fc[3] for fc in fl)
                                 for fl in face_frames)
                    face_score = inside / len(face_frames)
                else:
                    face_score = 0.0
                score = 1.5 * face_score + 0.6 * a + 1.5 * purity - 2.5 * cam_ov + 0.2 * sc
                if score > best_score:
                    best_score, best = score, (x0, y0, x1 - x0, y1 - y0)
    s = layout.src_w / float(W)
    x, y, w, h = best
    x, y = int(round(x * s)), int(round(y * s))
    w, h = int(round(w * s)), int(round(h * s))
    w, h = min(w, layout.src_w - x), min(h, layout.src_h - y)
    return x, y, w, h


def caption_margin_v(out_h: int) -> int:
    """Legenda centrada na divisória entre o conteúdo e o streamer (o lugar
    padrão nos cortes de react: não cobre o rosto de ninguém)."""
    frac = float(np.clip(getattr(config, "REACT_CONTENT_FRAC", 0.5), 0.3, 0.7))
    font = getattr(config, "CAPTION_FONT_SIZE", 104)
    return int(round(out_h * (1.0 - frac) - font * 0.5))


def hook_bottom_y(out_h: int) -> int:
    """Base do balão do título no layout react: logo acima da legenda."""
    font = getattr(config, "CAPTION_FONT_SIZE", 104)
    cap_top = out_h - caption_margin_v(out_h) - font
    return int(cap_top - 40)
