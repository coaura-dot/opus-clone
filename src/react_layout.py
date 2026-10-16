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


def _refine_vertical(E, box, face):
    """Gameplay: texto parado colado na facecam ("total subs", "subs today")
    era engolido pela caixa (achado no teste com o SMP). A borda de cima e
    de baixo de verdade é a linha horizontal mais perto do rosto que
    atravessa a caixa de lado a lado."""
    bx, by, bw, bh = box
    cx, cy, fw, fh = face
    H = E.shape[0]
    x0, x1 = int(bx + 0.05 * bw), int(bx + 0.95 * bw)
    if x1 - x0 < 10:
        return box
    cov = (E[:, x0:x1] > 0.5).mean(axis=1)
    cov = np.maximum(cov, np.maximum(np.roll(cov, 1), np.roll(cov, -1)))  # linha de 1-2 px tremida
    # 40%: o texto da live tapa parte da borda e roupa escura some contra
    # jogo escuro (medido no SMP: 0,40-0,54 na borda, 0 no resto)
    top, bottom = by, by + bh
    for y in range(int(cy - 0.9 * fh), int(by) + 2, -1):
        if 0 <= y < H and cov[y] >= 0.4:
            top = float(y)
            break
    for y in range(int(cy + 0.6 * fh), int(by + bh) - 2):
        if 0 <= y < H and cov[y] >= 0.4:
            bottom = float(y)
            break
    return (bx, top, bw, bottom - top)


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
    if relaxed:
        # gameplay: o streamer passa boa parte da live em tela cheia (câmera
        # grande + chat) e a facecam pequena só aparece nos trechos de jogo
        # (achado real: DrDonut, "HAVE I BEEN CLONED???") -- mais amostras
        n = max(n, 96)
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
    all_faces = [_faces(det, f) for f in frames]
    for i, f in enumerate(frames):
        for face in all_faces[i]:
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
    if relaxed:
        min_presence = min(min_presence, getattr(config, "GAMEPLAY_MIN_CAM_PRESENCE", 0.10))
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
        if relaxed and not live:
            # gameplay: streamer concentrado no jogo mexe pouco (medido: SMP,
            # mexe 0,040 e muda 7,2); boneco/foto/desenho mudam 1,4-1,8 e
            # coisas paradas do jogo, 0
            live = jitter >= 0.02 and appearance >= 4.0
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
        if relaxed:
            box = _refine_vertical(E, box, face)
        bx, by, bw, bh = box
        # webcam fechada no rosto (comum em gameplay) é estreita: 1,5x o rosto basta
        if bw < 1.5 * face[2] or bh < 1.4 * face[3] or bw * bh > 0.35 * W * H:
            if verbose:
                print(f"    [react] rosto fixo em ({face[0]:.0f},{face[1]:.0f}) mas caixa implausível {box}")
            continue
        # caixa de verdade: TODO lado achado (nenhum no chute) e pelo menos 1
        # borda de verdade (linha fixa ou fronteira parado|mexendo; a borda
        # da tela também fecha a caixa), e rosto de webcam (não miniatura).
        # Achado real (Netolab, canal do Felipe Neto): o boneco do relógio da
        # Bela e a Fera na estante virava "facecam" -- caixa com 3 lados no
        # chute e 1 na borda da tela, rosto de 4,5% da altura; e "mexia" porque
        # o vídeo corta entre câmeras e ele muda de lugar a cada uma. Depois
        # dele, um dos apresentadores nos closes (1 lado no chute). Medido nas
        # 12 facecams reais testadas (reacts, hp, speedrun, SMP, Casimiro,
        # webcam em faixa na lateral): 0 lados no chute, rosto de 10,6-23,5%.
        guesses = sum(1 for k in sides[:4] if k == "g")
        borders = sum(1 for k in sides[:4] if k in "lv")
        min_face = 0.035 if relaxed else getattr(config, "REACT_MIN_FACE_HEIGHT", 0.06)
        if guesses > 0 or borders < 1 or face[3] < min_face * H:
            if verbose:
                print(f"    [react] rosto fixo em ({face[0]:.0f},{face[1]:.0f}) sem caixa de facecam de verdade "
                      f"(lados={sides}, rosto {face[3] / H * 100:.1f}% da altura) -- boneco/quadro do cenário")
            continue
        if _content_over_cam(box, face, idx, all_faces, grays, W, H):
            if verbose:
                print(f"    [react] caixa {tuple(int(v) for v in box)} é o CONTEÚDO por cima da câmera "
                      f"do streamer (rosto maior fora dela, fundo fixo) -- não é facecam")
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
        ok = independence >= 0.25 or framed or (relaxed and off_center and presence >= min_presence)
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


def _content_over_cam(box, face, idx, all_faces, grays, W, H) -> bool:
    """A caixa com rosto é um vídeo (TikTok, Reels, YouTube) numa janela
    POR CIMA da câmera do streamer -- o contrário de uma facecam?

    Achado no teste (layout do orochidois: câmera dele na tela inteira e o
    vídeo reagido numa janela): o rosto de quem fala no vídeo da janela
    passava em todas as provas de facecam (vivo, caixa com moldura), o
    vídeo virava "react" com a "facecam" ausente e o corte saía só com o
    rosto do streamer.
    Facecam nunca corta; o vídeo da janela corta. É conteúdo quando:
      - fora da caixa há, em quase todo quadro, um rosto MAIOR, da mesma
        pessoa parada no mesmo lugar (o streamer na frente da webcam); e
      - a caixa troca de imagem inteira entre amostras (corte do vídeo)
        enquanto o resto da tela fica igual, ou é grande (12%+ da tela).
    (Uma entrevista sem cortes numa caixa sobre a câmera continua
    ambígua: aí vale a facecam, e o plano por cena cobre os trechos em que
    ela "some" -- ver content_window.py.)"""
    bx, by, bw, bh = box
    fh = face[3]
    big, frames = [], []
    for i in idx:
        out = [f for f in all_faces[i] if f[3] >= 1.3 * fh and
               not (bx - 0.02 * W <= f[0] <= bx + bw + 0.02 * W and by - 0.02 * H <= f[1] <= by + bh + 0.02 * H)]
        if out:
            big.append(max(out, key=lambda f: f[3]))
            frames.append(i)
    if len(frames) < max(4, 0.8 * len(idx)):
        return False
    arr = np.array(big, dtype=np.float64)
    cx, cy, fw, fhh = (float(np.median(arr[:, k])) for k in range(4))
    if fhh < 0.12 * H:
        return False
    # a MESMA pessoa parada no mesmo lugar (sentada na frente da webcam):
    # rosto de conteúdo pula de lugar e de tamanho a cada corte
    if float(np.median(np.abs(arr[:, 0] - cx))) > 0.3 * fw or float(np.median(np.abs(arr[:, 1] - cy))) > 0.3 * fhh \
            or float(np.median(np.abs(arr[:, 3] - fhh))) > 0.15 * fhh:
        return False
    # janela grande pra ASSISTIR (um quarto da tela ou mais) por cima da
    # webcam: facecam de react fica em 5-10% da tela
    if bw * bh >= 0.12 * W * H:
        return True
    inside = np.zeros((H, W), bool)
    inside[int(by):int(by + bh), int(bx):int(bx + bw)] = True
    pairs = cuts = 0
    # amostras seguidas (os cortes do vídeo ficam entre os planos com rosto)
    for a in range(min(idx), max(idx)):
        d = np.abs(grays[a + 1].astype(np.float32) - grays[a].astype(np.float32))
        d_in, d_out = float(d[inside].mean()), float(d[~inside].mean())
        pairs += 1
        cuts += int(d_in > 30.0 and d_out < 0.5 * d_in)
    # medido: janela de TikTok 5 cortes em 31 pares (dentro 44-147, fora
    # 17-33); facecam de react: nenhum (o streamer mexendo dá 5-20 dentro)
    return cuts >= max(2, 0.08 * pairs)


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


_TINY_W, _TINY_H = 64, 36   # miniatura usada pra achar os cortes do vídeo reagido


def _iter_frames(source_path: str, start: float, duration: float, rate: float, w: int, h: int):
    """Quadros do trecho, um por vez (não guarda o clipe inteiro na memória)."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", source_path,
           "-t", f"{max(duration, 0.1):.3f}", "-vf", f"fps={rate},scale={w}:{h}", "-an",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def _tiny_mask(cam_r, W: int, H: int) -> np.ndarray:
    """1 = conteúdo, 0 = facecam (o streamer mexendo não é corte do vídeo)."""
    m = np.ones((_TINY_H, _TINY_W), np.float32)
    x0, y0 = int(cam_r[0] * _TINY_W / W), int(cam_r[1] * _TINY_H / H)
    x1, y1 = int(np.ceil(cam_r[2] * _TINY_W / W)), int(np.ceil(cam_r[3] * _TINY_H / H))
    m[y0:y1, x0:x1] = 0.0
    return m


def _tiny_gray(img, mask) -> np.ndarray:
    t = cv2.resize(img, (_TINY_W, _TINY_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(t, cv2.COLOR_BGR2GRAY).astype(np.float32) * mask


def _overlap(a, b) -> float:
    """Área de a (x0, y0, x1, y1) que cai dentro de b, em fração de a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


@dataclass
class ContentShot:
    """Um plano do vídeo reagido (entre dois cortes dele) e como ele entra no
    painel de cima."""
    t0: float                          # início (s desde o começo do clipe)
    cut_diff: float                    # tamanho da troca de imagem no corte (acha o quadro exato no render)
    window: Tuple[int, int, int, int]  # recorte do conteúdo (x, y, w, h), px da fonte
    disp: Tuple[int, int, int, int]    # onde ele entra no painel (x, y, w, h), px da saída
    fills: bool                        # True = preenche o painel; False = inteiro, com fundo desfocado


@dataclass
class CamShot:
    """Enquadramento PARADO do streamer (tripé) a partir de `t0`."""
    t0: float
    crop: Tuple[float, float, float, float]  # recorte da facecam (x, y, w, h), px da fonte
    face: Tuple[float, float]                # centro do rosto (o zoom de edição puxa pra ele)


@dataclass
class Geometry:
    out_w: int
    out_h: int
    top: int         # y do painel do conteúdo
    content_h: int
    cam_x: int
    cam_y: int
    cam_w: int
    cam_h: int
    wide: bool       # conteúdo na largura inteira (painel baixo) + faixas de fundo desfocado

    @property
    def seam(self) -> int:
        return self.top + self.content_h

    @property
    def full(self) -> bool:
        """Os dois painéis cobrem a tela inteira (sem faixa de fundo)."""
        return self.top == 0 and self.cam_y + self.cam_h >= self.out_h and self.cam_w >= self.out_w


class ReactClipPlan:
    """Plano de um clipe em tela dividida.

    Achado real (react do Casimiro ao Show do Milhão): o painel de cima era
    UMA janela estreita fixa (63% da largura de um vídeo 16:9) e cortava a
    pergunta e a apresentadora; o de baixo seguia cada mexida de cabeça do
    streamer numa facecam pequena ampliada ~4x -- borrado e tremendo.
    Agora:
      - conteúdo: um enquadramento PARADO por plano do vídeo reagido,
        trocando exatamente no corte dele; cobre os rostos e o texto que
        importam (pergunta, legenda, placar). Se não cabe recortado, entra
        inteiro. Se o clipe tem muita cena larga, o painel já é da largura
        inteira (vídeo 16:9 inteiro em cima);
      - streamer: enquadramento de TRIPÉ (parado); facecam pequena aparece
        inteira, com o painel na proporção dela, pra ampliar o mínimo."""

    def __init__(self, layout: ReactLayout, times, present, geom: Geometry, shots: List[ContentShot],
                 cam_shots: List[CamShot], cam_up: float, area, mask, fps: float, rate: float):
        self.layout = layout
        self.times = np.asarray(times, dtype=np.float64)
        self.present = np.asarray(present, dtype=bool)
        self.geom = geom
        self.shots = shots
        self.cam_shots = cam_shots
        self.cam_up = cam_up                  # quanto a facecam é ampliada no painel
        self.area = area                      # área do vídeo reagido (x, y, w, h), px da fonte (fundo)
        self._mask = mask
        self._half = 0.5 / max(fps, 1.0)
        self._sample_dt = 1.0 / max(rate, 1e-3)
        self._cur = 0
        self._prev_tiny = None
        # zoom de edição (momento-chave) no streamer: facecam já bem ampliada
        # não aguenta muito mais zoom sem borrar
        umax = float(getattr(config, "REACT_CAM_MAX_UPSCALE", 2.4))
        self.cam_zoom_max = max(1.06, umax * 1.15 / max(cam_up, 1e-3))

    def active(self, t: float) -> bool:
        if len(self.times) == 0:
            return False
        i = int(np.clip(np.searchsorted(self.times, t) - 1, 0, len(self.times) - 1))
        return bool(self.present[i])

    @property
    def active_fraction(self) -> float:
        return float(self.present.mean()) if len(self.present) else 0.0

    def caption_margin_v(self) -> int:
        """Legenda centrada na divisória entre o conteúdo e o streamer."""
        font = getattr(config, "CAPTION_FONT_SIZE", 104)
        return int(round(self.geom.out_h - self.geom.seam - font * 0.5))

    def hook_bottom_y(self) -> int:
        """Base do balão do título: logo acima da legenda."""
        font = getattr(config, "CAPTION_FONT_SIZE", 104)
        return int(self.geom.out_h - self.caption_margin_v() - font - 40)

    def describe(self) -> str:
        g = self.geom
        inteiros = sum(1 for s in self.shots if not s.fills)
        txt = (f"conteúdo {'na largura inteira' if g.wide else 'em painel alto'} ({g.content_h}px), "
               f"streamer {g.cam_h}px parado (facecam ampliada {self.cam_up:.1f}x"
               f"{', ' + str(len(self.cam_shots)) + ' enquadramentos' if len(self.cam_shots) > 1 else ''}), "
               f"{len(self.shots)} plano(s) do vídeo reagido")
        if inteiros and not g.wide:
            txt += f", {inteiros} mostrado(s) inteiro(s)"
        return txt

    def _content_shot(self, frame, t: float) -> ContentShot:
        """Plano do vídeo reagido neste quadro. A análise (8 quadros/s) diz
        entre quais amostras foi o corte; aqui a troca acontece no quadro
        exato em que a imagem muda (senão o enquadramento novo entraria
        alguns quadros antes ou depois do corte)."""
        tiny = _tiny_gray(frame, self._mask)
        d = float(np.mean(np.abs(tiny - self._prev_tiny))) if self._prev_tiny is not None else 0.0
        self._prev_tiny = tiny
        shots, j = self.shots, self._cur
        # o tempo de cada amostra tem 1-2 quadros de imprecisão: a troca
        # forçada (corte gradual, sem pico) só vem meia amostra depois
        while j + 1 < len(shots) and t >= shots[j + 1].t0 + 0.5 * self._sample_dt:
            j += 1
        if j + 1 < len(shots):
            nxt = shots[j + 1]
            if t > nxt.t0 - self._sample_dt - 6 * self._half and d >= max(0.5 * nxt.cut_diff, 6.0):
                j += 1
        self._cur = j
        return shots[j]

    def _cam_at(self, t: float) -> CamShot:
        cam = self.cam_shots[0]
        for c in self.cam_shots[1:]:
            if t >= c.t0 - self._half:
                cam = c
        return cam

    def _background(self, frame) -> np.ndarray:
        """Fundo das faixas: o próprio vídeo reagido, bem desfocado e escuro."""
        g = self.geom
        x, y, w, h = self.area
        target = g.out_w / float(g.out_h)
        if w / float(h) > target:
            nw = int(round(h * target))
            x, w = x + (w - nw) // 2, nw
        else:
            nh = int(round(w / target))
            y, h = y + (h - nh) // 2, nh
        small = cv2.resize(frame[y:y + h, x:x + w], (max(g.out_w // 12, 4), max(g.out_h // 12, 4)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 2.0)
        small = cv2.convertScaleAbs(small, alpha=float(getattr(config, "FALLBACK_BG_DARKEN", 0.55)))
        return cv2.resize(small, (g.out_w, g.out_h), interpolation=cv2.INTER_LINEAR)

    def compose(self, frame, t: float, out_w: int, out_h: int, zoom: float = 1.0) -> np.ndarray:
        from .reframer import crop_scaled, hq_resize
        g = self.geom
        shot = self._content_shot(frame, t)
        if g.full and shot.fills:
            out = np.empty((out_h, out_w, 3), np.uint8)
        else:
            out = self._background(frame)

        # conteúdo: recorte parado do plano atual
        x, y, w, h = shot.window
        dx, dy, dw, dh = shot.disp
        out[g.top + dy:g.top + dy + dh, dx:dx + dw] = hq_resize(frame[y:y + h, x:x + w], dw, dh)

        # streamer: tripé; o zoom de edição (momento-chave) fecha no rosto
        cam = self._cam_at(t)
        cx, cy, cw, ch = cam.crop
        z = float(np.clip(zoom, 1.0, self.cam_zoom_max))
        if z > 1.001:
            nw, nh = cw / z, ch / z
            fx, fy = cam.face
            cx = float(np.clip(fx - nw / 2, cx, cx + cw - nw))
            cy = float(np.clip(fy - nh * 0.42, cy, cy + ch - nh))
            cw, ch = nw, nh
        out[g.cam_y:g.cam_y + g.cam_h, g.cam_x:g.cam_x + g.cam_w] = crop_scaled(
            frame, cx, cy, cw, ch, g.cam_w, g.cam_h, sharpen=True)

        # divisória fina entre os painéis
        sep = max(int(round(out_h * 0.002)), 2)
        out[g.seam - sep // 2: g.seam + (sep - sep // 2)] = (18, 18, 18)
        return out


def plan_clip(layout: ReactLayout, source_path: str, start: float, end: float,
              out_w: int, out_h: int, fps: float = 30.0) -> Optional[ReactClipPlan]:
    """Analisa o trecho do clipe (8 quadros/s, baixa resolução, sem guardar
    os quadros) e monta o plano da tela dividida."""
    from .shot_plan import _is_cut, _merge_short, _text_lines
    W, H = _analysis_size(layout.src_w, layout.src_h)
    rate = float(getattr(config, "REACT_ANALYSIS_FPS", 8.0))
    every = max(int(round(rate / 2.0)), 1)   # rosto/texto/facecam: 2 amostras por segundo
    det = _new_yunet()
    bx, by, bw, bh = layout.box_at(W)
    s = layout.src_w / float(W)
    thr = getattr(config, "REACT_CAM_MATCH_MAX_DIFF", 22.0)
    cam_r = (max(bx - 0.05 * bw, 0.0), max(by - 0.05 * bh, 0.0),
             min(bx + 1.05 * bw, float(W)), min(by + 1.05 * bh, float(H)))
    mask = _tiny_mask(cam_r, W, H)

    present, cam_faces, reg = [], [], []
    samples = {}      # amostra -> (rostos do conteúdo, linhas de texto)
    smalls = []       # cinza reduzido das amostras regulares (onde o vídeo reagido mexe)
    cuts = {}         # amostra -> tamanho da troca de imagem
    prev_tiny = prev_hist = None
    pending = -1
    n = 0
    for i, f in enumerate(_iter_frames(source_path, start, end - start, rate, W, H)):
        n = i + 1
        t_small = cv2.resize(f, (_TINY_W, _TINY_H), interpolation=cv2.INTER_AREA)
        tiny = cv2.cvtColor(t_small, cv2.COLOR_BGR2GRAY).astype(np.float32) * mask
        hist = cv2.calcHist([cv2.cvtColor(t_small, cv2.COLOR_BGR2HSV)], [0, 1, 2], None,
                            [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if prev_tiny is not None:
            d = float(np.mean(np.abs(tiny - prev_tiny)))
            if _is_cut(d, float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))):
                cuts[i] = d
                pending = i + 1   # 1ª amostra do plano novo pode pegar a transição
        prev_tiny, prev_hist = tiny, hist
        regular = i % every == 0
        if not regular and i != pending:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = _faces(det, f)
        content = [fc for fc in faces
                   if not (cam_r[0] <= fc[0] <= cam_r[2] and cam_r[1] <= fc[1] <= cam_r[3])]
        lines = [(x, y, x + w, y + h) for x, y, w, h in _text_lines(gray) if _letters(gray, x, y, w, h)]
        lines = [ln for ln in lines if _overlap(ln, cam_r) < 0.2]
        samples[i] = (content, lines)
        if regular:
            in_cam = [fc for fc in faces
                      if bx + 0.03 * bw < fc[0] < bx + 0.97 * bw and by < fc[1] < by + bh
                      and fc[3] < 0.95 * bh]
            diff = layout.cam_matches(gray)
            present.append(bool(in_cam) or (diff is not None and diff < thr))
            cam_faces.append(max(in_cam, key=lambda fc: fc[3]) if in_cam else None)
            reg.append(i)
            smalls.append(cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA))
    if not reg:
        return None

    present = np.array(present)
    min_len = max(int(round(getattr(config, "REACT_MIN_SEGMENT_SECONDS", 3.0) * rate / every)), 1)
    present = _smooth_runs(present, min_len)
    # tela dividida é do CLIPE INTEIRO ou de nada: ligar e desligar no meio
    # (achado real: "react em tela dividida (48% do clipe)") parecia defeito.
    # Só divide se a facecam está na tela em quase todo o trecho.
    if present.mean() < getattr(config, "REACT_MIN_CLIP_PRESENCE", 0.85):
        return None
    present = np.ones_like(present)
    times = np.array(reg, dtype=np.float64) / rate

    # planos do vídeo reagido (flash e chicote não contam como corte)
    edges = _merge_short([0] + sorted(cuts) + [n], max(int(round(0.5 * rate)), 1))
    spans = list(zip(edges, edges[1:]))
    is_mark = _static_text(smalls, spans, W)
    base = _letterbox(smalls, W, H)
    pos = {i: k for k, i in enumerate(reg)}
    needs, areas = [], []
    for a, b in spans:
        idx = [i for i in sorted(samples) if a <= i < b]
        if len(idx) > 1 and idx[0] == a and a > 0:
            idx = idx[1:]
        nd = _shot_needs(idx, samples, is_mark, W, H)
        ar = base
        if nd is None:
            # sem rosto nem texto: o vídeo reagido é onde a imagem mexe (o
            # player, numa live com navegador + chat em volta)
            ks = [pos[i] for i in reg if a <= i < b]
            if len(ks) >= 3:
                ar = _moving_box([smalls[k] for k in ks], cam_r, W, H, base) or base
        needs.append(nd)
        areas.append(ar)
    area = base

    # tudo daqui pra baixo em px da FONTE
    src_box = layout.cam_box
    fhs = [fc[3] * s for fc in cam_faces if fc is not None]
    face_h = float(np.median(fhs)) if fhs else layout.face[3]
    area_s = tuple(v * s for v in area)
    needs_s = [None if nd is None else tuple(v * s for v in nd) for nd in needs]
    areas_s = [tuple(v * s for v in ar) for ar in areas]
    durs = [(b - a) / rate for a, b in spans]
    geom, cam_crop = _geometry(out_w, out_h, src_box, face_h, area_s, needs_s, areas_s, durs)
    shots = [_place_shot(nd, ar, src_box, geom, layout.src_w, layout.src_h, a / rate, cuts.get(a, 0.0))
             for nd, ar, (a, _) in zip(needs_s, areas_s, spans)]
    track = [None if fc is None else (fc[0] * s, fc[1] * s) for fc in cam_faces]
    cam_shots = _cam_shots(src_box, cam_crop, track, times, layout.face)
    cam_up = geom.cam_w / max(cam_crop[0], 1e-3)
    ax0, ay0, ax1, ay1 = (int(round(v)) for v in area_s)
    return ReactClipPlan(layout, times, present, geom, shots, cam_shots, cam_up,
                         (ax0, ay0, ax1 - ax0, ay1 - ay0), mask, fps, rate)


def _letters(gray, x: int, y: int, w: int, h: int) -> bool:
    """Linha de texto tem traço vertical (as letras); listra de bandeira,
    LED do cenário e borda de mesa só têm traço horizontal."""
    box = gray[y:y + h, x:x + w].astype(np.float32)
    if box.size == 0:
        return False
    gx = float(np.mean(np.abs(cv2.Sobel(box, cv2.CV_32F, 1, 0, ksize=3))))
    gy = float(np.mean(np.abs(cv2.Sobel(box, cv2.CV_32F, 0, 1, ksize=3))))
    return gx >= 0.6 * gy


def _static_text(smalls, spans, W: int):
    """Marca d'água do canal, hashtag do programa, logo da live: texto que
    fica parado no mesmo lugar em TODOS os planos (as bordas das letras
    aparecem em quase toda amostra, troque o que trocar atrás). Não é motivo
    pra mostrar a tela inteira. A pergunta de um programa de TV some nos
    closes -- essa continua importante. Devolve uma função linha -> é marca?"""
    if len(spans) < 2 or len(smalls) < 4:
        return lambda ln: False
    k = smalls[0].shape[1] / float(W)
    persist = np.mean([cv2.Canny(g, 60, 150) > 0 for g in smalls], axis=0)
    persist = cv2.dilate(persist.astype(np.float32), np.ones((3, 3), np.uint8))

    def is_mark(ln) -> bool:
        if ln[2] - ln[0] > 0.35 * W:
            return False
        sub = persist[int(ln[1] * k):int(np.ceil(ln[3] * k)), int(ln[0] * k):int(np.ceil(ln[2] * k))]
        edges = sub[sub > 0.25]
        return edges.size >= 12 and float(np.mean(edges >= 0.8)) >= 0.5
    return is_mark


def _corner_bug(ln, W: int, H: int) -> bool:
    """Texto curto encostado num canto: logo/marca do canal do vídeo
    reagido, hashtag do programa (achado no teste: "Instituto Teotônio
    Vilela" no canto puxava o enquadramento até lá). Pergunta, título e
    legenda são mais largos ou mais pro meio."""
    x0, y0, x1, y1 = ln
    return (x1 - x0) < 0.2 * W and (x0 < 0.1 * W or x1 > 0.9 * W) and (y0 < 0.15 * H or y1 > 0.85 * H)


def _shot_needs(idx, samples, is_mark, W: int, H: int):
    """O que precisa aparecer neste plano (x0, y0, x1, y1), px da análise:
    rostos em destaque e texto (pergunta, legenda, placar). None = nada
    específico (cena aberta, paisagem): mostra o vídeo inteiro."""
    if not idx:
        return None
    k = len(idx)
    need_n = max(1, int(np.ceil(0.34 * k)))
    big = H * float(getattr(config, "REACT_CONTENT_FACE_MIN", 0.06))
    boxes = []
    faces = [fc for i in idx for fc in samples[i][0] if fc[3] >= big]
    for fc in faces:
        # rosto que aparece em boa parte do plano (1 detecção solta não conta)
        hits = sum(any(abs(g[0] - fc[0]) < 0.6 * fc[2] and abs(g[1] - fc[1]) < 0.6 * fc[3]
                       for g in samples[i][0]) for i in idx)
        if hits >= need_n:
            cx, cy, fw, fh = fc
            boxes.append((cx - 0.8 * fw, cy - 0.8 * fh, cx + 0.8 * fw, cy + 1.0 * fh))
    lines = [ln for i in idx for ln in samples[i][1]]
    for ln in lines:
        if is_mark(ln) or _corner_bug(ln, W, H):
            continue
        hits = sum(any(_iou(ln, o) >= 0.4 for o in samples[i][1]) for i in idx)
        if hits >= need_n:
            boxes.append(ln)
    if not boxes:
        return None
    arr = np.array(boxes, dtype=np.float64)
    px, py = 0.015 * W, 0.015 * H
    return (max(float(arr[:, 0].min()) - px, 0.0), max(float(arr[:, 1].min()) - py, 0.0),
            min(float(arr[:, 2].max()) + px, float(W)), min(float(arr[:, 3].max()) + py, float(H)))


def _letterbox(smalls, W: int, H: int) -> Tuple[float, float, float, float]:
    """Área do vídeo sem as barras pretas (x0, y0, x1, y1), px da análise."""
    full = (0.0, 0.0, float(W), float(H))
    if not smalls:
        return full
    sh, sw = smalls[0].shape[:2]
    k = sw / float(W)
    peak = np.max(np.stack(smalls), axis=0)   # barra preta: escura em TODA amostra
    dark = float(getattr(config, "LETTERBOX_DARK_LEVEL", 12)) + 4
    rows = np.nonzero(np.percentile(peak, 90, axis=1) > dark)[0]
    cols = np.nonzero(np.percentile(peak, 90, axis=0) > dark)[0]
    if len(rows) < 0.3 * sh or len(cols) < 0.3 * sw:
        return full
    x0, x1, y0, y1 = cols[0] / k, (cols[-1] + 1) / k, rows[0] / k, (rows[-1] + 1) / k
    # barra fina (moldura, 1-2% da tela) não vale a pena tirar
    x0 = 0.0 if x0 < 0.03 * W else x0
    y0 = 0.0 if y0 < 0.03 * H else y0
    x1 = float(W) if x1 > 0.97 * W else x1
    y1 = float(H) if y1 > 0.97 * H else y1
    return float(x0), float(y0), float(x1), float(y1)


def _moving_box(smalls, cam_r, W: int, H: int, base):
    """Maior região que mexe dentro de UM plano (x0, y0, x1, y1), px da
    análise -- o player do vídeo reagido numa live com navegador, chat e
    faixas em volta. Medida numa versão BORRADA da imagem (chat rolando
    muda em detalhe fino e some no borrão). None = não dá pra afirmar
    (região pequena, ou quase a tela toda)."""
    sh, sw = smalls[0].shape[:2]
    k = sw / float(W)
    act = np.std(np.stack([cv2.GaussianBlur(g, (0, 0), 3.0).astype(np.float32) for g in smalls]), axis=0)
    act[int(cam_r[1] * k):int(np.ceil(cam_r[3] * k)), int(cam_r[0] * k):int(np.ceil(cam_r[2] * k))] = 0.0
    hot = float(np.percentile(act, 95))
    if hot < 4.0:
        return None
    on = (act > max(3.0, 0.25 * hot)).astype(np.uint8)
    ks = max(int(0.03 * sw), 3)
    on = cv2.morphologyEx(on, cv2.MORPH_CLOSE, np.ones((ks, ks), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
    if n < 2:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (stats[j, c] / k for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                             cv2.CC_STAT_HEIGHT))
    base_area = (base[2] - base[0]) * (base[3] - base[1])
    if not (0.35 * base_area <= w * h <= 0.8 * base_area):
        return None
    return (max(x, base[0]), max(y, base[1]), min(x + w, base[2]), min(y + h, base[3]))


def _geometry(out_w: int, out_h: int, box, face_h: float, area, needs, areas, durs):
    """Tamanho dos painéis do clipe (px da saída) e o recorte da facecam
    (largura, altura em px da fonte).

    Streamer: o maior painel em que a facecam não passa de
    REACT_CAM_MAX_UPSCALE de ampliação. Facecam pequena (o normal em react:
    ~20% da largura da tela) não chega lá nem inteira -- aí ela entra
    INTEIRA, na proporção dela (ampliação mínima possível). Antes o recorte
    era mais fechado que a facecam (rosto em 30% do painel): ~4x, borrado.
    Conteúdo: painel alto (o resto da tela) quando o que importa cabe
    recortado; senão (pergunta de programa de TV, duas pessoas nas pontas,
    cena aberta) o vídeo inteiro na largura da tela."""
    bx, by, bw, bh = box
    umax = float(getattr(config, "REACT_CAM_MAX_UPSCALE", 2.4))
    uhard = float(getattr(config, "REACT_CAM_MAX_UPSCALE_HARD", 3.3))
    frac = float(np.clip(getattr(config, "REACT_CONTENT_FRAC", 0.5), 0.3, 0.7))
    cam_max_h = int(round(out_h * (1.0 - frac) / 2.0)) * 2
    tface = float(getattr(config, "REACT_CAM_FACE_FRAC", 0.30))
    cam_h = None
    for hs in range(cam_max_h, int(out_h * 0.3), -8):
        asp = out_w / float(hs)
        ch_max = min(float(bh), bw / asp)
        ch_min = hs / umax
        if ch_min <= ch_max:
            ch = float(np.clip(face_h / tface, ch_min, ch_max))
            cam_h, crop = hs, (ch * asp, ch)
            break
    cam_w = out_w
    if cam_h is None:
        # facecam pequena: inteira
        cw, ch = float(bw), float(bh)
        cam_h = int(round(out_w * ch / cw / 2.0)) * 2
        if cam_h > cam_max_h:
            cam_h = cam_max_h
            ch = cw * cam_h / out_w
        if out_w / cw > uhard:
            # pequena demais até pra isso (fonte 720p): painel mais estreito
            # que a tela, sobre o fundo desfocado, em vez de virar borrão
            cam_w = int(round(cw * uhard / 2.0)) * 2
            cam_h = int(round(ch * uhard / 2.0)) * 2
        crop = (cw, ch)

    ax0, ay0, ax1, ay1 = area
    aw, ah = ax1 - ax0, ay1 - ay0
    tall_h = out_h - cam_h
    wide_h = min(int(round(out_w * ah / aw / 2.0)) * 2, tall_h)
    total = max(sum(durs), 1e-6)
    wide_t = 0.0
    for nd, ar, d in zip(needs, areas, durs):
        # sem rosto/texto: o plano inteiro importa (paisagem, jogo, player).
        # Pede a largura inteira o plano que, no painel alto, ficaria pequeno
        # (pergunta + cenário de um programa de TV 16:9: 51% do painel); um
        # plano de mesa com 5 pessoas fica com ~75% -- dá pra mostrar inteiro
        if _fill_ratio(nd if nd is not None else ar, ar, out_w, tall_h) < 0.6:
            wide_t += d
    wide = wide_t >= float(getattr(config, "REACT_WIDE_LAYOUT_MIN_FRAC", 0.4)) * total \
        or tall_h <= 1.12 * wide_h
    content_h = wide_h if wide else tall_h
    rest = out_h - content_h - cam_h
    # faixa livre: um pouco mais embaixo (a interface do Shorts cobre o rodapé)
    top = int(round(rest * 0.45 / 2.0)) * 2
    geom = Geometry(out_w, out_h, top, content_h, (out_w - cam_w) // 2, top + content_h,
                    cam_w, cam_h, wide)
    return geom, crop


def _fill_ratio(need, area, pw: int, ph: int) -> float:
    """Quanto do painel pw x ph o plano ocupa mostrando `need` inteiro (1 =
    preenche; o resto é faixa desfocada)."""
    asp = pw / float(ph)
    aw, ah = area[2] - area[0], area[3] - area[1]
    fw = min(aw, ah * asp)
    if _fits(need, fw, fw / asp):
        return 1.0
    nw, nh = need[2] - need[0], need[3] - need[1]
    w = float(np.clip(max(nw, nh * asp), nw, aw))
    h = float(np.clip(max(nh, nw / asp), nh, ah))
    sc = min(pw / w, ph / h)
    return (w * sc) * (h * sc) / float(pw * ph)


def _fits(need, fw: float, fh: float) -> bool:
    """O que importa no plano cabe numa janela fw x fh (até 10% maior: a
    folga em volta do rosto/texto pode ser cortada)?"""
    return need is not None and need[2] - need[0] <= 1.1 * fw and need[3] - need[1] <= 1.1 * fh


def _place_shot(need, area, box, geom: Geometry, src_w: int, src_h: int, t0: float,
                cut_diff: float) -> ContentShot:
    """Enquadramento parado de um plano do vídeo reagido."""
    ax0, ay0, ax1, ay1 = area
    aw, ah = ax1 - ax0, ay1 - ay0
    pw, ph = geom.out_w, geom.content_h
    asp = pw / float(ph)
    fw, fh = min(aw, ah * asp), min(aw, ah * asp) / asp   # maior janela no formato do painel
    cam = (box[0], box[1], box[0] + box[2], box[1] + box[3])
    if need is None and fw >= 0.88 * aw and fh >= 0.88 * ah:
        need = area   # sem nada específico, mas quase no formato do painel: preenche (corta uma lasquinha)
    if _fits(need, fw, fh):
        # cabe: preenche o painel, cobrindo o que importa; foge da facecam se
        # der. Um pouco maior que o painel: corta só a folga em volta (a
        # caixa do rosto já inclui ombros e margem)
        lo_x, hi_x = max(ax0, need[2] - fw), min(need[0], ax1 - fw)
        lo_y, hi_y = max(ay0, need[3] - fh), min(need[1], ay1 - fh)
        if lo_x > hi_x:
            lo_x = hi_x = float(np.clip((need[0] + need[2] - fw) / 2.0, ax0, ax1 - fw))
        if lo_y > hi_y:
            lo_y = hi_y = float(np.clip((need[1] + need[3] - fh) / 2.0, ay0, ay1 - fh))
        cx = (need[0] + need[2]) / 2.0 - fw / 2.0
        cy = (need[1] + need[3]) / 2.0 - fh / 2.0
        best = None
        for x in np.linspace(lo_x, max(hi_x, lo_x), 13):
            for y in np.linspace(lo_y, max(hi_y, lo_y), 5):
                cost = _overlap((x, y, x + fw, y + fh), cam) * 10.0 + \
                    (abs(x - cx) / max(fw, 1.0) + abs(y - cy) / max(fh, 1.0))
                if best is None or cost < best[0]:
                    best = (cost, x, y)
        x, y = best[1], best[2]
        w, h = fw, fh
        disp = (0, 0, pw, ph)
        fills = True
    else:
        if need is None:
            x, y, w, h = ax0, ay0, aw, ah
        else:
            # só o que importa (+ o que o formato do painel deixar), inteiro
            nw, nh = need[2] - need[0], need[3] - need[1]
            w = float(np.clip(max(nw, nh * asp), nw, aw))
            h = float(np.clip(max(nh, nw / asp), nh, ah))
            x = float(np.clip((need[0] + need[2] - w) / 2.0, ax0, ax1 - w))
            y = float(np.clip((need[1] + need[3] - h) / 2.0, ay0, ay1 - h))
        sc = min(pw / w, ph / h)
        dw, dh = min(int(round(w * sc / 2.0)) * 2, pw), min(int(round(h * sc / 2.0)) * 2, ph)
        fills = dw >= pw - 2 and dh >= ph - 2
        if fills:
            dw, dh = pw, ph
        disp = ((pw - dw) // 2, (ph - dh) // 2, dw, dh)
    xi, yi = int(round(x)), int(round(y))
    wi, hi = min(int(round(w)), src_w - xi), min(int(round(h)), src_h - yi)
    return ContentShot(t0, cut_diff, (xi, yi, max(wi, 2), max(hi, 2)), disp, fills)


def _cam_shots(box, crop, track, times, default_face) -> List[CamShot]:
    """Enquadramento de TRIPÉ do streamer: parado. Só muda (em corte seco)
    se ele sair da posição por um tempo -- levantou, foi pro lado."""
    bx, by, bw, bh = box
    cw, ch = crop
    pts = [(t, p) for t, p in zip(times, track) if p is not None]
    if not pts:
        pts = [(0.0, (default_face[0], default_face[1]))]

    def place(sel):
        fx = float(np.median([p[0] for _, p in sel]))
        fy = float(np.median([p[1] for _, p in sel]))
        x = float(np.clip(fx - cw / 2.0, bx, bx + bw - cw))
        y = float(np.clip(fy - ch * 0.42, by, by + bh - ch))
        return (x, y, cw, ch), (fx, fy)

    if cw >= bw - 1.0 and ch >= bh - 1.0:
        # a facecam inteira: não tem pra onde mexer
        return [CamShot(0.0, *place(pts))]
    hold = 3.0
    segs, start = [], 0
    while start < len(pts):
        _, (fx, fy) = place([q for q in pts[start:] if q[0] < pts[start][0] + hold])
        end, out_since = len(pts), None
        for k in range(start, len(pts)):
            px, py = pts[k][1]
            if abs(px - fx) > 0.22 * cw or abs(py - fy) > 0.22 * ch:
                out_since = k if out_since is None else out_since
                if pts[k][0] - pts[out_since][0] >= 2.0:
                    end = out_since
                    break
            else:
                out_since = None
        end = max(end, start + 1)
        segs.append((start, end))
        start = end
    # trecho curto não vira enquadramento próprio (junta com o anterior)
    merged = []
    for a, b in segs:
        if merged and pts[b - 1][0] - pts[a][0] < hold:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if len(merged) > 1 and pts[merged[0][1] - 1][0] - pts[merged[0][0]][0] < hold:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    return [CamShot(0.0 if k == 0 else pts[a][0], *place(pts[a:b])) for k, (a, b) in enumerate(merged)]


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
