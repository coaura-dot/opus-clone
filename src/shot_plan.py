"""
Plano por cena: analisa o trecho do clipe ANTES de renderizar e decide o
enquadramento de cada plano (cena entre dois cortes) do vídeo de origem.

Por que existe (achado real, vídeo estilo documentário/vlog com muito corte,
zoom e câmera na mão): decidindo quadro a quadro, sem saber onde o plano
termina, o reenquadramento
  - trocava de layout no meio do plano (rosto -> fundo borrado -> rosto),
    porque o rosto demora uns quadros pra ser achado depois de cada corte;
  - somava o próprio zoom ao zoom que a câmera do vídeo já fazia ("zoom
    duplo") e perseguia o rosto que a câmera na mão já balançava;
  - mostrava as barras pretas de cinema do vídeo (com a legenda do vídeo
    original dentro delas) em cima e embaixo do recorte;
  - mostrava todo plano sem rosto (b-roll: aeroporto, carro, paisagem) como
    uma faixa fina no meio do fundo borrado;
  - cortava o texto de cartelas de título.

Aqui, com o plano inteiro em mãos:
  - os cortes são achados de antemão (flash e chicote não viram corte);
  - cada plano ganha UM layout do começo ao fim:
      single  um rosto: recorte 9:16 com tamanho FIXO no plano inteiro (o zoom
              da câmera original aparece como está) e posição travada, ou um
              caminho suavizado sem atraso quando o rosto anda de verdade;
      multi   duas ou mais pessoas que não cabem num recorte só: o
              rastreamento de quem está falando de sempre (reframer.py);
      broll   sem rosto: recorte 9:16 em tela cheia, parado, no ponto com
              mais detalhe da imagem;
      text    cartela/print com texto que não cabe no recorte: quadro
              inteiro sobre fundo borrado;
  - barras pretas (letterbox/pillarbox) são medidas por plano e ficam de
    fora do recorte.
"""
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config

AN_W = 640           # largura da cópia usada na análise
_TINY_W = 64         # largura da cópia usada pra achar corte
_FACE_EVERY = 3      # detecta rosto a cada N quadros (10/s a 30fps)
_SCENE_EVERY = 15    # barras/texto/detalhe a cada N quadros (2/s a 30fps)


@dataclass
class Shot:
    start: int                       # primeiro quadro (relativo ao início do clipe)
    end: int                         # quadro depois do último
    kind: str                        # "single" | "multi" | "broll" | "text"
    bounds: Tuple[int, int, int, int]  # área útil sem barras pretas (x0, y0, x1, y1), px da fonte
    crop_h: float = 0.0              # altura do recorte 9:16 (px da fonte)
    xs: Optional[np.ndarray] = None  # centro x do recorte por quadro (px da fonte)
    ys: Optional[np.ndarray] = None  # y do rosto por quadro (px da fonte)
    text_x: Optional[Tuple[int, int]] = None  # cartela: faixa horizontal com o texto (px da fonte)

    def pos(self, frame_idx: int) -> Tuple[float, float]:
        i = int(np.clip(frame_idx - self.start, 0, len(self.xs) - 1))
        return float(self.xs[i]), float(self.ys[i])


class ShotPlan:
    def __init__(self, shots: List[Shot], n_frames: int, full: Tuple[int, int, int, int]):
        self.shots = shots
        self.full = full
        self._idx = np.zeros(max(n_frames, 1), dtype=np.int32)
        for i, s in enumerate(shots):
            self._idx[s.start:s.end] = i

    def shot_at(self, frame_idx: int) -> Shot:
        return self.shots[int(self._idx[min(max(frame_idx, 0), len(self._idx) - 1)])]

    def is_cut(self, frame_idx: int) -> bool:
        return frame_idx > 0 and self.shot_at(frame_idx).start == frame_idx

    def summary(self) -> str:
        kinds = {}
        for s in self.shots:
            kinds[s.kind] = kinds.get(s.kind, 0) + 1
        names = {"single": "rosto", "multi": "conversa", "broll": "sem rosto", "text": "texto",
                 "facefit": "close gigante"}
        parts = ", ".join(f"{names.get(k, k)} {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        bars = sum(1 for s in self.shots if s.bounds != self.full)
        txt = f"{len(self.shots)} cena(s): {parts}"
        if bars:
            txt += f"; barras pretas tiradas em {bars}"
        return txt


# ---------------------------------------------------------------------------
# medidas por quadro

def _bars(profile: np.ndarray, dark: float) -> Tuple[int, int]:
    """Quantas linhas (ou colunas) escuras seguidas há em cada ponta."""
    lit = np.nonzero(profile > dark)[0]
    if len(lit) == 0:
        return len(profile), len(profile)
    return int(lit[0]), int(len(profile) - 1 - lit[-1])


def _text_lines(gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Linhas de texto grandes (cartela, título, print) -- (x, y, w, h)."""
    H, W = gray.shape[:2]
    if H < 40 or W < 40:
        return []
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 50, 7), 3)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (0.025 * H <= h <= 0.2 * H) or w < 2.5 * h or w < 0.08 * W:
            continue
        if area / float(w * h) < 0.45:
            continue
        box = gray[y:y + h, x:x + w]
        if float(box.std()) < 35:  # texto tem contraste forte com o fundo
            continue
        out.append((int(x), int(y), int(w), int(h)))
    return out


def _detail_profile(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return (np.abs(gx) + np.abs(gy)).sum(axis=0)


# ---------------------------------------------------------------------------

def build_plan(source_path: str, start: float, duration: float, src_w: int, src_h: int,
               fps: float, detector, verbose: bool = True) -> Optional[ShotPlan]:
    """Analisa o trecho [start, start+duration] e devolve o plano por cena
    (ou None se não deu pra analisar -- o reframer volta pro modo antigo)."""
    if detector is None or src_w <= 0 or src_h <= 0:
        return None
    an_w = min(AN_W, src_w)
    an_h = max(int(round(src_h * an_w / src_w / 2.0)) * 2, 2)
    sx, sy = src_w / an_w, src_h / an_h
    cmd = ["ffmpeg", "-v", "error", "-ss", str(max(start, 0.0)), "-i", str(source_path),
           "-t", str(max(duration, 0.05)), "-vf", f"fps={fps:.6f}",  # mesma grade de quadros do render
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{an_w}x{an_h}", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None

    min_score = getattr(config, "YUNET_SCORE_THRESHOLD", 0.6)
    tiny_h = max(int(round(an_h * _TINY_W / an_w)), 2)
    diffs, hdist = [], []
    faces = {}     # quadro -> [(cx, cy, w, h)] em px da análise
    scene = {}     # quadro -> (linhas, colunas, gray)  (medidas leves; gray só pra texto/detalhe)
    prev_tiny = prev_hist = None
    force_detect = 0
    since_cut = 10 ** 9
    frame_bytes = an_w * an_h * 3
    i = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(an_h, an_w, 3)
            tiny = cv2.resize(frame, (_TINY_W, tiny_h), interpolation=cv2.INTER_AREA)
            tiny_g = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY).astype(np.float32)
            hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            if prev_tiny is None:
                diffs.append(0.0)
                hdist.append(0.0)
            else:
                diffs.append(float(np.mean(np.abs(tiny_g - prev_tiny))))
                hdist.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
                if _is_cut(diffs[-1], hdist[-1]):
                    force_detect = 3  # quadros logo depois do corte: rosto do plano novo
                    since_cut = 0
            prev_tiny, prev_hist = tiny_g, hist

            if i % _FACE_EVERY == 0 or force_detect > 0:
                faces[i] = _faces(detector, frame, min_score)
                force_detect = max(force_detect - 1, 0)
            # barras/texto/detalhe: 2x por segundo e logo depois de cada corte
            # (plano curto também precisa de amostra; o 1º quadro pode ser flash)
            if i % _SCENE_EVERY == 0 or since_cut in (1, 6):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scene[i] = (np.median(gray, axis=1), np.median(gray, axis=0), gray)
            since_cut += 1
            i += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
    n = i
    if n < 2:
        return None

    cuts = [k for k in range(1, n) if _is_cut(diffs[k], hdist[k])]
    bounds_list = _merge_short([0] + cuts + [n], int(round(getattr(config, "SHOT_MIN_SECONDS", 0.4) * fps)))

    an_bounds = _all_bounds(bounds_list, scene, an_w, an_h)
    shots = [_plan_shot(a, b, faces, scene, bb, sx, sy, fps)
             for a, b, bb in zip(bounds_list[:-1], bounds_list[1:], an_bounds)]
    plan = ShotPlan(shots, n, (0, 0, src_w, src_h))
    if verbose:
        print(f"    [cenas] {plan.summary()}")
    return plan


def _is_cut(d: float, h: float) -> bool:
    thr = float(getattr(config, "SCENE_CUT_THRESHOLD", 28.0))
    return d > thr or (d > thr * 0.45 and h > 0.4)


def _faces(detector, frame, min_score) -> list:
    from .reframer import _detect_yunet
    return _detect_yunet(detector, frame, frame.shape[1], min_score)


def _merge_short(edges: List[int], min_len: int) -> List[int]:
    """Plano curto demais (flash, chicote, transição) entra no plano seguinte."""
    edges = sorted(set(edges))
    changed = True
    while changed and len(edges) > 2:
        changed = False
        for k in range(len(edges) - 1):
            if edges[k + 1] - edges[k] < min_len:
                if k + 1 < len(edges) - 1:
                    del edges[k + 1]      # junta com o seguinte
                elif k > 0:
                    del edges[k]          # último plano: junta com o anterior
                else:
                    break
                changed = True
                break
    return edges


def _shot_samples(scene, a, b) -> list:
    samples = [scene[k] for k in sorted(scene) if a <= k < b]
    if not samples and scene:
        samples = [scene[min(scene, key=lambda k: abs(k - a))]]
    return samples


def _profiles(samples):
    # mediana entre as amostras: um flash branco ou um quadro escuro isolado
    # não mudam a medida
    return (np.median(np.stack([s[0] for s in samples]), axis=0),
            np.median(np.stack([s[1] for s in samples]), axis=0))


def _bar_size(profile: np.ndarray, dark: float) -> int:
    """Barra de cinema é simétrica: vale a MENOR das duas pontas (uma roupa
    escura encostada na barra de baixo não pode comer a imagem). 0 = sem barra."""
    bar = min(_bars(profile, dark))
    return bar if 0.04 * len(profile) <= bar <= 0.25 * len(profile) else 0


def _all_bounds(edges, scene, an_w, an_h) -> list:
    """Área útil (sem barras pretas) de cada plano, em px da análise.

    Barra preta de verdade é preto puro (LETTERBOX_DARK_LEVEL); numa cena
    noturna a imagem inteira pode ficar quase tão escura quanto ela e a
    medida falha. Por isso a barra que aparece na MAIORIA do clipe vale
    também nos planos escuros, desde que ali as mesmas faixas estejam escuras."""
    full = (0, 0, an_w, an_h)
    if not getattr(config, "LETTERBOX_CROP_ENABLED", True):
        return [(full, full)] * (len(edges) - 1)
    dark = float(getattr(config, "LETTERBOX_DARK_LEVEL", 12))
    per = []
    for a, b in zip(edges[:-1], edges[1:]):
        smp = _shot_samples(scene, a, b)
        if not smp:
            per.append((b - a, None, None, 0, 0))
            continue
        rows, cols = _profiles(smp)
        per.append((b - a, rows, cols, _bar_size(rows, dark), _bar_size(cols, dark)))

    def majority(idx):
        votes = {}
        for p in per:
            if p[idx]:
                votes[p[idx]] = votes.get(p[idx], 0) + p[0]
        if not votes:
            return 0
        bar, dur = max(votes.items(), key=lambda kv: kv[1])
        near = sum(d for v, d in votes.items() if abs(v - bar) <= 2)
        return bar if near >= 0.5 * sum(p[0] for p in per) else 0

    def box(bx, by):
        x0, x1 = (bx + 2, an_w - bx - 2) if bx else (0, an_w)
        y0, y1 = (by + 2, an_h - by - 2) if by else (0, an_h)
        return x0, y0, x1, y1

    maj_y, maj_x = majority(3), majority(4)
    out = []
    for dur, rows, cols, by, bx in per:
        # a barra do clipe vale onde as mesmas faixas estão escuras: cobre a
        # cena noturna (barra não medida) e a cartela de fundo preto (barra
        # medida grande demais, cortando o texto que entra e sai)
        my = mx = 0
        if rows is not None:
            if maj_y and max(rows[:maj_y].max(), rows[-maj_y:].max()) <= 2 * dark:
                by = my = maj_y
            if maj_x and max(cols[:maj_x].max(), cols[-maj_x:].max()) <= 2 * dark:
                bx = mx = maj_x
        out.append((box(bx, by), box(mx, my)))
    return out


def _plan_shot(a, b, faces, scene, an_bounds, sx, sy, fps) -> Shot:
    samples = _shot_samples(scene, a, b)
    (bx0, by0, bx1, by1), clip_bounds = an_bounds
    bw, bh = bx1 - bx0, by1 - by0
    bounds = (int(round(bx0 * sx)), int(round(by0 * sy)), int(round(bx1 * sx)), int(round(by1 * sy)))
    content_h = bounds[3] - bounds[1]
    content_w = bounds[2] - bounds[0]
    aspect = config.TARGET_WIDTH / config.TARGET_HEIGHT
    full_crop_h = min(float(content_h), content_w / aspect)

    # rostos dentro da área útil, grandes o bastante pra contar
    fs = []
    for k in sorted(faces):
        if not a <= k < b:
            continue
        good = [f for f in faces[k] if bx0 <= f[0] <= bx1 and by0 <= f[1] <= by1 and f[3] >= 0.035 * bh]
        fs.append((k, good))
    with_face = [(k, g) for k, g in fs if g]
    presence = len(with_face) / max(len(fs), 1)

    # texto grande (cartela/título)
    text_hits, big_text_hits = 0, 0
    crop_w_an = full_crop_h * aspect / sx
    tx = []
    for s in samples:
        lines = _text_lines(s[2][by0:by1, bx0:bx1])
        tx += [(x, x + w) for (x, _, w, _) in lines]
        widths = [w for (_, _, w, _) in lines]
        if sum(widths) > 1.5 * crop_w_an or any(w > 0.95 * crop_w_an for w in widths):
            text_hits += 1
        if any(h >= 0.08 * bh and w > crop_w_an for (_, _, w, h) in lines):
            big_text_hits += 1
    is_text = samples and text_hits >= max(1, len(samples) / 2.0)
    is_big_text = samples and big_text_hits >= max(1, len(samples) / 2.0)

    n = b - a
    frames = np.arange(a, b, dtype=np.float64)
    min_crop_h = min(config.TARGET_HEIGHT / max(getattr(config, "SUBJECT_MAX_UPSCALE", 3.0), 1.0), full_crop_h)
    target_frac = getattr(config, "SUBJECT_TARGET_FACE_FRAC", 0.12)

    # close gigante em que o detector perde o rosto em boa parte das amostras
    # (rosto cortado pela borda, borrão de movimento): poucas detecções
    # enormes já bastam -- senão o plano virava "sem rosto" e o recorte caía
    # no meio do rosto do mesmo jeito
    crop_w_full = full_crop_h * aspect
    big = [(k, max(g, key=lambda f: f[2])) for k, g in with_face
           if max(f[2] for f in g) * sx > getattr(config, "SHOT_FACE_TOO_WIDE", 0.8) * crop_w_full]
    if 0 < presence < 0.4 and len(big) >= max(1, 0.15 * len(fs)) and not is_big_text:
        cx = float(np.median([f[0] for _, f in big])) * sx
        fw = float(np.median([f[2] for _, f in big])) * sx
        w = float(np.clip(fw * 1.25, crop_w_full, content_w))
        x0 = float(np.clip(cx - w / 2, bounds[0], bounds[2] - w))
        return Shot(a, b, "facefit", bounds, text_x=(int(round(x0)), int(round(x0 + w))))

    if presence >= 0.4 and not is_big_text:
        counts, spans = [], []
        for _, g in with_face:
            top_h = max(f[3] for f in g)
            main = [f for f in g if f[3] >= 0.6 * top_h]
            counts.append(len(main))
            spans.append((min(f[0] - f[2] / 2 for f in main), max(f[0] + f[2] / 2 for f in main),
                          float(np.median([f[3] for f in main]))))
        multi = float(np.median(counts)) >= 2
        if multi:
            # cabem todos num recorte só? (duas pessoas lado a lado no close)
            span_w = float(np.median([s[1] - s[0] for s in spans])) * sx
            med_h = float(np.median([s[2] for s in spans])) * sy
            crop_h = float(np.clip(med_h / target_frac, min_crop_h, full_crop_h))
            if span_w <= 0.8 * crop_h * aspect:
                track = []
                for k, g in with_face:
                    top_h = max(f[3] for f in g)
                    main = [f for f in g if f[3] >= 0.6 * top_h]  # sem os rostos pequenos do fundo
                    track.append((k, (min(f[0] for f in main) + max(f[0] for f in main)) / 2.0,
                                  float(np.mean([f[1] for f in main])), float(np.median([f[3] for f in main]))))
                return _single(a, b, frames, track, bounds, sx, sy, crop_h, full_crop_h, fps, min_crop_h,
                               target_frac)
            return Shot(a, b, "multi", bounds)
        # um rosto: segue o mesmo rosto ao longo do plano (o mais perto do anterior)
        track, prev = [], None
        for k, g in with_face:
            f = max(g, key=lambda f: f[3]) if prev is None else \
                min(g, key=lambda f: abs(f[0] - prev[0]) + abs(f[1] - prev[1]) - 0.5 * f[3])
            prev = f
            track.append((k, f[0], f[1], f[3], f[2]))
        # CLOSE GIGANTE: rosto mais largo que o recorte 9:16 (achado real,
        # Podpah: zoom de edição no rosto ocupando a tela) -- recortar só
        # mostrava nariz e boca por 12 s. Mostra o rosto INTEIRO, com folga,
        # ampliado até onde cabe, sobre fundo borrado.
        face_w = float(np.median([t[4] for t in track])) * sx
        crop_w_full = full_crop_h * aspect
        if face_w > getattr(config, "SHOT_FACE_TOO_WIDE", 0.8) * crop_w_full:
            cx = float(np.median([t[1] for t in track])) * sx
            w = float(np.clip(face_w * 1.25, crop_w_full, content_w))
            x0 = float(np.clip(cx - w / 2, bounds[0], bounds[2] - w))
            return Shot(a, b, "facefit", bounds, text_x=(int(round(x0)), int(round(x0 + w))))
        return _single(a, b, frames, track, bounds, sx, sy, None, full_crop_h, fps, min_crop_h, target_frac)

    if is_text or is_big_text:
        # cartela: só a barra do clipe inteiro (fundo preto da cartela não é barra)
        if (bx0, by0, bx1, by1) != clip_bounds:
            bx0, by0, bx1, by1 = clip_bounds
            bw = bx1 - bx0
            bounds = (int(round(bx0 * sx)), int(round(by0 * sy)), int(round(bx1 * sx)), int(round(by1 * sy)))
            content_w = bounds[2] - bounds[0]
            tx = [(t0 + (an_bounds[0][0] - bx0), t1 + (an_bounds[0][0] - bx0)) for t0, t1 in tx]
        # amplia só a faixa com o texto (com margem), em vez do quadro todo
        text_x = None
        if tx:
            lo, hi = min(t[0] for t in tx), max(t[1] for t in tx)
            margin = 0.05 * bw
            x0 = (bx0 + lo - margin) * sx
            x1 = (bx0 + hi + margin) * sx
            min_w = full_crop_h * aspect
            if x1 - x0 < min_w:
                c = (x0 + x1) / 2.0
                x0, x1 = c - min_w / 2.0, c + min_w / 2.0
            x0, x1 = max(x0, bounds[0]), min(x1, bounds[2])
            if x1 - x0 < 0.95 * content_w:
                text_x = (int(round(x0)), int(round(x1)))
        return Shot(a, b, "text", bounds, text_x=text_x)

    # sem rosto: tela cheia, parado no ponto com mais detalhe
    crop_w_px = full_crop_h * aspect
    cx = (bounds[0] + bounds[2]) / 2.0
    if samples:
        prof = np.median(np.stack([_detail_profile(s[2][by0:by1, bx0:bx1]) for s in samples]), axis=0)
        win = max(int(round(crop_w_px / sx)), 1)
        if win < len(prof):
            csum = np.convolve(prof, np.ones(win), mode="valid")
            pos = np.arange(len(csum)) + win / 2.0
            prior = 1.0 - 0.35 * np.abs(pos - len(prof) / 2.0) / (len(prof) / 2.0)  # leve preferência pelo centro
            cx = bounds[0] + float(pos[int(np.argmax(csum * prior))]) * sx
    # y: recorte encostado no topo da área útil (com a altura toda, é o único)
    return Shot(a, b, "broll", bounds, crop_h=full_crop_h,
                xs=np.full(n, cx), ys=np.full(n, bounds[1] + full_crop_h * config.HEADROOM_RATIO))


def _single(a, b, frames, track, bounds, sx, sy, crop_h, full_crop_h, fps, min_crop_h, target_frac) -> Shot:
    ks = np.array([t[0] for t in track], dtype=np.float64)
    xs = np.array([t[1] for t in track]) * sx
    ys = np.array([t[2] for t in track]) * sy
    hs = np.array([t[3] for t in track]) * sy
    # fora: posição longe da típica do plano (o plano curto de outra pessoa
    # que entrou junto, uma detecção solta)
    keep = np.abs(xs - np.median(xs)) <= 0.35 * float(np.median(hs)) / target_frac * \
        config.TARGET_WIDTH / config.TARGET_HEIGHT + 1e-6
    if keep.sum() >= max(2, 0.6 * len(xs)):
        ks, xs, ys, hs = ks[keep], xs[keep], ys[keep], hs[keep]
    if len(xs) >= 3:  # tira detecção solta (mão, reflexo)
        xs = _median3(xs)
        ys = _median3(ys)
    if crop_h is None:
        # tamanho FIXO no plano: o zoom que a câmera original faz aparece como
        # está, em vez de o programa compensar (ou somar) por cima
        crop_h = max(float(np.median(hs)) / target_frac, float(np.max(hs)) / 0.45)
        crop_h = float(np.clip(crop_h, min_crop_h, full_crop_h))
    crop_w = crop_h * config.TARGET_WIDTH / config.TARGET_HEIGHT
    px = np.interp(frames, ks, xs)
    py = np.interp(frames, ks, ys)
    sigma = max(getattr(config, "SHOT_PATH_SMOOTH_SECONDS", 0.5) * fps, 1.0)
    if np.ptp(px) <= getattr(config, "SHOT_LOCK_X", 0.15) * crop_w:
        px = np.full_like(px, float(np.median(xs)))   # câmera parada: o rosto mexe dentro do quadro
    else:
        px = _smooth(px, sigma)
    if np.ptp(py) <= getattr(config, "SHOT_LOCK_Y", 0.10) * crop_h:
        py = np.full_like(py, float(np.median(ys)))
    else:
        py = _smooth(py, sigma)
    return Shot(a, b, "single", bounds, crop_h=crop_h, xs=px, ys=py)


def _median3(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    out[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    return out


def _smooth(v: np.ndarray, sigma: float) -> np.ndarray:
    """Suavização sem atraso (olha pra frente e pra trás -- dá porque o
    plano inteiro já é conhecido)."""
    r = int(3 * sigma)
    if len(v) < 3 or r < 1:
        return v
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    padded = np.pad(v, r, mode="edge")
    return np.convolve(padded, k, mode="valid")
