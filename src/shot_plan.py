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
      split   tela dividida NO VÍDEO ORIGINAL: uma imagem (foto, print,
              vídeo reagido) de um lado e a câmera da pessoa do outro. Sai
              em pé com a imagem em cima e a pessoa embaixo (ver _find_split);
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
    content: Optional[Tuple[int, int, int, int]] = None  # split: área da imagem (x0, y0, x1, y1), px da fonte
    cam: Optional[Tuple[int, int, int, int]] = None      # split: área da câmera da pessoa
    face: Optional[Tuple[float, float, float]] = None    # split: rosto (cx, cy, altura), px da fonte

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
        names = {"single": "rosto", "multi": "conversa", "speaker": "conversa (corta pra quem fala)",
                 "broll": "sem rosto", "text": "texto", "facefit": "close gigante/cena inteira",
                 "split": "imagem + pessoa (tela dividida)"}
        parts = ", ".join(f"{names.get(k, k)} {n}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1]))
        bars = sum(1 for s in self.shots if s.bounds != self.full)
        txt = f"{len(self.shots)} cena(s): {parts}"
        cuts = sum(getattr(s, "switches", 0) for s in self.shots)
        if cuts:
            txt += f"; {cuts} corte(s) pra quem fala"
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
               fps: float, detector, verbose: bool = True, words=None) -> Optional[ShotPlan]:
    """Analisa o trecho [start, start+duration] e devolve o plano por cena
    (ou None se não deu pra analisar -- o reframer volta pro modo antigo).
    `words`: palavras da transcrição (tempo do vídeo original), pra
    confirmar tela dividida pela fala ("essa foto aqui")."""
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
    colprof = []   # quadro -> quanto cada coluna da miniatura mudou (troca de só METADE da tela)
    faces = {}     # quadro -> [(cx, cy, w, h)] em px da análise
    acts = {}      # quadro -> [atividade da boca de cada rosto] (None = sem quadro anterior)
    prev_det = None
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
                colprof.append(np.zeros(tiny_g.shape[1], np.float32))
            else:
                colprof.append(np.abs(tiny_g - prev_tiny).mean(axis=0))
                diffs.append(float(np.mean(np.abs(tiny_g - prev_tiny))))
                hdist.append(float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)))
                if _is_cut(diffs[-1], hdist[-1]):
                    force_detect = 3  # quadros logo depois do corte: rosto do plano novo
                    since_cut = 0
                    prev_det = None   # boca do plano anterior não compara com a do novo
            prev_tiny, prev_hist = tiny_g, hist

            if i % _FACE_EVERY == 0 or force_detect > 0:
                faces[i] = _faces(detector, frame, min_score)
                force_detect = max(force_detect - 1, 0)
                # quem está FALANDO: a boca muda entre uma detecção e outra (a
                # região dos olhos desconta o movimento da cabeça)
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cur = [(f, _face_patches(g, f)) for f in faces[i]]
                acts[i] = [_mouth_activity(f, pt, prev_det[1]) if prev_det else None for f, pt in cur]
                prev_det = (i, cur)
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

    def confirm_split(seam: int, a: int, b: int, still: bool) -> bool:
        """Borda reta de cima a baixo só vira tela dividida com uma 2ª prova:
        (1) em algum momento do clipe só UM lado dela trocou de imagem de
        repente (a foto mudou e a pessoa continuou -- isso não acontece com
        a lateral de uma estante); ou (2) a fala aponta pra imagem nesse
        plano ("essa moça aqui", "olha essa foto", "tá vendo") E o lado da
        imagem está parado como uma foto (achado num react real: "era esse
        óculos aqui" + a lateral de uma estante viravam tela dividida)."""
        ts = int(round(seam * _TINY_W / an_w))
        for prof in colprof:
            if ts - 2 <= 2 or ts + 2 >= len(prof) - 2:
                break
            left, right = float(prof[:ts - 1].mean()), float(prof[ts + 2:].mean())
            hi, lo = max(left, right), min(left, right)
            if hi > 18.0 and lo < 6.0 and lo < 0.25 * hi:
                return True
        if words and still:
            from .react_detector import points_at_image
            if points_at_image(words, start + a / fps - 1.5, start + b / fps + 1.5):
                return True
        return False

    shots = [_plan_shot(a, b, faces, scene, bb, sx, sy, fps, acts, confirm_split)
             for a, b, bb in zip(bounds_list[:-1], bounds_list[1:], an_bounds)]
    plan = ShotPlan(shots, n, (0, 0, src_w, src_h))
    if verbose:
        print(f"    [cenas] {plan.summary()}")
    return plan


def _is_cut(d: float, h: float) -> bool:
    thr = float(getattr(config, "SCENE_CUT_THRESHOLD", 28.0))
    return d > thr or (d > thr * 0.45 and h > 0.4)


def _face_patches(gray: np.ndarray, f):
    """(boca, olhos) de um rosto, em miniatura (pra comparar entre quadros)."""
    cx, cy, w, h = f
    H, W = gray.shape

    def patch(y0, y1):
        x0, x1 = int(max(cx - 0.3 * w, 0)), int(min(cx + 0.3 * w, W))
        y0, y1 = int(max(cy + y0 * h, 0)), int(min(cy + y1 * h, H))
        if x1 - x0 < 4 or y1 - y0 < 3:
            return None
        p = cv2.resize(gray[y0:y1, x0:x1], (16, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        return p - p.mean()
    return patch(0.12, 0.45), patch(-0.30, -0.05)


def _mouth_activity(f, pt, prev) -> Optional[float]:
    """Quanto a boca mudou desde a detecção anterior do MESMO rosto."""
    cx, cy, w, h = f
    best = None
    for g, gpt in prev:
        if abs(g[0] - cx) < 0.5 * w and abs(g[1] - cy) < 0.5 * h and 0.75 < g[3] / max(h, 1e-3) < 1.33:
            best = gpt
            break
    if best is None or pt[0] is None or best[0] is None:
        return None
    mouth = float(np.mean(np.abs(pt[0] - best[0])))
    eyes = float(np.mean(np.abs(pt[1] - best[1]))) if pt[1] is not None and best[1] is not None else 0.0
    return max(mouth - 0.8 * eyes, 0.0)


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


def _seam(gray: np.ndarray) -> Optional[int]:
    """Coluna de uma "costura" vertical: duas imagens diferentes encostadas
    (tela dividida da edição). Diferente de uma borda comum (batente de
    porta, braço, microfone): pega a altura INTEIRA e é fina e reta."""
    H, W = gray.shape
    if W < 60 or H < 40:
        return None
    g = gray.astype(np.int16)
    diff = np.abs(g[:, 2:] - g[:, :-2]) > 14           # diff[:, c] compara c e c+2
    cov = diff.mean(axis=0).astype(np.float32)          # fração da altura com borda
    # o normal da vizinhança (mediana de 21 colunas): textura cheia de
    # bordas (cabelo, folhagem) não vira costura
    base = cv2.medianBlur(np.round(cov * 255).astype(np.uint8)[None, :], 21)[0].astype(np.float32) / 255.0
    score = cov - base
    lo, hi = int(0.2 * W), int(0.8 * W)
    c = lo + int(np.argmax(score[lo:hi]))
    if cov[c] >= 0.7 and score[c] >= 0.4:
        return c + 1
    return None


def _find_split(samples, fs, box):
    """Tela dividida do vídeo original: (coluna da costura em px da análise,
    lado da imagem, lado da câmera, imagem parada?) ou None.

    Achado real (corte do Maicon Küster, "analisando perfis do tinder"): a
    edição põe a foto do perfil de um lado e a câmera dele do outro enquanto
    ele comenta ("essa moça aqui que botou a foto do casamento"). O recorte
    seguia o rosto dele e a foto sumia -- ou pulava pro rosto DA FOTO.
    Sem IA: a costura vertical reta e fixa no plano inteiro, um rosto VIVO
    (mexe) de um lado e, do outro, a imagem (sem rosto, rosto parado de foto
    ou rosto bem menor, de um vídeo reagido)."""
    bx0, by0, bx1, by1 = box
    if len(samples) < 2:
        return None
    cols = []
    for smp in samples:
        c = _seam(smp[2][by0:by1, bx0:bx1])
        if c is not None:
            cols.append(bx0 + c)
    if len(cols) < max(2, 0.7 * len(samples)):
        return None
    seam = int(np.median(cols))
    if sum(abs(c - seam) <= 3 for c in cols) < 0.7 * len(samples):
        return None
    # rostos de cada lado (o maior de cada lado, por quadro analisado)
    side = {"L": [], "R": []}
    for _, g in fs:
        for key, sel in (("L", [f for f in g if f[0] + 0.3 * f[2] < seam]),
                         ("R", [f for f in g if f[0] - 0.3 * f[2] > seam])):
            if sel:
                side[key].append(max(sel, key=lambda f: f[3]))
    n = max(len(fs), 1)

    def stats(lst):
        if not lst:
            return 0.0, 0.0, 0.0
        arr = np.array(lst, dtype=np.float64)
        h = float(np.median(arr[:, 3]))
        # o quanto o rosto anda (x + y, relativo ao tamanho dele). Medido no
        # teste: pessoa falando calma 0,009-0,04; rosto de FOTO 0,0000-0,0004
        jit = float(np.median(np.abs(arr[:, 0] - np.median(arr[:, 0])))
                    + np.median(np.abs(arr[:, 1] - np.median(arr[:, 1])))) / max(h, 1e-3)
        return len(lst) / n, h, jit
    pl, hl, jl = stats(side["L"])
    pr, hr, jr = stats(side["R"])
    # câmera = lado com rosto quase sempre e o rosto maior
    # câmera = lado com rosto VIVO (que mexe) quase sempre; o rosto de uma
    # foto fica parado (achado no teste: com a pessoa fora do quadro, o
    # rosto da foto do perfil virava "a câmera" e a foto ia pra baixo)
    live = getattr(config, "SPLIT_MIN_FACE_JITTER", 0.003)
    okl, okr = pl >= 0.5 and jl >= live, pr >= 0.5 and jr >= live
    if okl and (hl >= hr or not okr):
        cam, content, hc, pk, hk, jk = "L", "R", hl, pr, hr, jr
    elif okr:
        cam, content, hc, pk, hk, jk = "R", "L", hr, pl, hl, jl
    else:
        return None
    # do lado da imagem: nada de rosto, rosto parado (foto) ou rosto bem
    # menor (vídeo reagido). Dois rostos vivos do mesmo tamanho = duas
    # pessoas lado a lado numa chamada -- aí é conversa, não imagem.
    if pk >= 0.3 and hk >= 0.75 * hc and jk >= live:
        return None
    width = (seam - bx0) if content == "L" else (bx1 - seam)
    if width < 0.25 * (bx1 - bx0):
        return None
    # o lado da imagem está PARADO (foto, print)? Mede a mudança entre as
    # amostras (borradas: a compressão do vídeo não conta). Foto: ~0;
    # estante com a facecam de um react passando: bem mais.
    x0, x1 = (bx0, seam - 2) if content == "L" else (seam + 2, bx1)
    blur = [cv2.GaussianBlur(smp[2][by0:by1, x0:x1], (0, 0), 2.0).astype(np.float32) for smp in samples]
    change = float(np.median([np.mean(np.abs(p - q)) for p, q in zip(blur, blur[1:])])) if len(blur) > 1 else 0.0
    return seam, content, cam, change < getattr(config, "SPLIT_STILL_MAX_CHANGE", 1.5)


def _trim(samples, x0, x1, y0, y1):
    """Tira bordas lisas (moldura/fundo de cor única) em volta da imagem."""
    g = np.median(np.stack([s[2][y0:y1, x0:x1].astype(np.float32) for s in samples]), axis=0)
    cs, rs = g.std(axis=0), g.std(axis=1)
    a, b = 0, len(cs)
    while a < b - 10 and cs[a] < 6:
        a += 1
    while b > a + 10 and cs[b - 1] < 6:
        b -= 1
    c, d = 0, len(rs)
    while c < d - 10 and rs[c] < 6:
        c += 1
    while d > c + 10 and rs[d - 1] < 6:
        d -= 1
    return x0 + a, y0 + c, x0 + b, y0 + d


def _plan_shot(a, b, faces, scene, an_bounds, sx, sy, fps, acts=None, confirm_split=None) -> Shot:
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

    # tela dividida no vídeo original (imagem de um lado, pessoa do outro)
    split = _find_split(samples, fs, (bx0, by0, bx1, by1)) \
        if getattr(config, "SPLIT_LAYOUT_ENABLED", True) else None
    if split and confirm_split is not None and not confirm_split(split[0], a, b, split[3]):
        split = None
    if split:
        seam, cside = split[0], split[1]
        if cside == "L":
            cx0, cx1, mx0, mx1 = bx0, seam - 2, seam + 2, bx1
        else:
            cx0, cx1, mx0, mx1 = seam + 2, bx1, bx0, seam - 2
        tx0, ty0, tx1, ty1 = _trim(samples, cx0, cx1, by0, by1)
        cam_faces = [max((f for f in g if mx0 <= f[0] <= mx1), key=lambda f: f[3])
                     for _, g in with_face if any(mx0 <= f[0] <= mx1 for f in g)]
        arr = np.array(cam_faces, dtype=np.float64)
        face = (float(np.median(arr[:, 0])) * sx, float(np.median(arr[:, 1])) * sy, float(np.median(arr[:, 3])) * sy)
        return Shot(a, b, "split", bounds,
                    content=(int(round(tx0 * sx)), int(round(ty0 * sy)), int(round(tx1 * sx)), int(round(ty1 * sy))),
                    cam=(int(round(mx0 * sx)), bounds[1], int(round(mx1 * sx)), bounds[3]), face=face)

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
           if max(f[2] for f in g) * sx > getattr(config, "SHOT_FACE_TOO_WIDE", 1.2) * crop_w_full]
    if 0 < presence < 0.4 and len(big) >= max(1, 0.15 * len(fs)) and not is_big_text:
        cx = float(np.median([f[0] for _, f in big])) * sx
        fw = float(np.median([f[2] for _, f in big])) * sx
        w = float(np.clip(fw * 1.25, crop_w_full, content_w))
        x0 = float(np.clip(cx - w / 2, bounds[0], bounds[2] - w))
        return Shot(a, b, "facefit", bounds, text_x=(int(round(x0)), int(round(x0 + w))))

    # celular na mão / selfie (achado real: live com rostos grandes, tortos
    # e tremidos): o detector acha o rosto em só 10-30% dos quadros, mas ele
    # está lá o tempo todo -- poucas detecções já dizem ONDE a pessoa está.
    # Antes o plano virava "sem rosto" e o recorte cortava cabeças.
    sparse = presence < 0.4
    if (not sparse or (len(with_face) >= 4 and presence >= 0.06)) and not is_big_text:
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
                               target_frac, lock=sparse)
            return _speaker(a, b, frames, faces, acts or {}, (bx0, by0, bx1, by1), bounds, sx, sy,
                            fps, crop_h, full_crop_h, aspect)
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
        if face_w > getattr(config, "SHOT_FACE_TOO_WIDE", 1.2) * crop_w_full:
            cx = float(np.median([t[1] for t in track])) * sx
            w = float(np.clip(face_w * 1.25, crop_w_full, content_w))
            x0 = float(np.clip(cx - w / 2, bounds[0], bounds[2] - w))
            return Shot(a, b, "facefit", bounds, text_x=(int(round(x0)), int(round(x0 + w))))
        return _single(a, b, frames, track, bounds, sx, sy, None, full_crop_h, fps, min_crop_h, target_frac,
                       lock=sparse)

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


def _speaker(a, b, frames, faces, acts, an_box, bounds, sx, sy, fps, crop_h, full_crop_h, aspect) -> Shot:
    """Duas ou mais pessoas que não cabem num recorte só (plano aberto da
    mesa): CORTA SECO pra quem está falando, como um editor de multicâmera
    -- nada de câmera deslizando de um rosto pro outro nem fade cruzado
    (pedido do usuário: "cortes e transições ruins"). Quem fala = boca
    mexendo (ver _mouth_activity). Cada pessoa fica pelo menos
    SPEAKER_MIN_HOLD_SECONDS na tela antes de trocar.
    Grupo de 3+ sem ninguém falando claro: mostra a cena inteira."""
    bx0, by0, bx1, by1 = an_box
    bh = by1 - by0
    dets = []  # (quadro, rosto, atividade)
    for k in sorted(faces):
        if not a <= k < b:
            continue
        ak = acts.get(k) or [None] * len(faces[k])
        g = [(f, x) for f, x in zip(faces[k], ak) if bx0 <= f[0] <= bx1 and by0 <= f[1] <= by1 and f[3] >= 0.035 * bh]
        if g:
            top_h = max(f[3] for f, _ in g)
            dets += [(k, f, x) for f, x in g if f[3] >= 0.6 * top_h]
    det_frames = sorted({k for k, _, _ in dets})
    if not dets or not det_frames:
        return Shot(a, b, "multi", bounds)
    # pessoas = grupos de rostos pela posição horizontal
    fw = float(np.median([f[2] for _, f, _ in dets]))
    xs_sorted = sorted(f[0] for _, f, _ in dets)
    centers, grp = [], [xs_sorted[0]]
    for x in xs_sorted[1:]:
        if x - grp[-1] > 0.6 * fw:
            centers.append(float(np.median(grp)))
            grp = [x]
        else:
            grp.append(x)
    centers.append(float(np.median(grp)))
    people = []
    for c in centers:
        mine = [(k, f, x) for k, f, x in dets if abs(f[0] - c) <= 0.6 * fw]
        if len({k for k, _, _ in mine}) >= 0.25 * len(det_frames):
            people.append(mine)
    if len(people) < 2:
        return Shot(a, b, "multi", bounds)
    nP, nK = len(people), len(det_frames)
    pos = {k: i for i, k in enumerate(det_frames)}
    act = np.zeros((nP, nK))
    for p, mine in enumerate(people):
        for k, _, x in mine:
            if x is not None:
                act[p, pos[k]] = max(act[p, pos[k]], x)
    step = max(float(np.median(np.diff(det_frames))) if nK > 1 else 3.0, 1.0)
    win = max(int(round(1.0 * fps / step)), 1)  # ~1 s de média
    sm = np.stack([np.convolve(np.pad(r, (win // 2, win - 1 - win // 2), mode="edge"), np.ones(win) / win,
                               mode="valid") for r in act])
    clear = float(np.max(sm)) > getattr(config, "SPEAKER_MIN_ACTIVITY", 1.5)
    if not clear and nP >= 3:
        return Shot(a, b, "facefit", bounds, text_x=(bounds[0], bounds[2]))  # cena inteira
    hold = getattr(config, "SPEAKER_MIN_HOLD_SECONDS", 2.0) * fps
    # começa em quem mais fala no 1º segundo (ou o maior rosto)
    first = sm[:, :max(win, 1)].mean(axis=1)
    cur = int(np.argmax(first)) if first.max() > 0 else \
        int(np.argmax([np.median([f[3] for _, f, _ in m]) for m in people]))
    seg_start = a
    who = np.zeros(nK, dtype=int)
    for i, k in enumerate(det_frames):
        cand = int(np.argmax(sm[:, i]))
        if cand != cur and sm[cand, i] > 1.3 * sm[cur, i] + 0.3 and k - seg_start >= hold:
            cur = cand
            seg_start = k
        who[i] = cur
    # posição fixa de cada pessoa no plano (o recorte não fica "seguindo")
    px_of = [float(np.median([f[0] for _, f, _ in m])) * sx for m in people]
    py_of = [float(np.median([f[1] for _, f, _ in m])) * sy for m in people]
    # troca um pouco ANTES de a média de 1 s perceber (a fala já começou)
    lead = int(round(0.4 * fps))
    idx = np.searchsorted(det_frames, np.minimum(frames + lead, det_frames[-1]), side="right") - 1
    idx = np.clip(idx, 0, nK - 1)
    xs = np.array([px_of[who[i]] for i in idx], dtype=np.float64)
    ys = np.array([py_of[who[i]] for i in idx], dtype=np.float64)
    shot = Shot(a, b, "speaker", bounds, crop_h=crop_h, xs=xs, ys=ys)
    shot.switches = int(np.sum(np.diff(who) != 0))
    return shot


def _single(a, b, frames, track, bounds, sx, sy, crop_h, full_crop_h, fps, min_crop_h, target_frac,
            lock: bool = False) -> Shot:
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
    if lock:
        # poucas detecções (câmera tremida): recorte PARADO na posição típica
        # -- interpolar entre detecções esparsas fazia a câmera "nadar"
        crop_h = float(np.clip(crop_h / 0.85, min_crop_h, full_crop_h))  # mais folga: rosto mexe
        return Shot(a, b, "single", bounds, crop_h=crop_h, xs=np.full_like(px, float(np.median(xs))),
                    ys=np.full_like(py, float(np.median(ys))))
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
