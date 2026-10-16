"""
Janela de conteúdo POR CIMA da câmera (sem IA pesada).

Achados reais:
  - corte do orochidois ("influenciadores aos prantos com o fim das bets"):
    a câmera dele ocupa a tela e o vídeo reagido aparece numa janela; o
    recorte só tirava o zoom do rosto e o vídeo sumia;
  - podcast do Bam Margera: a edição abre janelas por cima da mesa (vídeo
    de câmera corporal, print de notícia, TikTok) enquanto eles comentam --
    o recorte ficava cortando entre as duas pessoas e a janela não aparecia.

A tela dividida de src/shot_plan.py precisa de uma divisa de cima a baixo.
Aqui a janela pode estar em qualquer lugar, de qualquer tamanho e com
qualquer coisa dentro (TikTok, Reels, vídeo do YouTube, post do Instagram,
tweet, print de notícia, foto):

  1. EVENTO: num quadro só, um RETÂNGULO da tela troca de imagem de repente
     e o resto continua (a janela abriu, o vídeo dentro dela cortou, a foto
     trocou) -- ou o contrário: a câmera dá um corte seco e um retângulo
     fica PARADO (a imagem entrou junto com o corte). Gente mexendo não
     conta: muda aos poucos. Chat subindo não conta: é a mesma imagem
     deslocada pra cima. Também olha até 30 s antes do clipe (a janela que
     abriu um pouco antes do trecho começar);
  2. a janela é o retângulo que se repete nesses eventos;
  3. em cada momento ela está na tela se a MOLDURA aparece: linha nítida em
     pelo menos 2 lados, perto de um evento dela;
  4. o trecho com a janela sai com ela em cima e a pessoa embaixo -- desde
     que tenha alguém FORA da janela (ou na frente dela).

Não é janela (achados nos testes): o chat da live (rola), a facecam de um
react (pequena, rosto vivo, nunca corta -- ver shot_plan._window_shot),
retângulos quaisquer num cenário cheio de linhas (só 1 lado nítido).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

GRID_W = 160          # largura da miniatura usada pra achar os eventos


@dataclass
class Window:
    rect: Tuple[int, int, int, int]          # (x0, y0, x1, y1) em px da análise
    events: List[int] = field(default_factory=list)   # quadros dos eventos
    rects: List[Tuple[int, int, int, int]] = field(default_factory=list)
    present: Dict[int, bool] = field(default_factory=dict)  # amostra -> borda visível?
    masks: List[np.ndarray] = field(default_factory=list)   # o que mudou em cada evento (grade)
    an_size: Tuple[int, int] = (0, 0)                        # (largura, altura) da análise
    kinds: List[str] = field(default_factory=list)           # "corte" (mudou dentro) | "parada" (ficou parada num corte)
    inner_cuts: int = 0                                      # cortes DENTRO dela com ela aberta antes e depois


def grid_size(an_w: int, an_h: int) -> Tuple[int, int]:
    return GRID_W, max(int(round(an_h * GRID_W / an_w)), 8)


def _scrolled(cur: np.ndarray, prev: np.ndarray, base: float) -> bool:
    """A mudança é a mesma imagem deslocada na vertical (chat subindo,
    página rolando)?"""
    h = cur.shape[0]
    for d in range(1, min(12, h // 3)):
        a = float(np.mean(np.abs(cur[:-d] - prev[d:])))
        b = float(np.mean(np.abs(cur[d:] - prev[:-d])))
        if min(a, b) < 0.45 * base:
            return True
    return False


def find_events(smalls: List[np.ndarray], lo: int = 1, hi: Optional[int] = None) -> List[Tuple[int, Tuple, np.ndarray]]:
    """[(quadro, retângulo na grade, o que mudou)] dos quadros em que só um
    retângulo da tela trocou de imagem de repente."""
    n = len(smalls)
    hi = n - 1 if hi is None else min(hi, n - 1)
    if n < 4:
        return []
    gh, gw = smalls[0].shape
    out = []
    for k in range(max(lo, 2), hi):
        cur = smalls[k].astype(np.int16)
        prev = smalls[k - 1].astype(np.int16)
        d0 = np.abs(cur - prev).astype(np.float32)
        d0b = cv2.blur(d0, (3, 3))
        m = d0b > 18.0
        frac = float(m.mean())
        if frac >= 0.5:
            # corte da câmera: a janela que ficou PARADA enquanto o resto
            # mudou (a imagem entrou junto com um corte seco da edição e não
            # tem evento próprio)
            ev = _still_window(d0, cur, gw, gh)
            if ev is not None:
                out.append((k, ev[0], ev[1], "parada"))
            continue
        if frac < 0.03:
            continue
        m8 = m.astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        if cnt < 2:
            continue
        j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = (int(v) for v in stats[j])
        if w < 0.15 * gw or h < 0.22 * gh or w * h > 0.85 * gw * gh:
            continue
        inside = m[y:y + h, x:x + w]
        if float(inside.mean()) < 0.4:
            continue
        out_px = float(m.sum() - inside.sum())
        out_area = float(gw * gh - w * h)
        if out_area > 0 and out_px / out_area > 0.03:
            continue
        base = float(d0[y:y + h, x:x + w].mean())
        if base < 12.0:
            continue
        # de repente: o quadro anterior e o seguinte quase não mudaram ali
        dm = float(np.abs(prev[y:y + h, x:x + w] - smalls[k - 2][y:y + h, x:x + w].astype(np.int16)).mean())
        dp = float(np.abs(smalls[k + 1][y:y + h, x:x + w].astype(np.int16) - cur[y:y + h, x:x + w]).mean())
        if dm > 0.35 * base or dp > 0.35 * base:
            continue
        if _scrolled(cur[y:y + h, x:x + w].astype(np.float32), prev[y:y + h, x:x + w].astype(np.float32), base):
            continue
        out.append((k, (x, y, x + w, y + h), m, "corte"))
    return out


def _still_window(d0: np.ndarray, cur: np.ndarray, gw: int, gh: int):
    """Num corte que mudou quase a tela toda: um retângulo que não mudou
    nada, com imagem dentro (não uma parede lisa), longe das bordas da tela
    em pelo menos dois lados. (retângulo na grade, máscara do retângulo)

    A parede lisa atrás do streamer também "não muda" num corte seco e
    encosta na janela: as regiões paradas são separadas pelas bordas da
    imagem (a moldura da janela é uma borda forte)."""
    still = cv2.blur(d0, (3, 3)) < 4.0
    c = cur.astype(np.float32)
    grad = np.abs(cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)) + np.abs(cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3))
    cut = (still & (grad < 100.0)).astype(np.uint8)
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(cut, connectivity=4)
    best = None
    for j in range(1, cnt):
        x, y, w, h, area = (int(v) for v in stats[j])
        if area < 0.015 * gw * gh or w < 0.15 * gw or h < 0.22 * gh or w * h > 0.7 * gw * gh:
            continue
        # a borda (cortada fora acima) volta pra dentro do retângulo
        x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + w + 1, gw), min(y + h + 1, gh)
        if float(still[y0:y1, x0:x1].mean()) < 0.85:
            continue
        touches = (x0 <= 1) + (y0 <= 1) + (x1 >= gw - 1) + (y1 >= gh - 1)
        if touches > 2:
            continue
        if float(c[y0:y1, x0:x1].std()) < 15.0:
            continue
        if best is None or (x1 - x0) * (y1 - y0) > best[0]:
            best = ((x1 - x0) * (y1 - y0), (x0, y0, x1, y1))
    if best is None:
        return None
    x0, y0, x1, y1 = best[1]
    m = np.zeros((gh, gw), bool)
    m[y0:y1, x0:x1] = still[y0:y1, x0:x1]
    return best[1], m


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def group(events, an_w: int, an_h: int) -> List[Window]:
    """Junta os eventos do mesmo retângulo (a mesma janela)."""
    gw, gh = grid_size(an_w, an_h)
    sx, sy = an_w / gw, an_h / gh
    wins: List[Window] = []
    for k, (x0, y0, x1, y1), m, kind in events:
        r = (int(round(x0 * sx)), int(round(y0 * sy)), int(round(x1 * sx)), int(round(y1 * sy)))
        for w in wins:
            if _iou(w.rect, r) >= 0.5:
                w.events.append(k)
                w.rects.append(r)
                w.masks.append(m)
                w.kinds.append(kind)
                w.rect = tuple(int(v) for v in np.median(np.array(w.rects), axis=0))
                break
        else:
            wins.append(Window(r, [k], [r], masks=[m], kinds=[kind]))
    return wins


def in_content(w: Window, f) -> bool:
    """Esse rosto é do conteúdo da janela? Fora do retângulo, não. Dentro:
    é do conteúdo se o lugar dele mudou junto quando a janela abriu/cortou.
    Achado real (podcast do Bam Margera): os apresentadores ficam recortados
    NA FRENTE da janela -- o rosto deles está dentro do retângulo, mas não
    muda quando o vídeo da janela corta."""
    cx, cy = f[0], f[1]
    an_w, an_h = w.an_size
    x0, y0, x1, y1 = w.rect
    mx, my = 0.02 * an_w, 0.02 * an_h
    if not (x0 - mx <= cx <= x1 + mx and y0 - my <= cy <= y1 + my):
        return False
    if not w.masks:
        return True
    gh, gw = w.masks[0].shape
    gx, gy = int(np.clip(cx * gw / an_w, 0, gw - 1)), int(np.clip(cy * gh / an_h, 0, gh - 1))
    r = max(int(round(0.25 * f[3] * gh / an_h)), 1)
    hits = [float(m[max(gy - r, 0):gy + r + 1, max(gx - r, 0):gx + r + 1].mean()) >= 0.5 for m in w.masks]
    return sum(hits) >= 0.5 * len(hits)


def _side_cov(gray: np.ndarray, side: str, pos: int, a: int, b: int) -> float:
    """Fração do lado (linha/coluna `pos`, de a até b) com borda nítida:
    imagem de um lado diferente da do outro."""
    H, W = gray.shape
    if b - a < 6:
        return 0.0
    g = gray
    if side in ("top", "bottom"):
        if pos < 3 or pos > H - 3:
            return -1.0
        o = g[pos - 3:pos - 1, a:b].astype(np.float32).mean(axis=0)
        i = g[pos + 1:pos + 3, a:b].astype(np.float32).mean(axis=0)
    else:
        if pos < 3 or pos > W - 3:
            return -1.0
        o = g[a:b, pos - 3:pos - 1].astype(np.float32).mean(axis=1)
        i = g[a:b, pos + 1:pos + 3].astype(np.float32).mean(axis=1)
    return float(np.mean(np.abs(o - i) > 10.0))


def _sides(rect, W: int, H: int, margin: int = 4):
    """Lados da janela que não encostam na borda da tela."""
    x0, y0, x1, y1 = rect
    out = []
    if y0 > margin:
        out.append(("top", y0, x0, x1))
    if y1 < H - margin:
        out.append(("bottom", y1, x0, x1))
    if x0 > margin:
        out.append(("left", x0, y0, y1))
    if x1 < W - margin:
        out.append(("right", x1, y0, y1))
    return out


def border_score(gray: np.ndarray, rect) -> float:
    """O quanto a moldura da janela aparece neste quadro (0..1): a fração
    dos lados livres (que não encostam na borda da tela) com a linha
    nítida em pelo menos 75% do comprimento. Precisa de 2 lados assim.
    Medido: janela de verdade, 2-4 lados com 0,80-1,00 (um lado pode ficar
    atrás do apresentador recortado na frente); retângulo qualquer num
    cenário cheio de detalhes (podcast), 1 lado e o resto 0,2-0,6."""
    H, W = gray.shape
    covs = [_side_cov(gray, s, p, a, b) for s, p, a, b in _sides(rect, W, H)]
    covs = [c for c in covs if c >= 0]
    strong = sum(1 for c in covs if c >= 0.75)
    if len(covs) < 2 or strong < 2:
        return 0.0
    return strong / float(len(covs))


def refine(rect, grays: List[np.ndarray]):
    """Acerta cada lado na linha da moldura (a grade dos eventos é grossa)."""
    if not grays:
        return rect
    H, W = grays[0].shape
    x0, y0, x1, y1 = rect
    r = max(int(round(W / GRID_W * 2)), 4)
    best = list(rect)
    for idx, side in enumerate(("left", "top", "right", "bottom")):
        pos = rect[idx]
        lim = W if side in ("left", "right") else H
        if pos <= 4 or pos >= lim - 4:
            continue
        a, b = (y0, y1) if side in ("left", "right") else (x0, x1)
        scores = []
        for p in range(max(pos - r, 3), min(pos + r, lim - 3) + 1):
            scores.append((float(np.median([_side_cov(g, side, p, a, b) for g in grays])), p))
        sc, p = max(scores)
        if sc >= 0.3:
            best[idx] = p
    return tuple(best)


def read_smalls(source_path: str, start: float, duration: float, fps: float,
                an_w: int, an_h: int) -> List[np.ndarray]:
    """Miniaturas cinza de um trecho (só pra achar eventos: barato)."""
    import subprocess
    if duration <= 0.5:
        return []
    gw, gh = grid_size(an_w, an_h)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(start, 0.0):.3f}", "-i", str(source_path), "-t", f"{duration:.3f}",
           "-vf", f"fps={fps:.6f},scale={gw}:{gh}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=max(60.0, 4 * duration)).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    n = len(raw) // (gw * gh)
    return [np.frombuffer(raw, np.uint8, gw * gh, i * gw * gh).reshape(gh, gw) for i in range(n)]


def detect(smalls, scene: Dict[int, tuple], an_w: int, an_h: int, fps: float,
           pre: Optional[List[np.ndarray]] = None) -> List[Window]:
    """Janelas de conteúdo do clipe, com a presença em cada amostra.
    `pre`: miniaturas de antes do clipe -- a janela que abriu um pouco antes
    do corte começar (um tweet lido por um minuto, um vídeo sem cortes)
    não tem evento nenhum dentro do clipe."""
    events = find_events(smalls)
    if pre:
        events = [(k - len(pre), r, m, t) for k, r, m, t in find_events(pre)] + events
    if not events:
        return []
    wins = group(events, an_w, an_h)
    keys = sorted(scene)
    out = []
    for w in wins:
        w.an_size = (an_w, an_h)
        # amostras logo depois dos eventos (a janela com certeza na tela)
        near = [scene[k][2] for k in keys if any(0 <= k - e <= int(1.5 * fps) for e in w.events)][:6] \
            or [scene[k][2] for k in keys[:6]]  # abriu antes do clipe: o começo do clipe
        w.rect = refine(w.rect, near)
        x0, y0, x1, y1 = w.rect
        if x1 - x0 < 0.15 * an_w or y1 - y0 < 0.22 * an_h:
            continue
        w.present = {k: border_score(scene[k][2], w.rect) > 0 for k in keys}
        flags = [w.present[k] for k in keys]
        # amostra solta (um quadro de transição) não muda a presença
        for i in range(1, len(flags) - 1):
            if flags[i - 1] == flags[i + 1] != flags[i]:
                w.present[keys[i]] = flags[i - 1]
        _anchor(w, keys, fps)
        if _scrolls(w, scene, keys, an_w):
            continue
        # corte do vídeo DENTRO da janela (aberta antes e depois): facecam
        # nunca tem isso
        lag = int(1.5 * fps)
        w.inner_cuts = sum(1 for e, t in zip(w.events, w.kinds) if t == "corte"
                           and any(w.present.get(k) for k in keys if e - lag <= k < e)
                           and any(w.present.get(k) for k in keys if e < k <= e + lag))
        if any(w.present.values()):
            out.append(w)
    return out


def _scrolls(w: Window, scene, keys: List[int], an_w: int) -> bool:
    """Janela estreita que muda ROLANDO (chat da live: mensagem nova empurra
    as outras pra cima) não é conteúdo. Achado no teste: o chat ficava
    parado num corte seco da câmera, igual a um print colado, e ia pra cima
    da tela. TikTok trocando por deslize rola uma vez ou outra; chat, o
    tempo todo."""
    x0, y0, x1, y1 = w.rect
    if x1 - x0 > 0.35 * an_w or y1 - y0 < 40:
        return False
    on = [k for k in keys if w.present.get(k)]
    scroll = other = 0
    for k1, k2 in zip(on, on[1:]):
        a = scene[k1][2][y0 + 3:y1 - 3, x0 + 3:x1 - 3].astype(np.float32)
        b = scene[k2][2][y0 + 3:y1 - 3, x0 + 3:x1 - 3].astype(np.float32)
        base = float(np.mean(np.abs(a - b)))
        if base < 3.0:
            continue
        best = min(min(float(np.mean(np.abs(a[:-d] - b[d:]))), float(np.mean(np.abs(a[d:] - b[:-d]))))
                   for d in range(2, min(40, a.shape[0] // 3)))
        if best < 0.5 * base:
            scroll += 1
        else:
            other += 1
    return scroll >= 3 and scroll >= other


def _anchor(w: Window, keys: List[int], fps: float) -> None:
    """Cada trecho em que a moldura aparece precisa de um evento da janela
    junto (ela abriu, cortou ou ficou parada num corte) -- ou ter aberto
    antes do clipe. Achado no teste (podcast Inteligência Ltda): um
    retângulo achado num corte de câmera "aparecia" em amostras soltas,
    longe do evento, por causa das linhas do cenário."""
    runs, cur = [], None
    for k in keys:
        if w.present.get(k):
            cur = [k, k] if cur is None else [cur[0], k]
        elif cur is not None:
            runs.append(cur)
            cur = None
    if cur is not None:
        runs.append(cur)
    before = any(e < 0 for e in w.events)
    for a, b in runs:
        ok = any(a - fps <= e <= b + fps for e in w.events) or (before and a == keys[0])
        if not ok:
            for k in keys:
                if a <= k <= b:
                    w.present[k] = False


def transitions(wins: List[Window], fps: float) -> List[int]:
    """Quadros em que uma janela abre ou fecha (viram fronteira de plano)."""
    out = []
    for w in wins:
        keys = sorted(w.present)
        for p, q in zip(keys, keys[1:]):
            if w.present[p] == w.present[q]:
                continue
            # o evento entre as duas amostras é o quadro exato da troca
            ev = [e for e in w.events if p < e <= q]
            out.append(ev[0] if ev else q)
    return sorted(set(out))


def window_in(wins: List[Window], a: int, b: int, scene_keys: List[int], fps: float) -> Optional[Window]:
    """A janela que está na tela no plano [a, b) inteiro (ou quase)."""
    keys = [k for k in scene_keys if a <= k < b]
    if not keys:
        return None
    best = None
    for w in wins:
        on = sum(1 for k in keys if w.present.get(k))
        if on < max(1, 0.7 * len(keys)):
            continue
        # conteúdo de verdade: mudou por dentro (evento no meio dela, não só
        # abrir/fechar) ou ficou aberta um tempo (inventário de jogo abre e
        # fecha rápido; print de notícia fica)
        inner = [e for e in w.events if a + 2 < e < b - 2]
        if not inner and (b - a) < 2.0 * fps:
            continue
        area = (w.rect[2] - w.rect[0]) * (w.rect[3] - w.rect[1])
        if best is None or area > best[0]:
            best = (area, w)
    return best[1] if best else None
