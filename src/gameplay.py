"""
Modo GAMEPLAY: jogo na tela inteira com a facecam do streamer num canto.

Pedido do usuário (corte do DrDonut, Short lRqCQhnmX10): "momentos em que
não está acontecendo nada no jogo, e o streamer está falando, foca nele.
Quando ocorre ação no jogo, como crystal pvp, pvp no geral, tiros, parkour,
ou alguém falando no jogo via voice chat, a câmera pegar o jogo".

O que saía antes: o reenquadramento comum tratava o vídeo como podcast --
ora um close borrado do rosto da facecam, ora um recorte do meio da tela com
a facecam cortada pela metade em cima -- sem relação nenhuma com o que
acontecia no jogo.

Agora, em canal marcado como gameplay ("...|game" na lista de canais):
  1. a facecam é achada com o detector do react, numa versão mais solta
     (react_layout.detect_react_layout(relaxed=True));
  2. `plan_clip` olha o clipe a 4 quadros/s e decide, trecho a trecho, entre
     DOIS planos de tela cheia:
       - STREAMER: close do streamer (recorte da facecam, rosto parado no
         plano), quando o jogo está calmo e ele está falando;
       - JOGO: o jogo em tela cheia, centrado na mira, sem pegar a facecam,
         quando tem AÇÃO (o jogo mexe muito: PvP, tiro, parkour, explosão
         -- medido pelo quanto a imagem do jogo muda) ou quando quem fala é
         OUTRA pessoa (voice chat: tem fala na transcrição e a boca do
         streamer está parada, igual quando ele está quieto);
  3. cada plano fica pelo menos GAMEPLAY_MIN_HOLD_SECONDS (2 s) na tela,
     com corte seco (sem pisca-pisca nem transição mole); o corte pro jogo
     entra um pouquinho ANTES da ação (o clipe já é conhecido inteiro).

Tem a mesma interface do plano do react (active/compose), então o
reframer.py usa sem mudança nenhuma.
"""
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .react_layout import ReactLayout, _analysis_size, _faces

RATE = 8.0           # amostras por segundo (boca: quadros a 125 ms um do outro)
_MOTION_LAG = 2      # movimento do jogo: compara com 2 amostras atrás (250 ms)
_MOTION_W = 480      # largura da imagem usada pra medir o movimento do jogo
GAME, STREAMER = 1, 0


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


def _runs(lab: np.ndarray) -> List[Tuple[int, int]]:
    out, a = [], 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[a]:
            out.append((a, i))
            a = i
    return out


def _enforce_hold(lab: np.ndarray, min_len: int) -> np.ndarray:
    """Nenhum plano mais curto que `min_len` amostras: ação curta estica
    (o jogo fica na tela o tempo mínimo); conversa curta no meio da ação
    vira jogo."""
    lab = lab.copy()
    n = len(lab)
    for _ in range(n + 1):  # cada volta conserta pelo menos um plano curto
        runs = _runs(lab)
        if len(runs) <= 1:
            break
        short = [(a, b) for a, b in runs if b - a < min_len]
        if not short:
            break
        a, b = short[0]
        if lab[a] == GAME:
            if b < n:
                lab[a:min(a + min_len, n)] = GAME   # estica pra frente
            else:
                lab[max(b - min_len, 0):b] = GAME   # fim do clipe: estica pra trás
        else:
            lab[a:b] = GAME if (a > 0 and lab[a - 1] == GAME) or (b < n and lab[b] == GAME) else lab[a]
    return lab


class GameplayClipPlan:
    """Plano de um clipe de gameplay: em cada instante, STREAMER ou JOGO."""
    split_screen = False

    def __init__(self, layout: ReactLayout, times, views, faces, game_window, analysis_w: int):
        self.layout = layout
        self.times = np.asarray(times, dtype=np.float64)
        self.views = np.asarray(views, dtype=np.int8)
        self.faces = faces                    # (cx, cy, fh) na fonte, travado por plano
        self.game_window = game_window        # (x, y, w, h) na fonte
        self._s = layout.src_w / float(analysis_w)

    def _i(self, t: float) -> int:
        if len(self.times) == 0:
            return 0
        return int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 1))

    def active(self, t: float) -> bool:
        return len(self.times) > 0

    @property
    def active_fraction(self) -> float:
        return 1.0 if len(self.times) else 0.0

    def view_at(self, t: float) -> int:
        return int(self.views[self._i(t)]) if len(self.views) else GAME

    @property
    def streamer_fraction(self) -> float:
        return float(np.mean(self.views == STREAMER)) if len(self.views) else 0.0

    @property
    def switches(self) -> int:
        return int(np.sum(self.views[1:] != self.views[:-1])) if len(self.views) > 1 else 0

    def compose(self, frame, t: float, out_w: int, out_h: int, zoom: float = 1.0) -> np.ndarray:
        if self.view_at(t) == STREAMER:
            return self._streamer(frame, t, out_w, out_h, zoom)
        return self._game(frame, out_w, out_h, zoom)

    def _game(self, frame, out_w, out_h, zoom):
        x, y, w, h = self.game_window
        z = max(zoom, 1.0)
        if z > 1.001:
            # zoom de edição puxando pro centro da tela (a mira)
            fw, fh = w / z, h / z
            cx = float(np.clip(frame.shape[1] / 2.0, x + fw / 2, x + w - fw / 2))
            cy = float(np.clip(frame.shape[0] / 2.0, y + fh / 2, y + h - fh / 2))
            x, y, w, h = int(cx - fw / 2), int(cy - fh / 2), int(fw), int(fh)
        crop = frame[y:y + h, x:x + w]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA if w > out_w else cv2.INTER_CUBIC)

    def _streamer(self, frame, t, out_w, out_h, zoom):
        """Close do streamer. Facecam grande: tela cheia. Facecam pequena
        (o normal em gameplay): ampliar até a tela cheia passaria de 5-7x e
        ficava borrado (teste real: speedrun 6,6x; SMP com o rosto cortado no
        queixo e na testa) -- aí o painel da facecam, nítido, com o rosto em
        destaque, sobre um fundo desfocado dela mesma."""
        bx, by, bw, bh = self.layout.cam_box
        # 2% pra dentro: borda/texto colado na facecam (achado: "total subs" do SMP)
        bx, by, bw, bh = bx + 0.02 * bw, by + 0.02 * bh, 0.96 * bw, 0.96 * bh
        fcx, fcy, ffh = self.faces[self._i(t)]
        frac = getattr(config, "GAMEPLAY_FACE_FRAC", 0.34)
        max_up = getattr(config, "GAMEPLAY_MAX_UPSCALE", 3.5)
        want = frac * out_h / max(ffh, 1.0) * max(zoom, 1.0)   # escala que deixa o rosto no tamanho certo
        fill = max(out_h / float(bh), out_w / float(bw))        # menor escala que enche a tela com a facecam
        sc = max(want, fill)
        if sc <= max_up:
            cw, ch = out_w / sc, out_h / sc
            x0 = int(np.clip(fcx - cw / 2, bx, bx + bw - cw))
            y0 = int(np.clip(fcy - ch * 0.42, by, by + bh - ch))
            crop = frame[y0:y0 + max(int(ch), 2), x0:x0 + max(int(cw), 2)]
            return _upscale(crop, out_w, out_h, sc)
        sc = min(want, max_up)
        cw, ch = min(float(bw), out_w / sc), min(float(bh), out_h / sc)
        x0 = int(np.clip(fcx - cw / 2, bx, bx + bw - cw))
        y0 = int(np.clip(fcy - ch * 0.45, by, by + bh - ch))
        crop = frame[y0:y0 + max(int(ch), 2), x0:x0 + max(int(cw), 2)]
        pw, ph = min(int(round(cw * sc)), out_w), min(int(round(ch * sc)), out_h)
        panel = _upscale(crop, pw, ph, sc)
        # fundo: a própria facecam cobrindo a tela, bem desfocada e escurecida
        k = max(out_w / float(crop.shape[1]), out_h / float(crop.shape[0]))
        small = cv2.resize(crop, (max(int(crop.shape[1] * k / 12), 2), max(int(crop.shape[0] * k / 12), 2)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.0)
        bg = cv2.resize(small, (int(np.ceil(crop.shape[1] * k)), int(np.ceil(crop.shape[0] * k))),
                        interpolation=cv2.INTER_LINEAR)
        oy, ox = (bg.shape[0] - out_h) // 2, (bg.shape[1] - out_w) // 2
        out = (bg[oy:oy + out_h, ox:ox + out_w].astype(np.float32) * 0.55).astype(np.uint8)
        px = (out_w - pw) // 2
        py = int(np.clip(round(out_h * 0.44 - ph / 2), 0, out_h - ph))
        out[py:py + ph, px:px + pw] = panel
        return out


def _upscale(crop, w, h, scale):
    out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if scale > 2.2:
        # facecam ampliada várias vezes fica mole: realce leve de nitidez
        soft = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = cv2.addWeighted(out, 1.45, soft, -0.45, 0)
    return out


def _game_window(layout: ReactLayout, W: int, H: int, out_w: int, out_h: int) -> Tuple[int, int, int, int]:
    """Janela FIXA do jogo pro clipe: a maior que não pega a facecam e fica
    mais perto do centro da tela (mira / personagem)."""
    aspect = out_w / float(out_h)
    bx, by, bw, bh = layout.box_at(W)
    mx, my = 0.06 * bw + 2, 0.06 * bh + 2
    cx0, cy0, cx1, cy1 = max(bx - mx, 0), max(by - my, 0), min(bx + bw + mx, W), min(by + bh + my, H)
    cam_area = max((cx1 - cx0) * (cy1 - cy0), 1.0)
    best, best_score = None, -1e9
    for sc in (1.0, 0.92, 0.84, 0.76, 0.68, 0.6):
        h = H * sc
        w = h * aspect
        if w > W:
            w, h = float(W), W / aspect
        for x in np.linspace(0, W - w, 33):
            for y in np.linspace(0, H - h, 1 if h >= H - 1 else 9):
                ix = max(0.0, min(x + w, cx1) - max(x, cx0))
                iy = max(0.0, min(y + h, cy1) - max(y, cy0))
                ov = ix * iy / cam_area
                off = abs(x + w / 2 - W / 2) / w + abs(y + h / 2 - H / 2) / h
                score = sc - 1.2 * off - 6.0 * ov
                if score > best_score:
                    best_score, best = score, (x, y, w, h)
    s = layout.src_w / float(W)
    x, y, w, h = (int(round(v * s)) for v in best)
    return x, y, min(w, layout.src_w - x), min(h, layout.src_h - y)


def plan_clip(layout: ReactLayout, source_path: str, start: float, end: float,
              out_w: int, out_h: int, words=None, log=print) -> Optional[GameplayClipPlan]:
    """Analisa o trecho do clipe e decide o plano de cada instante.
    `words`: palavras da transcrição (tempo absoluto do vídeo)."""
    from .reframer import _new_yunet
    from .shot_plan import _face_patches, _mouth_activity

    W, H = _analysis_size(layout.src_w, layout.src_h)
    mw = min(_MOTION_W, W)
    mh = max(int(round(H * mw / W / 2.0)) * 2, 2)
    bx, by, bw, bh = layout.box_at(W)
    # região do jogo pra medir movimento: tudo menos a facecam (com folga)
    gmask = np.ones((mh, mw), bool)
    k = mw / float(W)
    gmask[int(max(by - 0.1 * bh, 0) * k):int(min(by + 1.1 * bh, H) * k),
          int(max(bx - 0.1 * bw, 0) * k):int(min(bx + 1.1 * bw, W) * k)] = False
    if gmask.mean() < 0.2:
        return None
    # recorte da facecam pra achar o rosto (ampliado se for pequeno)
    pad = 0.08
    c0x, c0y = int(max(bx - pad * bw, 0)), int(max(by - pad * bh, 0))
    c1x, c1y = int(min(bx + (1 + pad) * bw, W)), int(min(by + (1 + pad) * bh, H))
    up = 2.0 if (c1x - c0x) < 360 else 1.0
    det = _new_yunet()

    motion, faces, mouth = [], [], []
    smalls, prev_det = [], []
    for f in _iter_frames(source_path, start, end - start, RATE, W, H):
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(f, (mw, mh), interpolation=cv2.INTER_AREA),
                                              cv2.COLOR_BGR2GRAY), (0, 0), 2.0).astype(np.float32)
        smalls.append(small)
        if len(smalls) > _MOTION_LAG:
            motion.append(float(np.abs(small - smalls.pop(0))[gmask].mean()))
        else:
            motion.append(None)
        cam = f[c0y:c1y, c0x:c1x]
        if up != 1.0:
            cam = cv2.resize(cam, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
        found = [fc for fc in _faces(det, cam) if fc[3] >= 0.25 * bh * up] if det is not None else []
        if found:
            fc = max(found, key=lambda v: v[3])
            fc = (c0x + fc[0] / up, c0y + fc[1] / up, fc[2] / up, fc[3] / up)
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            pt = _face_patches(gray, fc)
            mouth.append(_mouth_activity(fc, pt, prev_det))
            prev_det = [(fc, pt)]
            faces.append(fc)
        else:
            mouth.append(None)
            prev_det = []
            faces.append(None)
    n = len(motion)
    if n < 4:
        return None
    m = np.array([v if v is not None else np.nan for v in motion], np.float64)
    first_ok = next((i for i in range(n) if not np.isnan(m[i])), None)
    if first_ok is None:
        return None
    m[:first_ok] = m[first_ok]
    times = np.arange(n) / RATE

    # --- AÇÃO no jogo: a imagem do jogo muda muito (média de 0,75 s) ---
    k = int(round(0.75 * RATE))
    ms = np.convolve(np.pad(m, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    med = float(np.median(ms))
    a_abs = getattr(config, "GAMEPLAY_ACTION_MOTION", 26.0)
    a_rel = getattr(config, "GAMEPLAY_ACTION_RELATIVE", 1.8)
    action = (ms >= a_abs) | ((ms >= a_rel * med) & (ms >= 0.55 * a_abs))

    # --- fala: de quem? ---
    speech = np.zeros(n, bool)
    phrases = []
    if words:
        ws = sorted((w.start - start, w.end - start) for w in words
                    if w.end > start and w.start < end)
        for a, b in ws:
            speech[int(max(a - 0.1, 0) * RATE):int(np.ceil((b + 0.1) * RATE))] = True
            if phrases and a - phrases[-1][1] <= 0.5:
                phrases[-1][1] = b
                phrases[-1][2] += 1
            else:
                phrases.append([a, b, 1])
    # boca do streamer na facecam: sinal fraco (rosto pequeno, microfone na
    # frente, cabeça mexendo). Medido no teste: ele gritando "go, go, go"
    # com a boca "parada" metade das amostras. Por isso a regra do voice chat
    # só vale quando a boca dele separa BEM fala de silêncio neste clipe, e
    # só marca frase longa em que a boca fica parada quase o tempo todo.
    mo = np.array([v if v is not None else np.nan for v in mouth], np.float64)
    other = np.zeros(n, bool)
    quiet = mo[~speech & ~np.isnan(mo)]
    talk = mo[speech & ~np.isnan(mo)]
    base = float(np.median(quiet)) if len(quiet) >= 12 else None
    reliable = (base is not None and len(talk) >= 16 and float(np.median(talk)) >= 2.0 * base + 0.3
                and float(np.mean(talk > 1.5 * base)) >= 0.5)
    n_other = 0
    if reliable and getattr(config, "GAMEPLAY_VOICE_CHAT", True):
        for a, b, nw in phrases:
            if b - a < 1.2 or nw < 3:
                continue
            i0, i1 = int(a * RATE), int(np.ceil(b * RATE))
            vals = mo[i0:i1]
            vals = vals[~np.isnan(vals)]
            seen = len(vals) / max(i1 - i0, 1)
            # boca tão parada quanto quando ele está quieto = outra pessoa falando
            if len(vals) >= 8 and seen >= 0.6 and float(np.mean(vals <= 1.2 * base)) >= 0.8 \
                    and float(np.percentile(vals, 90)) <= 2.0 * base:
                other[i0:i1] = True
                n_other += 1

    # facecam sumiu (streamer tirou a câmera / tela de transição): jogo
    seen_face = np.array([fc is not None for fc in faces])
    near = np.convolve(seen_face.astype(float), np.ones(int(2 * RATE) + 1), mode="same") > 0  # rosto a até 1 s

    lab = np.full(n, -1, np.int8)
    lab[speech] = STREAMER
    lab[action | other | ~near] = GAME
    # corte pro jogo entra 0,25 s antes da ação e sai 0,5 s depois dela
    g = lab == GAME
    gd = g.copy()
    lead, tail = int(round(0.25 * RATE)), int(round(0.5 * RATE))
    for j in range(1, lead + 1):
        gd[:-j] |= g[j:]
    for j in range(1, tail + 1):
        gd[j:] |= g[:-j]
    lab[gd] = GAME
    # calmo e ninguém falando: segura o plano que estava
    if (lab == -1).all():
        lab[:] = GAME
    for i in range(n):
        if lab[i] == -1 and i > 0:
            lab[i] = lab[i - 1]
    first = next((i for i in range(n) if lab[i] != -1), None)
    if first:
        lab[:first] = lab[first]
    lab = _enforce_hold(lab, max(int(round(getattr(config, "GAMEPLAY_MIN_HOLD_SECONDS", 2.0) * RATE)), 1))

    # rosto do streamer: TRAVADO por plano (câmera parada no close)
    s = layout.src_w / float(W)
    default = (layout.face[0] / s, layout.face[1] / s, layout.face[3] / s)
    locked = [default] * n
    for a, b in _runs(lab):
        if lab[a] != STREAMER:
            continue
        fs = [faces[i] for i in range(a, b) if faces[i] is not None] or \
             [fc for fc in faces if fc is not None] or [None]
        if fs[0] is None:
            v = default
        else:
            arr = np.array(fs)
            v = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])), float(np.median(arr[:, 3])))
        for i in range(a, b):
            locked[i] = v
    locked = [(cx * s, cy * s, fh * s) for cx, cy, fh in locked]

    window = _game_window(layout, W, H, out_w, out_h)
    plan = GameplayClipPlan(layout, times, lab, locked, window, W)
    plan.n_action = int(_count_runs(action))
    plan.n_other = n_other
    plan.mouth_reliable = reliable
    plan.debug = {"motion": ms, "speech": speech, "mouth": mo, "other": other, "action": action,
                  "face": seen_face, "base": base}
    return plan


def _count_runs(flags: np.ndarray) -> int:
    return int(np.sum(flags[1:] & ~flags[:-1]) + (1 if len(flags) and flags[0] else 0))
