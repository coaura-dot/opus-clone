"""
Efeitos de edição de corte viral, planejados a partir da FALA do clipe (não
aleatórios): zoom nos momentos-chave, "impacto" (flash + tremida + separação
RGB) nos momentos mais fortes, emoji quando uma palavra-chave é dita, e uma
abertura com zoom out + flicker.

`plan_effects` roda uma vez por clipe (timestamps já na linha do tempo final,
sem as pausas cortadas) e devolve um FxPlan; o reenquadramento consulta o
plano quadro a quadro (`zoom_at`, `apply_frame_effects`).

Momento-chave = palavra com peso: número/dinheiro, palavra de impacto
(legenda verde), palavra com emoji, fim de frase com "!" ou pergunta forte,
somado ao quanto a voz está mais alta que o normal naquele instante. Os
melhores viram zoom (com espaçamento mínimo pra não virar enjoo); os mais
fortes de todos viram impacto.
"""
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .emoji_map import EMOJI_KEYWORDS, EMOJI_PHRASES

EMOJI_DIR = Path(__file__).resolve().parent.parent / "assets" / "emoji"


def intensity() -> float:
    """Multiplicador global da força dos efeitos (FX_INTENSITY): 1.0 = como
    calibrado, 0.7 = 30% mais suave. Afeta zoom, abertura, flash, tremida,
    RGB e flicker (não a quantidade de efeitos)."""
    return max(float(getattr(config, "FX_INTENSITY", 1.0)), 0.0)


def _norm(word: str) -> str:
    w = unicodedata.normalize("NFKD", word.lower())
    w = "".join(c for c in w if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", w)


# chave mais longa primeiro: "videogame" tem que ganhar de "video"
_KEYS_LONGEST_FIRST = sorted(EMOJI_KEYWORDS, key=len, reverse=True)


def emoji_for(word: str) -> Optional[List[str]]:
    """Grupo de emojis pra uma palavra falada (ver emoji_map.py), ou None."""
    w = _norm(word)
    if len(w) < 3:
        return None
    for key in _KEYS_LONGEST_FIRST:
        pool = EMOJI_KEYWORDS[key]
        if key.endswith("="):
            if w == key[:-1]:
                return pool
        elif w.startswith(key):
            return pool
    return None


def phrase_emoji_for(words, i: int) -> Optional[List[str]]:
    """Grupo de emojis de uma expressão ("meu deus", "sei lá") que TERMINA
    na palavra `i`, ou None. Grupo vazio = expressão que bloqueia emoji
    ("todo mundo" não é 🌎)."""
    for n in (3, 2):
        if i - n + 1 < 0:
            continue
        phrase = " ".join(_norm(w.text) for w in words[i - n + 1:i + 1])
        if phrase in EMOJI_PHRASES:
            return EMOJI_PHRASES[phrase]
    return None


_NEGATIONS = {"nao", "nunca", "nenhum", "nenhuma", "sem", "jamais", "nem", "not", "no", "never"}


@dataclass
class FxPlan:
    zooms: List[Tuple[float, float, float]] = field(default_factory=list)   # (início, fim, nível)
    impacts: List[float] = field(default_factory=list)
    emojis: List[Tuple[float, float, str]] = field(default_factory=list)    # (início, fim, código)
    intro: bool = True
    _emoji_cache: dict = field(default_factory=dict, repr=False)

    # ----- zoom ---------------------------------------------------------
    def zoom_at(self, t: float) -> float:
        z = 1.0
        ease_in = getattr(config, "FX_ZOOM_IN_SECONDS", 0.22)
        ease_out = getattr(config, "FX_ZOOM_OUT_SECONDS", 0.45)
        for t0, t1, level in self.zooms:
            if t < t0 or t > t1 + ease_out:
                continue
            if t < t0 + ease_in:
                p = (t - t0) / ease_in
                # ease-out-back: passa um pouco do alvo e volta ("overshoot")
                c1 = 1.70158
                k = 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2
            elif t <= t1:
                k = 1.0
            else:
                p = (t - t1) / ease_out
                k = 1 - (p * p * (3 - 2 * p))  # smoothstep de volta
            z = max(z, 1.0 + (level - 1.0) * k)
        if self.intro:
            dur = getattr(config, "FX_INTRO_ZOOM_SECONDS", 0.7)
            if t < dur:
                p = t / dur
                z *= 1.0 + (getattr(config, "FX_INTRO_ZOOM", 1.18) - 1.0) * intensity() * (1 - p) ** 3
        return z

    # ----- efeitos no quadro já composto ---------------------------------
    def apply_frame_effects(self, frame: np.ndarray, t: float, out_h: int) -> np.ndarray:
        out = frame
        # impacto: flash + tremida + separação RGB, decaindo rápido
        for ti in self.impacts:
            dt = t - ti
            if 0 <= dt < 0.30:
                decay = (1 - dt / 0.30) ** 2
                if dt < 0.18:
                    k = intensity()
                    gain = 1.0 + getattr(config, "FX_FLASH_STRENGTH", 0.45) * k * (1 - dt / 0.18) ** 2
                    out = cv2.convertScaleAbs(out, alpha=gain, beta=18 * k * (1 - dt / 0.18))
                amp = getattr(config, "FX_SHAKE_PX", 14) * intensity() * decay
                if amp >= 1:
                    rng = np.random.default_rng(int(t * 1000))
                    dx, dy = rng.uniform(-amp, amp, 2)
                    m = np.float32([[1, 0, dx], [0, 1, dy]])
                    out = cv2.warpAffine(out, m, (out.shape[1], out.shape[0]),
                                         borderMode=cv2.BORDER_REFLECT)
                if dt < 0.14:
                    shift = int(round(getattr(config, "FX_RGB_SPLIT_PX", 10) * intensity() * (1 - dt / 0.14)))
                    if shift:
                        out = out.copy()
                        out[:, :, 2] = np.roll(out[:, :, 2], shift, axis=1)   # vermelho pra direita
                        out[:, :, 0] = np.roll(out[:, :, 0], -shift, axis=1)  # azul pra esquerda
                break
        # flicker de abertura (película antiga, só nos primeiros instantes)
        if self.intro and t < getattr(config, "FX_INTRO_FLICKER_SECONDS", 0.45):
            rng = np.random.default_rng(int(t * 30))
            amp = 0.2 * intensity()
            out = cv2.convertScaleAbs(out, alpha=float(rng.uniform(1 - amp, 1 + amp)), beta=0)
        # emoji acima da legenda
        for t0, t1, code in self.emojis:
            if t0 <= t <= t1:
                out = self._draw_emoji(out, code, t - t0, t1 - t, out_h)
                break
        return out

    def _emoji_rgba(self, code: str) -> Optional[np.ndarray]:
        if code not in self._emoji_cache:
            img = cv2.imread(str(EMOJI_DIR / f"{code}.png"), cv2.IMREAD_UNCHANGED)
            self._emoji_cache[code] = img if img is not None and img.shape[2] == 4 else None
        return self._emoji_cache[code]

    def _sticker(self, code: str, img: np.ndarray, size: int):
        """Emoji no tamanho `size` já em estilo "adesivo": contorno branco +
        sombra suave (um emoji escuro -- 💣, 💀 -- sumia em fundo escuro),
        com margem transparente pra o contorno não ser cortado na borda.
        Em cache: o tamanho só muda nos ~0.2s do "pop" de entrada, e refazer
        dilatação/desfoque todo quadro custava ~7ms por quadro."""
        key = (code, size)
        if key not in self._emoji_cache:
            sprite = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            pad = max(size // 10, 4)
            sprite = cv2.copyMakeBorder(sprite, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0, 0))
            a_full = sprite[:, :, 3].astype(np.float32) / 255.0
            k = max(size // 24, 2)
            stroke = cv2.dilate(a_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                  (2 * k + 1, 2 * k + 1)))
            stroke = cv2.GaussianBlur(stroke, (0, 0), sigmaX=1.0)
            shadow = cv2.GaussianBlur(stroke, (0, 0), sigmaX=max(size / 20.0, 1.5))
            self._emoji_cache[key] = (sprite, a_full, stroke, shadow)
        return self._emoji_cache[key]

    def _draw_emoji(self, frame, code: str, since: float, left: float, out_h: int):
        img = self._emoji_rgba(code)
        if img is None:
            return frame
        base = getattr(config, "FX_EMOJI_SIZE", 170)
        # entrada com "pop" (0.2 -> 1.15 -> 1.0), leve flutuação, saída com fade
        if since < 0.12:
            s = 0.2 + (1.15 - 0.2) * (since / 0.12)
        elif since < 0.22:
            s = 1.15 - 0.15 * ((since - 0.12) / 0.10)
        else:
            s = 1.0
        alpha = min(1.0, left / 0.15) if left < 0.15 else 1.0
        size = max(int(base * s), 8)
        sprite, a_full, stroke, shadow = self._sticker(code, img, size)
        h, w = frame.shape[:2]
        # a legenda fica com a BASE em CAPTION_MARGIN_V; o emoji vai logo acima.
        # folga acima da legenda: o grupo entra crescendo até 108% e a
        # palavra falada até 125% — com 10px o emoji encostava no texto
        cap_top = out_h - getattr(config, "CAPTION_MARGIN_V", 560) - int(getattr(config, "CAPTION_FONT_SIZE", 135) * 0.95)
        cy = cap_top - base // 2 - 60 + int(6 * math.sin(since * 6.0))
        cx = w // 2 + int(8 * math.sin(since * 3.0))
        sh, sw = sprite.shape[:2]
        x0, y0 = cx - sw // 2, cy - sh // 2
        x1, y1 = x0 + sw, y0 + sh
        sx0, sy0 = max(0, -x0), max(0, -y0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return frame
        ys, xs = slice(sy0, sy0 + (y1 - y0)), slice(sx0, sx0 + (x1 - x0))
        rgb = sprite[ys, xs, :3].astype(np.float32)
        a = a_full[ys, xs, None] * alpha
        st = stroke[ys, xs, None] * alpha
        sd = shadow[ys, xs, None] * alpha
        out = frame  # o quadro composto já é um array novo a cada frame: desenha nele
        region = out[y0:y1, x0:x1].astype(np.float32)
        region *= 1.0 - 0.45 * sd                       # sombra
        region = region * (1 - st) + 255.0 * st          # contorno branco
        region = region * (1 - a) + rgb * a              # emoji
        out[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)
        return out


def _word_weight(text: str, emphasis: set) -> float:
    bare = re.sub(r"[^\wà-öø-ÿ%$]", "", text.lower())
    if not bare:
        return 0.0
    w = 0.0
    if re.search(r"\d", bare) or "%" in text or "$" in text:
        w += 1.5
    if bare in emphasis:
        w += 1.5
    if emoji_for(text):
        w += 1.0
    if text.rstrip().endswith("!"):
        w += 1.2
    if text.rstrip().endswith("?") and len(bare) > 3:
        w += 0.6
    return w


def plan_effects(words, duration: float, energies: Optional[np.ndarray] = None,
                 energy_hop: float = 0.1, hook: bool = True) -> FxPlan:
    """`words`: palavras com timestamps na linha do tempo FINAL do clipe."""
    from .captioner import _emphasis_set
    plan = FxPlan(intro=getattr(config, "FX_INTRO_ENABLED", True))
    if not getattr(config, "FX_ENABLED", True):
        plan.intro = False
        return plan
    emphasis = _emphasis_set()
    e_mean = float(energies.mean()) if energies is not None and len(energies) else 0.0
    e_std = float(energies.std()) if energies is not None and len(energies) else 0.0

    scored = []
    for i, w in enumerate(words):
        weight = _word_weight(w.text, emphasis)
        if energies is not None and e_std > 1e-6:
            idx = min(int(w.start / energy_hop), len(energies) - 1)
            z = (float(energies[idx]) - e_mean) / e_std
            weight += max(min(z, 3.0), 0.0) * 0.6
        if weight >= getattr(config, "FX_KEY_MOMENT_MIN_WEIGHT", 1.5):
            scored.append((weight, i))
    scored.sort(reverse=True)

    # zoom nos momentos-chave: melhores primeiro, respeitando espaçamento
    min_gap = getattr(config, "FX_ZOOM_MIN_GAP_SECONDS", 3.5)
    start_after = 1.2 if plan.intro else 0.3  # não briga com o zoom de abertura
    taken: List[float] = []
    for weight, i in scored:
        t0 = max(words[i].start - 0.08, 0.0)
        if t0 < start_after or t0 > duration - 1.0:
            continue
        if any(abs(t0 - t) < min_gap for t in taken):
            continue
        # segura até o fim da frase (ou no máx. 2.2s)
        t1 = words[i].end
        for w in words[i + 1:]:
            if w.start - t1 > 0.35 or w.end - t0 > 2.2:
                break
            t1 = w.end
            if w.text.rstrip().endswith((".", "!", "?")):
                break
        level = 1.0 + min(0.08 + 0.025 * weight, getattr(config, "FX_ZOOM_MAX", 1.16) - 1.0) * intensity()
        plan.zooms.append((t0, max(t1, t0 + 0.6), level))
        taken.append(t0)
    plan.zooms.sort()

    # impacto: só os momentos mais fortes, bem espaçados
    imp_gap = getattr(config, "FX_IMPACT_MIN_GAP_SECONDS", 8.0)
    imp_min = getattr(config, "FX_IMPACT_MIN_WEIGHT", 3.0)
    for weight, i in scored:
        t = words[i].start
        if weight < imp_min or t < 0.8:
            continue
        if any(abs(t - x) < imp_gap for x in plan.impacts):
            continue
        plan.impacts.append(t)
    plan.impacts.sort()

    # emojis: primeira palavra-chave de cada trecho, espaçados
    if getattr(config, "FX_EMOJI_ENABLED", True):
        emo_gap = getattr(config, "FX_EMOJI_MIN_GAP_SECONDS", 2.5)
        hold = getattr(config, "FX_EMOJI_SECONDS", 1.3)
        repeat_gap = getattr(config, "FX_EMOJI_REPEAT_GAP_SECONDS", 25.0)
        last = -1e9
        last_by_code: dict = {}
        hook_until = getattr(config, "HOOK_SECONDS", 3.2) if hook else 0.0
        for i, w in enumerate(words):
            pool = phrase_emoji_for(words, i)
            if pool is None:
                pool = emoji_for(w.text)
            if not pool or w.start - last < emo_gap or w.start < hook_until * 0.5:
                continue  # sem emoji, ou expressão que bloqueia ("todo mundo")
            prev = [_norm(x.text) for x in words[max(0, i - 2):i]]
            if any(x in _NEGATIONS for x in prev):
                continue  # "não tem vitória" não ganha troféu
            # do grupo, o emoji usado há mais tempo (variedade), fora os que
            # acabaram de aparecer
            ready = [c for c in pool if (EMOJI_DIR / f"{c}.png").exists()
                     and w.start - last_by_code.get(c, -1e9) >= repeat_gap]
            if not ready:
                continue
            code = min(ready, key=lambda c: last_by_code.get(c, -1e9))
            plan.emojis.append((w.start, min(w.start + hold, duration), code))
            last = last_by_code[code] = w.start
    return plan
