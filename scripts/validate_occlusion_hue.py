#!/usr/bin/env python3
"""
Re-valida (e ajuda a calibrar) o discriminador de oclusão por matiz/hue
descrito no RELATORIO_PROXIMOS_PASSOS.txt (item 2), usando exatamente a
mesma metodologia: duas janelas de um clipe real — uma em que a pessoa
rastreada está FALANDO de verdade, outra em que só está com a mão perto
do queixo/boca (ou outro gesto) — e compara a atividade média descontada
(a mesma conta que `_FaceActivityTracker.sample_motion` faz em
reframer.py) entre as duas.

Uso:
    python3 scripts/validate_occlusion_hue.py <clipe.mp4> \\
        --winA 9.0 3.0 --winB 23.0 2.2

--winA é a janela "falando de verdade" (start_seconds duration_seconds),
--winB é a janela "só gesto/oclusão". Espera-se ratio(A/B) bem MAIOR que 1
(quanto maior, melhor a discriminação). O relatório mediu ~1.11x no modo
antigo (contraste em cinza) e ~2.2x no modo hue.

Se o ratio sair menor que o esperado no SEU clipe, ajuste
FACE_OCCLUSION_HUE_STD_MIN / FACE_OCCLUSION_SAT_FLOOR em src/config.py e
rode de novo — este script usa os valores de config.py atuais, então dá
pra iterar rápido sem tocar no pipeline principal.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.reframer import _new_cascades, _detect_all_faces  # noqa: E402
from src import config  # noqa: E402


def _extract_frames(video_path: str, start: float, duration: float):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    n = max(int(round(duration * fps)), 1)
    frames = []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def _mouth_patch(frame, face, ps=24):
    cx, cy, fw, fh = face
    h, w = frame.shape[:2]
    rw, rh = max(fw * 0.7, 10), max(fh * 0.5, 10)
    x0 = int(np.clip(cx - rw / 2, 0, max(w - 1, 0)))
    y0 = int(np.clip(cy - fh * 0.05, 0, max(h - 1, 0)))
    x1 = int(np.clip(x0 + rw, 0, w))
    y1 = int(np.clip(y0 + rh, 0, h))
    if x1 <= x0 or y1 <= y0:
        return None, None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gp = cv2.resize(gray[y0:y1, x0:x1], (ps, ps), interpolation=cv2.INTER_AREA).astype(np.float32)
    hp = cv2.resize(hsv[y0:y1, x0:x1], (ps, ps), interpolation=cv2.INTER_AREA)
    return gp, hp


def _window_activity(frames, cascades, use_hue: bool):
    hue_min = config.FACE_OCCLUSION_HUE_STD_MIN
    sat_floor = config.FACE_OCCLUSION_SAT_FLOOR
    contrast_min = config.FACE_OCCLUSION_CONTRAST_MIN

    prev_gray = None
    scores = []
    n_no_face = 0
    for frame in frames:
        faces = _detect_all_faces(cascades, frame)
        if not faces:
            n_no_face += 1
            prev_gray = None
            continue
        face = max(faces, key=lambda f: f[2] * f[3])
        gray_patch, hsv_patch = _mouth_patch(frame, face)
        if gray_patch is None:
            continue
        if prev_gray is not None:
            raw = float(np.mean(np.abs(gray_patch - prev_gray)))
            if use_hue:
                hue = hsv_patch[:, :, 0].astype(np.float32)
                sat = hsv_patch[:, :, 1].astype(np.float32)
                hue_std = float(np.std(hue))
                mean_sat = float(np.mean(sat))
                sat_conf = float(np.clip(mean_sat / max(sat_floor, 1e-3), 0.0, 1.0))
                factor = float(np.clip(hue_std / max(hue_min, 1e-3), 0.12, 1.0)) * (0.4 + 0.6 * sat_conf)
            else:
                contrast = float(np.std(gray_patch))
                factor = float(np.clip(contrast / max(contrast_min, 1e-3), 0.12, 1.0))
            scores.append(raw * factor)
        prev_gray = gray_patch
    return scores, n_no_face


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="caminho do clipe de teste")
    ap.add_argument("--winA", nargs=2, type=float, required=True, metavar=("START", "DUR"),
                     help="janela 'falando de verdade' (segundos)")
    ap.add_argument("--winB", nargs=2, type=float, required=True, metavar=("START", "DUR"),
                     help="janela 'só gesto/oclusão' (segundos)")
    args = ap.parse_args()

    cascades = _new_cascades()
    framesA = _extract_frames(args.video, *args.winA)
    framesB = _extract_frames(args.video, *args.winB)
    if not framesA or not framesB:
        sys.exit("Não consegui ler frames de uma das janelas — confira os timestamps/caminho do vídeo.")

    for label, use_hue in [("hue (novo, FACE_OCCLUSION_USE_HUE=True)", True),
                            ("contraste em cinza (antigo)", False)]:
        scoresA, missA = _window_activity(framesA, cascades, use_hue)
        scoresB, missB = _window_activity(framesB, cascades, use_hue)
        meanA = float(np.mean(scoresA)) if scoresA else 0.0
        meanB = float(np.mean(scoresB)) if scoresB else 0.0
        ratio = meanA / meanB if meanB > 1e-9 else float("inf")
        print(f"\n=== modo: {label} ===")
        print(f"  winA (falando):  atividade média = {meanA:.3f}  "
              f"({len(scoresA)} amostras, {missA} frames sem rosto)")
        print(f"  winB (oclusão):  atividade média = {meanB:.3f}  "
              f"({len(scoresB)} amostras, {missB} frames sem rosto)")
        print(f"  ratio A/B = {ratio:.2f}x  "
              f"{'(bom — bem maior que 1x)' if ratio > 1.8 else '(fraco — considere recalibrar)'}")


if __name__ == "__main__":
    main()
