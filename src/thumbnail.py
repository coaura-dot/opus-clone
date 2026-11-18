"""
Capa (thumbnail) do Short: o quadro de mais IMPACTO do próprio clipe --
careta, susto, gargalhada, boca aberta, rosto grande e nítido.

Pedido do usuário (v47, depois do vídeo "a forma certa de enviar Shorts":
"capa chamativa aumenta a taxa de cliques"): "faz ele escolher um frame de
impacto, careta, ou algo assim do vídeo pra pôr de thumb".

Como escolhe (tudo local, ~5-10 s por clipe):
  1. amostra o clipe pronto a 4 quadros/s (pula o começo e o fim);
  2. rosto (YuNet, o mesmo do reenquadramento) + EXPRESSÃO (modelo FER+ do
     ONNX Model Zoo, licença MIT, 35 MB baixado uma vez em models/expressao):
     quanto menos "cara neutra", melhor -- surpresa, gargalhada, medo, nojo
     e raiva pesam mais. Sorriso só vale inteiro com DENTES à mostra
     (gargalhada); sorriso fechado olhando pra baixo não é careta;
  3. ajuda: rosto GRANDE (em plano aberto a expressão não aparece na capa e
     o modelo erra), de frente, nítido (sem borrão de movimento) e pico de
     som (grito, risada) perto do quadro;
  4. atrapalha: olho fechado sem estar rindo (piscada no meio da fala);
  5. sem rosto (gameplay, tela): o quadro mais nítido e colorido no pico de
     som;
  6. refina em volta do melhor momento (12 quadros/s) pra pegar o quadro
     exato do pico da careta.
Grava clip_XX.thumb.jpg (1080x1920, o quadro do próprio vídeo, com a
legenda) e o tempo em segundos no .meta.json ("thumb_time").
"""
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "expressao"
MODEL_NAME = "emotion-ferplus-8.onnx"
MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
MODEL_SIZE = 35040571
MODEL_SHA256 = "a2a2ba6a335a3b29c21acb6272f962bd3d47f84952aaffa03b60986e04efa61c"
# saída do FER+: neutro, felicidade, surpresa, tristeza, raiva, nojo, medo, desprezo
_IMPACT = np.array([0.0, 0.8, 1.0, 0.35, 0.8, 0.9, 0.9, 0.4], dtype=np.float32)

AW, AH = 540, 960          # análise em meia resolução (o vídeo é 1080x1920)
FPS = 4.0

_state = {"session": None, "failed_at": 0.0, "lock": threading.Lock(), "yunet": None}


def thumb_path(video: str) -> Path:
    return Path(video).with_suffix(".thumb.jpg")


# ------------------------------------------------------------ modelo ---
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _model(log) -> Optional[Path]:
    final = MODEL_DIR / MODEL_NAME
    if final.exists() and final.stat().st_size == MODEL_SIZE:
        return final
    if time.time() - _state["failed_at"] < 3600:
        return None
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        log(f"    [capa] baixando o modelo de expressões ({MODEL_SIZE / 1024 ** 2:.0f} MB, uma vez só)...")
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AutoClipper"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                out.write(buf)
        if part.stat().st_size != MODEL_SIZE or _sha256(part) != MODEL_SHA256:
            part.unlink()
            raise OSError("arquivo baixado não confere")
        os.replace(part, final)
        return final
    except Exception as e:
        _state["failed_at"] = time.time()
        log(f"    [capa] não consegui baixar o modelo de expressões ({e.__class__.__name__}); "
            "escolho a capa sem ele (rosto grande e nítido no pico de som).")
        return None


def _session(log):
    with _state["lock"]:
        if _state["session"] is None:
            path = _model(log)
            if path is None:
                return None
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.log_severity_level = 3
                _state["session"] = ort.InferenceSession(str(path), sess_options=opts,
                                                         providers=["CPUExecutionProvider"])
            except Exception as e:
                _state["failed_at"] = time.time()
                log(f"    [capa] modelo de expressões não abriu ({e.__class__.__name__}).")
                return None
        return _state["session"]


def _yunet():
    if _state["yunet"] is None:
        from .reframer import _new_yunet
        _state["yunet"] = _new_yunet() or False
    return _state["yunet"] or None


# ---------------------------------------------------------- medidas ---
def _audio_rms(video: str, step: float = 0.25) -> Optional[np.ndarray]:
    try:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                              "-f", "s16le", "-"], capture_output=True, timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32)
    hop = int(16000 * step)
    n = len(x) // hop
    if n < 4:
        return None
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-6)


def _frames(video: str, start: float, end: float, fps: float):
    """(tempo, quadro BGR em AWxAH), lendo um por vez (não carrega tudo na memória)."""
    cmd =["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
           "-vf", f"fps={fps},scale={AW}:{AH}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    size = AW * AH * 3
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        k = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield start + k / fps, np.frombuffer(buf, np.uint8).reshape(AH, AW, 3)
            k += 1


def _faces(det, frame) -> list:
    """Rostos do YuNet (caixa x, y, w, h + 5 pontos: olhos, nariz, cantos da
    boca), em pixels do quadro de análise."""
    det.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = det.detect(frame)
    if faces is None:
        return []
    return [[float(v) for v in f] for f in faces if f[-1] >= 0.7]


def _face_feats(gray, f) -> dict:
    """De frente? Dentes à mostra? Boca aberta? Olhos abertos?"""
    x, y, w, h = f[:4]
    rex, rey, lex, ley, _, _, rmx, rmy, lmx, lmy = f[4:14]
    g = gray.astype(np.float32)
    face = g[int(max(y + 0.2 * h, 0)):int(y + 0.8 * h), int(max(x + 0.2 * w, 0)):int(x + 0.8 * w)]
    med = float(np.median(face)) if face.size else 128.0
    mw = max(abs(lmx - rmx), 1.0)
    mx, my = (lmx + rmx) / 2, (lmy + rmy) / 2
    mouth = g[int(max(my - 0.25 * mw, 0)):int(my + 0.45 * mw), int(max(mx - 0.45 * mw, 0)):int(mx + 0.45 * mw)]
    ew = 0.22 * w

    def eye(ex, ey):
        e = g[int(max(ey - 0.5 * ew, 0)):int(ey + 0.5 * ew), int(max(ex - 0.7 * ew, 0)):int(ex + 0.7 * ew)]
        return float((e < 0.6 * med).mean()) if e.size else 0.0
    return {"frontal": float(np.clip((abs(lex - rex) / max(w, 1) - 0.2) / 0.15, 0, 1)),
            "teeth": float((mouth > 1.25 * med).mean()) if mouth.size else 0.0,
            "open": float((mouth < 0.55 * med).mean()) if mouth.size else 0.0,
            "eyes": min(eye(rex, rey), eye(lex, ley))}


def _expression(sess, gray, box, feats) -> float:
    """0..1: o quanto a cara NÃO é neutra (peso maior pra careta/susto/riso)."""
    x, y, w, h = box[:4]
    side = max(w, h) * 1.15
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x1, y1 = int(min(cx + side / 2, gray.shape[1])), int(min(cy + side / 2, gray.shape[0]))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return 0.0
    import cv2
    crop = cv2.resize(gray[y0:y1, x0:x1], (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    logits = sess.run(None, {sess.get_inputs()[0].name: crop[None, None]})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    w_ = _IMPACT.copy()
    # sorriso só vale inteiro com dentes à mostra (gargalhada); surpresa, com a boca aberta
    w_[1] *= float(np.clip((feats["teeth"] - 0.05) / 0.15, 0.25, 1.0))
    w_[2] *= float(np.clip((feats["open"] + feats["teeth"]) / 0.3, 0.4, 1.0))
    return float((p * w_).sum())


def _sharp(gray, box=None) -> float:
    """Nitidez (variância do Laplaciano). No rosto, medida com o rosto em
    128x128: rosto minúsculo de plano aberto não pode parecer "mais nítido"
    que um close de pele lisa."""
    import cv2
    if box is not None:
        x, y, w, h = [int(v) for v in box[:4]]
        gray = gray[max(y, 0):max(y + h, 1), max(x, 0):max(x + w, 1)]
        if gray.size < 100:
            return 0.0
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    if gray.size < 100:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _colorful(frame) -> float:
    b, g, r = [frame[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def _score_frames(video, start, end, fps, sess, det, rms, step):
    import cv2
    rows = []
    hi = float(np.percentile(rms, 90)) if rms is not None else 1.0
    for t, f in _frames(video, start, end, fps):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 25:  # quadro escuro/preto
            continue
        loud = 0.0
        if rms is not None:
            a, b = int(max(t - 0.5, 0) / step), int((t + 0.25) / step) + 1
            if a < len(rms):
                loud = float(min(rms[a:max(b, a + 1)].max() / (hi + 1e-6), 1.5))
        faces = _faces(det, f) if det is not None else []
        if faces:
            box = max(faces, key=lambda b: b[2] * b[3])
            ft = _face_feats(gray, box)
            rows.append(dict(ft, t=t, face=True, expr=_expression(sess, gray, box, ft) if sess else 0.0,
                             size=box[3] / AH, sharp=_sharp(gray, box), loud=loud))
        else:
            rows.append({"t": t, "face": False, "expr": 0.0, "size": 0.0, "sharp": _sharp(gray),
                         "loud": loud, "color": _colorful(f)})
    return rows


def _rank(rows) -> list:
    """Nota de cada quadro (0..~1.3)."""
    if not rows:
        return []
    face_sharp = np.median([r["sharp"] for r in rows if r["face"]] or [1.0]) or 1.0
    any_sharp = np.median([r["sharp"] for r in rows]) or 1.0
    colors = [r.get("color", 0.0) for r in rows if not r["face"]]
    color_ref = np.percentile(colors, 90) if colors else 1.0
    for r in rows:
        if r["face"]:
            sharp = min(r["sharp"] / face_sharp, 1.5) / 1.5
            size = min(r["size"] / 0.3, 1.0)           # rosto com 30% da altura: close de capa
            # rosto pequeno (plano aberto): a expressão não aparece na capa e o modelo erra
            trust = float(np.clip((r["size"] - 0.07) / 0.13, 0.0, 1.0)) * (0.5 + 0.5 * r["frontal"])
            s = 0.5 * r["expr"] * trust + 0.2 * size + 0.1 * r["frontal"] + 0.1 * sharp \
                + 0.1 * min(r["loud"], 1.0)
            if r["sharp"] < 0.45 * face_sharp:
                s -= 0.25                              # borrão de movimento: careta tremida não serve
            if r["eyes"] < 0.08 and r["teeth"] < 0.15:
                s -= 0.3                               # olho fechado sem estar rindo: piscada
            # rosto (grande) ganha de quadro sem rosto; rosto minúsculo de plano aberto, não
            r["score"] = s + 0.25 * float(np.clip((r["size"] - 0.05) / 0.1, 0.0, 1.0))
        else:
            sharp = min(r["sharp"] / any_sharp, 1.5) / 1.5
            r["score"] = 0.4 * sharp + 0.3 * min(r.get("color", 0.0) / (color_ref + 1e-6), 1.0) \
                + 0.3 * min(r["loud"], 1.0)
    return rows


def pick(video: str, meta: Optional[dict] = None, log=print) -> Optional[dict]:
    """Escolhe o quadro e grava a capa. {"time", "score", "why"} ou None."""
    import cv2
    try:
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(video)], capture_output=True, text=True, timeout=30).stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if dur < 2:
        return None
    sess, det = _session(log), _yunet()
    rms, step = _audio_rms(video), 0.25
    # começo (título-gancho, se tiver) e o último segundo ficam de fora
    lead = 2.5 if (meta or {}).get("hook_burned") else 0.8
    start, end = min(lead, dur * 0.2), max(dur - 1.0, dur * 0.8)
    rows = _rank(_score_frames(video, start, end, FPS, sess, det, rms, step))
    if not rows:
        return None
    best = max(rows, key=lambda r: r["score"])
    # refina: o pico exato da careta está entre as amostras
    fine = _rank(_score_frames(video, max(best["t"] - 0.3, start), min(best["t"] + 0.35, end), 12.0,
                               sess, det, rms, step) + [dict(best)])
    best = max(fine or [best], key=lambda r: r["score"])
    out = thumb_path(video)
    tmp = out.with_suffix(".tmp.jpg")
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{best['t']:.3f}", "-i", str(video), "-frames:v", "1",
                        "-q:v", "3", str(tmp)], capture_output=True, timeout=60)
    if r.returncode != 0 or not tmp.exists():
        return None
    if tmp.stat().st_size > 1_900_000:  # o YouTube aceita até 2 MB
        img = cv2.imread(str(tmp))
        cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    os.replace(tmp, out)
    why = (f"rosto, expressão {best['expr']:.2f}" if best["face"] else "sem rosto: quadro nítido no pico de som")
    return {"time": round(best["t"], 2), "score": round(best["score"], 3), "why": why}


def ensure(video: str, meta_path: Optional[str] = None, log=print) -> Optional[Path]:
    """A capa do clipe (escolhe agora se ainda não tem). None se não deu."""
    out = thumb_path(video)
    if out.exists():
        return out
    meta = {}
    if meta_path:
        try:
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    try:
        got = pick(video, meta, log)
    except Exception as e:  # capa nunca derruba a postagem
        log(f"    [capa] não consegui escolher a capa ({e.__class__.__name__}: {e}).")
        return None
    if not got:
        return None
    if meta_path and meta:
        meta["thumb_time"] = got["time"]
        try:
            Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
    log(f"    [capa] quadro em {int(got['time']) // 60}:{got['time'] % 60:04.1f} ({got['why']})")
    return out
