"""
De quem é essa voz? (o streamer ou outra pessoa -- voice chat do jogo)

Pedido do usuário (corte do DrDonut, Short -Jxy2H_GLkE): "os players no game
estão falando com ele e a câmera está focada no rosto dele". Em gameplay,
quando outro jogador fala, a câmera tem que mostrar o jogo.

O que NÃO funcionou (medido no SMP do Tommyinnit, ~60 frases):
  - tom de voz (F0): o mesmo streamer vai de 130 Hz falando calmo a 350 Hz
    gritando -- metade das frases dele virava "outra pessoa";
  - boca na facecam: rosto pequeno, microfone na frente; sinal fraco.

O que funciona: a "impressão digital" da voz (embedding de locutor) com um
modelo pequeno e pronto (WeSpeaker ResNet34, 26 MB, licença Apache-2.0),
rodando no processador pelo onnxruntime -- que já vem instalado junto com a
transcrição (faster-whisper). ~20 ms por frase; não usa placa de vídeo.

  1. cada frase (palavras da transcrição) vira um vetor de 256 números;
  2. a voz PRINCIPAL do clipe é o grupo maior de frases parecidas entre si
     (o streamer fala mais que todo mundo no clipe dele);
  3. frase com voz bem diferente da principal = outra pessoa.

O modelo é baixado uma vez (models/voz/), conferido por tamanho e SHA-256.
Sem internet ou sem o onnxruntime, a função devolve None e o modo gameplay
segue só com a regra da boca (como antes).
"""
import hashlib
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "voz"
MODEL_NAME = "wespeaker_resnet34_LM.onnx"
MODEL_URL = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx"
MODEL_SIZE = 26530309
MODEL_SHA256 = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
SR = 16000

_state = {"session": None, "failed_at": 0.0, "lock": threading.Lock()}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def model_path(log=print) -> Optional[Path]:
    """Baixa (uma vez) e confere o modelo. None se não deu."""
    final = MODEL_DIR / MODEL_NAME
    if final.exists() and final.stat().st_size == MODEL_SIZE:
        return final
    if time.time() - _state["failed_at"] < 3600:
        return None
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        log(f"    [voz] baixando o modelo de vozes ({MODEL_SIZE / 1024 ** 2:.0f} MB, uma vez só)...")
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
        log(f"    [voz] não consegui baixar o modelo de vozes ({e.__class__.__name__}); sigo sem ele.")
        return None


def _session(log=print):
    with _state["lock"]:
        if _state["session"] is not None:
            return _state["session"]
        path = model_path(log)
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
            log(f"    [voz] modelo de vozes indisponível ({e.__class__.__name__}); sigo sem ele.")
            return None
        return _state["session"]


# --- fbank no padrão do Kaldi (o que o WeSpeaker espera) ------------------
_MEL = {}


def _mel_banks(n_fft: int, n_mels: int = 80, low: float = 20.0, high: float = SR / 2.0) -> np.ndarray:
    key = (n_fft, n_mels)
    if key in _MEL:
        return _MEL[key]

    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f) / 700.0)
    m_lo, m_hi = mel(low), mel(high)
    delta = (m_hi - m_lo) / (n_mels + 1)
    freqs = np.arange(n_fft // 2 + 1) * SR / n_fft
    mf = mel(freqs)
    banks = np.zeros((n_mels, len(freqs)), np.float32)
    for i in range(n_mels):
        left, center, right = m_lo + i * delta, m_lo + (i + 1) * delta, m_lo + (i + 2) * delta
        up = (mf - left) / (center - left)
        down = (right - mf) / (right - center)
        banks[i] = np.maximum(0.0, np.minimum(up, down))
    _MEL[key] = banks
    return banks


def fbank(x: np.ndarray) -> np.ndarray:
    """80 bandas log-mel, janela 25 ms, passo 10 ms, média tirada (CMN)."""
    x = x.astype(np.float64) * 32768.0
    win, hop, n_fft = 400, 160, 512
    if len(x) < win:
        return np.zeros((0, 80), np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(axis=1, keepdims=True)
    fr = np.concatenate([fr[:, :1] * (1 - 0.97), fr[:, 1:] - 0.97 * fr[:, :-1]], axis=1)
    fr = fr * np.hamming(win)[None, :]
    spec = np.abs(np.fft.rfft(fr, n=n_fft, axis=1)) ** 2
    feats = np.log(np.maximum(spec @ _mel_banks(n_fft).T, np.finfo(np.float32).eps))
    feats -= feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


def load_audio(path: str, start: float, end: float) -> Optional[np.ndarray]:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(start, 0):.3f}", "-t", f"{max(end - start, 0.1):.3f}",
           "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return x if len(x) > SR // 2 else None


def embed(x: np.ndarray, log=print) -> Optional[np.ndarray]:
    sess = _session(log)
    if sess is None:
        return None
    f = fbank(x)
    if len(f) < 50:  # menos de ~0,5 s de fala: não dá pra dizer de quem é
        return None
    e = sess.run(None, {"feats": f[None]})[0][0]
    return e / (np.linalg.norm(e) + 1e-9)


def who_speaks(audio: np.ndarray, phrases: List[Tuple[float, float]], log=print,
               min_len: float = 1.0, ref: Optional[List[bool]] = None) -> Optional[List[Optional[float]]]:
    """Pra cada frase (início, fim em s no tempo de `audio`): o quanto a voz
    parece a voz PRINCIPAL do clipe (cosseno: ~0,6-0,9 mesma pessoa,
    ~0-0,3 outra). None na frase curta demais; None no total se o modelo
    não está disponível. `ref`: só essas frases definem a voz principal
    (ex.: as de dentro do clipe, sem o trecho de depois)."""
    embs, idx = [], []
    for i, (a, b) in enumerate(phrases):
        if b - a < min_len:
            continue
        seg = audio[int(a * SR):int(b * SR)]
        e = embed(seg, log)
        if e is None:
            if _state["session"] is None:
                return None
            continue
        embs.append(e)
        idx.append(i)
    out: List[Optional[float]] = [None] * len(phrases)
    if len(embs) < 2:
        return out
    E = np.stack(embs)
    use = np.array([ref is None or bool(ref[i]) for i in idx])
    if use.sum() < 2:
        use[:] = True
    Er = E[use]
    S = Er @ Er.T
    # voz principal: a frase que mais se parece com as outras (o "centro" do
    # grupo maior), e a média das que se parecem com ela
    medoid = int(np.argmax((S > 0.5).sum(axis=1) + 0.01 * S.sum(axis=1)))
    members = S[medoid] > 0.5
    main = Er[members].mean(axis=0)
    main /= np.linalg.norm(main) + 1e-9
    for k, i in enumerate(idx):
        out[i] = float(E[k] @ main)
    return out


def embed_phrases(audio: np.ndarray, phrases: List[Tuple[float, float]], log=print,
                  min_len: float = 1.0) -> Optional[List[Optional[np.ndarray]]]:
    """A "impressão digital" de cada frase (None na curta demais); None no
    total se o modelo não está disponível."""
    out: List[Optional[np.ndarray]] = []
    for a, b in phrases:
        if b - a < min_len:
            out.append(None)
            continue
        e = embed(audio[int(a * SR):int(b * SR)], log)
        if e is None and _state["session"] is None:
            return None
        out.append(e)
    return out
