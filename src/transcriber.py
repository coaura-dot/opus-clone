"""
Transcrição com timestamps por palavra usando faster-whisper.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    segments: List[Segment]

    @property
    def words(self) -> List[Word]:
        ws = []
        for seg in self.segments:
            ws.extend(seg.words)
        return ws

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


def _norm_token(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def _drop_repetition_loops(words: List[Word]) -> List[Word]:
    """Remove "loops" de alucinação do Whisper: a mesma frase curta repetida
    em sequência dezenas de vezes (achado real num podcast: "Foi 99." 62
    vezes seguidas, num trecho de risada/música) — que ia parar na legenda e
    na seleção de cortes. Uma sequência de 2-6 palavras repetida 3+ vezes
    seguidas (ou 1 palavra repetida 4+ vezes — "não, não, não" é fala
    normal) fica só com a primeira ocorrência."""
    toks = [_norm_token(w.text) for w in words]
    out: List[Word] = []
    i = 0
    while i < len(words):
        skipped = False
        for n in range(1, 7):  # período mais curto primeiro (senão "foi 99" x3 vira um bloco de 6 mantido inteiro)
            chunk = toks[i:i + n]
            if len(chunk) < n or not any(chunk):
                continue
            reps = 1
            while toks[i + reps * n:i + (reps + 1) * n] == chunk:
                reps += 1
            if reps >= (4 if n == 1 else 3):
                out.extend(words[i:i + n])
                i += reps * n
                skipped = True
                break
        if not skipped:
            out.append(words[i])
            i += 1
    return out


# símbolo que o Whisper às vezes devolve como "palavra" separada: "100 %"
# aparecia assim na legenda e no título
_SYMBOL_ONLY_RE = re.compile(r"^[%°ºª]+[.,!?;:]*$")


def _attach_symbols(words: List[Word]) -> List[Word]:
    out: List[Word] = []
    for w in words:
        if out and _SYMBOL_ONLY_RE.match(w.text.strip()):
            prev = out[-1]
            out[-1] = Word(start=prev.start, end=max(prev.end, w.end), text=prev.text.rstrip() + w.text.strip())
        else:
            out.append(w)
    return out


# contexto do vídeo (título, canal) usado como "dica" pro Whisper -- acerta
# mais nomes próprios e gírias. main.py define antes de transcrever.
_PROMPT = {"text": None}


def set_context(*parts) -> None:
    txt = ". ".join(p.strip() for p in parts if p and p.strip())
    _PROMPT["text"] = txt or None


def _speech_mask(audio_path: str):
    """Fala/silêncio a cada 10 ms (energia do áudio, limiar adaptativo)."""
    import subprocess
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
           "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = 160
    n = len(x) // hop
    if n < 50:
        return None
    rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    floor, loud = np.percentile(db, 10), np.percentile(db, 90)
    thr = max(floor + 0.35 * (loud - floor), floor + 6.0)
    mask = db > thr
    # fecha buracos curtos (< 60 ms) e tira estalos (< 40 ms)
    m = mask.copy()
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if 0 < i and j < n and j - i < 6:
                m[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i < 4:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def _onset_strength(audio_path: str):
    """Força de "começo de som" a cada 10 ms (subida de energia), suavizada."""
    import subprocess
    import numpy as np
    cmd = ["ffmpeg", "-v", "error", "-i", str(audio_path), "-ac", "1", "-ar", "16000",
           "-f", "s16le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.int16).astype(np.float32)
    hop = 160
    n = len(x) // hop
    if n < 100:
        return None
    db = 10 * np.log10(np.mean((x[:n * hop] / 32768.0).reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = np.convolve(db, np.ones(3) / 3, mode="same")
    rise = np.clip(np.diff(db, prepend=db[0]), 0, None)
    return np.convolve(rise, np.ones(5), mode="same")  # +-20 ms


def fix_offsets(words: List[Word], audio_path: str, window_s: float = 20.0,
                max_shift: float = 0.6) -> List[Word]:
    """Corrige atraso/adiantamento CONSTANTE dos tempos do Whisper, trecho a
    trecho (~20 s): testa deslocamentos de -0,6 a +0,6 s e fica com o que
    melhor encaixa os começos e fins de FRASE (palavra depois/antes de uma
    pausa) nos começos e fins de fala do áudio -- e, de desempate, os
    começos de palavra nos ataques de sílaba. O whisper.cpp erra assim (a
    legenda inteira de um trecho vinha uns décimos atrasada) e ajuste
    palavra por palavra não pega esse caso. Só usa ataque de sílaba sozinho
    não dá: as sílabas se repetem a cada ~0,2 s e um deslocamento errado
    também "encaixa"."""
    import numpy as np
    if len(words) < 8:
        return words
    try:
        env = _onset_strength(audio_path)
        m = _speech_mask(audio_path)
    except Exception:
        return words
    if env is None or m is None:
        return words
    n = min(len(env), len(m))
    mf = m[:n].astype(np.float32)
    env = env[:n] / (np.percentile(env[:n], 99) + 1e-9)
    shifts = np.arange(-int(max_shift * 100), int(max_shift * 100) + 1, 2)
    starts = np.array([w.start for w in words])
    ends = np.array([w.end for w in words])
    gap_before = np.r_[10.0, starts[1:] - ends[:-1]]
    gap_after = np.r_[starts[1:] - ends[:-1], 10.0]
    out = [Word(w.start, w.end, w.text) for w in words]

    def frac(idx, a, b):
        """fração de fala entre idx+a e idx+b (quadros de 10 ms)"""
        vals = []
        for k in range(a, b):
            j = np.clip(idx + k, 0, n - 1)
            vals.append(mf[j])
        return np.mean(vals, axis=0)

    centers, offs = [], []
    t = starts[0]
    while t <= starts[-1]:
        sel = np.nonzero((starts >= t) & (starts < t + window_s))[0]
        ph_s = sel[gap_before[sel] > 0.25]   # começos de frase
        ph_e = sel[gap_after[sel] > 0.25]    # fins de frase
        off = 0.0
        if len(sel) >= 8 and len(ph_s) + len(ph_e) >= 3:
            fs = np.round(starts[sel] * 100).astype(int)
            fps_ = np.round(starts[ph_s] * 100).astype(int)
            fpe = np.round(ends[ph_e] * 100).astype(int)
            scores = []
            for sh in shifts:
                sc = 0.0
                if len(fps_):
                    sc += float(np.mean(frac(fps_ + sh, 0, 10) - frac(fps_ + sh, -15, -3)))
                if len(fpe):
                    sc += float(np.mean(frac(fpe + sh, -10, 0) - frac(fpe + sh, 5, 15)))
                idx = np.clip(fs + sh, 0, n - 1)
                sc += 0.3 * float(np.mean(env[idx]))
                scores.append(sc)
            scores = np.array(scores)
            best = int(np.argmax(scores))
            zero = scores[len(shifts) // 2]
            if scores[best] - zero > 0.08:  # só mexe com ganho claro
                off = shifts[best] / 100.0
        centers.append(t + window_s / 2)
        offs.append(off)
        t += window_s / 2  # janelas com metade sobreposta
    if not any(offs):
        return out
    o = np.array(offs)
    if len(o) >= 3:  # mediana de 3 janelas vizinhas: um trecho estranho não puxa sozinho
        o = np.r_[o[0], np.median(np.stack([o[:-2], o[1:-1], o[2:]]), axis=0), o[-1]]
    per_word = np.interp(starts, centers, o)
    for w, d in zip(out, per_word):
        w.start += float(d)
        w.end += float(d)
    for a, b in zip(out, out[1:]):
        if a.end > b.start:
            a.end = max(b.start, a.start + 0.05)
    return out


def snap_to_speech(words: List[Word], audio_path: str) -> List[Word]:
    """Acerta o começo de cada palavra no começo REAL da fala (energia do
    áudio). Os tempos do Whisper (principalmente do whisper.cpp) chegam
    adiantados ou atrasados em uns décimos de segundo, e a legenda aparecia
    antes de a pessoa falar ou depois dela já ter falado. Mexe só onde o
    áudio é claro: palavra que começa no silêncio vai pro próximo começo de
    fala (até 0,5 s depois); palavra depois de uma pausa recua pro começo da
    fala (até 0,3 s antes). Ninguém passa da palavra vizinha."""
    if not words:
        return words
    try:
        m = _speech_mask(audio_path)
    except Exception:
        return words
    if m is None:
        return words
    import numpy as np
    n = len(m)
    onsets = np.nonzero(m & ~np.r_[False, m[:-1]])[0]
    out = [Word(w.start, w.end, w.text) for w in words]
    for i, w in enumerate(out):
        f = int(round(w.start * 100))
        if f >= n:
            break
        prev_end = out[i - 1].end if i else 0.0
        next_start = out[i + 1].start if i + 1 < len(out) else float("inf")
        new = None
        if not m[max(f, 0)]:
            k = np.searchsorted(onsets, f)
            if k < len(onsets) and onsets[k] - f <= 50:
                new = onsets[k] / 100.0
        elif i == 0 or w.start - prev_end > 0.12:
            k = np.searchsorted(onsets, f, side="right") - 1
            if k >= 0 and f - onsets[k] <= 30:
                new = onsets[k] / 100.0
        if new is not None:
            new = min(max(new, prev_end), next_start - 0.02)
            if new > w.start - 0.5 and new < next_start:
                shift = new - w.start
                w.start = new
                if shift > 0:
                    w.end = min(max(w.end, w.start + 0.08), max(next_start, w.start + 0.08))
        if w.end <= w.start:
            w.end = w.start + 0.08
    return out


def _clean_transcript(t: "Transcript") -> "Transcript":
    # o filtro roda na sequência de palavras do vídeo INTEIRO (um loop
    # costuma atravessar vários segmentos do Whisper) e depois devolve cada
    # palavra mantida ao seu segmento de origem
    owner = {id(w): i for i, seg in enumerate(t.segments) for w in seg.words}
    kept = {id(w) for w in _drop_repetition_loops(t.words)}
    segments = []
    for i, seg in enumerate(t.segments):
        words = _attach_symbols([w for w in seg.words if id(w) in kept and owner[id(w)] == i])
        if len(words) != len(seg.words):
            seg = Segment(start=seg.start, end=seg.end,
                          text=" ".join(w.text for w in words), words=words)
        if seg.words or (seg.text.strip() and not t.segments[i].words):
            segments.append(seg)
    return Transcript(language=t.language, segments=segments)


def transcribe(audio_path: str, model_size: str = "small",
               device: str = "auto", compute_type: str = "auto") -> Transcript:
    """Ponto de entrada único usado pelo resto do pipeline. Escolhe o motor
    de transcrição de acordo com config.TRANSCRIBE_ENGINE:
      - "faster-whisper" (padrão): CPU ou GPU NVIDIA (CUDA) via CTranslate2.
      - "whispercpp": CPU ou GPU AMD/Intel/NVIDIA via Vulkan (whisper.cpp) —
        é o caminho que de fato acelera a RX 580 no Windows.
    Os parâmetros model_size/device/compute_type só valem para o motor
    "faster-whisper" (mantidos aqui por compatibilidade); o motor
    "whispercpp" lê sua própria configuração (WHISPERCPP_*) de config.py."""
    from . import config
    engine = getattr(config, "TRANSCRIBE_ENGINE", "faster-whisper")
    if engine == "whispercpp":
        from .transcriber_whispercpp import transcribe as transcribe_whispercpp
        from .transcriber_whispercpp import missing_setup
        problem = missing_setup(config.WHISPERCPP_MODEL, config.WHISPERCPP_BIN)
        if problem:
            print(f"    [aviso] whisper.cpp indisponível ({problem}) — usando "
                  "faster-whisper. Confira WHISPERCPP_BIN/WHISPERCPP_MODEL em config.py.")
            return _finish(_transcribe_faster_whisper(audio_path, model_size, device, compute_type), audio_path)
        return _finish(transcribe_whispercpp(
            audio_path,
            model_path=config.WHISPERCPP_MODEL,
            bin_path=config.WHISPERCPP_BIN,
            language=getattr(config, "WHISPERCPP_LANGUAGE", "auto"),
            threads=getattr(config, "WHISPERCPP_THREADS", 0),
            use_gpu=getattr(config, "WHISPERCPP_USE_GPU", True),
            gpu_device=getattr(config, "WHISPERCPP_GPU_DEVICE", 0),
            prompt=_PROMPT["text"],
            beam_size=getattr(config, "WHISPERCPP_BEAM_SIZE", 5),
            use_dtw=getattr(config, "WHISPERCPP_DTW", True),
        ), audio_path)
    return _finish(_transcribe_faster_whisper(audio_path, model_size, device, compute_type), audio_path)


def _finish(t: "Transcript", audio_path: str) -> "Transcript":
    from . import config
    t = _clean_transcript(t)
    if getattr(config, "CAPTION_SNAP_TO_SPEECH", True):
        # 1) atraso constante por trecho, no vídeo inteiro; 2) palavra por palavra
        all_words = t.words
        fixed = fix_offsets(all_words, audio_path) if all_words else all_words
        remap = {id(w): f for w, f in zip(all_words, fixed)}
        segments = []
        for seg in t.segments:
            words = [remap.get(id(w), w) for w in seg.words]
            words = snap_to_speech(words, audio_path) if words else words
            segments.append(Segment(start=words[0].start if words else seg.start,
                                    end=words[-1].end if words else seg.end,
                                    text=seg.text, words=words))
        t = Transcript(language=t.language, segments=segments)
    return t


def _transcribe_faster_whisper(audio_path: str, model_size: str = "small",
                                device: str = "auto", compute_type: str = "auto") -> Transcript:
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    print(f"[2/6] Transcrevendo áudio (modelo Whisper '{model_size}')...")

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    from . import config
    cpu_threads = config.WHISPER_CPU_THREADS
    if cpu_threads <= 0 and device == "cpu":
        # estima núcleos físicos (assume SMT 2x) pra aproveitar bem CPUs com
        # muitos núcleos, como Xeons de servidor, sem sobrecarregar com
        # threads lógicas demais
        n_logical = os.cpu_count() or 4
        cpu_threads = min(max(n_logical // 2, 1), 16) if n_logical > 4 else n_logical

    # beam_size=5 (padrão do faster-whisper) explora 5 hipóteses de texto em
    # paralelo pra cada trecho — mais preciso, mas ~4-5x mais lento que
    # busca gulosa (beam_size=1), sem ganho de precisão que realmente
    # importe pra este uso (achar os melhores cortes + legendar, não
    # transcrição jurídica/legendagem profissional). É o ajuste de maior
    # impacto disponível sem trocar de hardware.
    beam_size = getattr(config, "WHISPER_BEAM_SIZE", 1)

    # BatchedInferencePipeline agrupa os trechos de fala (já isolados pelo
    # VAD) em lotes e processa em paralelo, em vez de um de cada vez em
    # sequência — em CPUs com muitos núcleos (como o Xeon 2680 v4, 14
    # núcleos/28 threads) isso usa o paralelismo disponível de forma muito
    # mais eficiente do que rodar tudo numa fila única. Cai para o pipeline
    # sequencial normal (mais lento, porém universal) se a versão instalada
    # do faster-whisper não tiver esse recurso.
    batch_size = getattr(config, "WHISPER_BATCH_SIZE", 8)
    use_batched = getattr(config, "WHISPER_BATCHED", True) and batch_size > 1

    print(f"    -> device={device}  compute_type={compute_type}  beam_size={beam_size}"
          + (f"  cpu_threads={cpu_threads}  batch_size={batch_size if use_batched else '(desativado)'}"
             if device == "cpu" else ""))

    model = WhisperModel(
        model_size, device=device, compute_type=compute_type,
        cpu_threads=cpu_threads if device == "cpu" else 0,
        num_workers=config.WHISPER_NUM_WORKERS,
    )

    transcribe_kwargs = dict(word_timestamps=True, vad_filter=True, beam_size=beam_size)
    if _PROMPT["text"]:
        transcribe_kwargs["initial_prompt"] = _PROMPT["text"][:220]
    if use_batched:
        try:
            pipeline = BatchedInferencePipeline(model=model)
            raw_segments, info = pipeline.transcribe(
                audio_path, batch_size=batch_size, **transcribe_kwargs,
            )
        except Exception as e:
            print(f"    [aviso] BatchedInferencePipeline falhou ({e}), "
                  f"usando transcrição sequencial...")
            raw_segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    else:
        raw_segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    segments = []
    for rs in raw_segments:
        words = [
            Word(start=w.start, end=w.end, text=w.word.strip())
            for w in (rs.words or [])
            if w.word and w.word.strip()
        ]
        segments.append(Segment(start=rs.start, end=rs.end, text=rs.text, words=words))

    print(f"    -> Idioma detectado: {info.language} ({len(segments)} segmentos)")
    return Transcript(language=info.language, segments=segments)
