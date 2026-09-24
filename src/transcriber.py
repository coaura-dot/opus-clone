"""
Transcrição com timestamps por palavra usando faster-whisper.
"""
import os
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


def _clean_transcript(t: "Transcript") -> "Transcript":
    # o filtro roda na sequência de palavras do vídeo INTEIRO (um loop
    # costuma atravessar vários segmentos do Whisper) e depois devolve cada
    # palavra mantida ao seu segmento de origem
    owner = {id(w): i for i, seg in enumerate(t.segments) for w in seg.words}
    kept = {id(w) for w in _drop_repetition_loops(t.words)}
    segments = []
    for i, seg in enumerate(t.segments):
        words = [w for w in seg.words if id(w) in kept and owner[id(w)] == i]
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
            return _clean_transcript(_transcribe_faster_whisper(audio_path, model_size, device, compute_type))
        return _clean_transcript(transcribe_whispercpp(
            audio_path,
            model_path=config.WHISPERCPP_MODEL,
            bin_path=config.WHISPERCPP_BIN,
            language=getattr(config, "WHISPERCPP_LANGUAGE", "auto"),
            threads=getattr(config, "WHISPERCPP_THREADS", 0),
            use_gpu=getattr(config, "WHISPERCPP_USE_GPU", True),
            gpu_device=getattr(config, "WHISPERCPP_GPU_DEVICE", 0),
        ))
    return _clean_transcript(_transcribe_faster_whisper(audio_path, model_size, device, compute_type))


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
