"""
Pipeline para vídeos longos (podcast de horas, entrevista longa etc.).

Problema que resolve: transcrever um vídeo de 1h40 (ou mais) inteiro, só
para usar 3-4 minutos dele em clipes, desperdiça a maior parte do tempo de
processamento — a transcrição costuma ser a etapa mais lenta do pipeline.

Estratégia (ver TOPIC_BOUNDARY_*/LONG_VIDEO_*/CHUNK_* em config.py):
  1. O vídeo já foi baixado por INTEIRO (main.py não muda isso). Acima de
     LONG_VIDEO_THRESHOLD_SECONDS, ele é tratado como uma sequência de
     blocos LÓGICOS de CHUNK_DURATION_SECONDS — apenas recortes de tempo
     sobre o arquivo já em disco (via extract_audio_segment, que só busca a
     posição e copia áudio, sem recodificar vídeo), não arquivos separados
     nem vídeo dividido fisicamente.
  2. Os blocos são embaralhados (ordem aleatória, sem repetição) e
     processados um de cada vez: extrai o áudio do bloco, transcreve só
     aquele trecho, roda select_clips() nele (igual ao vídeo inteiro, só
     que em escala de bloco) e filtra os candidatos pelo viral score
     mínimo (CHUNK_VIRAL_SCORE_MIN).
  3. Se o bloco tem candidato(s) bons o bastante, eles são entregues ao
     consumidor (main.py, que monta os clipes) IMEDIATAMENTE — e a
     transcrição do PRÓXIMO bloco já começa em uma thread de fundo antes
     disso, para que o tempo gasto montando o clipe (recorte vertical +
     legenda + música, a etapa mais cara depois da transcrição) se
     sobreponha ao tempo de transcrever o bloco seguinte, em vez de rodar
     tudo em sequência estrita ("pipeline eficiente" pedido pelo usuário).
  4. Se nenhum bloco do vídeo inteiro atingir o score mínimo (limiar
     calibrado alto demais para aquele conteúdo específico), em vez de
     falhar sem gerar nada, usa como último recurso os melhores candidatos
     vistos mesmo abaixo do limiar — sempre avisando no console quando
     isso acontece.

HONESTIDADE: o valor de CHUNK_VIRAL_SCORE_MIN é um palpite inicial (ver
comentário em config.py), não uma calibração validada contra vídeos longos
reais (não havia nenhum disponível neste ambiente de desenvolvimento) —
espere precisar ajustar depois de rodar contra o seu conteúdo de verdade.
"""
import queue
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import config
from .clip_selector import ClipCandidate, select_clips
from .transcriber import Transcript, Word, transcribe as _default_transcribe
from .video_editor import extract_audio_segment


def is_long_video(duration: float) -> bool:
    return duration > config.LONG_VIDEO_THRESHOLD_SECONDS


def build_chunks(duration: float, chunk_seconds: float = None) -> List[Tuple[float, float]]:
    """Divide [0, duration) em blocos de `chunk_seconds` (padrão:
    CHUNK_DURATION_SECONDS). O último bloco fica mais curto se a duração
    total não for múltipla exata — nunca é descartado, mesmo que sobrem só
    alguns segundos, para não perder conteúdo do fim do vídeo."""
    size = chunk_seconds or config.CHUNK_DURATION_SECONDS
    if duration <= 0 or size <= 0:
        return []
    chunks = []
    t = 0.0
    while t < duration:
        chunks.append((t, min(t + size, duration)))
        t += size
    return chunks


@dataclass
class _ChunkResult:
    chunk_index: int
    start: float
    end: float
    candidates: List[ClipCandidate] = field(default_factory=list)  # timestamps ABSOLUTOS
    error: Optional[str] = None


def _transcribe_and_score_chunk(
    source_path: str,
    chunk_index: int,
    start: float,
    end: float,
    total_duration: float,
    work_dir: Path,
    n_needed: int,
    transcribe_fn: Callable[..., Transcript] = None,
) -> _ChunkResult:
    """Extrai o áudio do bloco [start, end), transcreve e pontua com
    select_clips(). Retorna candidatos com timestamps já convertidos para
    ABSOLUTO (tempo no vídeo de origem, somando `start`) e com `.words`
    preenchido (também absoluto) para que build_clip consiga gerar a
    legenda sem precisar do transcript do vídeo inteiro.

    `total_duration` é a duração do VÍDEO INTEIRO (não do bloco) -- usado
    só pra informar select_clips() se este bloco é o primeiro/último do
    vídeo de verdade (ver V7 em clip_selector.py: um bloco do MEIO não
    pode tratar sua própria borda como início/fim de assunto automático,
    só um corte de tempo arbitrário).

    `transcribe_fn` é injetável (usado pelos testes, para não depender de
    faster-whisper/whisper.cpp reais); em produção usa
    src.transcriber.transcribe normalmente."""
    transcribe_fn = transcribe_fn or _default_transcribe
    chunk_audio = work_dir / f"chunk_{chunk_index}_audio.wav"
    try:
        extract_audio_segment(source_path, start, end, str(chunk_audio))
        transcript = transcribe_fn(
            str(chunk_audio),
            model_size=config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        chunk_duration = end - start
        try:
            local_candidates = select_clips(
                transcript, str(chunk_audio), chunk_duration, n_needed,
                is_first_segment=(start <= 0.01),
                is_last_segment=(end >= total_duration - 0.01),
            )
        except RuntimeError:
            local_candidates = []  # bloco sem nenhum candidato válido (ex.: silêncio/curto demais)

        abs_words = [Word(start=w.start + start, end=w.end + start, text=w.text)
                     for w in transcript.words]
        for c in local_candidates:
            c.start += start
            c.end += start
            c.words = abs_words

        return _ChunkResult(chunk_index, start, end, candidates=local_candidates)
    except Exception as e:
        return _ChunkResult(chunk_index, start, end, candidates=[], error=str(e))
    finally:
        try:
            if chunk_audio.exists():
                chunk_audio.unlink()
        except Exception:
            pass


def iter_long_video_clip_batches(
    source_path: str,
    total_duration: float,
    n_clips: int,
    work_dir: Path,
    transcribe_fn: Callable[..., Transcript] = None,
):
    """Gerador que produz, um bloco de cada vez, listas de ClipCandidate
    (timestamps absolutos no vídeo de origem, `.words` já preenchido) que
    passaram no critério de CHUNK_VIRAL_SCORE_MIN — na ordem em que os
    blocos terminam de ser processados (aleatória, ver build_chunks +
    shuffle abaixo), não a ordem cronológica do vídeo.

    A transcrição do bloco SEGUINTE começa em uma thread de fundo assim que
    o resultado do bloco atual sai da fila para o consumidor — como a fila
    tem tamanho 1, a thread de fundo nunca fica mais de um bloco à frente
    do consumidor (não desperdiça transcrição em blocos que talvez nem
    sejam necessários), mas também nunca fica ociosa esperando o consumidor
    terminar de montar o clipe atual.

    Para de produzir assim que `n_clips` candidatos já foram entregues, ou
    quando os blocos acabam. Se NENHUM bloco atingir o score mínimo, usa os
    melhores candidatos vistos (mesmo abaixo do limiar) como último
    recurso, para nunca terminar com zero clipes só por causa de um limiar
    calibrado alto demais para aquele vídeo específico — mas avisa no
    console quando isso acontece."""
    chunks = build_chunks(total_duration)
    random.shuffle(chunks)
    if not chunks:
        raise RuntimeError("Vídeo sem duração válida para dividir em blocos.")

    result_queue: "queue.Queue[Optional[_ChunkResult]]" = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    def worker():
        for i, (start, end) in enumerate(chunks):
            if stop_event.is_set():
                break
            res = _transcribe_and_score_chunk(
                source_path, i, start, end, total_duration, work_dir, n_clips,
                transcribe_fn=transcribe_fn,
            )
            result_queue.put(res)
            if stop_event.is_set():
                break
        result_queue.put(None)  # sentinela: acabaram os blocos (ou foi interrompido)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    delivered = 0
    leftover: List[ClipCandidate] = []  # candidatos vistos mas não entregues ainda
    chunks_seen = 0

    try:
        while True:
            res = result_queue.get()
            if res is None:
                break
            chunks_seen += 1
            if res.error:
                print(f"    [aviso] bloco {res.chunk_index} "
                      f"({res.start/60:.1f}-{res.end/60:.1f}min) falhou ao processar: {res.error}")
                continue

            good = [c for c in res.candidates if c.score >= config.CHUNK_VIRAL_SCORE_MIN]
            if good:
                good.sort(key=lambda c: c.score, reverse=True)
                take = good[: max(n_clips - delivered, 0)]
                if take:
                    delivered += len(take)
                    yield take
                leftover.extend(good[len(take):])
                if delivered >= n_clips:
                    stop_event.set()
                    break
            else:
                leftover.extend(res.candidates)

        if delivered < n_clips and leftover:
            leftover.sort(key=lambda c: c.score, reverse=True)
            fallback = leftover[: n_clips - delivered]
            if fallback:
                print(f"    [aviso] só {delivered} clipe(s) atingiram o viral score mínimo "
                      f"({config.CHUNK_VIRAL_SCORE_MIN}) em {chunks_seen} bloco(s) analisado(s); "
                      f"completando com {len(fallback)} candidato(s) abaixo do limiar. "
                      f"Considere baixar CHUNK_VIRAL_SCORE_MIN em config.py se isso for frequente.")
                delivered += len(fallback)
                yield fallback

        if delivered == 0:
            raise RuntimeError(
                f"Nenhum candidato de corte foi encontrado em {chunks_seen} bloco(s) "
                "analisados do vídeo."
            )
    finally:
        stop_event.set()
