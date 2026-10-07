"""
Transcrição via whisper.cpp — permite usar a GPU AMD (RX 580 incluída) no
Windows para acelerar a transcrição, via o backend Vulkan do ggml. O motor
padrão do projeto (faster-whisper/CTranslate2) só sabe rodar em CPU ou GPU
NVIDIA (CUDA); não existe backend AMD para ele em nenhum SO. O whisper.cpp,
por outro lado, compila com um backend Vulkan — API gráfica multiplataforma
que a RX 580 suporta nativamente no Windows (driver Adrenalin) — então a
GPU passa a fazer de verdade a multiplicação de matriz da rede neural.

Não existe binding Python oficial pré-compilado com Vulkan para Windows, e
compilar um a partir do código-fonte é bem mais complicado que compilar só
o executável de linha de comando. Por isso este módulo chama o binário
`whisper-cli` via subprocess (a forma testada e documentada pelo próprio
projeto de usá-lo por fora do C++) e faz o parsing da saída JSON (`-ojf`)
para os mesmos dataclasses (Transcript/Segment/Word) usados pelo resto do
pipeline — nenhum outro módulo do projeto precisa saber qual motor gerou a
transcrição.

Todos os detalhes abaixo (nomes de flag, formato do JSON, nome do binário,
flag de build do Vulkan) foram conferidos direto no código-fonte do
whisper.cpp (não de memória) em https://github.com/ggerganov/whisper.cpp:
  - flags do CLI:        examples/cli/cli.cpp (parsing de argumentos)
  - schema do JSON:      função output_json() em examples/cli/cli.cpp
  - timestamp por palavra: técnica "-ml 1 -sow" documentada no próprio
    projeto em examples/generate-karaoke.sh
  - flag de build Vulkan: GGML_VULKAN em CMakeLists.txt
  - taxa de amostragem exigida: WHISPER_SAMPLE_RATE=16000 em include/whisper.h

O que NÃO foi possível testar (documentado com honestidade, não simulado):
rodar de fato a inferência numa RX 580 real — o ambiente onde este código
foi escrito não tem GPU nem acesso à Hugging Face (onde os modelos .bin
ficam hospedados) para baixar pesos reais. A integração foi validada lendo
o código-fonte da ferramenta, não executando-a ponta a ponta. Teste na sua
máquina e me avise se algum flag/formato tiver mudado numa versão mais nova.
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .transcriber import Transcript, Segment, Word
from .utils import run

# pausa mínima (segundos) entre o fim de uma palavra e o início da próxima
# para começar um novo "segmento" (frase) no objeto Transcript. Com
# "-ml 1 -sow", o whisper.cpp devolve cada PALAVRA como um item separado no
# JSON (é a técnica documentada pelo próprio projeto pra conseguir
# timestamp por palavra, já que ele não tem uma opção "word_timestamps=True"
# direta como o faster-whisper) — esses itens de 1 palavra são reagrupados
# aqui só para preencher Transcript.segments/full_text de forma razoável.
# O resto do pipeline (seleção de clipes, legendas) usa transcript.words,
# então esse agrupamento não afeta o resultado final.
_SEGMENT_GAP_S = 0.6


def _resample_to_16k_mono(src_wav: str, dst_wav: str):
    """whisper.cpp exige áudio PCM 16kHz mono. O áudio já extraído pelo
    resto do pipeline está em 44100Hz — reamostra numa cópia temporária
    (rápido: é só um resample de áudio, não decodifica vídeo)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src_wav),
           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-map_metadata", "-1",
           "-fflags", "+bitexact", "-flags:a", "+bitexact", str(dst_wav)]
    run(cmd)


def _group_into_segments(words: list) -> list:
    if not words:
        return []
    groups, current = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        gap = w.start - prev.end
        ends_sentence = prev.text.strip().endswith((".", "!", "?"))
        if gap > _SEGMENT_GAP_S or ends_sentence:
            groups.append(current)
            current = [w]
        else:
            current.append(w)
    groups.append(current)
    return [
        Segment(start=g[0].start, end=g[-1].end,
                text=" ".join(w.text for w in g), words=g)
        for g in groups
    ]


# melhores modelos primeiro: se algum destes estiver na MESMA pasta do
# modelo configurado, ele é usado no lugar (só baixar o arquivo -- ver
# README). O "small" erra muita palavra em português (pedido do usuário:
# "tá errando muito palavras"); o large-v3-turbo erra bem menos e, com só 4
# camadas no decodificador, roda em tempo razoável na RX 580.
_BETTER_MODELS = [
    "ggml-large-v3-turbo.bin", "ggml-large-v3-turbo-q8_0.bin", "ggml-large-v3-turbo-q5_0.bin",
    "ggml-large-v3.bin", "ggml-large-v3-q5_0.bin", "ggml-medium.bin", "ggml-medium-q5_0.bin",
]


def best_model(model_path: str) -> str:
    """O melhor modelo disponível na pasta do modelo configurado."""
    from . import config
    if not getattr(config, "WHISPERCPP_AUTO_BEST_MODEL", True):
        return model_path
    folder = Path(model_path).parent
    for name in _BETTER_MODELS:
        cand = folder / name
        if cand.exists():
            return str(cand)
    return model_path


def _dtw_preset(model_path: str) -> Optional[str]:
    """Nome do preset de alinhamento DTW do whisper.cpp pro modelo
    (ggml-large-v3-turbo-q5_0.bin -> large.v3.turbo)."""
    import re
    m = re.match(r"ggml-(tiny|base|small|medium|large-v1|large-v2|large-v3-turbo|large-v3)(\.en)?",
                 Path(model_path).name)
    if not m:
        return None
    return m.group(1).replace("-", ".") + (m.group(2) or "")


def missing_setup(model_path: str, bin_path: str = "whisper-cli") -> Optional[str]:
    """Retorna None se o binário e o modelo do whisper.cpp existem, ou uma
    descrição curta do que está faltando. Usado por transcriber.transcribe
    pra cair no faster-whisper em vez de travar quando o whisper.cpp não
    está instalado nesta máquina (ex.: config.py copiado de outro PC)."""
    if not (shutil.which(bin_path) or Path(bin_path).exists()):
        return f"binário '{bin_path}' não encontrado"
    if not Path(model_path).exists():
        return f"modelo '{model_path}' não encontrado"
    return None


def transcribe(audio_path: str, model_path: str, bin_path: str = "whisper-cli",
               language: str = "auto", threads: int = 0,
               use_gpu: bool = True, gpu_device: int = 0,
               extra_args: Optional[list] = None, prompt: Optional[str] = None,
               beam_size: int = 5, use_dtw: bool = True) -> Transcript:
    """Transcreve via whisper.cpp. `bin_path` pode ser só o nome do
    executável (se estiver no PATH) ou o caminho completo pro
    whisper-cli.exe. `model_path` precisa apontar pra um arquivo .bin no
    formato ggml (baixado manualmente — veja README.md).

    `prompt`: texto de contexto (título do vídeo, nome do canal) -- o Whisper
    acerta mais nomes próprios e gírias que aparecem nele.
    `use_dtw`: tempo de cada palavra pelo alinhamento DTW (bem mais preciso
    que o padrão do whisper.cpp, que deixava a legenda adiantada/atrasada)."""
    model_path = best_model(model_path)
    print(f"[2/6] Transcrevendo áudio (whisper.cpp — modelo '{Path(model_path).name}')...")

    resolved_bin = shutil.which(bin_path) or (bin_path if Path(bin_path).exists() else None)
    if not resolved_bin:
        raise FileNotFoundError(
            f"binário do whisper.cpp não encontrado ('{bin_path}'). Configure "
            f"WHISPERCPP_BIN em config.py com o caminho completo do "
            f"whisper-cli.exe (veja README.md, seção 'Transcrição via GPU/Vulkan')."
        )
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"modelo do whisper.cpp não encontrado em '{model_path}'. Baixe um "
            f"modelo .bin (ex.: ggml-small.bin) e configure WHISPERCPP_MODEL "
            f"em config.py (veja README.md)."
        )

    with tempfile.TemporaryDirectory(prefix="whispercpp_") as tmp:
        wav16k = str(Path(tmp) / "audio_16k.wav")
        _resample_to_16k_mono(audio_path, wav16k)

        out_prefix = str(Path(tmp) / "out")
        # threads: 0 (ou None) deve deixar o whisper.cpp decidir usando os
        # núcleos reais da CPU, não um valor fixo — "threads or 8" trocava
        # silenciosamente 0 por 8, contrariando o próprio comentário do
        # config.py.
        n_threads = threads if threads else (os.cpu_count() or 8)
        base = [
            resolved_bin, "-m", str(model_path), "-f", wav16k,
            "-l", language, "-t", str(n_threads),
            "-ml", "1", "-sow",         # 1 "segmento" por palavra -> timestamp por palavra
            "-ojf", "-of", out_prefix,  # JSON completo, com offsets em ms
        ]
        if not use_gpu:
            base.append("-ng")          # força CPU mesmo se o binário tiver Vulkan/CUDA
        if extra_args:
            base += extra_args
        quality = []
        if beam_size and beam_size > 1:
            quality += ["-bs", str(beam_size)]
        if prompt:
            quality += ["--prompt", prompt[:220]]
        dtw = _dtw_preset(model_path) if use_dtw else None
        # tentativas: um whisper-cli mais antigo pode não conhecer algum
        # parâmetro (sai com erro na hora) -- tira os opcionais aos poucos
        # em vez de falhar a transcrição. DTW não funciona junto com flash
        # attention, daí o -nfa.
        attempts = []
        if dtw:
            attempts += [base + quality + ["-dtw", dtw, "-nfa"], base + quality + ["-dtw", dtw]]
        attempts += [base + quality, base]

        # seleção de GPU no backend Vulkan do whisper.cpp é feita por
        # variável de ambiente, não por flag de linha de comando (o
        # whisper-cli não tem -dev/--device — confirmado contra o -h
        # oficial). GGML_VK_VISIBLE_DEVICES é a variável correta.
        env = os.environ.copy()
        if use_gpu and gpu_device:
            env["GGML_VK_VISIBLE_DEVICES"] = str(gpu_device)

        json_path = Path(out_prefix + ".json")
        for cmd in attempts:
            if json_path.exists():
                json_path.unlink()
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                  encoding="utf-8", errors="replace")
            if proc.returncode == 0 and json_path.exists():
                break
            if cmd is not attempts[-1]:
                print("    [aviso] whisper.cpp recusou parâmetros extras (versão antiga?) -- "
                      "tentando de novo com menos opções")
        if proc.returncode != 0 or not json_path.exists():
            raise RuntimeError(
                f"whisper.cpp falhou (código {proc.returncode}). Saída:\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}\n\n"
                f"Comando executado: {' '.join(cmd)}"
            )
        data = json.loads(json_path.read_text(encoding="utf-8"))

    words = []
    for item in data.get("transcription", []):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        off = item.get("offsets", {})
        start = off.get("from", 0) / 1000.0
        end = max(off.get("to", 0) / 1000.0, start + 0.05)
        # DTW: tempo do 1º token de texto da palavra (t_dtw em centésimos
        # de segundo; -1 = sem DTW)
        for tok in item.get("tokens") or []:
            ttext = (tok.get("text") or "").strip()
            if not ttext or ttext.startswith("[_") or ttext.startswith("<|"):
                continue
            t_dtw = tok.get("t_dtw", -1)
            if isinstance(t_dtw, (int, float)) and t_dtw >= 0:
                dstart = t_dtw / 100.0
                if abs(dstart - start) < 1.5:  # proteção contra valor absurdo
                    start = dstart
                    end = max(end, start + 0.05)
            break
        words.append(Word(start=start, end=end, text=text))
    # ordem e sem sobreposição (o DTW mexe só no começo de cada palavra)
    words.sort(key=lambda w: w.start)
    for a, b in zip(words, words[1:]):
        if a.end > b.start:
            a.end = max(b.start, a.start + 0.05)

    segments = _group_into_segments(words)
    detected_lang = data.get("result", {}).get("language", language)
    print(f"    -> Idioma detectado: {detected_lang} ({len(segments)} segmentos, "
          f"{len(words)} palavras)")
    return Transcript(language=detected_lang, segments=segments)
