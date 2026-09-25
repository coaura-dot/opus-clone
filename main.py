#!/usr/bin/env python3
"""
Auto Clipper — gerador automático de cortes virais (estilo Opus Clip)
======================================================================
Baixa um vídeo do YouTube, transcreve, seleciona automaticamente os
melhores trechos, reenquadra para vertical seguindo o rosto do orador,
adiciona legendas estilo karaokê, música de fundo com ducking e
pequenos efeitos de zoom — tudo automaticamente.

Uso:
    python main.py
    python main.py --url URL --clips N
"""
import argparse
import ctypes
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import config
from src import hwaccel
from src import long_video
from src import react_detector
from src.utils import ensure_ffmpeg, ensure_dir, check_dependency, video_info
from src.downloader import download_youtube_video, get_video_title
from src.transcriber import transcribe
from src.clip_selector import select_clips
from src.video_editor import extract_audio, build_clip


def _force_exit(code: int):
    """Encerra o processo IMEDIATAMENTE — usado tanto no fim normal do
    programa quanto pelo watchdog (ver _start_watchdog) se algo travar no
    meio do processamento.

    HISTÓRICO (pra quem ler isso depois): a v1 dessa função usava só
    os._exit(code) — resolvia em teoria (mata o processo sem esperar o
    shutdown "educado" do interpretador), mas foi reportado continuar
    travando mesmo assim. Isso aponta pra uma causa mais funda que
    os._exit() não cobre: no Windows, tanto o encerramento normal quanto
    os._exit() (que por baixo dos panos chama a mesma ExitProcess do
    Windows) ainda esperam o sistema notificar TODAS as DLLs carregadas no
    processo (DLL_PROCESS_DETACH) antes de morrer de verdade — e se
    alguma DLL nativa carregada (driver de GPU é o suspeito clássico;
    aqui você tá usando AMD AMF) tiver uma rotina de finalização travada/
    em deadlock, o processo INTEIRO fica preso exatamente nesse ponto,
    depois de já ter impresso tudo, sem responder a Ctrl+C (é uma trava
    dentro do carregador de DLL do próprio Windows, não em código Python
    que o interpretador consiga interromper).

    CORREÇÃO (v2): TerminateProcess (API do Windows, chamada direto via
    ctypes, não a função _exit()/os._exit() do C) mata o processo no
    nível do sistema operacional SEM notificar nenhuma DLL — é o mesmo
    mecanismo que o Gerenciador de Tarefas usa pra "Finalizar tarefa".
    Não tem como ficar mais forçado que isso sem ser de fora do processo.

    HONESTIDADE: continuo sem conseguir reproduzir esse travamento aqui
    (sem Windows real, sem AMD AMF, sem faster-whisper/whisper.cpp
    instalados neste ambiente) — é a correção mais forte disponível pra
    esse tipo de sintoma (processo Windows preso na hora de sair), não
    uma confirmação de que era exatamente isso. Se ainda travar depois
    dessa versão, o travamento não é mais no ENCERRAMENTO do processo (já
    que isso mata até isso) — seria durante o PROCESSAMENTO em si (ver
    _start_watchdog abaixo, que cobre esse caso também).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
            # TerminateProcess mata o processo antes desta linha rodar --
            # esse "return" só existe pra deixar claro que a função não
            # continua (e como fallback se, por algum motivo, a chamada
            # acima falhar silenciosamente em vez de matar o processo).
            return
        except Exception:
            pass  # ctypes indisponível/falhou por algum motivo -- cai pro os._exit abaixo
    os._exit(code)


def _start_watchdog(timeout_seconds: float):
    """Rede de segurança contra QUALQUER travamento (não só na saída) —
    inclusive um processo ffmpeg/whisper.cpp que nunca retorna no meio do
    processamento. Se o programa inteiro não terminar sozinho dentro de
    `timeout_seconds`, essa thread força o encerramento com _force_exit().

    Roda numa thread daemon=True (não segura a saída sozinha; se o
    programa terminar normalmente antes do timeout, essa thread morre
    junto com o processo sem nunca disparar). timeout_seconds vem de
    config.WATCHDOG_TIMEOUT_SECONDS — ajuste lá se o seu conteúdo real
    (vídeos muito longos, muitos clipes) legitimamente demorar mais que
    o padrão."""
    def _watchdog():
        time.sleep(timeout_seconds)
        print(f"\n[watchdog] o programa passou de {timeout_seconds/60:.0f} min sem terminar "
              "sozinho -- isso não é normal, forçando encerramento agora (provável softlock; "
              "ver RELATORIO_PROXIMOS_PASSOS.txt, item 11). Se o seu vídeo/quantidade de clipes "
              "legitimamente precisa de mais tempo que isso, suba "
              "config.WATCHDOG_TIMEOUT_SECONDS.")
        _force_exit(1)
    threading.Thread(target=_watchdog, daemon=True).start()


def ask_int(prompt: str, default: int, min_v: int = 1, max_v: int = 20) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and min_v <= int(raw) <= max_v:
            return int(raw)
        print(f"   Digite um número entre {min_v} e {max_v}.")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--url", default=None)
    parser.add_argument("--clips", type=int, default=None)
    cli_args, _ = parser.parse_known_args()

    print("=" * 62)
    print("  AUTO CLIPPER — Cortes virais automáticos (estilo Opus Clip)")
    print("=" * 62)

    _start_watchdog(getattr(config, "WATCHDOG_TIMEOUT_SECONDS", 90 * 60))

    ensure_ffmpeg()
    if not check_dependency("yt-dlp"):
        print("[ERRO] yt-dlp não encontrado. Instale com: pip install yt-dlp")
        sys.exit(1)

    print(f"  Encode de vídeo: {hwaccel.describe()}")

    if cli_args.url:
        url = cli_args.url.strip()
        print(f"\nCole o link do vídeo do YouTube: {url}")
    else:
        url = input("\nCole o link do vídeo do YouTube: ").strip()
    if not url:
        print("Nenhum link informado. Encerrando.")
        sys.exit(1)

    if cli_args.clips is not None:
        n_clips = cli_args.clips
        print(f"Quantos clipes você quer gerar? [3]: {n_clips}")
    else:
        n_clips = ask_int("Quantos clipes você quer gerar?", default=3, min_v=1, max_v=15)

    work_dir = ensure_dir(config.WORK_DIR)
    output_dir = ensure_dir(config.OUTPUT_DIR)

    t0 = time.time()
    try:
        source_path = download_youtube_video(url, str(work_dir))
        # título do vídeo original, só pro crédito no .post.txt de cada
        # clipe -- se falhar (rede, vídeo privado...), segue sem ele
        try:
            source_title = get_video_title(url)
        except Exception:
            source_title = None
        info = video_info(source_path)
        print(f"    Duração: {info['duration']/60:.1f} min | "
              f"{info['width']}x{info['height']} | {info['fps']:.1f}fps")

        # Modo REACT (RELATORIO item 14): classifica o vídeo INTEIRO antes
        # de gerar qualquer clipe -- se for detectado como reação a
        # conteúdo de tela, liga o rastreamento de tela (item 3a, que por
        # padrão fica desligado) dinamicamente só pra este vídeo. NÃO
        # VALIDADO contra vídeo de reação real (ver src/react_detector.py e
        # RELATORIO_PROXIMOS_PASSOS.txt).
        if getattr(config, "REACT_MODE_AUTO_DETECT", True):
            print("    Analisando se o vídeo é do tipo 'reaction'...")
            react_report = react_detector.detect_react_video(str(source_path), info["duration"])
            if react_report.is_react:
                config.VIDEO_IN_VIDEO_ENABLED = True
                print(f"    -> Detectado como REACT (tela plausível em "
                      f"{react_report.screen_fraction*100:.0f}% das checagens) -- "
                      f"câmera vai seguir o vídeo reagido quando ele estiver tocando.")
            else:
                print(f"    -> Vídeo comum (tela plausível em só "
                      f"{react_report.screen_fraction*100:.0f}% das checagens).")

        if long_video.is_long_video(info["duration"]):
            # Vídeo longo (> LONG_VIDEO_THRESHOLD_SECONDS): não transcreve o
            # vídeo inteiro de uma vez — transcreve por blocos aleatórios,
            # montando o clipe de cada bloco bom assim que ele fica pronto,
            # enquanto o bloco seguinte já é transcrito em segundo plano.
            # Ver src/long_video.py para os detalhes do pipeline.
            print(f"\n    Vídeo longo detectado ({info['duration']/60:.1f} min > "
                  f"{config.LONG_VIDEO_THRESHOLD_SECONDS/60:.0f} min) — transcrevendo por "
                  f"blocos de {config.CHUNK_DURATION_SECONDS/60:.0f} min em vez do vídeo "
                  "inteiro de uma vez.\n")
            chunk_work_dir = ensure_dir(work_dir / "chunks")
            results = []
            clip_index = 0
            # Construído sequencialmente (um clipe de cada vez) por
            # simplicidade/robustez: o ganho de eficiência pedido já vem do
            # bloco seguinte sendo transcrito em segundo plano ENQUANTO este
            # clipe é montado (ver long_video.iter_long_video_clip_batches);
            # paralelizar também a montagem dos clipes entre si (como no
            # ramo de vídeo curto abaixo, via hwaccel) é possível de somar
            # depois, mas não é necessário para o pedido original.
            for batch in long_video.iter_long_video_clip_batches(
                    str(source_path), info["duration"], n_clips, chunk_work_dir):
                for cand in batch:
                    clip_index += 1
                    final_path = build_clip(
                        str(source_path), cand, clip_index, cand.words,
                        str(work_dir), str(output_dir),
                        info["width"], info["height"], info["fps"],
                        source_title=source_title, source_url=url,
                    )
                    results.append((final_path, cand))
        else:
            full_audio = work_dir / "full_audio.wav"
            extract_audio(str(source_path), str(full_audio))

            transcript = transcribe(
                str(full_audio),
                model_size=config.WHISPER_MODEL_SIZE,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE_TYPE,
            )

            candidates = select_clips(transcript, str(full_audio), info["duration"], n_clips)

            n_workers = hwaccel.resolve_parallel_clips(len(candidates))
            hwaccel.set_parallel_workers(n_workers)
            results = [None] * len(candidates)

            if n_workers > 1:
                print(f"\nProcessando {len(candidates)} clipe(s) — até {n_workers} em paralelo "
                      f"(aproveitando os núcleos da CPU)...\n")
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {
                        pool.submit(
                            build_clip, str(source_path), cand, i + 1,
                            transcript.words, str(work_dir), str(output_dir),
                            info["width"], info["height"], info["fps"],
                            source_title=source_title, source_url=url,
                        ): i
                        for i, cand in enumerate(candidates)
                    }
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        results[idx] = (fut.result(), candidates[idx])
            else:
                for i, cand in enumerate(candidates, 1):
                    final_path = build_clip(
                        str(source_path), cand, i, transcript.words, str(work_dir), str(output_dir),
                        info["width"], info["height"], info["fps"],
                        source_title=source_title, source_url=url,
                    )
                    results[i - 1] = (final_path, cand)

        elapsed = time.time() - t0
        # watchdog CURTO (independente do de _start_watchdog(90min) lá em
        # cima, que cobre travas durante o PROCESSAMENTO) — o usuário
        # relatou o processo ficando preso bem AQUI, depois de já ter
        # impresso tudo, sem responder Ctrl+C/Enter, mesmo já existindo o
        # _force_exit(0) logo abaixo com TerminateProcess. Não consigo
        # confirmar a causa raiz exata sem reproduzir num Windows real com
        # essa GPU/driver — mas essa rede de segurança curta (poucos
        # segundos, não os 90min do watchdog de processamento) garante que,
        # não importa o que trave depois deste ponto (o próprio
        # TerminateProcess incluso), o processo morre logo em seguida em
        # vez de ficar preso indefinidamente.
        _start_watchdog(getattr(config, "EXIT_WATCHDOG_SECONDS", 6))
        total_clip_seconds = sum(cand.duration for _, cand in results)
        speed = total_clip_seconds / elapsed if elapsed > 0 else 0.0
        print("\n" + "=" * 62)
        print(f"  CONCLUÍDO em {elapsed/60:.1f} min — {len(results)} clipe(s) gerado(s)")
        print(f"  Velocidade média: {speed:.1f}x tempo real "
              f"({total_clip_seconds:.0f}s de clipes em {elapsed:.0f}s de processamento)")
        print("=" * 62)
        for path, cand in results:
            print(f"  • {Path(path).name}")
            print(f"      {cand.start:.1f}s -> {cand.end:.1f}s  |  score {cand.score:.1f}")
            print(f"      \"{cand.title}\"")
        print(f"\nArquivos salvos em: {output_dir.resolve()}")

    except Exception as e:
        print(f"\n[ERRO] {e}")
        traceback.print_exc()
        _force_exit(1)
    else:
        _force_exit(0)


if __name__ == "__main__":
    main()
