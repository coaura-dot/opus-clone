"""
Orquestra a montagem final de cada clipe: reenquadramento vertical seguindo
o rosto, legendas estilo karaokê, música de fundo com ducking e efeitos de
zoom — tudo em um único passe de encode de vídeo, gerando um arquivo .mp4
pronto para publicar.
"""
from pathlib import Path

from .audio import audio_energy
from .effects import find_energy_peaks
from .reframer import render_vertical_clip
from .captioner import generate_ass
from .music import mix_with_music, pick_music_track, load_track_credit
from .post_kit import write_post_kit
from .react_detector import find_reference_times
from .utils import run, ensure_dir, sanitize_filename
from . import config


FONTS_DIR = str(Path(__file__).resolve().parent.parent / "assets" / "fonts")


def extract_audio(video_path: str, out_path: str):
    """Extrai o áudio completo de um vídeo (usado uma vez, para a
    transcrição do vídeo inteiro)."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
           "-vn", "-ac", "1", "-ar", "44100", str(out_path)]
    run(cmd)


def extract_audio_segment(source_path: str, start: float, end: float, out_path: str):
    """Extrai só o áudio de um trecho [start, end] direto do vídeo de
    origem. Não decodifica/recodifica vídeo — é uma operação leve, mesmo em
    fontes longas, porque o ffmpeg busca a posição no contêiner e descarta
    os pacotes de vídeo sem processá-los."""
    duration = max(end - start, 0.1)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-ss", str(max(start, 0.0)), "-i", str(source_path), "-t", str(duration),
           "-vn", "-ac", "1", "-ar", "44100", str(out_path)]
    run(cmd)


def _format_ts(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _write_context_dump(transcript_words, candidate, txt_path: str,
                         context_seconds: float = 60.0):
    """Salva um .txt ao lado do clipe com a transcrição de
    `context_seconds` ANTES + o trecho usado + `context_seconds` DEPOIS,
    pra conferir de relance (sem precisar assistir o vídeo de origem
    inteiro de novo) se o corte pegou o assunto certo desde o começo até
    o final, ou se cortou no meio de alguma parte -- pedido explícito do
    usuário: "recorte na transcrição o trecho utilizado, o texto de um
    minuto antes e um minuto depois, pra sabermos se deu certo a
    contextualização ou se ele cortou no meio".

    `transcript_words` precisa ter timestamps ABSOLUTOS no vídeo de
    origem (mesma convenção já usada em todo o resto do pipeline) -- tanto
    o caso normal (transcript.words do vídeo inteiro) quanto o de vídeo
    longo (candidate.words, documentado com a mesma convenção em
    clip_selector.ClipCandidate) já garantem isso."""
    before, used, after = [], [], []
    ctx_start = candidate.start - context_seconds
    ctx_end = candidate.end + context_seconds
    for w in transcript_words:
        if w.end < ctx_start or w.start > ctx_end:
            continue
        if w.start < candidate.start:
            before.append(w.text)
        elif w.start >= candidate.end:
            after.append(w.text)
        else:
            used.append(w.text)

    lines = [
        f"Clipe usado: {_format_ts(candidate.start)} -> {_format_ts(candidate.end)} "
        f"({candidate.duration:.1f}s)",
        f"Contexto exibido: {context_seconds:.0f}s antes e {context_seconds:.0f}s depois",
        "",
        f"--- {context_seconds:.0f}s ANTES do clipe "
        f"(a partir de {_format_ts(ctx_start)}) ---",
        " ".join(before) if before else "(sem transcrição disponível aqui -- "
                                          "provavelmente é o início do vídeo)",
        "",
        f">>> TRECHO USADO NO CLIPE ({_format_ts(candidate.start)} -> "
        f"{_format_ts(candidate.end)}) <<<",
        " ".join(used) if used else "(vazio -- não deveria acontecer)",
        "",
        f"--- {context_seconds:.0f}s DEPOIS do clipe "
        f"(até {_format_ts(ctx_end)}) ---",
        " ".join(after) if after else "(sem transcrição disponível aqui -- "
                                        "provavelmente é o fim do vídeo)",
    ]
    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")


def build_clip(source_path: str, candidate, clip_index: int, transcript_words,
                work_dir: str, output_dir: str,
                src_w: int = None, src_h: int = None, src_fps: float = None,
                source_title: str = None, source_url: str = None) -> str:
    work = ensure_dir(Path(work_dir) / f"clip_{clip_index}")
    ensure_dir(output_dir)

    if src_w is None or src_h is None or src_fps is None:
        from .utils import video_info
        info = video_info(source_path)
        src_w, src_h, src_fps = info["width"], info["height"], info["fps"]

    print(f"[4/6] Clip {clip_index}: extraindo áudio do trecho "
          f"({candidate.start:.1f}s - {candidate.end:.1f}s)...")
    voice_audio = work / "voice.wav"
    extract_audio_segment(source_path, candidate.start, candidate.end, str(voice_audio))

    energies = audio_energy(str(voice_audio), candidate.duration)
    peaks_local = find_energy_peaks(energies, hop=0.5)  # já relativo ao início do clipe

    # curva de energia em resolução mais fina, só para o reenquadramento
    # confirmar qual rosto está falando de verdade (ver reframer.py) — o
    # hop de 0.5s usado acima para os zoom punches é grosso demais pra essa
    # finalidade.
    face_gate_hop = getattr(config, "FACE_AUDIO_GATE_HOP", 0.1)
    face_gate_energies = audio_energy(str(voice_audio), candidate.duration,
                                       hop=face_gate_hop)

    print(f"[5/6] Clip {clip_index}: gerando legendas e mixando música...")
    ass_path = work / "captions.ass"
    generate_ass(transcript_words, clip_offset=candidate.start, output_path=str(ass_path),
                 clip_duration=candidate.duration)

    mixed_audio = work / "mixed.wav"
    music_track = pick_music_track()
    mix_with_music(str(voice_audio), candidate.duration, str(mixed_audio), track=music_track)

    # Modo REACT (RELATORIO item 14): frases tipo "olha a camisa dele" no
    # trecho deste clipe forçam um zoom breve na tela reagida quando
    # renderizado — sem efeito nenhum se este vídeo não tiver nenhuma tela
    # detectável (screen_tracker fica None em render_vertical_clip).
    reference_times = find_reference_times(transcript_words, candidate.start, candidate.end)

    filename = f"clip_{clip_index:02d}_{sanitize_filename(candidate.title)}.mp4"
    final_path = Path(output_dir) / filename
    context_txt_path = final_path.with_suffix(".contexto.txt")
    _write_context_dump(transcript_words, candidate, str(context_txt_path))
    post_txt_path = final_path.with_suffix(".post.txt")
    write_post_kit(str(post_txt_path), candidate.text, load_track_credit(music_track),
                   source_title, source_url)

    print(f"[6/6] Clip {clip_index}: reenquadrando + legendas + áudio "
          f"(passe único de encode)...")
    render_vertical_clip(
        source_path, candidate.start, candidate.end,
        src_w=src_w, src_h=src_h, fps=src_fps,
        output_path=str(final_path),
        ass_path=str(ass_path), audio_path=str(mixed_audio),
        fonts_dir=FONTS_DIR, zoom_peak_times=peaks_local,
        audio_energy=face_gate_energies, audio_energy_hop=face_gate_hop,
        reference_times=reference_times,
    )

    # render_vertical_clip só levanta exceção se o ffmpeg terminar com
    # código de erro -- mas um código de sucesso não garante por si só que
    # o arquivo final ficou de verdade no disco (relatado: "Concluído"
    # impresso, mas o .mp4 não estava em output\). Confirmação explícita
    # aqui, em vez de confiar só no returncode: se isso disparar, pelo
    # menos você vê um erro de verdade em vez de um "Concluído" mentiroso.
    if not final_path.exists():
        raise RuntimeError(
            f"O ffmpeg terminou sem erro, mas o arquivo final não foi "
            f"encontrado em {final_path} -- algo apagou/moveu ele logo em "
            f"seguida, ou o caminho de saída não bateu com o esperado."
        )
    size = final_path.stat().st_size
    if size < 10_000:  # um clipe vertical de poucos segundos já passa
                        # disso fácil -- abaixo desse tamanho é sinal de
                        # arquivo truncado/vazio, não um vídeo de verdade
        raise RuntimeError(
            f"O arquivo final ({final_path}) ficou suspeito demais pequeno "
            f"({size} bytes) -- o encode provavelmente falhou/foi "
            f"interrompido no meio, mesmo o ffmpeg tendo retornado sem erro."
        )

    print(f"    -> Concluído: {final_path}")
    print(f"    -> Contexto da transcrição salvo em: {context_txt_path}")
    print(f"    -> Título/descrição/hashtags pra postar: {post_txt_path}")
    return str(final_path)
