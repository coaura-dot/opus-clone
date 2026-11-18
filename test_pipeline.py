"""Teste de integração do pipeline SEM depender do YouTube/Whisper reais.
Gera um vídeo sintético localmente (não precisa de internet) e usa uma
transcrição falsa para validar de ponta a ponta:
clip_selector -> reframer (passe único: recorte+legenda+áudio) -> video_editor
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.transcriber import Transcript, Segment, Word
from src.clip_selector import select_clips
from src.video_editor import extract_audio, build_clip
from src.utils import video_info, ensure_dir, run

SOURCE = ".autoclip_work/synthetic_source.mp4"
WORK = ".autoclip_work/test"
OUT = ".autoclip_work/test_output"


def ensure_synthetic_source(path: str, duration: float = 20.0):
    """Gera um vídeo de teste (padrão de cores em movimento + tom de áudio)
    100% localmente via ffmpeg, sem precisar baixar nada. Recriado apenas se
    ainda não existir."""
    if Path(path).exists():
        return
    ensure_dir(Path(path).parent)
    print(f"Gerando vídeo sintético de teste ({duration:.0f}s)...")
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", "60",
        "-c:a", "aac", str(path),
    ]
    run(cmd)

FAKE_SENTENCES = [
    (0.5, 4.0, "você sabia que ninguém te conta esse segredo importante"),
    (4.5, 8.0, "esse é o maior erro que as pessoas cometem todos os dias"),
    (8.5, 12.5, "isso mudou completamente a forma como eu penso sobre dinheiro"),
    (13.0, 16.5, "aqui estão três dicas que realmente funcionam de verdade"),
    (17.0, 19.5, "e por isso você nunca deveria ignorar esse conselho"),
]


def build_fake_transcript() -> Transcript:
    segments = []
    for start, end, text in FAKE_SENTENCES:
        words_txt = text.split()
        step = (end - start) / len(words_txt)
        words = []
        for i, w in enumerate(words_txt):
            ws = start + i * step
            we = ws + step * 0.85
            suffix = "." if i == len(words_txt) - 1 else ""
            words.append(Word(start=ws, end=we, text=w + suffix))
        segments.append(Segment(start=start, end=end, text=text, words=words))
    return Transcript(language="pt", segments=segments)


def main():
    ensure_dir(WORK)
    ensure_dir(OUT)
    ensure_synthetic_source(SOURCE)

    info = video_info(SOURCE)
    print("Fonte:", info)

    full_audio = Path(WORK) / "full_audio.wav"
    extract_audio(SOURCE, str(full_audio))

    transcript = build_fake_transcript()

    # relaxa a duração mínima só para o teste (vídeo de 20s / clipes de 3-5s)
    from src import config
    config.MIN_CLIP_DURATION = 3
    config.MAX_CLIP_DURATION = 10
    config.IDEAL_CLIP_DURATION = 4
    config.MIN_GAP_BETWEEN_CLIPS = 1
    # TOPIC_BOUNDARY_GRACE_SECONDS (padrão 12s) precisa ser escalado junto
    # com MAX_CLIP_DURATION acima -- 12s de folga é proporcional para um
    # MAX_CLIP_DURATION real de 75s (~16%), mas num vídeo de teste de 20s
    # com MAX=10s isso permitiria uma janela "chegar até o fim do vídeo"
    # (que sempre conta como fim de assunto, ver clip_selector.py) e
    # engolir quase o vídeo inteiro num candidato só. Mantendo a mesma
    # proporção (~16% do MAX_CLIP_DURATION deste teste).
    config.TOPIC_BOUNDARY_GRACE_SECONDS = 1.5

    candidates = select_clips(transcript, str(full_audio), info["duration"], n_clips=2)
    assert len(candidates) >= 1, "Nenhum candidato selecionado!"

    for i, cand in enumerate(candidates, 1):
        final_path = build_clip(SOURCE, cand, i, transcript.words, WORK, OUT,
                                 info["width"], info["height"], info["fps"])
        out_info = video_info(final_path)
        print(f"Clip {i} ->", out_info)
        assert out_info["width"] == config.TARGET_WIDTH
        assert out_info["height"] == config.TARGET_HEIGHT
        assert out_info["duration"] > 0

    print("\nTESTE OK: pipeline completo rodou sem erros e gerou vídeos verticais válidos.")


if __name__ == "__main__":
    main()
