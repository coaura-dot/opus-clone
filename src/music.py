"""
Seleção e mixagem automática de música de fundo, com "ducking" (redução
automática do volume da música sempre que há fala) para o áudio ficar
com aparência profissional sem trabalho manual.

Faixas em assets/music/ só são usadas se estiverem listadas em
assets/music/track_drops.txt (formato "Título - M:SS" por linha, o mesmo
formato exportado por sites tipo ytmp3.gg) — o "M:SS" é o tempo do
"drop"/refrão de cada música, o trecho mais chamativo, que é onde a trilha
começa a tocar no clipe (em vez de começar pela introdução, que costuma ser
mais lenta/menos "viral"). Faixas sem entrada correspondente no manifesto,
ou que o ffprobe não reconhece como áudio de verdade (arquivo corrompido,
página de erro salva como .mp3 etc. — comum em downloads em lote de sites
de conversão), são ignoradas.
"""
import random
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .utils import run


@dataclass
class MusicTrack:
    path: Path
    drop_seconds: float  # onde começar a tocar (0.0 = do início, p/ trilha ambiente)


def _decode_mangled_unicode(s: str) -> str:
    """Alguns nomes de arquivo (dependendo do site/ferramenta de origem)
    vêm com acentos quebrados no formato literal '#U00e3' em vez do
    caractere real 'ã' — um escape que não foi decodificado direito na
    exportação. Decodifica esse padrão de volta pro caractere unicode
    real antes de normalizar (ex: 'Xonad#U00e3o' -> 'Xonadão')."""
    return re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)


def _normalize_title(s: str) -> str:
    """Normaliza um título pra comparação: minúsculas, sem acento, sem
    pontuação, espaços colapsados. Deixa 'Take Me To Church' e
    'take_me_to_church (Official Video)' comparáveis."""
    s = _decode_mangled_unicode(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[_\-.]+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_timestamp(s: str) -> Optional[float]:
    """Converte 'M:SS' ou 'H:MM:SS' em segundos. Retorna None se não bater
    com o formato esperado."""
    parts = s.strip().split(":")
    if not (2 <= len(parts) <= 3) or not all(p.isdigit() for p in parts):
        return None
    parts = [int(p) for p in parts]
    if len(parts) == 2:
        m, sec = parts
        h = 0
    else:
        h, m, sec = parts
    if sec >= 60 or m >= 60:
        return None
    return float(h * 3600 + m * 60 + sec)


def _load_drop_manifest() -> dict:
    """Lê assets/music/track_drops.txt e retorna {título_normalizado: segundos}.
    Linhas que não batem com 'Título - M:SS' são ignoradas silenciosamente
    (ex: linha em branco, cabeçalho)."""
    manifest_path = Path(config.MUSIC_DIR) / "track_drops.txt"
    manifest = {}
    if not manifest_path.exists():
        return manifest
    line_re = re.compile(r"^(.*\S)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$")
    for raw_line in manifest_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = line_re.match(raw_line.strip())
        if not m:
            continue
        title, ts = m.group(1), m.group(2)
        seconds = _parse_timestamp(ts)
        if seconds is None:
            continue
        manifest[_normalize_title(title)] = seconds
    return manifest


def _match_title(filename_stem: str, manifest: dict):
    """Tenta casar o nome do arquivo (sem extensão) com um título do
    manifesto (track_drops.txt ou track_credits.txt) e devolve o valor
    associado a ele (segundos do drop, ou a linha de crédito). Nomes de arquivo baixados de conversores costumam ter lixo
    extra — artista, '(Official Video)', qualidade, e principalmente
    features tipo 'ft. Fulano' ENCAIXADOS NO MEIO do título (ex: título
    'Rockabye (SHAKED Remix)' vira arquivo 'Rockabye ft. Sean Paul &
    Anne-Marie (SHAKED Remix)') — por isso o match não pode exigir o
    título inteiro contíguo. Em vez disso, verifica se as palavras do
    título aparecem, em ordem, dentro do nome do arquivo (podendo ter
    outras palavras entre elas) — uma 'subsequência', não uma substring
    exata. Quando mais de um título do manifesto bate, fica com o mais
    longo/específico (ex: prefere 'Don't Let Me Down (Illenium Remix)' a
    só 'Don't Let Me Down' quando o arquivo realmente é o remix)."""
    norm_name = _normalize_title(filename_stem)
    if not norm_name:
        return None
    if norm_name in manifest:
        return manifest[norm_name]

    name_tokens = norm_name.split()
    best_seconds, best_len = None, 0
    for title, seconds in manifest.items():
        if len(title) < 4:
            continue  # título curto demais pra um match por subsequência ser confiável
        title_tokens = title.split()
        idx = 0
        for tok in name_tokens:
            if idx < len(title_tokens) and tok == title_tokens[idx]:
                idx += 1
        if idx == len(title_tokens) and len(title) > best_len:
            best_seconds, best_len = seconds, len(title)
    return best_seconds


def _is_real_audio(path: Path) -> bool:
    """Confirma via ffprobe que o arquivo é áudio de verdade (tem stream de
    áudio e duração > 0) — filtra downloads corrompidos/incompletos ou
    páginas de erro salvas com extensão .mp3, comuns em lote vindo de sites
    de conversão."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=15,
        )
        text = out.stdout.decode(errors="ignore")
        if "codec_type=audio" not in text:
            return False
        dm = re.search(r"duration=([\d.]+)", text)
        return bool(dm and float(dm.group(1)) > 1.0)
    except Exception:
        return False


def list_usable_tracks() -> list:
    """Varre assets/music/, valida cada arquivo (áudio de verdade + tempo de
    drop listado no manifesto) e retorna a lista de MusicTrack utilizáveis.
    Arquivos que falham em qualquer uma das duas checagens são ignorados —
    por pedido explícito: só entra faixa real com drop conhecido."""
    music_dir = Path(config.MUSIC_DIR)
    if not music_dir.exists():
        return []
    candidates = (list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
                  + list(music_dir.glob("*.m4a")))
    manifest = _load_drop_manifest()
    tracks = []
    for path in candidates:
        drop = _match_title(path.stem, manifest)
        if drop is None:
            continue  # sem tempo de drop listado -> não entra
        if not _is_real_audio(path):
            continue  # não é áudio de verdade -> não entra
        tracks.append(MusicTrack(path=path, drop_seconds=drop))
    return tracks


def pick_music_track() -> Optional[MusicTrack]:
    """Escolhe aleatoriamente uma trilha utilizável de assets/music/, se
    houver alguma."""
    tracks = list_usable_tracks()
    return random.choice(tracks) if tracks else None


def _generate_ambient_bed(duration: float, output_path: str):
    """Fallback: sintetiza uma trilha ambiente simples (acordes suaves) caso
    nenhuma faixa em assets/music/ passe nas checagens acima (pasta vazia,
    nenhum arquivo com drop listado no manifesto, ou arquivos inválidos).
    Não depende de internet nem de arquivos externos.

    OBS: aqui a trilha só é normalizada (sem pré-atenuação agressiva) — o
    volume relativo final é decidido em UM único lugar, por MUSIC_VOLUME_DB
    em mix_with_music(). Antes esta função já aplicava volume=0.15 e depois
    mix_with_music() aplicava outros -22dB (~0.08x) em cima disso, deixando
    a trilha praticamente inaudível (~0.012x) mesmo antes do ducking."""
    duration = max(duration, 0.5)
    fadeout_start = max(duration - 2, 0)
    filter_complex = (
        f"[0:a][1:a][2:a]amix=inputs=3:duration=longest,"
        f"afade=t=in:d=2,afade=t=out:st={fadeout_start}:d=2[out]"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency=110:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=164.81:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-filter_complex", filter_complex,
        "-map", "[out]", "-ar", "44100", str(output_path),
    ]
    run(cmd)


def _db_to_factor(db: float) -> float:
    return 10 ** (db / 20)


def _integrated_lufs(path: str) -> Optional[float]:
    """Loudness integrada (LUFS, EBU R128) de um arquivo de áudio, ou None
    se não der pra medir (arquivo mudo, ffmpeg sem o filtro etc.)."""
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                           "-af", "ebur128", "-f", "null", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    matches = re.findall(r"^\s+I:\s+(-?[\d.]+) LUFS", proc.stderr.decode(errors="ignore"), re.M)
    if not matches:
        return None
    value = float(matches[-1])
    return value if value > -70 else None


def _music_gain_db(voice_audio: str, music_path: str) -> float:
    """Ganho a aplicar na trilha pra ela ficar MUSIC_BELOW_VOICE_DB abaixo
    da voz DESTE clipe (antes do ducking). Um valor fixo em dB dependia de
    quão alto o podcast foi gravado: medido em clipes reais do Flow, a voz
    vinha a -28 LUFS e a música ficava só ~9 dB abaixo dela — alta demais,
    ainda mais com phonk (muito grave)."""
    below = getattr(config, "MUSIC_BELOW_VOICE_DB", None)
    if below is not None:
        voice_i = _integrated_lufs(voice_audio)
        music_i = _integrated_lufs(music_path)
        if voice_i is not None and music_i is not None:
            return float(np.clip((voice_i - below) - music_i, -60.0, 6.0))
    return config.MUSIC_VOLUME_DB


def load_track_credit(track: Optional[MusicTrack]) -> Optional[str]:
    """Linha de crédito da trilha, lida de assets/music/track_credits.txt
    (formato "Título | crédito" por linha, casado com o nome do arquivo do
    mesmo jeito que track_drops.txt). Trilhas com licença de atribuição (ex.:
    CC BY) PRECISAM desse crédito na descrição do post; None = trilha sem
    crédito cadastrado (ou nenhuma trilha, só o fundo sintetizado)."""
    if track is None:
        return None
    credits_path = Path(config.MUSIC_DIR) / "track_credits.txt"
    if not credits_path.exists():
        return None
    credits = {}
    for raw_line in credits_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        title, sep, credit = raw_line.partition("|")
        if sep and title.strip() and credit.strip():
            credits[_normalize_title(title)] = credit.strip()
    return _match_title(track.path.stem, credits)


_AUTO = object()


def mix_with_music(voice_audio: str, duration: float, output_path: str,
                   track=_AUTO) -> str:
    """Mixa o áudio de voz original com uma trilha de fundo (do usuário ou
    gerada automaticamente), aplicando ducking para a música abaixar
    automaticamente sempre que há fala. `track` permite a quem chama
    escolher a trilha antes (pra saber qual foi usada e creditar no post);
    sem ele, sorteia aqui mesmo. None = fundo sintetizado."""
    if track is _AUTO:
        track = pick_music_track()
    work_dir = Path(output_path).parent
    music_path = work_dir / "_music_bed.wav"

    if track is not None:
        # -ar/-ac explícitos: sem isso, a trilha do usuário mantém a taxa de
        # amostragem original do arquivo (ex: 48kHz, ou até 192kHz em
        # algumas trilhas royalty-free) enquanto o áudio de voz sai sempre
        # em 44100Hz mono — o amix/sidechaincompress então tinha que
        # reamostrar silenciosamente nos bastidores, o que já é desperdício
        # de CPU e, com alguns builds de ffmpeg, gerava artefatos.
        #
        # -ss antes de -i faz o ffmpeg começar a ler a trilha já a partir do
        # "drop" (trecho mais chamativo) em vez do início/introdução —
        # assim a música já entra animada desde o primeiro segundo do
        # clipe, em vez de começar por um trecho arrastado. -stream_loop -1
        # garante que, se o clipe for mais longo que o resto da música
        # depois do drop, ela recomeça do início do arquivo em vez de
        # cortar/silenciar.
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-stream_loop", "-1", "-ss", str(max(track.drop_seconds, 0.0)),
            "-i", str(track.path),
            "-t", str(duration), "-ar", "44100", "-ac", "1", str(music_path),
        ]
        run(cmd)
    else:
        _generate_ambient_bed(duration, str(music_path))

    music_vol = _db_to_factor(_music_gain_db(voice_audio, str(music_path)))
    ratio = getattr(config, "MUSIC_DUCKING_RATIO", 4)
    threshold = getattr(config, "MUSIC_DUCKING_THRESHOLD", 0.08)
    # loudnorm normaliza o volume final para -14 LUFS (padrão usado por
    # TikTok/YouTube/Instagram), deixando o áudio com volume consistente e
    # "profissional" entre clipes diferentes, em vez de depender de quão
    # alto/baixo a fala original ou a trilha escolhida estavam.
    loudnorm = f"loudnorm=I={config.LOUDNESS_TARGET_LUFS}:TP=-1.5:LRA=11"
    if config.MUSIC_DUCKING:
        # ratio/threshold mais suaves que antes (era ratio=8, threshold=0.05):
        # aquilo abaixava a música quase a zero durante QUALQUER fala, e como
        # um clipe falado quase não tem silêncio, a música nunca voltava a
        # ficar audível — na prática sumia do clipe inteiro.
        filter_complex = (
            f"[1:a]volume={music_vol}[music];"
            f"[music][0:a]sidechaincompress=threshold={threshold}:ratio={ratio}:"
            f"attack=15:release=400:makeup=1[ducked];"
            f"[0:a][ducked]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1,"
            f"volume=2,{loudnorm}[out]"
        )
    else:
        filter_complex = (
            f"[1:a]volume={music_vol}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1,"
            f"volume=2,{loudnorm}[out]"
        )

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(voice_audio), "-i", str(music_path),
        "-filter_complex", filter_complex,
        "-map", "[out]", "-ar", "44100", str(output_path),
    ]
    run(cmd)
    return output_path
