import platform
import subprocess
import shutil
import sys
import re
import json
from pathlib import Path


def check_dependency(binary: str) -> bool:
    return shutil.which(binary) is not None


def ensure_ffmpeg():
    if not check_dependency("ffmpeg") or not check_dependency("ffprobe"):
        print("[ERRO] ffmpeg/ffprobe não encontrados no PATH.")
        if platform.system() == "Windows":
            print("Instale com um dos comandos abaixo (PowerShell) e abra um terminal novo depois:")
            print("  winget install Gyan.FFmpeg")
            print("  ou: choco install ffmpeg   (se você usa Chocolatey)")
            print("Alternativa manual: baixe o build 'full' ou 'essentials' em")
            print("  https://www.gyan.dev/ffmpeg/builds/  e adicione a pasta bin\\ ao PATH do sistema")
            print("(os builds do gyan.dev já vêm com suporte a h264_amf, usado pela sua RX 580)")
        elif platform.system() == "Darwin":
            print("Instale com: brew install ffmpeg")
        else:
            print("Instale com: sudo apt install ffmpeg   (Debian/Ubuntu)")
        sys.exit(1)


def run(cmd, **kwargs):
    """Executa um comando de shell e lança exceção com stderr legível em caso de erro."""
    result = subprocess.run(
        [str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Comando falhou ({result.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"{result.stderr.decode(errors='ignore')[-4000:]}"
        )
    return result


def probe(path):
    """Retorna metadados (duração, resolução, fps) via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = run(cmd).stdout
    return json.loads(out)


def video_info(path):
    data = probe(path)
    v = next(s for s in data["streams"] if s["codec_type"] == "video")
    duration = float(data["format"].get("duration") or v.get("duration") or 0)
    fps_raw = v.get("avg_frame_rate", "30/1")
    num, den = fps_raw.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    return {
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "duration": duration,
    }


def sanitize_filename(name: str, max_len: int = 60) -> str:
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] or "clip"


def fmt_time(seconds: float) -> str:
    """Converte segundos para HH:MM:SS.cc (formato de timestamp ASS)."""
    seconds = max(seconds, 0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
