"""
Download de vídeos do YouTube via yt-dlp.
"""
from pathlib import Path
from .utils import run, ensure_dir


def download_youtube_video(url: str, work_dir: str) -> Path:
    out_dir = ensure_dir(work_dir)
    out_template = str(out_dir / "source.%(ext)s")

    # A pasta de trabalho é reaproveitada entre execuções (não é criada uma
    # nova a cada run), e por padrão o yt-dlp NÃO baixa de novo um arquivo
    # já existente com o mesmo nome de destino — ele silenciosamente pula o
    # download e fica com o vídeo antigo, mesmo se a URL agora é outra. Sem
    # isso, rodar o programa duas vezes seguidas com vídeos diferentes podia
    # processar o vídeo ERRADO (o da execução anterior) sem avisar nada.
    # Apagando qualquer "source.*" restante de uma execução anterior antes
    # de baixar, garantimos que o vídeo processado é sempre o da URL atual.
    for stale in out_dir.glob("source.*"):
        stale.unlink()

    print(f"[1/6] Baixando vídeo: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", out_template,
        url,
    ]
    run(cmd)

    candidates = list(out_dir.glob("source.*"))
    mp4s = [c for c in candidates if c.suffix == ".mp4"]
    if not mp4s:
        raise RuntimeError("Download concluído, mas nenhum arquivo .mp4 foi encontrado.")
    video_path = mp4s[0]
    print(f"    -> Salvo em {video_path}")
    return video_path


def get_video_title(url: str) -> str:
    cmd = ["yt-dlp", "--get-title", "--no-playlist", url]
    out = run(cmd).stdout.decode(errors="ignore").strip()
    return out or "video"
