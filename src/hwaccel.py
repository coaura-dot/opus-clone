"""
Detecção automática de aceleração de hardware para encode de vídeo.

Suporta:
  - AMF (Windows, GPUs AMD — inclui a RX 580: usa o mesmo encoder de vídeo
    dedicado (VCE) que o VAAPI usa no Linux, só que pelo driver AMD/AMF
    nativo do Windows, sem precisar de nenhum device Linux)
  - VAAPI (Linux, GPUs AMD e Intel via driver mesa/radeonsi)
  - NVENC (GPUs NVIDIA, funciona em qualquer SO)
  - CPU / libx264 (fallback universal, sempre funciona)

A detecção é 100% automática: testa de verdade se o encoder de hardware
funciona (não só se "existe" no ffmpeg) fazendo um encode mínimo de
verificação, e cai para CPU sozinho se qualquer coisa falhar — sem exigir
nenhuma configuração manual do usuário. Se um encode de hardware falhar
durante o uso real (ex: driver com algum problema específico), o pipeline
também detecta isso e refaz aquele passo em CPU automaticamente.
"""
import os
import platform
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

from . import config


def _has_encoder(name: str) -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
        )
        return name.encode() in out.stdout
    except Exception:
        return False


def _vaapi_device_works(device: str) -> bool:
    """Faz um encode de verificação real (1 frame minúsculo) para confirmar
    que o driver VAAPI funciona de ponta a ponta neste sistema, e não
    apenas que o ffmpeg foi compilado com suporte a ele."""
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-vaapi_device", device,
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
            "-vf", "format=nv12,hwupload",
            "-c:v", "h264_vaapi", "-frames:v", "1",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _nvenc_works() -> bool:
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
            "-c:v", "h264_nvenc", "-frames:v", "1",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _amf_works() -> bool:
    """Faz um encode de verificação real (1 frame minúsculo) via AMF —
    o caminho de GPU da AMD no Windows (driver Adrenalin + runtime AMF).
    Não precisa de nenhum device explícito, ao contrário do VAAPI: o
    ffmpeg encontra a GPU AMD sozinho através do driver DirectX/AMF."""
    try:
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
            "-c:v", "h264_amf", "-frames:v", "1",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


@lru_cache(maxsize=1)
def detect_encoder() -> dict:
    """Detecta e testa de verdade o melhor encoder disponível. Resultado
    fica em cache (só testa uma vez por execução do programa)."""
    mode = getattr(config, "ENCODER_MODE", "auto")

    if mode == "cpu":
        return {"mode": "cpu"}

    if mode in ("auto", "nvenc") and _has_encoder("h264_nvenc") and _nvenc_works():
        return {"mode": "nvenc"}
    if mode == "nvenc":
        return {"mode": "cpu"}

    if mode in ("auto", "amf") and platform.system() == "Windows":
        if _has_encoder("h264_amf") and _amf_works():
            return {"mode": "amf"}
    if mode == "amf":
        return {"mode": "cpu"}

    if mode in ("auto", "vaapi") and platform.system() == "Linux":
        device = getattr(config, "VAAPI_DEVICE", "/dev/dri/renderD128")
        if Path(device).exists() and _has_encoder("h264_vaapi") and _vaapi_device_works(device):
            return {"mode": "vaapi", "device": device}
    if mode == "vaapi":
        return {"mode": "cpu"}

    return {"mode": "cpu"}


_force_cpu_lock = threading.Lock()
_force_cpu_after_failure = False

_parallel_workers_lock = threading.Lock()
_active_parallel_workers = 1


def set_parallel_workers(n: int):
    """Informa ao módulo quantos clipes serão processados simultaneamente
    nesta execução. Usado para dividir as threads de encode da CPU entre os
    workers e evitar oversubscription (ex: 4 clipes em paralelo, cada um
    tentando usar 14 threads, num Xeon de 28 threads — o SO passaria mais
    tempo trocando de contexto do que codificando)."""
    global _active_parallel_workers
    with _parallel_workers_lock:
        _active_parallel_workers = max(1, int(n))


def mark_hw_failed():
    """Chamado quando um encode de hardware falha em uso real (não no
    teste de verificação). A partir daí, todos os próximos encodes desta
    execução usam CPU, evitando ficar tentando e falhando repetidamente."""
    global _force_cpu_after_failure
    with _force_cpu_lock:
        _force_cpu_after_failure = True


def describe() -> str:
    enc = detect_encoder()
    names = {
        "amf": "GPU AMD via AMF (Windows — usa o encoder de vídeo dedicado da placa, ex: RX 580)",
        "vaapi": "GPU via VAAPI (AMD/Intel — usa o encoder de vídeo dedicado da placa)",
        "nvenc": "GPU NVIDIA (NVENC)",
        "cpu": "CPU (libx264)",
    }
    return names.get(enc["mode"], enc["mode"])


def _amf_quality_for_preset(preset: str) -> str:
    """Converte o preset no estilo libx264 (usado no resto do código) para
    o parâmetro -quality do AMF, que só aceita speed/balanced/quality."""
    fast = {"ultrafast", "superfast", "veryfast", "faster"}
    slow = {"slow", "slower", "veryslow"}
    if preset in fast:
        return "speed"
    if preset in slow:
        return "quality"
    return "balanced"


def encoder_args(extra_vf: str = None, crf: int = None, preset: str = None,
                  force_cpu: bool = False):
    """Monta os argumentos de ffmpeg (pré-input, filtro de vídeo, codec de
    saída) para o encoder ativo, encaixando corretamente um filtro de vídeo
    extra (ex: legendas .ass) tanto no caminho de hardware quanto no de CPU.

    `crf`/`preset` default para config.ENCODE_CRF/ENCODE_PRESET quando não
    informados — mantém um único lugar (config.py) pra ajustar qualidade x
    tamanho de arquivo, em vez de valores fixos espalhados pelo código.

    Retorna (pre_input_args: list, video_filter: str|None, codec_args: list).
    """
    if crf is None:
        crf = getattr(config, "ENCODE_CRF", 20)
    if preset is None:
        preset = getattr(config, "ENCODE_PRESET", "medium")

    use_cpu = force_cpu or _force_cpu_after_failure
    enc = {"mode": "cpu"} if use_cpu else detect_encoder()

    vf_parts = [extra_vf] if extra_vf else []
    maxrate = getattr(config, "ENCODE_MAXRATE_MBPS", None)
    rate_cap = []
    if maxrate:
        rate_cap = ["-maxrate", f"{maxrate}M", "-bufsize", f"{maxrate * 2}M"]

    if enc["mode"] == "vaapi":
        pre = ["-vaapi_device", enc["device"]]
        vf_parts.append("format=nv12,hwupload")
        vf = ",".join(vf_parts)
        # qp é o equivalente ao crf no encoder de hardware (menor = melhor qualidade)
        codec = ["-c:v", "h264_vaapi", "-qp", str(crf), *rate_cap]
        return pre, vf, codec

    if enc["mode"] == "nvenc":
        vf = ",".join(vf_parts) if vf_parts else None
        codec = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf), *rate_cap]
        return [], vf, codec

    if enc["mode"] == "amf":
        vf = ",".join(vf_parts) if vf_parts else None
        quality = _amf_quality_for_preset(preset)
        # rc=cqp (QP constante) é o equivalente do crf do libx264 no AMF:
        # qp_i/qp_p usam a mesma escala 0-51 (menor = melhor qualidade).
        codec = ["-c:v", "h264_amf", "-usage", "transcoding", "-quality", quality,
                 "-rc", "cqp", "-qp_i", str(crf), "-qp_p", str(crf), *rate_cap]
        return [], vf, codec

    vf = ",".join(vf_parts) if vf_parts else None
    n_threads = str(_default_cpu_threads())
    codec = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-threads", n_threads]
    return [], vf, codec


def is_hardware_active() -> bool:
    if _force_cpu_after_failure:
        return False
    return detect_encoder()["mode"] != "cpu"


def _default_cpu_threads() -> int:
    """Estima quantas threads de encode cada clipe deve usar. Parte dos
    núcleos físicos (assume SMT 2x em CPUs com mais de 4 threads lógicas,
    o que cobre a maioria dos processadores desktop/servidor modernos,
    incluindo Xeons como o E5-2680 v4, que tem 14 núcleos/28 threads) e
    depois divide pelo número de clipes rodando em paralelo nesta execução,
    para não deixar threads demais disputando os mesmos núcleos físicos."""
    n = os.cpu_count() or 4
    physical_guess = max(n // 2, 1) if n > 4 else n
    with _parallel_workers_lock:
        workers = _active_parallel_workers
    per_clip = max(physical_guess // workers, 2)
    return min(per_clip, 16)


def resolve_parallel_clips(n_clips: int) -> int:
    """Decide quantos clipes processar simultaneamente. Em CPUs com muitos
    núcleos (ex: Xeon 14c/28t) processar vários clipes ao mesmo tempo
    aproveita os núcleos que ficariam ociosos processando um de cada vez.
    Quando o encode é feito via GPU, limita a concorrência para não
    sobrecarregar o único chip de vídeo da placa."""
    setting = getattr(config, "PARALLEL_CLIPS", "auto")
    if isinstance(setting, int) and setting > 0:
        return max(1, min(setting, n_clips))

    cpu_n = os.cpu_count() or 4
    if is_hardware_active():
        cap = getattr(config, "MAX_PARALLEL_CLIPS_HW", 2)
    else:
        cap = getattr(config, "MAX_PARALLEL_CLIPS_CPU", 4)
        cap = min(cap, max(cpu_n // 2, 1))
    return max(1, min(n_clips, cap))
