"""
Modos de início: limite de uso da GPU (25%, 50%, 70% ou sem limite).

Pedido do usuário: "implemente modos de Start, um que usa só 25% da gpu,
outro que usa 50, outro que usa 70, e outro irrestrito" -- pra deixar o PC
usável (jogar, assistir, trabalhar) com o piloto rodando.

Quem usa a GPU aqui são programas separados que o Auto Clipper chama:
  - whisper.cpp (whisper-cli, Vulkan): a transcrição;
  - ffmpeg com encoder da placa (h264_amf na RX 580, nvenc, qsv): o vídeo final.
Placa de vídeo não tem um "use só X%" que funcione em qualquer marca, então
o limite é feito no TEMPO: a cada 1 segundo, esses programas trabalham a
fração pedida e ficam PAUSADOS o resto (25% = 0,25 s trabalhando, 0,75 s
parados). Na média a GPU fica nessa porcentagem do que usaria. Pausar não
estraga nada: o programa só continua de onde parou. O resto (baixar,
recortar, analisar rosto) é CPU e não é limitado.

O modo escolhido fica salvo em autopilot_data/modo_gpu.txt: quando o piloto
religa (ou depois de uma atualização) ele volta no mesmo modo. E vale AO
VIVO: a interface (interface.py) troca o modo com o piloto ligado e a
edição em andamento passa pro modo novo em ~1 s, sem parar nada.

Tempo: com 25% a transcrição e o encode levam até ~4x mais; os limites de
tempo (watchdog, tempo máximo por vídeo) crescem junto pra nada ser
cortado no meio por "demora".
"""
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Optional

ENV = "AUTOCLIPPER_GPU_LIMIT"
ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "autopilot_data" / "modo_gpu.txt"
MODES = (25, 50, 70, 100)
_PERIOD = 1.0  # segundos de cada ciclo trabalha/pausa

_GPU_ARGS = ("_amf", "_nvenc", "_qsv", "_vaapi", "_vulkan", "-hwaccel")
_state = {"limit": 100, "thread": None, "installed": False, "slowest": 1.0}
_procs: "weakref.WeakSet" = weakref.WeakSet()
_paused: set = set()
_lock = threading.Lock()


def _parse(v) -> Optional[int]:
    try:
        n = int(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return n if 5 <= n <= 100 else (100 if n > 100 else None)


def configured_limit() -> int:
    """Modo salvo (o último escolhido: interface, atalho ou --gpu) >
    variável de ambiente > config."""
    try:
        n = _parse(SAVED.read_text(encoding="utf-8"))
    except OSError:
        n = None
    if n is None:
        n = _parse(os.environ.get(ENV))
    if n is None:
        try:
            from . import config
            n = _parse(getattr(config, "GPU_LIMIT_PERCENT", 100))
        except Exception:
            n = None
    return n or 100


def save_limit(n: int) -> None:
    os.environ[ENV] = str(n)
    try:
        SAVED.parent.mkdir(parents=True, exist_ok=True)
        SAVED.write_text(f"{n}\n", encoding="utf-8")
    except OSError:
        pass


def describe(n: Optional[int] = None) -> str:
    n = configured_limit() if n is None else n
    return "GPU sem limite" if n >= 100 else f"GPU limitada a {n}%"


def time_factor(n: Optional[int] = None) -> float:
    """Quanto mais tempo as etapas na GPU podem levar neste modo."""
    n = configured_limit() if n is None else n
    return 1.0 if n >= 100 else min(100.0 / n, 4.0)


def slowest_factor() -> float:
    """O maior `time_factor` desde que este processo ligou (o modo pode ter
    sido trocado no meio): os tempos-limite usam este."""
    f = max(_state["slowest"], time_factor())
    _state["slowest"] = f
    return f


def _uses_gpu(args) -> bool:
    if isinstance(args, (str, bytes)):
        parts = [args.decode(errors="ignore") if isinstance(args, bytes) else args]
    else:
        parts = [os.fspath(a) if not isinstance(a, bytes) else a.decode(errors="ignore") for a in args]
    if not parts:
        return False
    prog = os.path.basename(parts[0]).lower()
    if "whisper" in prog:
        return True
    if "ffmpeg" in prog:
        return any(g in p for p in parts[1:] for g in _GPU_ARGS)
    return False


# --- pausar / continuar um processo -------------------------------------
if sys.platform == "win32":
    import ctypes

    _ntdll = ctypes.WinDLL("ntdll")
    _ntdll.NtSuspendProcess.argtypes = (ctypes.c_void_p,)
    _ntdll.NtResumeProcess.argtypes = (ctypes.c_void_p,)
    _ntdll.NtSuspendProcess.restype = _ntdll.NtResumeProcess.restype = ctypes.c_long

    def _pause(p) -> bool:
        return _ntdll.NtSuspendProcess(ctypes.c_void_p(int(p._handle))) == 0

    def _cont(p) -> bool:
        return _ntdll.NtResumeProcess(ctypes.c_void_p(int(p._handle))) == 0
else:
    def _pause(p) -> bool:
        os.kill(p.pid, signal.SIGSTOP)
        return True

    def _cont(p) -> bool:
        os.kill(p.pid, signal.SIGCONT)
        return True


def _resume(p) -> None:
    with _lock:
        if id(p) not in _paused:
            return
        _paused.discard(id(p))
    try:
        _cont(p)
    except Exception:
        pass


def _resume_all() -> None:
    for p in list(_procs):
        _resume(p)


def _alive():
    out = []
    for p in list(_procs):
        try:
            if p.returncode is None and p.poll() is None:
                out.append(p)
        except Exception:
            pass
    return out


def _loop() -> None:
    last_read = 0.0
    while True:
        if time.time() - last_read >= 1.0:
            # modo trocado na interface vale na hora
            last_read = time.time()
            _state["limit"] = configured_limit()
            _state["slowest"] = max(_state["slowest"], time_factor(_state["limit"]))
        lim = _state["limit"]
        on = _PERIOD * lim / 100.0
        procs = _alive()
        if not procs or lim >= 100:
            time.sleep(0.2)
            continue
        time.sleep(on)
        procs = _alive()
        with _lock:
            for p in procs:
                if id(p) in _paused:
                    continue
                try:
                    if _pause(p):
                        _paused.add(id(p))
                except Exception:
                    pass
        time.sleep(_PERIOD - on)
        for p in procs:
            _resume(p)


class _ThrottledPopen(subprocess.Popen):
    """Popen que coloca programas de GPU no ciclo trabalha/pausa."""

    def __init__(self, args, *a, **kw):
        super().__init__(args, *a, **kw)
        self._gpu_limited = False
        self._no_scale = False
        if _uses_gpu(args):
            # registra mesmo sem limite agora: o modo pode mudar no meio
            self._gpu_limited = True
            _procs.add(self)

    def _scale(self, timeout):
        if timeout is None or not getattr(self, "_gpu_limited", False) or self._no_scale:
            return timeout
        return timeout * time_factor(25)  # o pior caso: o modo pode baixar no meio

    def communicate(self, input=None, timeout=None):
        timeout = self._scale(timeout)
        self._no_scale = True
        try:
            return super().communicate(input, timeout)
        finally:
            self._no_scale = False

    def wait(self, timeout=None):
        return super().wait(self._scale(timeout))

    def send_signal(self, sig):
        _resume(self)  # processo pausado não morre direito no Linux
        return super().send_signal(sig)

    def terminate(self):
        _resume(self)
        return super().terminate()

    def kill(self):
        _resume(self)
        return super().kill()


def install(limit: Optional[int] = None, log=print) -> int:
    """Liga o controle de GPU neste processo (e nos que ele abrir). Fica
    ligado mesmo "sem limite": o modo pode ser trocado na interface com o
    programa rodando. Devolve o % atual."""
    n = configured_limit() if limit is None else (_parse(limit) or 100)
    os.environ[ENV] = str(n)  # o operário (main.py) herda
    _state["limit"] = n
    _state["slowest"] = max(_state["slowest"], time_factor(n))
    if _state["installed"]:
        return n
    subprocess.Popen = _ThrottledPopen
    _state["installed"] = True
    t = threading.Thread(target=_loop, name="limite-gpu", daemon=True)
    _state["thread"] = t
    t.start()
    atexit.register(_resume_all)
    if n <= 50:
        try:
            from . import config
            config.MAX_PARALLEL_CLIPS_HW = 1  # um clipe por vez no encoder da placa
        except Exception:
            pass
    return n
