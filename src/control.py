"""
Controle do piloto automático por fora (interface.py, atalhos .bat).

Pedido do usuário: "crie uma interface pela qual posso usar o programa, e
ligar e desligar por botões de acordo com o modo que eu quiser".

Como a interface fala com o piloto (que roda em outro processo):
  - autopilot_data/piloto_ligado.json: o piloto escreve a cada 5 s ("estou
    vivo", pid, modo da GPU). Arquivo velho (> 20 s) = piloto desligado ou
    caiu. Também impede dois pilotos ao mesmo tempo (interface + .bat):
    os dois mexeriam na mesma fila.
  - autopilot_data/parar.pedido: a interface cria; o piloto vê em ~1 s e
    para como no Ctrl+C (mata a edição em andamento, salva tudo e sai). O
    vídeo que estava sendo editado volta pra lista e é feito de novo depois.
  - autopilot_data/modo_gpu.txt (src/throttle.py): trocar o modo com o
    piloto ligado vale na hora, sem parar nada.
"""
import _thread
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "autopilot_data"
HEARTBEAT = DATA_DIR / "piloto_ligado.json"
STOP = DATA_DIR / "parar.pedido"
_BEAT_EVERY = 5.0
_FRESH = 20.0

_state = {"thread": None, "stop": threading.Event(), "started": 0.0, "stopping": False}


def _pid_alive(pid: int) -> bool:
    """O processo ainda existe? (piloto que caiu de repente não apaga o
    arquivo de "estou vivo": sem esta conferência a interface achava que
    ele estava ligado por até 20 s e demorava a religar)"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        k32.CloseHandle.argtypes = (wintypes.HANDLE,)
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # Linux: processo que já terminou mas ainda não foi "recolhido"
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return True


def running_info() -> Optional[dict]:
    """Dados do piloto ligado (pid, modo, desde quando) ou None."""
    try:
        info = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if time.time() - float(info.get("beat", 0)) > _FRESH:
        return None
    try:
        if not _pid_alive(int(info.get("pid", -1))):
            return None
    except Exception:
        pass
    return info


def request_stop() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STOP.write_text(f"{time.time():.0f}\n", encoding="utf-8")


def clear_stop() -> None:
    try:
        STOP.unlink()
    except OSError:
        pass


def stop_requested() -> bool:
    return STOP.exists()


def _write_beat(mode: str) -> None:
    from .throttle import configured_limit
    info = {"pid": os.getpid(), "started": _state["started"], "beat": time.time(),
            "gpu": configured_limit(), "mode": mode, "stopping": _state["stopping"]}
    tmp = HEARTBEAT.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(info), encoding="utf-8")
        os.replace(tmp, HEARTBEAT)
    except OSError:
        pass


def _watch(mode: str) -> None:
    last_beat = 0.0
    last_int = 0.0
    while not _state["stop"].is_set():
        now = time.time()
        if now - last_beat >= _BEAT_EVERY:
            _write_beat(mode)
            last_beat = now
        if STOP.exists():
            if not _state["stopping"]:
                _state["stopping"] = True
                _write_beat(mode)
            # igual ao Ctrl+C; repete a cada 10 s caso algum trecho engula o
            # primeiro (a interface mata à força se passar de 90 s)
            if now - last_int >= 10.0:
                last_int = now
                _thread.interrupt_main()
        _state["stop"].wait(1.0)


def start(mode: str = "auto") -> bool:
    """Chamado pelo piloto ao ligar. False = já tem outro piloto ligado."""
    other = running_info()
    if other and int(other.get("pid", -1)) != os.getpid():
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clear_stop()  # pedido de parar velho (de antes de ligar) não vale
    _state.update(started=time.time(), stopping=False)
    _state["stop"].clear()
    _write_beat(mode)
    t = threading.Thread(target=_watch, args=(mode,), name="controle", daemon=True)
    _state["thread"] = t
    t.start()
    return True


def finish() -> None:
    """Chamado pelo piloto ao sair (sempre, num finally)."""
    _state["stop"].set()
    clear_stop()
    try:
        info = json.loads(HEARTBEAT.read_text(encoding="utf-8"))
        if int(info.get("pid", -1)) == os.getpid():
            HEARTBEAT.unlink()
    except (OSError, ValueError):
        pass
