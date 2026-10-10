#!/usr/bin/env python3
"""
Auto Clipper -- INTERFACE (janela com botões)
=============================================
Pedido do usuário: "crie uma interface pela qual posso usar o programa, e
ligar e desligar por botões de acordo com o modo que eu quiser".

    Duplo clique em AUTO_CLIPPER.bat (ou no atalho "Auto Clipper" da área
    de trabalho). `interface.py --ligar` já abre ligando, no último modo.

  - 5 botões de modo: SÓ POSTAR (não baixa nem edita: só posta a fila, do
    maior score pro menor, conferindo no canal antes) e 25%, 50%, 70% da GPU
    ou sem limite. Com o piloto desligado, o botão LIGA nesse modo; com ele
    ligado, TROCA o modo na hora (nos de GPU a edição em andamento continua,
    só muda o ritmo -- ver src/throttle.py).
  - Ligar / Desligar: desligar para em segundos, salvando tudo (o vídeo que
    estava sendo editado volta pra lista).
  - Painel "Agora": postados hoje, fila, próximo post, último post, o que o
    piloto está fazendo; e o log ao vivo.
  - Cortar um link, conectar o canal, abrir as pastas.

O piloto roda num processo separado, sem janela. Se ele cair, a interface
religa em 60 s (como o INICIAR_AUTOMATICO.bat). Fechar a interface pergunta
se é pra desligar o piloto ou deixar ele rodando sozinho; ao abrir de novo,
ela mostra o piloto que já estava ligado. Um piloto aberto pelos atalhos
.bat também aparece aqui e pode ser desligado pelo botão.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except ImportError:  # Python instalado sem o "tcl/tk"
    print("Esta instalação do Python não tem o tkinter (a parte de janelas).\n"
          "Reinstale o Python pelo python.org marcando \"tcl/tk and IDLE\", ou use o "
          "INICIAR_AUTOMATICO.bat / AUTO_CLIPPER_MENU.bat.")
    sys.exit(1)

from src import config
from src import control
from src import throttle
from src import autopilot as ap

WIN = sys.platform == "win32"
LOG = ap.DATA_DIR / "autopilot.log"
STATE = ap.DATA_DIR / "state.json"
OUT_DIR = ROOT / getattr(config, "AUTOPILOT_OUTPUT_DIR", "output/autopiloto")
MANUAL_DIR = ROOT / "output" / "postar_a_mao"

POST = "postar"  # modo SÓ POSTAR (não baixa nem edita: só posta a fila)
MODES = [(POST, "Só postar", "quase não usa o PC"), (25, "25%", "PC livre"), (50, "50%", "equilibrado"),
         (70, "70%", "rápido"), (100, "Sem limite", "o mais rápido")]
RESTART_AFTER = 60     # piloto caiu: religa depois disso (igual ao .bat)
FORCE_KILL_AFTER = 90  # pediu pra desligar e não desligou: mata à força

# cores
BG, CARD, CARD2, LINE = "#0f172a", "#1e293b", "#273449", "#334155"
TEXT, MUTED, DIM = "#e2e8f0", "#94a3b8", "#64748b"
GREEN, GREEN_D = "#22c55e", "#15803d"
RED, RED_D = "#ef4444", "#b91c1c"
AMBER, BLUE, BLUE_D = "#f59e0b", "#3b82f6", "#1d4ed8"
TEAL, TEAL_D = "#0d9488", "#0f766e"
FONT = "Segoe UI" if WIN else "DejaVu Sans"
MONO = "Consolas" if WIN else "DejaVu Sans Mono"


def _python() -> str:
    """O Python do programa (o da .venv, se existir), com janela de console
    (pythonw não serve: o whisper.cpp/ffmpeg abririam janelas soltas)."""
    env = os.environ.get("AUTOCLIPPER_PYTHON")
    if env:
        return env
    for c in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if c.exists():
            return str(c)
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        exe = exe[:-len("pythonw.exe")] + "python.exe"
    return exe


CONSOLE = ap.DATA_DIR / "piloto_saida.txt"


def _spawn(args, capture: bool = False) -> subprocess.Popen:
    """Roda `autopilot.py args` sem janela.

    A saída (print, erros) vai pra autopilot_data/piloto_saida.txt. Achado
    real (v30 no PC do usuário): a interface roda no pythonw, que não tem
    console; sem um destino pra saída, o Windows entregava ao piloto uma
    saída "quebrada" e o primeiro print derrubava ele em 1 s ("O piloto caiu
    (código 1)" sem parar). No Windows o piloto roda num console SEM
    JANELA (CREATE_NO_WINDOW): o ffmpeg/whisper.cpp que ele abre herdam esse
    console em vez de cada um piscar uma janela preta na tela (e o Terminal
    do Windows 11 não tem janela nenhuma pra mostrar)."""
    cmd = [_python(), "-u", str(ROOT / "autopilot.py")] + [str(a) for a in args]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    kw = {"cwd": str(ROOT), "env": env, "stdin": subprocess.DEVNULL}
    out = None
    if capture:
        kw.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                  encoding="utf-8", errors="replace")
    else:
        ap.DATA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if CONSOLE.exists() and CONSOLE.stat().st_size > 5 * 1024 * 1024:
                CONSOLE.replace(CONSOLE.with_suffix(".old.txt"))
        except OSError:
            pass
        out = open(CONSOLE, "ab")
        out.write(f"\n===== {datetime.now():%d/%m %H:%M:%S} autopilot.py {' '.join(str(a) for a in args)}\n"
                  .encode("utf-8"))
        out.flush()
        kw.update(stdout=out, stderr=subprocess.STDOUT)
    if WIN:
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        kw["startupinfo"] = si
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    try:
        return subprocess.Popen(cmd, **kw)
    finally:
        if out is not None:
            out.close()  # o piloto ficou com a cópia dele


def _crash_reason() -> str:
    """A última linha de erro que o piloto escreveu antes de cair."""
    try:
        with open(CONSOLE, "rb") as f:
            f.seek(max(CONSOLE.stat().st_size - 16 * 1024, 0))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = [ln.strip() for ln in tail if ln.strip()]
    for ln in reversed(tail):
        if ln.startswith("=====") and "autopilot.py" in ln:
            break
        if ("Error" in ln or "Exception" in ln or "erro" in ln.lower()) and not ln.startswith("File "):
            return ln[:220]
    return tail[-1][:220] if tail else ""


def _kill_tree(pid: int) -> None:
    try:
        if WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except OSError:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _open(path) -> None:
    p = Path(path)
    if not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
    try:
        if WIN:
            os.startfile(str(p))  # noqa: abre no Explorer / programa padrão
        else:
            subprocess.Popen(["xdg-open", str(p)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        messagebox.showerror("Auto Clipper", f"Não consegui abrir {p}:\n{e}")


def _current_mode():
    """POST (só postar) ou o % de GPU do modo completo -- o salvo."""
    return POST if control.post_only() else throttle.configured_limit()


def _apply_mode(m) -> None:
    if m == POST:
        control.set_post_only(True)
    else:
        throttle.save_limit(m)
        control.set_post_only(False)


def _mode_label(m) -> str:
    return "só postar" if m == POST else ("sem limite" if m >= 100 else f"{m}%")


def _mode_desc(m) -> str:
    return "modo SÓ POSTAR" if m == POST else throttle.describe(m)


def _ago(ts: float) -> str:
    m = int(max(time.time() - ts, 0) // 60)
    if m < 1:
        return "agora há pouco"
    if m < 60:
        return f"há {m} min"
    h = m // 60
    return f"há {h} h" if h < 24 else f"há {h // 24} dia(s)"


def _when(ts: float) -> str:
    if ts - time.time() < 60:
        return "agora"
    d = datetime.fromtimestamp(ts)
    return f"{d:%H:%M}" if d.date() == datetime.now().date() else f"amanhã {d:%H:%M}"


class Button(tk.Label):
    """Botão chapado com cor (o tk.Button do Windows ignora parte das cores)."""

    def __init__(self, parent, text, command, bg=CARD2, fg=TEXT, hover=None, font=None, **kw):
        super().__init__(parent, text=text, bg=bg, fg=fg, cursor="hand2",
                         font=font or (FONT, 10), padx=kw.pop("padx", 14), pady=kw.pop("pady", 8), **kw)
        self._bg, self._hover, self._cmd, self._enabled = bg, hover or LINE, command, True
        self.bind("<Enter>", lambda e: self._enabled and self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))
        self.bind("<Button-1>", lambda e: self._enabled and self._cmd())

    def style(self, bg=None, fg=None, hover=None, text=None, enabled=None):
        if bg is not None:
            self._bg = bg
        if hover is not None:
            self._hover = hover
        if enabled is not None:
            self._enabled = enabled
            self.config(cursor="hand2" if enabled else "arrow")
        self.config(bg=self._bg, **({"fg": fg} if fg else {}), **({"text": text} if text else {}))


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.proc = None        # piloto aberto por esta janela
        self.link_proc = None   # "cortar um link" aberto por esta janela
        self.want_on = False    # o usuário quer o piloto ligado (religa se cair)
        self.stop_at = None     # quando pediu pra desligar
        self.restart_at = None  # piloto caiu: religa nessa hora
        self.log_pos = None
        self.last_line = ""
        self._build()
        info = control.running_info()
        if info and info.get("mode") == "auto":
            self.want_on = True
            self._note(f"Piloto já estava ligado (desde {datetime.fromtimestamp(info.get('started', 0)):%d/%m %H:%M}).")
        self._tick()
        self._stats()

    # ------------------------------------------------------------ tela ---
    def _card(self, parent, title):
        outer = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE)
        tk.Label(outer, text=title, bg=CARD, fg=MUTED, font=(FONT, 9, "bold")).pack(anchor="w", padx=16, pady=(12, 4))
        body = tk.Frame(outer, bg=CARD)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        return outer, body

    def _build(self):
        r = self.root
        r.title("Auto Clipper")
        r.configure(bg=BG)
        r.geometry("940x780")
        r.minsize(760, 620)

        head = tk.Frame(r, bg=BG)
        head.pack(fill="x", padx=20, pady=(16, 10))
        left = tk.Frame(head, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="Auto Clipper", bg=BG, fg=TEXT, font=(FONT, 20, "bold")).pack(anchor="w")
        tk.Label(left, text="Acha vídeo bombando, corta, edita e posta no seu canal", bg=BG, fg=MUTED,
                 font=(FONT, 10)).pack(anchor="w")
        self.status = tk.Label(head, text="", bg=CARD, fg=TEXT, font=(FONT, 11, "bold"), padx=16, pady=8)
        self.status.pack(side="right")

        # --- ligar / modos
        card, body = self._card(r, "PILOTO AUTOMÁTICO")
        card.pack(fill="x", padx=20, pady=(0, 10))
        modes = tk.Frame(body, bg=CARD)
        modes.pack(fill="x")
        self.mode_btns = {}
        for i, (n, label, sub) in enumerate(MODES):
            b = Button(modes, f"{label}\n{sub}", lambda n=n: self._mode_click(n), font=(FONT, 11, "bold"),
                       pady=10, justify="center")
            b.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            modes.columnconfigure(i, weight=1, uniform="m")
            self.mode_btns[n] = b
        row = tk.Frame(body, bg=CARD)
        row.pack(fill="x", pady=(10, 0))
        self.power = Button(row, "", self._power_click, font=(FONT, 12, "bold"), pady=10, padx=26)
        self.power.pack(side="left")
        self.hint = tk.Label(row, text="", bg=CARD, fg=MUTED, font=(FONT, 9), justify="left", wraplength=560)
        self.hint.pack(side="left", padx=14)

        # --- agora
        card, body = self._card(r, "AGORA")
        card.pack(fill="x", padx=20, pady=(0, 10))
        grid = tk.Frame(body, bg=CARD)
        grid.pack(fill="x")
        self.kpi = {}
        for i, (key, label) in enumerate([("today", "Postados hoje"), ("queue", "Clipes na fila"),
                                          ("next", "Próximo post"), ("manual", "Pra postar à mão hoje")]):
            box = tk.Frame(grid, bg=CARD2)
            box.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            grid.columnconfigure(i, weight=1, uniform="k")
            v = tk.Label(box, text="—", bg=CARD2, fg=TEXT, font=(FONT, 18, "bold"))
            v.pack(anchor="w", padx=12, pady=(8, 0))
            tk.Label(box, text=label, bg=CARD2, fg=MUTED, font=(FONT, 9)).pack(anchor="w", padx=12, pady=(0, 8))
            self.kpi[key] = v
        self.last_post = tk.Label(body, text="", bg=CARD, fg=TEXT, font=(FONT, 10), anchor="w", cursor="hand2")
        self.last_post.pack(fill="x", pady=(10, 0))
        self.last_post.bind("<Button-1>", lambda e: self._open_last())
        self.doing = tk.Label(body, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w", justify="left")
        self.doing.pack(fill="x", pady=(4, 0))

        # --- ferramentas (antes do log: o log é que estica)
        tools = tk.Frame(r, bg=BG)
        tools.pack(side="bottom", fill="x", padx=20, pady=(0, 16))
        for text, cmd in [("✂  Cortar um link…", self._cut_link), ("Conectar canal do YouTube", self._login),
                          ("Pasta: postar à mão", lambda: _open(MANUAL_DIR)), ("Pasta dos cortes", lambda: _open(OUT_DIR)),
                          ("▶  YouTube Studio", lambda: webbrowser.open("https://studio.youtube.com"))]:
            Button(tools, text, cmd, font=(FONT, 9), pady=7, padx=12).pack(side="left", padx=(0, 8))

        # --- log
        card, body = self._card(r, "O QUE O PILOTO ESTÁ FAZENDO (LOG)")
        card.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        wrap = tk.Frame(body, bg=CARD)
        wrap.pack(fill="both", expand=True)
        self.log = tk.Text(wrap, bg="#0b1220", fg="#cbd5e1", font=(MONO, 9), relief="flat", wrap="word",
                           insertbackground=TEXT, padx=10, pady=8, height=12, highlightthickness=0)
        sb = tk.Scrollbar(wrap, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("err", foreground="#fca5a5")
        self.log.tag_configure("ok", foreground="#86efac")
        self.log.tag_configure("act", foreground="#93c5fd")
        self.log.tag_configure("dim", foreground=DIM)
        self.log.tag_configure("ui", foreground="#fde68a")
        self.log.configure(state="disabled")

        r.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------- log ---
    def _append(self, lines):
        if not lines:
            return
        at_end = self.log.yview()[1] > 0.98
        self.log.configure(state="normal")
        for ln in lines:
            body = ln[17:] if ln.startswith("[") and len(ln) > 17 else ln
            tag = ("err" if "[!]" in ln or "erro" in body.lower()[:40]
                   else "ok" if "postado" in body.lower() or "liberado" in body.lower()
                   else "act" if body.lstrip().startswith((">>", "..")) or "LIGADO" in body
                   else "dim" if body.startswith("    |") else "ui" if body.startswith("[interface]") else "")
            self.log.insert("end", ln + "\n", tag)
        n = int(self.log.index("end-1c").split(".")[0])
        if n > 2500:
            self.log.delete("1.0", f"{n - 2000}.0")
        self.log.configure(state="disabled")
        if at_end:
            self.log.see("end")

    def _note(self, msg: str):
        self._append([f"[{datetime.now():%d/%m %H:%M:%S}] [interface] {msg}"])

    def _read_log(self):
        try:
            size = LOG.stat().st_size
        except OSError:
            return
        if self.log_pos is None or size < self.log_pos:  # primeira leitura ou log girou
            self.log_pos = max(size - 48 * 1024, 0)
            skip_partial = self.log_pos > 0
        else:
            skip_partial = False
        if size == self.log_pos:
            return
        try:
            with open(LOG, "rb") as f:
                f.seek(self.log_pos)
                data = f.read(size - self.log_pos)
        except OSError:
            return
        cut = data.rfind(b"\n")
        if cut < 0:
            return
        self.log_pos += cut + 1
        lines = data[:cut].decode("utf-8", errors="replace").splitlines()
        if skip_partial and lines:
            lines = lines[1:]
        lines = [ln.rstrip() for ln in lines]
        for ln in reversed(lines):
            if ln.strip():
                self.last_line = ln
                break
        self._append(lines)

    # ---------------------------------------------------------- piloto ---
    def _start(self, mode):
        _apply_mode(mode)
        control.clear_stop()
        try:
            self.proc = _spawn(["--auto", "--so-postar"] if mode == POST else ["--auto", "--gpu", mode])
        except OSError as e:
            messagebox.showerror("Auto Clipper", f"Não consegui ligar o piloto:\n{e}")
            return
        self.want_on, self.stop_at, self.restart_at = True, None, None
        self._note(f"Ligando o piloto ({_mode_desc(mode)})...")
        self._refresh_controls()

    def _stop(self):
        self.want_on, self.restart_at = False, None
        if control.running_info() or (self.proc and self.proc.poll() is None):
            control.request_stop()
            self.stop_at = time.time()
            self._note("Desligando o piloto (a edição em andamento para e o vídeo volta pra lista)...")
        self._refresh_controls()

    def _mode_click(self, mode):
        info = control.running_info()
        starting = self.proc is not None and self.proc.poll() is None
        if info or starting:
            cur = _current_mode()
            if cur != mode:
                _apply_mode(mode)
                if mode == POST:
                    self._note("Modo SÓ POSTAR: o piloto para de buscar e editar (a edição em andamento para e "
                               "o vídeo volta pra lista) e só posta a fila, do maior score pro menor.")
                elif cur == POST:
                    self._note(f"Modo completo ({throttle.describe(mode)}): o piloto volta a buscar, editar e postar.")
                else:
                    self._note(f"Modo trocado pra {throttle.describe(mode)} -- vale na hora, sem parar a edição.")
            self._refresh_controls()
            return
        if self.link_proc is not None and self.link_proc.poll() is None:
            return
        self._start(mode)

    def _power_click(self):
        if self._is_on() or self.restart_at:
            if self.restart_at and not self._is_on():
                self.want_on, self.restart_at = False, None
                self._note("Religamento cancelado.")
                self._refresh_controls()
                return
            self._stop()
        elif not (self.link_proc is not None and self.link_proc.poll() is None):
            self._start(_current_mode())

    def _is_on(self) -> bool:
        info = control.running_info()
        if info and info.get("mode") == "auto":
            return True
        return self.proc is not None and self.proc.poll() is None

    def _tick(self):
        try:
            self._watch_procs()
            self._read_log()
            self._refresh_controls()
        finally:
            self.root.after(1000, self._tick)

    def _watch_procs(self):
        now = time.time()
        info = control.running_info()
        if self.proc is not None and self.proc.poll() is not None:
            code = self.proc.returncode
            self.proc = None
            if self.want_on and self.stop_at is None and code not in (0, None):
                self.restart_at = now + RESTART_AFTER
                why = _crash_reason()
                self._note(f"O piloto caiu (código {code}){': ' + why if why else ''}. Religando em "
                           f"{RESTART_AFTER} s... (detalhes em autopilot_data\\piloto_saida.txt)")
            elif self.want_on and self.stop_at is None and not info:
                self.want_on = False  # saiu sozinho (ex.: já tinha outro piloto ligado)
        if self.restart_at and now >= self.restart_at and not info:
            self.restart_at = None
            self._start(_current_mode())
        if self.stop_at is not None:
            alive = info or (self.proc is not None and self.proc.poll() is None)
            if not alive:
                self.stop_at = None
                control.clear_stop()
                self._note("Piloto desligado.")
            elif now - self.stop_at > FORCE_KILL_AFTER:
                pid = (info or {}).get("pid") or (self.proc.pid if self.proc else None)
                if pid:
                    _kill_tree(int(pid))
                if self.proc is not None:
                    try:
                        self.proc.wait(timeout=5)
                    except Exception:
                        pass
                    self.proc = None  # morto por nós: não é "caiu", não religa
                try:
                    control.HEARTBEAT.unlink()
                except OSError:
                    pass
                self.stop_at = None
                self.want_on = False
                control.clear_stop()
                self._note("O piloto não respondeu: encerrado à força.")
                return
        if self.link_proc is not None and self.link_proc.poll() is not None:
            self.link_proc = None
            self._note("Corte do link terminou (veja acima o que foi postado).")
        # piloto aberto por fora (atalho .bat) que ligou ou desligou
        if info and info.get("mode") == "auto" and not self.want_on and self.stop_at is None:
            self.want_on = True
        if not info and self.proc is None and self.want_on and not self.restart_at and self.stop_at is None:
            self.want_on = False

    def _refresh_controls(self):
        info = control.running_info()
        mode = _current_mode()
        starting = self.proc is not None and self.proc.poll() is None and not info
        link = (info and info.get("mode") == "link") or (self.link_proc is not None and self.link_proc.poll() is None)
        on = bool(info and info.get("mode") == "auto")
        if link:
            st, color = "●  CORTANDO UM LINK", AMBER
        elif self.stop_at is not None or (info and info.get("stopping")):
            st, color = "●  DESLIGANDO...", AMBER
        elif on:
            st, color = ("●  LIGADO  ·  só postando" if mode == POST else
                         f"●  LIGADO  ·  {'GPU sem limite' if mode >= 100 else f'GPU {mode}%'}"), GREEN
        elif starting:
            st, color = "●  LIGANDO...", AMBER
        elif self.restart_at:
            st, color = f"●  CAIU · religando em {int(self.restart_at - time.time())} s", RED
        else:
            st, color = "○  DESLIGADO", MUTED
        self.status.config(text=st, fg=color)

        busy = on or starting or link
        for n, b in self.mode_btns.items():
            sel = n == mode
            acc, acc_d = (TEAL, TEAL_D) if n == POST else (BLUE, BLUE_D)
            b.style(bg=acc if sel else CARD2, hover=acc_d if sel else LINE, fg="#ffffff" if sel else TEXT,
                    enabled=not link)
        if self.stop_at is not None:
            self.power.style(text="■  Desligando…", bg=LINE, hover=LINE, fg=MUTED, enabled=False)
            self.hint.config(text="Parando a edição em andamento e salvando tudo...")
        elif on or starting:
            self.power.style(text="■  Desligar", bg=RED, hover=RED_D, fg="#ffffff", enabled=True)
            self.hint.config(text=("Só postando: não baixa nem edita nada; posta a fila do maior score pro menor, "
                                   "conferindo no canal antes de cada post." if mode == POST else
                                   "Clique em outro modo pra trocar na hora (a edição continua, só muda o ritmo "
                                   "da placa de vídeo). Desligar salva tudo; o vídeo em edição volta pra lista."))
        elif self.restart_at:
            self.power.style(text="✕  Cancelar religar", bg=CARD2, hover=LINE, fg=TEXT, enabled=True)
            self.hint.config(text="O piloto caiu e vai ser religado sozinho.")
        else:
            self.power.style(text=f"▶  Ligar ({_mode_label(mode)})", bg=GREEN_D,
                             hover=GREEN, fg="#ffffff", enabled=not link)
            self.hint.config(text="Clique num modo pra ligar nele. Só postar: não baixa nem edita, só posta a "
                                  "fila (maior score primeiro). 25%: PC livre, edição ~4x mais lenta. Sem limite: "
                                  "o mais rápido.")
        line = self.last_line[17:] if self.last_line.startswith("[") else self.last_line
        line = line.strip().lstrip("| ").strip()
        self.doing.config(text=("Agora: " + line[:150]) if line and (busy or self.stop_at) else
                          ("Último registro: " + line[:140]) if line else "")

    # ---------------------------------------------------------- painel ---
    def _stats(self):
        try:
            self._fill_stats()
        except Exception:
            pass
        finally:
            self.root.after(3000, self._stats)

    def _fill_stats(self):
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        now = time.time()
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        posted = st.get("posted", [])
        queue = st.get("queue", [])
        target = ap.daily_limit()
        today = sum(1 for p in posted if p.get("at", 0) >= midnight)
        manual = sum(1 for m in st.get("manual", []) if m.get("at", 0) >= midnight)
        self.kpi["today"].config(text=f"{today} / {target}")
        self.kpi["queue"].config(text=str(len(queue)))
        self.kpi["manual"].config(text=str(manual))
        on = self._is_on()
        if not on:
            nxt = "desligado"
        elif not queue:
            nxt = "fila vazia"
        else:
            q = st.get("quota", {})
            uploads = q.get("uploads", 0) if q.get("day") == ap._pacific_day() else 0
            if st.get("blocked_until", 0) > now:
                ts = st["blocked_until"]
            elif uploads >= target:
                ts = ap._next_pacific_midnight()
            else:
                ts = max(st.get("last_post_at", 0) + ap.post_gap_seconds(), now)
            nxt = _when(ts)
        self.kpi["next"].config(text=nxt, font=(FONT, 18 if len(nxt) <= 6 else 13, "bold"))
        if posted:
            p = posted[-1]
            self._last_id = p.get("youtube_id")
            extra = f"  ·  bloqueado: {p['blocked']}" if p.get("blocked") else ""
            self.last_post.config(text=f"Último post: {p.get('title', '')[:90]}  ({_ago(p.get('at', 0))}){extra}  ↗",
                                  fg="#fca5a5" if p.get("blocked") else TEXT)
        else:
            self._last_id = None
            self.last_post.config(text="Nenhum vídeo postado ainda.", fg=MUTED)

    def _open_last(self):
        if getattr(self, "_last_id", None):
            webbrowser.open(f"https://youtube.com/shorts/{self._last_id}")

    # ------------------------------------------------------ ferramentas ---
    def _cut_link(self):
        if self._is_on() or control.running_info():
            messagebox.showinfo("Auto Clipper", "Desligue o piloto antes de cortar um link: os dois mexem "
                                "na mesma fila de clipes.\n\nDepois é só ligar de novo.")
            return
        url = simpledialog.askstring("Cortar um link", "Link do vídeo do YouTube (ou caminho de um arquivo do PC):",
                                     parent=self.root)
        if not url or not url.strip():
            return
        n = simpledialog.askinteger("Cortar um link", "Quantos cortes?", parent=self.root, minvalue=1, maxvalue=10,
                                    initialvalue=getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3))
        if not n:
            return
        try:
            self.link_proc = _spawn(["--url", url.strip(), "--clips", n])
        except OSError as e:
            messagebox.showerror("Auto Clipper", str(e))
            return
        self._note(f"Cortando {url.strip()} ({n} corte(s)); os clipes são postados assim que ficarem prontos.")

    def _login(self):
        try:
            proc = _spawn(["--login"], capture=True)
        except OSError as e:
            messagebox.showerror("Auto Clipper", str(e))
            return
        self._note("Abrindo o navegador pra conectar o canal (escolha a conta do canal e autorize)...")

        def wait():
            out = proc.communicate()[0] or ""
            lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("=")]
            msg = "\n".join(lines[-6:]) or "Terminou."
            ok = "Conectado ao canal" in out
            self.root.after(0, lambda: (messagebox.showinfo if ok else messagebox.showwarning)("Conectar canal", msg))
        threading.Thread(target=wait, daemon=True).start()

    # ---------------------------------------------------------- fechar ---
    def _on_close(self):
        if self._is_on() and self.stop_at is None:
            ans = messagebox.askyesnocancel(
                "Fechar o Auto Clipper",
                "O piloto está ligado.\n\n"
                "Sim: desligar o piloto e fechar.\n"
                "Não: fechar só esta janela -- o piloto continua trabalhando sozinho "
                "(abra a interface de novo pra ver ou desligar).\n"
                "Cancelar: voltar.")
            if ans is None:
                return
            if ans:
                control.request_stop()
        self.root.destroy()


def main():
    if WIN:
        try:  # texto nítido em tela com escala (125%, 150%...)
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    app = App(root)
    if "--ligar" in sys.argv[1:] and not control.running_info():
        # aberta pela atualização: volta a trabalhar sozinha, no último modo
        root.after(800, lambda: app._start(_current_mode()))
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # aberta pelo pythonw (sem console): erro na tela + arquivo pra mandar
        import traceback
        err = traceback.format_exc()
        try:
            ap.DATA_DIR.mkdir(parents=True, exist_ok=True)
            (ap.DATA_DIR / "interface_erro.txt").write_text(err, encoding="utf-8")
            messagebox.showerror("Auto Clipper", "A interface deu erro ao abrir:\n\n" + err[-900:] +
                                 "\n\n(salvo em autopilot_data\\interface_erro.txt)")
        except Exception:
            print(err)
        sys.exit(1)
