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
  - Quantos Shorts por dia (− e +; padrão 6, vale na hora).
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
from src import accounts
from src import channel_lists

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


class ChannelsWindow:
    """Lista branca (canais que o piloto corta) e lista negra (nunca cortar)
    -- src/channel_lists.py. Pedido do usuário (v48). Salva na hora; o piloto
    ligado usa na próxima busca e tira da fila os clipes da lista negra."""

    _open = None

    def __init__(self, app):
        if ChannelsWindow._open is not None and ChannelsWindow._open.win.winfo_exists():
            ChannelsWindow._open.win.lift()
            ChannelsWindow._open.win.focus_force()
            return
        ChannelsWindow._open = self
        self.app = app
        w = self.win = tk.Toplevel(app.root)
        w.title("Canais: cortar / não cortar")
        w.configure(bg=BG)
        w.geometry("980x600")
        w.minsize(760, 460)
        tk.Label(w, text="Canais", bg=BG, fg=TEXT, font=(FONT, 16, "bold")).pack(anchor="w", padx=20, pady=(14, 0))
        tk.Label(w, text="Vale na hora, com o piloto ligado. Lista negra ganha da branca: o canal some da busca e os "
                         "clipes dele que estão na fila não são postados.",
                 bg=BG, fg=MUTED, font=(FONT, 9), wraplength=900, justify="left").pack(anchor="w", padx=20, pady=(2, 10))
        cols = tk.Frame(w, bg=BG)
        cols.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        cols.columnconfigure(0, weight=1, uniform="c")
        cols.columnconfigure(1, weight=1, uniform="c")
        self.white = self._column(cols, 0, "LISTA BRANCA -- canais que o piloto corta",
                                  "Cole o @ ou o link do canal (ex.: @flowpodcast)", GREEN, self._add_white,
                                  self._del_white, extras=True)
        self.black = self._column(cols, 1, "LISTA NEGRA -- nunca cortar",
                                  "@, link ou o NOME do canal (ex.: Felipe Neto)", RED, self._add_black,
                                  self._del_black)
        self._fill()

    def _column(self, parent, col, title, hint, color, add, delete, extras=False):
        outer = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE)
        outer.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0))
        tk.Label(outer, text=title, bg=CARD, fg=color, font=(FONT, 9, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
        row = tk.Frame(outer, bg=CARD)
        row.pack(fill="x", padx=14)
        entry = tk.Entry(row, bg=CARD2, fg=TEXT, insertbackground=TEXT, relief="flat", font=(FONT, 10))
        entry.pack(side="left", fill="x", expand=True, ipady=6)
        entry.bind("<Return>", lambda e: add())
        Button(row, "Adicionar", add, font=(FONT, 9, "bold"), pady=5, padx=12).pack(side="left", padx=(8, 0))
        tk.Label(outer, text=hint, bg=CARD, fg=DIM, font=(FONT, 8)).pack(anchor="w", padx=14, pady=(3, 0))
        opts = {}
        if extras:
            orow = tk.Frame(outer, bg=CARD)
            orow.pack(fill="x", padx=14, pady=(4, 0))
            for key, text in (("en", "canal em inglês"), ("game", "gameplay (facecam + jogo)")):
                v = tk.BooleanVar(value=False)
                tk.Checkbutton(orow, text=text, variable=v, bg=CARD, fg=MUTED, selectcolor=CARD2,
                               activebackground=CARD, activeforeground=TEXT, font=(FONT, 9)).pack(side="left",
                                                                                                 padx=(0, 12))
                opts[key] = v
        lb_wrap = tk.Frame(outer, bg=CARD)
        lb_wrap.pack(fill="both", expand=True, padx=14, pady=(8, 6))
        lb = tk.Listbox(lb_wrap, bg="#0b1220", fg=TEXT, selectbackground=BLUE_D, relief="flat", highlightthickness=0,
                        font=(FONT, 10), activestyle="none")
        sb = tk.Scrollbar(lb_wrap, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        lb.bind("<Delete>", lambda e: delete())
        brow = tk.Frame(outer, bg=CARD)
        brow.pack(fill="x", padx=14, pady=(0, 12))
        Button(brow, "Remover selecionado", delete, font=(FONT, 9), pady=4, padx=10).pack(side="left")
        count = tk.Label(brow, text="", bg=CARD, fg=DIM, font=(FONT, 9))
        count.pack(side="right")
        return {"entry": entry, "list": lb, "opts": opts, "count": count, "items": []}

    def _fill(self):
        d = channel_lists.load()
        for side, items, fmt in ((self.white, d["white"], channel_lists.label),
                                 (self.black, d["black"], lambda b: b.get("canal", ""))):
            side["items"] = list(items)
            lb = side["list"]
            lb.delete(0, "end")
            for it in items:
                lb.insert("end", fmt(it))
            side["count"].config(text=f"{len(items)} canal(is)")

    def _selected(self, side):
        sel = side["list"].curselection()
        return side["items"][sel[0]] if sel else None

    def _add_white(self):
        text = self.white["entry"].get().strip()
        if not text:
            return
        if not channel_lists.looks_like_channel(text):
            messagebox.showinfo("Lista branca", "Cole o @ do canal (ex.: @flowpodcast) ou o link dele "
                                "(youtube.com/@... ou youtube.com/channel/...).", parent=self.win)
            return
        if channel_lists.is_black(text):
            if not messagebox.askyesno("Lista branca", f"\"{text}\" está na lista negra. Tirar de lá e pôr na "
                                       "lista branca?", parent=self.win):
                return
        o = self.white["opts"]
        e = channel_lists.add_white(text, en=o["en"].get(), game=o["game"].get())
        self.white["entry"].delete(0, "end")
        self._fill()
        self.app._note(f"Lista branca: {channel_lists.label(e)} adicionado -- conferindo o canal no YouTube...")
        threading.Thread(target=self._check, args=(e,), daemon=True).start()

    def _check(self, e):
        """Confere se o canal existe e anota o nome (em segundo plano)."""
        try:
            from src import discovery
            vids = discovery._list_channel(channel_lists.spec(e), 3)
            name = vids[0]["channel"] if vids else None
            msg = (f"Lista branca: canal \"{name}\" conferido ({len(vids)} vídeo(s) recentes)." if vids
                   else "Lista branca: o canal não tem vídeos públicos (confira o link).")
            if name:
                channel_lists.set_name(e["canal"], name)
        except Exception as ex:
            text = str(ex)
            if "404" in text or "does not exist" in text or "not found" in text.lower():
                msg = (f"Lista branca: não achei \"{e['canal']}\" no YouTube -- confira o @ ou o link "
                       "(ficou na lista; remova se estiver errado).")
            else:
                msg = f"Lista branca: não deu pra conferir \"{e['canal']}\" agora; fica na lista mesmo assim."

        def done():
            self.app._note(msg)
            if self.win.winfo_exists():
                self._fill()
        self.app.root.after(0, done)

    def _del_white(self):
        it = self._selected(self.white)
        if it is None:
            return
        channel_lists.remove_white(it["canal"])
        self._fill()
        self.app._note(f"Lista branca: {channel_lists.label(it)} removido (os clipes dele que já estão na fila "
                       "continuam; pra tirar também, ponha na lista negra).")

    def _add_black(self):
        text = self.black["entry"].get().strip()
        if not text:
            return
        if len(channel_lists.norm(text)) < 3:
            messagebox.showinfo("Lista negra", "Nome curto demais: escreva o nome do canal, o @ ou o link.",
                                parent=self.win)
            return
        gone = channel_lists.add_black(text)
        self.black["entry"].delete(0, "end")
        self._fill()
        self.app._note(f"Lista negra: \"{text}\" adicionado -- nunca mais cortado; clipes dele na fila saem."
                       + (f" Saiu da lista branca: {', '.join(gone)}." if gone else ""))

    def _del_black(self):
        it = self._selected(self.black)
        if it is None:
            return
        channel_lists.remove_black(it.get("canal"))
        self._fill()
        self.app._note(f"Lista negra: \"{it.get('canal')}\" removido.")


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
        # com vários canais a lista de canais ocupa espaço: usa a altura da tela
        r.geometry(f"940x{max(min(r.winfo_screenheight() - 90, 1000), 700)}")
        r.minsize(760, 620)

        head = tk.Frame(r, bg=BG)
        head.pack(fill="x", padx=20, pady=(16, 10))
        left = tk.Frame(head, bg=BG)
        left.pack(side="left")
        tk.Label(left, text="Auto Clipper", bg=BG, fg=TEXT, font=(FONT, 20, "bold")).pack(anchor="w")
        tk.Label(left, text="Acha vídeo bombando, corta, edita e posta nos seus canais", bg=BG, fg=MUTED,
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
        # quantos Shorts por dia (pedido do usuário; padrão 6) -- vale na hora
        prow = tk.Frame(body, bg=CARD)
        prow.pack(fill="x", pady=(12, 0))
        self.posts_label = tk.Label(prow, text="Shorts por dia:", bg=CARD, fg=TEXT, font=(FONT, 11, "bold"))
        self.posts_label.pack(side="left")
        Button(prow, "−", lambda: self._posts_step(-1), font=(FONT, 13, "bold"), padx=14, pady=2).pack(
            side="left", padx=(12, 0))
        self.posts_val = tk.Label(prow, text="", bg=CARD, fg=TEXT, font=(FONT, 15, "bold"), width=3)
        self.posts_val.pack(side="left")
        Button(prow, "+", lambda: self._posts_step(1), font=(FONT, 13, "bold"), padx=14, pady=2).pack(side="left")
        self.posts_hint = tk.Label(prow, text="", bg=CARD, fg=MUTED, font=(FONT, 9), justify="left", anchor="w",
                                   wraplength=430)
        self.posts_hint.pack(side="left", padx=14)

        # --- canais do YouTube (v46: vários canais; o piloto alterna entre eles)
        card, body = self._card(r, "CANAIS DO YOUTUBE")
        card.pack(fill="x", padx=20, pady=(0, 10))
        self.acc_box = tk.Frame(body, bg=CARD)
        self.acc_box.pack(fill="x")
        arow = tk.Frame(body, bg=CARD)
        arow.pack(fill="x", pady=(6, 0))
        Button(arow, "+ Adicionar canal", self._login, bg=TEAL, hover=TEAL_D, font=(FONT, 9, "bold"),
               pady=5, padx=12).pack(side="left")
        self.acc_hint = tk.Label(arow, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w", justify="left",
                                 wraplength=650)
        self.acc_hint.pack(side="left", padx=12)
        self._acc_rows, self._acc_sig, self._login_proc = {}, None, None

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
        lp = tk.Frame(body, bg=CARD)
        lp.pack(fill="x", pady=(10, 0))
        # quadros do vídeo ORIGINAL do último post: quando a câmera sai errada,
        # dá pra ver (e mandar) o que tinha na tela antes do reenquadramento
        self.sheet_link = tk.Label(lp, text="", bg=CARD, fg=MUTED, font=(FONT, 9, "underline"), cursor="hand2")
        self.sheet_link.pack(side="right")
        self.sheet_link.bind("<Button-1>", lambda e: self._open_sheet())
        self.last_post = tk.Label(lp, text="", bg=CARD, fg=TEXT, font=(FONT, 10), anchor="w", cursor="hand2")
        self.last_post.pack(side="left", fill="x", expand=True)
        self.last_post.bind("<Button-1>", lambda e: self._open_last())
        # freio de alcance (src/reach.py): o YouTube parou de entregar os Shorts
        self.alert = tk.Label(body, text="", bg=CARD, fg=AMBER, font=(FONT, 9, "bold"), anchor="w", justify="left",
                              wraplength=820)
        self.alert.pack(fill="x", pady=(4, 0))
        self.doing = tk.Label(body, text="", bg=CARD, fg=MUTED, font=(FONT, 9), anchor="w", justify="left")
        self.doing.pack(fill="x", pady=(4, 0))

        # --- ferramentas (antes do log: o log é que estica)
        tools = tk.Frame(r, bg=BG)
        tools.pack(side="bottom", fill="x", padx=20, pady=(0, 16))
        for text, cmd in [("✂  Cortar um link…", self._cut_link),
                          ("☰  Canais: cortar / não cortar", lambda: ChannelsWindow(self)),
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
        if "accounts" not in st:  # state de antes da v46 (o piloto converte ao ligar)
            st["accounts"] = {accounts.FIRST: {k: st[k] for k in ("reach_brake", "auth_down", "last_post_at")
                                               if k in st}}
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        posted = st.get("posted", [])
        queue = st.get("queue", [])
        accs = accounts.load()
        active = [a for a in accs if a.get("on", True)]
        down = {a["id"] for a in active if self._login_down(st, a)}
        live = [a["id"] for a in active if a["id"] not in down]
        target = ap.total_per_day(len(active))
        today = sum(1 for p in posted if p.get("at", 0) >= midnight)
        when = ap.projected(st, live) if live else {}
        self._fill_accounts(st, accs, down, when)
        manual = sum(1 for m in st.get("manual", []) if m.get("at", 0) >= midnight)
        self.kpi["today"].config(text=f"{today} / {target}")
        self.kpi["queue"].config(text=str(len(queue)))
        self.kpi["manual"].config(text=str(manual))
        on = self._is_on()
        if not on:
            nxt = "desligado"
        elif not queue:
            nxt = "fila vazia"
        elif not live:
            nxt = "sem canal"
        else:
            nxt = _when(max(min(when.values()), now)) if when else "—"
        self.kpi["next"].config(text=nxt, font=(FONT, 18 if len(nxt) <= 6 else 13, "bold"))
        if posted:
            p = posted[-1]
            self._last_id = p.get("youtube_id")
            extra = f"  ·  bloqueado: {p['blocked']}" if p.get("blocked") else ""
            self.last_post.config(text=f"Último post: {p.get('title', '')[:90]}  ({_ago(p.get('at', 0))}){extra}  ↗",
                                  fg="#fca5a5" if p.get("blocked") else TEXT)
            sheet = Path(p["video"]).with_suffix(".origem.jpg") if p.get("video") else None
            self._sheet = sheet if sheet is not None and sheet.exists() else None
        else:
            self._last_id = None
            self._sheet = None
            self.last_post.config(text="Nenhum vídeo postado ainda.", fg=MUTED)
        self.sheet_link.config(text="ver quadros do original" if self._sheet else "")
        self._show_posts()
        brakes = [(a, ap.reach.brake_of(st, a["id"])) for a in active]
        brakes = [(a, b) for a, b in brakes if b]
        if down:
            names = ", ".join(accounts.label(a) for a in active if a["id"] in down)
            self.alert.config(fg="#fca5a5", text=f"⚠ O login do YouTube caiu em: {names} -- nada é postado nele. "
                              "Clique em \"Reconectar\" no canal (se cai a cada 7 dias, publique o app no Google "
                              "Cloud: README, \"Postagem automática\", passo 3).")
        elif len(brakes) == 1 and len(active) == 1:
            b = brakes[0][1]
            self.alert.config(fg=AMBER, text=f"⚠ O YouTube parou de entregar os Shorts: os últimos tiveram "
                                             f"{', '.join(map(str, b.get('recent', [])))} views (antes "
                                             f"~{b.get('base', 0):.0f}). Freio ligado: 1 post a cada "
                                             f"{ap.reach.brake_hours():.0f} h até voltar.")
        elif brakes:
            self.alert.config(fg=AMBER, text="⚠ Freio de alcance (o YouTube parou de entregar os últimos Shorts) em: "
                                             + ", ".join(accounts.label(a) for a, _ in brakes)
                                             + f" -- 1 post a cada {ap.reach.brake_hours():.0f} h nesses canais.")
        elif not accs:
            self.alert.config(fg=AMBER, text="⚠ Nenhum canal conectado: clique em \"+ Adicionar canal\".")
        else:
            self.alert.config(text="")

    def _login_down(self, st: dict, acc: dict) -> bool:
        """Login do canal caído (e não reconectado depois do aviso)?"""
        a = ap.sub(st, acc["id"]).get("auth_down")
        try:
            mt = accounts.token_file(acc).stat().st_mtime
        except OSError:
            mt = 0
        return bool(a) and mt <= a.get("at", 0)

    def _fill_accounts(self, st: dict, accs: list, down: set, when: dict):
        """Uma linha por canal: situação, posts nas últimas 24 h, próximo post."""
        sig = tuple((a["id"], a.get("on", True)) for a in accs)
        if sig != self._acc_sig:
            for w in self.acc_box.winfo_children():
                w.destroy()
            self._acc_rows = {}
            for i, a in enumerate(accs):
                row = tk.Frame(self.acc_box, bg=CARD2)
                row.pack(fill="x", pady=(0 if i == 0 else 3, 0))
                dot = tk.Label(row, text="●", bg=CARD2, fg=MUTED, font=(FONT, 11))
                dot.pack(side="left", padx=(10, 6))
                name = tk.Label(row, text="", bg=CARD2, fg=TEXT, font=(FONT, 10, "bold"), anchor="w", width=22)
                name.pack(side="left")
                for text, cmd in [("Remover", lambda a=a: self._acc_remove(a)),
                                  ("Ativar" if not a.get("on", True) else "Pausar", lambda a=a: self._acc_toggle(a)),
                                  ("Reconectar", lambda a=a: self._login(a))]:
                    Button(row, text, cmd, bg=CARD, font=(FONT, 8), pady=1, padx=8).pack(side="right", padx=(0, 6),
                                                                                       pady=3)
                info = tk.Label(row, text="", bg=CARD2, fg=MUTED, font=(FONT, 9), anchor="w")
                info.pack(side="left", fill="x", expand=True, padx=(8, 0))
                self._acc_rows[a["id"]] = (dot, name, info)
            self._acc_sig = sig
        per = ap.daily_limit()
        for a in accs:
            dot, name, info = self._acc_rows[a["id"]]
            name.config(text=accounts.label(a)[:30])
            n24 = len(ap.posts_last_24h(st, a["id"]))
            if not a.get("on", True):
                dot.config(fg=DIM)
                info.config(text="pausado -- não recebe posts", fg=DIM)
            elif a["id"] in down:
                dot.config(fg=RED)
                info.config(text="login caiu -- clique em Reconectar", fg="#fca5a5")
            else:
                braked = ap.reach.braked(st, a["id"])
                dot.config(fg=AMBER if braked else GREEN)
                nxt = f" · próximo {_when(max(when[a['id']], time.time()))}" if a["id"] in when and self._is_on() else ""
                info.config(text=f"{n24}/{per} em 24 h{nxt}" + (" · freio de alcance" if braked else ""),
                            fg=AMBER if braked else MUTED)
        if not accs:
            self.acc_hint.config(text="Conecte o canal do YouTube: o navegador abre, você escolhe a conta e o canal.")
        elif len(accs) == 1:
            self.acc_hint.config(text="Mais canais: o piloto alterna entre eles (cada clipe vai pra um canal só).")
        else:
            self.acc_hint.config(text=f"{len([a for a in accs if a.get('on', True)])} canais ativos: o piloto alterna "
                                      "entre eles. Cada clipe vai pra um canal só.")

    def _posts_step(self, d: int):
        n = int(min(max(ap.daily_limit() + d, 1), 24))
        control.set_posts_per_day(n)
        self._show_posts()
        self._fill_stats()

    def _show_posts(self):
        """Valor e explicação do "Shorts por dia" (a fila pode mudar por fora)."""
        n = ap.daily_limit()
        n_acc = len(accounts.active())

        def fmt(sec):
            mins = int(round(sec / 60))
            return f"{mins // 60} h" + (f" {mins % 60} min" if mins % 60 else "") if mins >= 60 else f"{mins} min"
        gap = fmt(ap.post_gap_seconds())
        skip = getattr(config, "AUTOPILOT_SKIP_HOURS", None)
        skip_txt = f" · nada das {skip[0]}h às {skip[1]}h" if skip else ""
        self.posts_val.config(text=str(n))
        if n_acc > 1:
            self.posts_label.config(text="Shorts por dia (cada canal):")
            self.posts_hint.config(text=f"{n_acc} canais × {n} = {n * n_acc} por dia · 1 post a cada "
                                        f"{fmt(ap.global_gap_seconds(n_acc))}, alternando (cada canal: 1 a cada {gap})"
                                        + skip_txt
                                        + (" · cuidado: acima de 6 por canal o alcance caiu" if n > 6 else ""),
                                   fg=AMBER if n > 6 else MUTED)
            return
        self.posts_label.config(text="Shorts por dia:")
        if n > 6:
            # achado real (9-10/out): com 1 por hora todos os Shorts ficaram com 0-6 views
            self.posts_hint.config(text=f"1 a cada {gap}{skip_txt} · cuidado: acima de 6 o canal parou de pegar views",
                                   fg=AMBER)
        else:
            self.posts_hint.config(text=f"1 a cada {gap}{skip_txt}"
                                        + (" · recomendado" if n == 6 else " · recomendado: 6"), fg=MUTED)

    def _open_last(self):
        if getattr(self, "_last_id", None):
            webbrowser.open(f"https://youtube.com/shorts/{self._last_id}")

    def _open_sheet(self):
        if getattr(self, "_sheet", None):
            _open(self._sheet)

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

    def _login(self, acc: dict = None):
        """Conecta um canal: o Google mostra a lista de contas/canais. Canal
        novo entra na lista; canal que já está nela só troca o login."""
        if self._login_proc is not None and self._login_proc.poll() is None:
            messagebox.showinfo("Conectar canal", "Já tem uma janela de login aberta no navegador: termine "
                                "(ou feche) aquela primeiro.")
            return
        try:
            proc = _spawn(["--login"], capture=True)
        except OSError as e:
            messagebox.showerror("Auto Clipper", str(e))
            return
        self._login_proc = proc
        who = f" \"{accounts.label(acc)}\"" if acc else ""
        self._note(f"Abrindo o navegador pra conectar o canal{who}: escolha a conta e o CANAL certo na lista "
                   "do Google e autorize...")

        def wait():
            out = proc.communicate()[0] or ""
            lines = [ln for ln in out.splitlines() if ln.strip() and not ln.startswith("=")]
            msg = "\n".join(lines[-6:]) or "Terminou."
            ok = "Conectado ao canal" in out
            if ok and control.running_info():
                msg += "\n\nO piloto ligado já pega o login novo e começa a postar nele em instantes."
            self.root.after(0, lambda: ((messagebox.showinfo if ok else messagebox.showwarning)("Conectar canal", msg),
                                        self._fill_stats()))
        threading.Thread(target=wait, daemon=True).start()

    def _acc_toggle(self, acc: dict):
        on = not acc.get("on", True)
        accounts.set_on(acc["id"], on)
        self._note(f"Canal \"{accounts.label(acc)}\" " + ("ativado: volta a receber posts." if on else
                                                             "pausado: não recebe posts até ativar de novo."))
        self._fill_stats()

    def _acc_remove(self, acc: dict):
        name = accounts.label(acc)
        if not messagebox.askyesno("Remover canal", f"Tirar \"{name}\" da lista?\n\nO piloto para de postar nele e "
                                   "o login dele é apagado deste PC. Os vídeos já postados continuam no canal."):
            return
        accounts.remove(acc["id"])
        self._note(f"Canal \"{name}\" removido da lista.")
        self._fill_stats()

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
