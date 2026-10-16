"""
Piloto automático: acha vídeo -> baixa -> edita -> posta no YouTube, sem parar.

Desenho (pensado pra rodar dias seguidos sem ninguém olhando):
  - cada vídeo é editado num PROCESSO SEPARADO (main.py como "operário"),
    com tempo máximo: se algo travar (ffmpeg, driver da GPU, whisper), o
    processo é morto e o piloto segue pro próximo vídeo; um erro num vídeo
    nunca derruba o loop.
  - tudo que importa fica salvo em autopilot_data/state.json (fila de
    clipes prontos, o que já foi postado, vídeos já usados, cota do dia):
    desligou o PC no meio, ao abrir de novo continua de onde parou.
  - postagem espaçada (AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS) e só dentro do
    horário configurado; fora dele continua produzindo clipes pra fila.
  - respeita a cota diária da API do YouTube; acabou a cota, espera zerar
    (meia-noite do horário do Pacífico).
  - vários canais (src/accounts.py): alterna entre eles, cada um no ritmo
    dele (4 por dia em cada com vários canais: 6 canais = 1 post por hora).
  - mantém o PC acordado enquanto roda (Windows).
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import config
from . import reach
from .accounts import FIRST, label, posted_by, sub

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "autopilot_data"


# ---------------------------------------------------------------- log ---
class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def __call__(self, msg: str = ""):
        line = f"[{datetime.now():%d/%m %H:%M:%S}] {msg}" if msg else ""
        with self._lock:
            # a tela é só um espelho do arquivo: saída sem console/quebrada
            # (piloto aberto pela interface) nunca derruba o piloto
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                try:
                    print(line.encode("ascii", "replace").decode(), flush=True)
                except Exception:
                    pass
            except Exception:
                pass
            try:
                if self.path.exists() and self.path.stat().st_size > 20 * 1024 * 1024:
                    self.path.replace(self.path.with_suffix(".old.log"))
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


# -------------------------------------------------------------- estado ---
def _pacific_day(now: Optional[float] = None) -> str:
    """A cota da API do YouTube zera à meia-noite do horário do Pacífico."""
    ts = now or time.time()
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(ts, ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:  # Windows sem o pacote tzdata: aproxima com UTC-8
        return datetime.fromtimestamp(ts, timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


def _next_pacific_midnight(now: Optional[float] = None) -> float:
    ts = now or time.time()
    day = _pacific_day(ts)
    t = ts
    while _pacific_day(t) == day:
        t += 600
    return t + 300  # folga de 5 min


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"sources": {}, "queue": [], "posted": [], "quota": {"day": "", "uploads": 0},
                     "blocked_until": 0, "last_post_at": 0}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                path.replace(path.with_suffix(".corrompido.json"))

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        text = json.dumps(self.data, ensure_ascii=False, indent=1)
        # troca atômica: nunca fica um arquivo pela metade. No Windows a troca
        # dá "Access is denied" se outro programa está com o arquivo aberto
        # naquele instante (a interface lendo, antivírus) -- achado real
        # (14/out, WinError 5 em state.tmp -> state.json): tenta de novo.
        for i in range(40):
            try:
                tmp.write_text(text, encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except PermissionError:
                if i == 39:
                    raise
                time.sleep(0.25)

    # --- vídeos de origem
    def seen_ids(self) -> set:
        now = time.time()
        out = set()
        for vid, s in self.data["sources"].items():
            # falhou poucas vezes e faz tempo: pode tentar de novo
            if s.get("status") == "failed" and s.get("attempts", 0) < 2 and now - s.get("at", 0) > 12 * 3600:
                continue
            out.add(vid)
        return out

    def recent_channels(self) -> dict:
        now, out = time.time(), {}
        for s in self.data["sources"].values():
            if s.get("status") == "done" and now - s.get("at", 0) < 24 * 3600 and s.get("channel"):
                out[s["channel"]] = out.get(s["channel"], 0) + 1
        return out

    def mark_source(self, vid: str, status: str, **kw):
        s = self.data["sources"].setdefault(vid, {})
        s.update(kw, status=status, at=time.time(), attempts=s.get("attempts", 0) + (status == "failed"))
        self.save()

    # --- cota
    def uploads_today(self) -> int:
        q = self.data["quota"]
        if q.get("day") != _pacific_day():
            q.update(day=_pacific_day(), uploads=0)
        return q["uploads"]

    def count_upload(self):
        self.uploads_today()
        self.data["quota"]["uploads"] += 1
        self.data["last_post_at"] = time.time()
        self.save()


# ------------------------------------------------------- PC acordado ---
def keep_awake(on: bool = True):
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
    except Exception:
        pass


# -------------------------------------------------------- operário ---
class YoutubeBlocked(Exception):
    """O YouTube recusou o download (anti-robô / excesso de pedidos). Não é
    culpa do vídeo: o piloto pausa os downloads em vez de queimar a lista."""


_BLOCK_MARKERS = ("Sign in to confirm you", "HTTP Error 429", "Too Many Requests")


class Interrupted(Exception):
    """Edição interrompida porque o modo mudou pra SÓ POSTAR (não é falha do vídeo)."""


_PRIO = {"low": None}
_AUTO = {"on": False}
# login do YouTube caído: o piloto acorda assim que o arquivo do login do
# canal mudar (botão "Reconectar" da interface), sem esperar 1 h. Canal novo
# adicionado com o piloto ligado também acorda (contas.json mudou).
_AUTH = {}            # canal -> data do arquivo de login quando caiu
_DEAD = {}            # canal -> o último erro foi login morto (False = rede etc.: tenta em 10 min)
_REG = {"mtime": None}


def _token_mtime(acc: str = FIRST):
    from . import accounts
    a = accounts.get(acc)
    try:
        return accounts.token_file(a).stat().st_mtime if a else None
    except OSError:
        return None


def _auth_lost(state, log: Log, msg: str, acc: str = FIRST) -> None:
    """Login morto: avisa (log + interface) e espera um login novo. No log,
    o mesmo aviso do mesmo canal no máximo a cada 6 h."""
    _AUTH[acc] = _token_mtime(acc)
    if state is None:
        log(f"  [!] YouTube [{label(acc)}]: {msg}")
        return
    me = sub(state, acc)
    old = me.get("auth_down") or {}
    # login trocado depois do aviso e ainda ruim: aviso novo (a interface
    # esconde o aviso mais velho que o arquivo do login)
    fresh = (_AUTH[acc] or 0) > old.get("at", 0)
    if old.get("msg") != msg or fresh or time.time() - old.get("logged", 0) > 6 * 3600:
        log(f"  [!] YouTube [{label(acc)}]: {msg}")
        me["auth_down"] = {"at": time.time() if old.get("msg") != msg or fresh else old.get("at"), "msg": msg,
                           "logged": time.time()}
        state.save()


def _auth_back(state, acc: str = FIRST) -> None:
    _AUTH.pop(acc, None)
    if state is not None and sub(state, acc).pop("auth_down", None) is not None:
        state.save()


def _relogged(acc: Optional[str] = None) -> bool:
    """Algum canal caído ganhou login novo (ou entrou canal novo na lista)?"""
    from . import accounts
    if acc is not None:
        return acc in _AUTH and _token_mtime(acc) != _AUTH[acc]
    return any(_token_mtime(a) != m for a, m in list(_AUTH.items())) or \
        (_REG["mtime"] is not None and accounts.mtime() != _REG["mtime"])


def low_priority(on: bool) -> None:
    """Modo só postar: o piloto fica com prioridade baixa no PC (jogo,
    navegador e o resto passam na frente). Volta ao normal no modo completo
    (no Windows a edição herdaria a prioridade baixa)."""
    if _PRIO["low"] == on:
        return
    _PRIO["low"] = on
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000 if on else 0x20)  # BELOW_NORMAL / NORMAL
        else:
            os.setpriority(os.PRIO_PROCESS, 0, 10 if on else 0)
    except Exception:
        pass


class LocalFailure(Exception):
    """A edição falhou por um problema no PC (ex.: modelo do whisper.cpp que
    não carrega), não por causa do vídeo: o vídeo não é descartado."""


# achado real: modelo do whisper.cpp baixado pela metade derrubou TODOS os
# vídeos de uma noite, e cada um foi marcado como "falhou"
_LOCAL_MARKERS = ("failed to initialize whisper context", "failed to load model")


def run_worker(url: str, n_clips: int, out_dir: Path, log: Log, lang: Optional[str] = None,
               game: bool = False) -> Optional[list]:
    """Roda main.py num processo separado pra editar um vídeo. Devolve a
    lista de clipes gerados: None se a edição falhou, [] se terminou bem mas
    nenhum trecho prestou (ex.: o juiz de IA reprovou todos)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "resultado.json"
    if results.exists():
        results.unlink()
    cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--url", url, "--clips", str(n_clips),
           "--out", str(out_dir), "--results", str(results)]
    if lang:
        cmd += ["--lang", lang]  # canal em outro idioma (ex.: "...|en" na lista de canais)
    if game:
        cmd += ["--game"]  # canal de gameplay ("...|game"): câmera streamer x jogo
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, **kw)

    seen_block, seen_local = [], []

    def _pump():
        repeated = 0
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if any(m in line for m in _LOCAL_MARKERS):
                seen_local.append(line)
            if any(m in line for m in _BLOCK_MARKERS):
                seen_block.append(line)
                repeated += 1
                if repeated > 3:
                    continue  # não enche o log com o mesmo erro repetido
            if line.strip() and not line.lstrip().startswith(("File \"", "~~~", "^^^", "...<", "raise ", "run(cmd)")):
                log(f"    | {line}")
    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    from .throttle import slowest_factor
    # com a GPU limitada (modo 25/50/70%) a edição demora mais: o limite
    # cresce junto (e o modo pode ser trocado no meio, pela interface)
    base = getattr(config, "AUTOPILOT_WORKER_TIMEOUT_MINUTES", 150) * 60
    started = time.time()
    try:
        # espera em passos curtos: no Windows um wait() longo segura o Ctrl+C
        # até a edição acabar (achado real: o piloto "desligou sozinho" logo
        # depois de terminar um vídeo -- era um Ctrl+C apertado bem antes)
        from . import control
        while proc.poll() is None:
            if _AUTO["on"] and control.post_only():
                log("    Modo SÓ POSTAR ligado: parando esta edição (o vídeo volta pra lista).")
                _kill_tree(proc)
                raise Interrupted()
            timeout = base * slowest_factor()
            if time.time() - started > timeout:
                log(f"    [!] edição passou de {timeout // 60:.0f} min -- encerrando esse vídeo e seguindo.")
                _kill_tree(proc)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        log("    Parando a edição em andamento...")
        _kill_tree(proc)
        raise
    reader.join(timeout=10)
    if not results.exists():
        if seen_block:
            raise YoutubeBlocked(seen_block[0].strip()[:200])
        if seen_local:
            raise LocalFailure("o whisper.cpp não conseguiu carregar o modelo de transcrição")
        log(f"    [!] a edição terminou sem gerar clipes (código {proc.returncode}).")
        return None
    try:
        clips = json.loads(results.read_text(encoding="utf-8")).get("clips", [])
    except ValueError:
        return None
    return [c for c in clips if Path(c["video"]).exists() and Path(c["meta"]).exists()]


def _kill_tree(proc: subprocess.Popen):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass


# ------------------------------------------------------------ disco ---
def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / 1024 ** 3
    except OSError:
        return 999.0


def make_room(state: State, out_root: Path, log: Log) -> bool:
    """Apaga clipes JÁ POSTADOS (mais antigos primeiro) se o disco estiver
    acabando. False = continua sem espaço (não produz mais até liberar)."""
    need = getattr(config, "AUTOPILOT_MIN_FREE_GB", 5)
    if free_gb(out_root) >= need:
        return True
    # primeiro os já públicos; os ainda privados guardam o arquivo pra um
    # possível repost depois da auditoria (src/release.py)
    for p in sorted(state.data["posted"], key=lambda x: (x.get("privacy") != "public", x.get("at", 0))):
        for key in ("video", "video_sem_musica"):
            f = p.get(key)
            if f and Path(f).exists():
                Path(f).unlink()
                log(f"    disco quase cheio: apaguei {Path(f).name} (já postado)")
        if free_gb(out_root) >= need:
            return True
    log(f"    [!] menos de {need} GB livres no disco e nada mais pra apagar -- pausando a produção.")
    return False


# -------------------------------------------------------------- postar ---
def _hour_ok(h: int) -> bool:
    """Posta nessa hora? AUTOPILOT_POST_HOURS (janela do dia) menos
    AUTOPILOT_SKIP_HOURS (v47: o vídeo que o usuário mandou aplicar diz que a
    maioria posta ao meio-dia e à tarde; horário fora do comum entrega mais
    em canal novo)."""
    start, end = getattr(config, "AUTOPILOT_POST_HOURS", (0, 24))
    ok = start <= h < end if start <= end else (h >= start or h < end)
    skip = getattr(config, "AUTOPILOT_SKIP_HOURS", None)
    if ok and skip:
        a, b = skip
        ok = not (a <= h < b if a <= b else (h >= a or h < b))
    return ok


def _in_post_window(ts: Optional[float] = None) -> bool:
    return _hour_ok(datetime.fromtimestamp(ts if ts is not None else time.time()).hour)


def _next_open(ts: float) -> float:
    """A primeira hora em que pode postar, a partir de `ts`."""
    if _in_post_window(ts):
        return ts
    t = datetime.fromtimestamp(ts).replace(minute=0, second=0, microsecond=0)
    for _ in range(48):
        t += timedelta(hours=1)
        if _hour_ok(t.hour):
            return t.timestamp()
    return ts


def daily_limit() -> int:
    """Shorts por dia EM CADA CANAL: o que foi escolhido na interface
    (autopilot_data/posts_por_dia.txt) ou AUTOPILOT_POSTS_PER_DAY (6). Já foi
    24 num canal só por pedido do usuário; voltou a 6 porque com 1 post por
    hora o YouTube parou de entregar os Shorts (src/reach.py). Com vários
    canais (src/accounts.py) o padrão vira 4 em cada: 6 canais = 24 por dia,
    1 por hora no total."""
    from . import control
    want = control.posts_per_day() or getattr(config, "AUTOPILOT_POSTS_PER_DAY", 6)
    return max(want, 0)


def total_per_day(n_channels: Optional[int] = None) -> int:
    """Shorts por dia somando todos os canais ligados."""
    if n_channels is None:
        from . import accounts
        n_channels = len(accounts.active())
    return daily_limit() * max(n_channels, 1)


def _window_hours() -> float:
    return float(sum(1 for h in range(24) if _hour_ok(h)) or 24)


def _base_gap() -> float:
    minimum = getattr(config, "AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS", 60) * 60
    limit = daily_limit()
    spread = _window_hours() * 3600 / limit if limit else minimum
    return max(minimum, spread)


def post_gap_seconds(state=None, acc: str = FIRST) -> float:
    """Intervalo entre posts NO MESMO CANAL: espalha o limite do dia pela
    janela de postagem inteira (com 6/dia em 24h: 1 a cada 4h; com 4/dia: 1
    a cada 6h), nunca menos que AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS. Sem
    isso, "1 por hora" com cota de 6 postava tudo de madrugada e parava.
    Com o freio de alcance do canal ligado (src/reach.py), no máximo 1 a
    cada AUTOPILOT_REACH_BRAKE_HOURS. `state`: State ou o dict do state.json."""
    gap = _base_gap()
    if state is not None and reach.braked(state, acc):
        gap = max(gap, reach.brake_hours() * 3600)
    return gap


def global_gap_seconds(n_live: int) -> float:
    """Entre posts de canais DIFERENTES: os canais se revezam dentro do
    intervalo de cada um (6 canais com 1 post a cada 6 h em cada = 1 post
    por hora no total, alternando)."""
    return _base_gap() / max(n_live, 1)


def posts_last_24h(data, acc: str) -> list:
    now = time.time()
    return sorted(p.get("at", 0) for p in posted_by(data, acc)
                  if not p.get("found_on_channel") and now - p.get("at", 0) < 86400)


def account_next(data, acc: str) -> float:
    """Quando o canal pode postar de novo, pelo ritmo dele (sem contar os outros)."""
    me = sub(data, acc)
    t = max(me.get("last_post_at", 0) + post_gap_seconds(data, acc), me.get("blocked_until", 0))
    recent, limit = posts_last_24h(data, acc), daily_limit()
    if limit <= 0:
        return time.time() + 86400
    if len(recent) >= limit:  # já postou o limite nas últimas 24 h (ex.: link colado na mão)
        t = max(t, recent[-limit] + 86400)
    return t


def next_slot(data, live: list) -> tuple:
    """(canal, hora) do próximo post: o canal que espera há mais tempo, sem
    dois posts mais perto que global_gap_seconds. `live`: ids dos canais com
    login funcionando."""
    d = getattr(data, "data", data)
    now = time.time()
    if not live:
        return None, now + 1800
    best = min(live, key=lambda a: (account_next(d, a), sub(d, a).get("last_post_at", 0)))
    t = max(account_next(d, best), d.get("last_post_at", 0) + global_gap_seconds(len(live)),
            d.get("blocked_until", 0))
    if not _in_post_window(t):
        t = _next_open(t)  # fora do horário (ou nas horas puladas): na primeira hora livre
    return best, t


def projected(data, live: list) -> dict:
    """{canal: hora do próximo post dele} simulando o revezamento (pra
    interface mostrar a ordem certa: só um canal posta de cada vez)."""
    import copy
    d = getattr(data, "data", data) or {}
    now = time.time()
    sim = {"accounts": copy.deepcopy(d.get("accounts", {})), "last_post_at": d.get("last_post_at", 0),
           "blocked_until": d.get("blocked_until", 0),
           "posted": [dict(p) for p in d.get("posted", []) if now - p.get("at", 0) < 86400]}
    out = {}
    for _ in range(2 * len(live)):
        acc, t = next_slot(sim, live)
        if acc is None:
            break
        out.setdefault(acc, t)
        if len(out) == len(live):
            break
        sub(sim, acc)["last_post_at"] = t
        sim["last_post_at"] = t
        sim["posted"].append({"account": acc, "at": t})
    return out


def can_post_now(state: State, ignore_schedule: bool = False, live: Optional[list] = None) -> tuple:
    """(pode?, hora da próxima tentativa) -- algum canal pode postar agora?"""
    now = time.time()
    if now < state.data.get("blocked_until", 0):
        return False, state.data["blocked_until"]
    if ignore_schedule:
        return True, now
    if live is None:
        from . import accounts
        live = [a["id"] for a in accounts.active()] or [FIRST]
    _, t = next_slot(state, live)
    return t <= now, t


def _prio(c: dict) -> int:
    """Prioridade de postagem: 2 = link colado na mão, 1 = repostagem de vídeo
    travado (filas antigas marcavam isso só com score 10**5/10**6)."""
    if c.get("priority"):
        return int(c["priority"])
    sc = c.get("score", 0) or 0
    return 2 if sc >= 10 ** 6 else 1 if sc >= 10 ** 5 else 0


# ------------------------------------------------- banco de clipes ---
def _channel_of(state: State, item: dict) -> str:
    src = state.data["sources"].get(item.get("source_id") or "", {})
    return item.get("channel") or src.get("channel") or item.get("source_id") or "?"


def _learned_weights(state: State) -> Optional[dict]:
    from .virality import current_weights
    learned = state.data.get("feedback", {}).get("weights")
    return current_weights(learned) if learned else None


def expire_queue(state: State, log: Log) -> int:
    """Clipe parado na fila há dias (sempre perdendo pros melhores) sai:
    abre espaço pra conteúdo novo."""
    days = getattr(config, "AUTOPILOT_QUEUE_MAX_AGE_DAYS", 4)
    if not days:
        return 0
    now = time.time()
    old = [c for c in state.data["queue"] if not _prio(c) and now - c.get("added", now) > days * 86400]
    if old:
        state.data["queue"] = [c for c in state.data["queue"] if c not in old]
        state.save()
        log(f"  Fila: {len(old)} clipe(s) com mais de {days} dias sem ser escolhido(s) saíram "
            "(abre espaço pra conteúdo novo).")
    return len(old)


def pool_ready(state: State, log: Log) -> bool:
    """Hora de postar: a fila já tem clipes de podcasts diferentes o bastante
    pra escolher o mais viral? Pedido do usuário: gerar vários clipes de
    vários podcasts no tempo em que não pode postar e postar o melhor -- não
    o primeiro que ficou pronto. Se o banco não enche (YouTube bloqueando
    download, por exemplo), posta o melhor que tiver depois de
    AUTOPILOT_POOL_MAX_WAIT_HOURS."""
    queue = state.data["queue"]
    if any(_prio(c) for c in queue):
        return True  # link colado na mão / repostagem: já escolhidos
    need_n = getattr(config, "AUTOPILOT_POOL_MIN", 6)
    need_src = getattr(config, "AUTOPILOT_POOL_MIN_SOURCES", 3)
    n, srcs = len(queue), len({_channel_of(state, c) for c in queue})
    now = time.time()
    if n >= need_n and srcs >= need_src:
        return True
    since = state.data.get("pool_wait_since") or now
    if not state.data.get("pool_wait_since"):
        state.data["pool_wait_since"] = since
        state.save()
        limit = since + getattr(config, "AUTOPILOT_POOL_MAX_WAIT_HOURS", 3) * 3600
        log(f"  Hora de postar, mas a fila tem {n} clipe(s) de {srcs} podcast(s): junto pelo menos "
            f"{need_n} de {need_src} podcasts diferentes pra postar o mais viral "
            f"(no máximo até {datetime.fromtimestamp(limit):%H:%M}).")
    if now - since >= getattr(config, "AUTOPILOT_POOL_MAX_WAIT_HOURS", 3) * 3600:
        return True
    return False


def _pool_deadline(state: State) -> float:
    since = state.data.get("pool_wait_since")
    return since + getattr(config, "AUTOPILOT_POOL_MAX_WAIT_HOURS", 3) * 3600 if since else 0.0


def _diversity_penalty(state: State, item: dict, acc: str = FIRST) -> float:
    """Não posta o mesmo vídeo/podcast em sequência NO MESMO CANAL (achado
    real: os 2 últimos posts eram do mesmo vídeo do Market Makers)."""
    posted = posted_by(state, acc)
    if not posted:
        return 0.0
    ch, now = _channel_of(state, item), time.time()
    last = posted[-1]
    pen = 0.0
    if last.get("source_id") and last.get("source_id") == item.get("source_id"):
        pen += 12.0
    elif _channel_of(state, last) == ch:
        pen += 6.0
    pen += 3.0 * sum(1 for p in posted if now - p.get("at", 0) < 86400 and _channel_of(state, p) == ch)
    return pen


def _ai_compare(state: State, queue: list, key, log: Log) -> None:
    """Juiz de IA ligado: compara os finalistas (de podcasts diferentes)
    entre si antes de postar -- nota de potencial viral relativa."""
    from . import ai_judge
    if not ai_judge.enabled():
        return
    finalists, per = [], {}
    for c in sorted((c for c in queue if not _prio(c)), key=key):
        ch = _channel_of(state, c)
        if per.get(ch, 0) >= 2:
            continue
        per[ch] = per.get(ch, 0) + 1
        finalists.append(c)
        if len(finalists) >= getattr(config, "AI_COMPARE_FINALISTS", 5):
            break
    if len(finalists) < 2 or all("ai_compare" in c for c in finalists):
        return
    items = []
    for c in finalists:
        try:
            m = json.loads(Path(c["meta"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            m = {}
        items.append({"title": m.get("title", ""), "text": m.get("text") or m.get("description", ""),
                      "duration": (m.get("end", 0) or 0) - (m.get("start", 0) or 0),
                      "source": m.get("source_title") or _channel_of(state, c)})
    verdicts = ai_judge.compare(items)
    if not verdicts:
        return
    for c, v, it in zip(finalists, verdicts, items):
        c["ai_compare"], c["ai_compare_reason"] = v["score"], v["reason"]
        log(f"    [IA] {v['score']:3d}/100 \"{it['title'][:60]}\" -- {v['reason']}")
    state.save()


def _quota_hit(state: State, e, key, log: Log, acc: str = FIRST) -> None:
    """O YouTube recusou por limite. Limite do CANAL (uploadLimitExceeded):
    só esse canal para até a meia-noite do Pacífico. Cota do PROJETO da API:
    todos param e o que faltava pra meta do dia vai pra pasta "postar à mão"."""
    until = _next_pacific_midnight()
    day = _pacific_day()
    if "uploadLimitExceeded" in str(e):
        me = sub(state, acc)
        me["blocked_until"] = until
        state.save()
        log(f"    [!] {e} -- o canal \"{label(acc)}\" chegou no limite de uploads dele hoje; volto a postar "
            f"nele em {datetime.fromtimestamp(until):%d/%m %H:%M} (os outros canais continuam).")
        if me.get("limit_explained_day") != day:
            me["limit_explained_day"] = day
            state.save()
            log("        Esse é o limite de uploads do CANAL no dia (não da API). Canal verificado por "
                "telefone tem limite maior: youtube.com/verify")
        return
    state.data["blocked_until"] = until
    state.save()
    done = state.uploads_today()
    log(f"    [!] {e} -- {done} postado(s) hoje pela API; volto a postar em "
        f"{datetime.fromtimestamp(until):%d/%m %H:%M}.")
    if state.data.get("quota_explained_day") != day:
        state.data["quota_explained_day"] = day
        state.save()
        log("        A cota do projeto da API do YouTube acabou por hoje (o padrão é 100 uploads por dia "
            "por projeto, somando todos os canais). Se acontecer sempre, peça aumento (é grátis): "
            "https://support.google.com/youtube/contact/yt_api_form. Vários projetos pra somar cota é "
            "proibido pelas regras da API -- o programa não faz.")
    from . import manual_post
    manual_post.export(state, total_per_day() - done, day, key, log)


def post_next(state: State, service, log: Log, ignore_schedule: bool = False,
              strict: bool = False, acc: str = FIRST) -> bool:
    """Posta o melhor clipe da fila no canal `acc`. True se postou. `strict`
    (modo só postar): ordem pura da nota, do maior pro menor, sem a regra de
    variedade de podcast."""
    from . import youtube_uploader as yt
    from . import channel_check
    queue = state.data["queue"]
    if not queue:
        return False
    # clipes da fila antiga (antes da nota de qualidade): avalia agora. Os
    # com prioridade (repostagem de vídeo travado, link colado na mão) não
    # passam pelo filtro: já foram escolhidos.
    for it in list(queue):
        if "quality" not in it and not _prio(it) and not judge(state, it, log):
            queue.remove(it)
    state.save()
    if not queue:
        return False
    # os mais promissores ainda sem nota de viralidade: analisa agora (no
    # máximo 3, pra não atrasar a postagem -- o normal é já terem sido
    # analisados no tempo ocioso, ver score_idle)
    from .virality import rank_key
    pending = sorted((c for c in queue if "viral" not in c and not _prio(c)),
                     key=lambda c: -c.get("quality", 50))
    # só postar: a ordem tem que ser a da nota de verdade -- mede todos os
    # que faltam (uma vez por clipe, poucos segundos cada)
    for c in (pending if strict else pending[:3]):
        score_viral(state, c, log)
    # prioridade primeiro (repostagem / link colado na mão); depois a maior
    # nota combinada (viralidade + qualidade)
    weights = _learned_weights(state)

    # mesmo vídeo de origem postado há pouco: vai pro fim (achado: 3 Shorts do
    # mesmo vídeo em menos de 24 h deixavam o canal repetitivo). Vale também
    # no só postar -- a ordem da nota continua entre os outros
    gap = getattr(config, "AUTOPILOT_SAME_SOURCE_GAP_HOURS", 24) * 3600
    now = time.time()
    recent_src = {p.get("source_id") for p in posted_by(state, acc)
                  if gap and p.get("source_id") and now - p.get("at", 0) < gap}

    def key(c):
        div = 0.0 if strict else _diversity_penalty(state, c, acc)
        same = 1 if (c.get("source_id") in recent_src and not _prio(c)) else 0
        return (-_prio(c), same, -(rank_key(c, weights) - div), c.get("added", 0))
    _ai_compare(state, queue, key, log)
    queue.sort(key=key)
    item = queue[0]
    if not Path(item["video"]).exists():
        log(f"    arquivo sumiu, tirando da fila: {item['video']}")
        queue.pop(0)
        state.save()
        return False
    if _old_hook(item):
        log(f"    clipe antigo com o título no começo, tirando da fila: {Path(item['video']).name}")
        queue.pop(0)
        state.save()
        return False
    meta = json.loads(Path(item["meta"]).read_text(encoding="utf-8"))
    # já está no canal? (pedido do usuário: "sempre verifica no canal se já
    # postou o vídeo em questão" -- ver src/channel_check.py)
    # (repostagem de vídeo travado como privado tem o MESMO título de propósito)
    try:
        dup = None if item.get("replaces") else channel_check.already_posted(service, state, meta["title"], acc)
    except channel_check.CheckFailed as e:
        if str(e).startswith("quotaExceeded"):
            _quota_hit(state, e, key, log, acc)
            return False
        if yt.dead_login(e):
            raise yt.AuthError(yt.DEAD_LOGIN_HINT) from e
        sub(state, acc)["blocked_until"] = time.time() + 10 * 60
        state.save()
        log(f"    [!] não consegui conferir no canal se \"{meta['title']}\" já foi postado ({e}); "
            "confiro de novo em 10 min (não posto às cegas).")
        return False
    if dup:
        queue.pop(0)
        if not any(p.get("youtube_id") == dup["id"] for p in state.data["posted"]):
            state.data["posted"].append({"youtube_id": dup["id"], "title": meta["title"], "at": dup["at"],
                                         "source_id": item.get("source_id"), "video": item["video"],
                                         "channel": _channel_of(state, item), "found_on_channel": True,
                                         "account": dup.get("account") or acc})
        state.save()
        log(f"    \"{meta['title']}\" já está no canal ({dup['where']}): https://youtube.com/shorts/{dup['id']} "
            "-- tirei da fila, vou pro próximo.")
        return False
    if not _prio(item):
        others = len({_channel_of(state, c) for c in queue})
        log(f"  >> Escolhido entre {len(queue)} clipe(s) de {others} podcast(s): nota final "
            f"{rank_key(item, weights):.0f}" + (f" (IA: {item['ai_compare']}/100)" if "ai_compare" in item else ""))
    # capa: o quadro de mais impacto do clipe (src/thumbnail.py)
    from . import post_kit, thumbnail
    thumb = thumbnail.ensure(item["video"], item.get("meta"), log) \
        if getattr(config, "YOUTUBE_THUMBNAIL", True) else None
    # v47 (vídeo que o usuário mandou aplicar): título em maiúsculas, só
    # #shorts, descrição com palavras-chave; sobe PRIVADO e o próprio YouTube
    # deixa público depois de processar em HD
    up = post_kit.upgrade_meta(dict(meta))
    up["title"] = post_kit.youtube_title(meta["title"])
    delay = getattr(config, "YOUTUBE_PUBLISH_DELAY_MINUTES", 25)
    public_at = time.time() + delay * 60 if delay and getattr(config, "YOUTUBE_PRIVACY", "public") == "public" \
        else None
    publish_at = datetime.fromtimestamp(public_at, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z") \
        if public_at else None
    log(f"  >> Postando no YouTube [{label(acc)}]: \"{up['title']}\"")
    try:
        vid = yt.upload_video(service, item["video"], up, publish_at=publish_at, progress=log)
    except yt.QuotaExceeded as e:
        _quota_hit(state, e, key, log, acc)
        return False
    except yt.AuthError:
        raise
    except yt.UploadError as e:
        item["attempts"] = item.get("attempts", 0) + 1
        log(f"    [!] falhou ({e}); tentativa {item['attempts']}/3")
        if item["attempts"] >= 3:
            queue.pop(0)
            log("    desisti desse clipe.")
        sub(state, acc)["blocked_until"] = time.time() + 15 * 60
        state.save()
        return False
    queue.pop(0)
    if thumb is not None:
        try:
            yt.set_thumbnail(service, vid, thumb)
            log("    capa posta (quadro de impacto do clipe)")
        except yt.ThumbNotAllowed as e:
            me = sub(state, acc)
            if me.get("thumb_warned_day") != _pacific_day():
                me["thumb_warned_day"] = _pacific_day()
                log(f"    [capa] {e} (o vídeo saiu com a capa automática do YouTube)")
        except yt.UploadError as e:
            log(f"    [capa] não consegui pôr a capa ({e}); o vídeo saiu com a capa automática.")
    try:
        from .release import privacy_of
        privacy = privacy_of(service, [vid]).get(vid) or "?"
    except Exception:
        privacy = "?"
    if public_at and privacy in ("private", "?"):
        privacy = "scheduled"  # fica público sozinho em public_at (src/release.py confere)
    entry = {"youtube_id": vid, "title": meta["title"], "at": time.time(),
             "source_id": item.get("source_id"), "video": item["video"],
             "video_sem_musica": item.get("video_sem_musica"), "privacy": privacy,
             "channel": _channel_of(state, item), "account": acc}
    # a nota que ele teve, pra conferir com as views reais (src/feedback.py)
    for k in ("viral", "viral_parts", "quality", "ai_score", "ai_compare"):
        if k in item:
            entry[k] = item[k]
    state.data.pop("pool_wait_since", None)
    if item.get("replaces"):
        entry["replaces"] = item["replaces"]
    if privacy == "?":
        del entry["privacy"]  # confere depois
    if public_at:
        entry["publish_at"] = public_at
    state.data["posted"].append(entry)
    sub(state, acc)["last_post_at"] = time.time()
    state.count_upload()
    note = "" if privacy in ("public", "?") else f" [{privacy} -- libera sozinho quando a auditoria sair]"
    if privacy == "scheduled":
        note = f" [fica público às {datetime.fromtimestamp(public_at):%H:%M}, depois de processar em HD]"
    log(f"    postado: https://youtube.com/shorts/{vid}  "
        f"({len(posts_last_24h(state, acc))}/{daily_limit()} nas últimas 24 h nesse canal){note}")
    if getattr(config, "AUTOPILOT_DELETE_POSTED_FILES", False):
        for f in (item["video"], item.get("video_sem_musica")):
            if f and Path(f).exists():
                Path(f).unlink()
    return True


# ------------------------------------------------------------ produzir ---
def judge(state: State, item: dict, log: Log) -> bool:
    """Dá a nota de qualidade (src/quality.py). False = não vale postar."""
    from . import quality
    if "quality" in item:
        return item["quality"] >= quality.min_quality() and not item.get("reject")
    try:
        q, reasons, reject = quality.assess_file(item["video"], item["meta"])
    except Exception as e:  # avaliação nunca derruba o piloto
        log(f"    [!] não consegui avaliar o clipe ({e.__class__.__name__}: {e}) -- fica na fila")
        item["quality"] = 50
        return True
    item["quality"], item["quality_reasons"] = q, reasons
    item["reject"] = reject
    ok = not reject and q >= quality.min_quality()
    try:
        title = json.loads(Path(item["meta"]).read_text(encoding="utf-8")).get("title", "")
    except (OSError, ValueError):
        title = Path(item["video"]).name
    why = f" ({', '.join(reasons)})" if reasons else ""
    log(f"    {'✓' if ok else '✗ descartado'} nota {q}/100: \"{title}\"{why}")
    if not ok:
        state.data.setdefault("rejected", []).append(
            {"video": item["video"], "title": title, "quality": q, "reasons": reasons,
             "source_id": item.get("source_id"), "at": time.time()})
        state.data["rejected"] = state.data["rejected"][-200:]
    return ok


def score_viral(state: State, item: dict, log: Log) -> None:
    """Nota de viralidade de um clipe da fila (src/virality.py)."""
    from . import virality
    try:
        meta = json.loads(Path(item["meta"]).read_text(encoding="utf-8"))
        v, parts = virality.analyze(item["video"], meta, weights=_learned_weights(state))
        if meta.get("ai_score") is not None:
            item["ai_score"] = meta["ai_score"]
    except Exception as e:  # análise nunca derruba o piloto
        log(f"    [!] não consegui medir a viralidade ({e.__class__.__name__}: {e})")
        item["viral"] = 50
        state.save()
        return
    item["viral"], item["viral_parts"] = v, parts
    state.save()
    why = virality.summary(parts)
    log(f"    viralidade {v}/100 -> nota final {virality.rank_key(item, _learned_weights(state)):.0f}: "
        f"\"{meta.get('title', '')}\"" + (f" ({why})" if why else ""))


def score_idle(state: State, log: Log, until: float) -> int:
    """Tempo ocioso (esperando a hora de postar): analisa a viralidade dos
    clipes da fila que ainda não têm nota, os de melhor qualidade primeiro,
    até `until`. Devolve quantos analisou."""
    todo = sorted((c for c in state.data["queue"] if "viral" not in c and not _prio(c)),
                  key=lambda c: -c.get("quality", 50))
    if not todo:
        _idle_thumbs(state, until)
        return 0
    log(f"  >> Tempo livre: medindo a viralidade de {len(todo)} clipe(s) da fila...")
    done = 0
    for c in todo:
        if time.time() > until - 90:
            break
        if not Path(c["video"]).exists():
            c["viral"] = 0  # sumiu: não tenta de novo a cada pausa (sai da fila na hora de postar)
            state.save()
            continue
        score_viral(state, c, log)
        done += 1
    if done:
        from .virality import rank_key
        w = _learned_weights(state)
        best = max(state.data["queue"], key=lambda c: rank_key(c, w) - _diversity_penalty(state, c))
        try:
            t = json.loads(Path(best["meta"]).read_text(encoding="utf-8")).get("title", "")
        except (OSError, ValueError):
            t = Path(best["video"]).name
        log(f"    próximo a postar (mais viral da fila): \"{t}\" -- nota {rank_key(best, w):.0f}")
    _idle_thumbs(state, until)
    return done


def _idle_thumbs(state: State, until: float) -> None:
    """Tempo livre: já escolhe a capa (src/thumbnail.py) dos clipes da fila,
    pra não atrasar a hora do post. Sem log: a capa aparece no post."""
    if not getattr(config, "YOUTUBE_THUMBNAIL", True):
        return
    from . import thumbnail
    for c in list(state.data["queue"]):
        if time.time() > until - 120:
            return
        if Path(c["video"]).exists() and not thumbnail.thumb_path(c["video"]).exists():
            thumbnail.ensure(c["video"], c.get("meta"), log=lambda m: None)


def prune_blocked(state: State, log: Log) -> int:
    """Tira da fila os clipes de vídeos com palavra bloqueada
    (AUTOPILOT_BLOCK_WORDS) -- inclusive os feitos antes do bloqueio existir."""
    from .discovery import blocked
    keep, gone = [], []
    for c in state.data["queue"]:
        src = state.data["sources"].get(c.get("source_id") or "", {})
        try:
            title = json.loads(Path(c["meta"]).read_text(encoding="utf-8")).get("source_title")
        except (OSError, ValueError):
            title = None
        word = None if _prio(c) else blocked(src.get("title"), src.get("channel"), title)
        if not word and not _prio(c):
            ch = (src.get("channel") or "").lower()
            word = next((n for n in getattr(config, "AUTOPILOT_DROP_QUEUED_FROM", [])
                         if n and n.lower() in ch), None)
        (gone if word else keep).append((c, word))
    if gone:
        state.data["queue"] = [c for c, _ in keep]
        state.save()
        log(f"  Fila: {len(gone)} clipe(s) de vídeo bloqueado ou de canal que saiu da lista removido(s) "
            f"({', '.join(sorted({w for _, w in gone}))}).")
    return len(gone)


def _old_hook(c: dict) -> bool:
    """Clipe feito antes de desligar o título-gancho (balão + whoosh no
    arquivo, src/hook_check.py)? Confere uma vez só por clipe."""
    if getattr(config, "HOOK_ENABLED", False) or c.get("hook_ok"):
        return False
    from .hook_check import burned_hook
    video = c.get("video")
    meta = c.get("meta") or (str(Path(video).with_suffix(".meta.json")) if video else None)
    try:
        r = burned_hook(video, meta)
    except Exception:  # conferência nunca derruba o piloto
        r = None
    if r is False:
        c["hook_ok"] = True
    return bool(r)


def prune_hook_clips(state: State, log: Log) -> int:
    """Tira da fila os clipes antigos com o título-gancho no começo e não
    libera os vídeos privados antigos que têm ele (ficam privados)."""
    if getattr(config, "HOOK_ENABLED", False):
        return 0
    from .release import _pending
    gone = [c for c in state.data["queue"] if _old_hook(c)]
    if gone:
        state.data["queue"] = [c for c in state.data["queue"] if c not in gone]
    locked = []
    for p in _pending(state):
        if p.get("video") and Path(p["video"]).exists() and _old_hook(p):
            p["gave_up"] = p["old_hook"] = True
            locked.append(p)
    state.save()
    if gone:
        log(f"  Fila: {len(gone)} clipe(s) antigo(s) com o título no começo removido(s) "
            "(feitos antes de desligar o título; os novos saem sem).")
    if locked:
        log(f"  {len(locked)} vídeo(s) privado(s) antigo(s) com o título no começo não vão ser "
            "liberados (ficam privados; dá pra apagar no YouTube Studio).")
    return len(gone) + len(locked)


def produce(state: State, source: dict, out_root: Path, log: Log, n_clips: Optional[int] = None) -> int:
    vid = source["id"]
    n_clips = n_clips or getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3)
    log(f"  >> Editando: \"{source.get('title', vid)}\" -- {source['url']}")
    t0 = time.time()
    clips = run_worker(source["url"], n_clips, out_root / vid, log, lang=source.get("lang"),
                       game=bool(source.get("game")))
    if clips is None:
        state.mark_source(vid, "failed", title=source.get("title"), channel=source.get("channel"))
        return 0
    if not clips:
        # editou certo, mas nenhum trecho valia a pena: não tenta de novo
        log("    nenhum trecho deste vídeo valeu um corte -- próximo vídeo.")
        state.mark_source(vid, "done", title=source.get("title"), channel=source.get("channel"), clips=0)
        return 0
    kept = 0
    for c in clips:
        no_music = Path(c["video"]).with_name(Path(c["video"]).stem + "_sem_musica.mp4")
        item = {"video": c["video"], "meta": c["meta"], "score": c.get("score", 0),
                "source_id": vid, "added": time.time(),
                "video_sem_musica": str(no_music) if no_music.exists() else None}
        if judge(state, item, log):
            state.data["queue"].append(item)
            kept += 1
    state.mark_source(vid, "done", title=source.get("title"), channel=source.get("channel"),
                      clips=kept)
    log(f"    {kept} de {len(clips)} clipe(s) aprovados pra postar em {(time.time() - t0) / 60:.0f} min -- "
        f"fila: {len(state.data['queue'])}")
    return kept


# --------------------------------------------------------- manutenção ---
_REPAIR = {"at": 0.0}


def repair_models(log: Log) -> None:
    """Modelo do whisper.cpp baixado pela metade: baixa de novo em segundo
    plano (src/whisper_models.py). No máximo a cada 6 h."""
    if getattr(config, "TRANSCRIBE_ENGINE", "") != "whispercpp" or time.time() - _REPAIR["at"] < 6 * 3600:
        return
    _REPAIR["at"] = time.time()
    try:
        from . import whisper_models
        from .transcriber_whispercpp import _BETTER_MODELS
        if getattr(config, "WHISPERCPP_AUTO_BEST_MODEL", True):
            whisper_models.start_repair(config.WHISPERCPP_MODEL, _BETTER_MODELS, log)
    except Exception as e:  # manutenção nunca derruba o piloto
        log(f"  [modelo] não consegui conferir os modelos ({e.__class__.__name__}: {e})")


def update_ytdlp(state: State, log: Log) -> None:
    """Atualiza o yt-dlp uma vez por dia: o YouTube muda direto e versão
    velha falha com 403 / "unable to extract yt initial data" (log real)."""
    if not getattr(config, "AUTOPILOT_UPDATE_YTDLP", True):
        return
    today = datetime.now().strftime("%Y-%m-%d")
    if state.data.get("ytdlp_checked") == today:
        return
    state.data["ytdlp_checked"] = today
    state.save()
    cmd = [sys.executable, "-m", "pip", "install", "-U", "--quiet", "--disable-pip-version-check",
           "yt-dlp[default]"]
    try:
        before = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True,
                                text=True, timeout=60).stdout.strip()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        after = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"], capture_output=True,
                               text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        log(f"  [yt-dlp] não consegui atualizar agora ({e.__class__.__name__}).")
        return
    if r.returncode != 0:
        log("  [yt-dlp] não consegui atualizar agora (sem internet?); tento amanhã.")
    elif after and after != before:
        log(f"  [yt-dlp] atualizado: {before or '?'} -> {after}")


def _forgive_local_failures(state: State, log: Log) -> None:
    """Uma vez só: vídeos que "falharam" nas últimas 48 h podem ter caído
    pelo modelo de transcrição quebrado (não por culpa do vídeo) -- voltam
    pra lista."""
    if state.data.get("forgave_v19"):
        return
    now, n = time.time(), 0
    for vid, s in list(state.data["sources"].items()):
        if s.get("status") == "failed" and now - s.get("at", 0) < 48 * 3600:
            del state.data["sources"][vid]
            n += 1
    state.data["forgave_v19"] = True
    state.save()
    if n:
        log(f"  {n} vídeo(s) que falharam pelo modelo de transcrição quebrado voltaram pra lista.")


# ------------------------------------------------------------ loop ---
def _sleep_until(ts: float, log: Log, why: str):
    wait = max(ts - time.time(), 0)
    if wait <= 0:
        return
    if wait > 120:
        log(f"  .. {why} -- próxima ação às {datetime.fromtimestamp(ts):%H:%M}")
    from . import control
    mode = control.post_only()
    end = time.time() + wait
    while time.time() < end:
        # passos curtos: o botão "Desligar" da interface (src/control.py)
        # para o piloto em ~2 s mesmo no meio de uma espera longa; trocar
        # entre "só postar" e o modo completo acorda o piloto na hora
        time.sleep(max(min(2.0, end - time.time()), 0))
        if control.post_only() != mode or _relogged():
            return


def _get_service(log: Log, upload: bool, state=None, acc: Optional[dict] = None):
    """Conecta um canal (sem abrir navegador). None se o login não serve."""
    if not upload:
        return None
    from . import accounts
    from . import youtube_uploader as yt
    acc = acc or accounts.get(FIRST) or {"id": FIRST, "token": accounts.LEGACY_TOKEN}
    aid = acc["id"]
    try:
        svc = yt.get_service(interactive=False, token=accounts.token_file(acc))
        cid, name, problem = yt.channel_info(svc)
        if not problem:
            problem = accounts.learn(aid, cid, name)
        if problem:
            _auth_lost(state, log, problem, aid)
            return None
        log(f"  YouTube conectado: canal \"{name}\"")
        _auth_back(state, aid)
        return svc
    except yt.UploadError as e:
        _DEAD[aid] = yt.dead_login(e) or "sem login" in str(e) or "falta o arquivo" in str(e)
        _auth_lost(state, log, str(e), aid)
        return None


def _sync_services(svcs: dict, next_try: dict, state, log: Log, upload: bool) -> list:
    """Conecta os canais da lista (canal novo, login novo, uma vez por hora os
    caídos), renova o acesso dos conectados e solta os que saíram da lista
    ou foram pausados. Devolve os ids dos canais prontos pra postar."""
    from . import accounts
    from . import youtube_uploader as yt
    accs = accounts.active() if upload else []
    _REG["mtime"] = accounts.mtime()
    ids = [a["id"] for a in accs]
    for gone in [k for k in svcs if k not in ids]:
        svcs.pop(gone, None)
        _AUTH.pop(gone, None)
    now = time.time()
    for a in accs:
        aid = a["id"]
        if svcs.get(aid) is None:
            if now >= next_try.get(aid, 0) or _relogged(aid):
                if _relogged(aid):
                    log(f"  Login novo do canal \"{label(a)}\": conectando...")
                _DEAD[aid] = True
                svcs[aid] = _get_service(log, upload, state, a)
                next_try[aid] = now + (3600 if _DEAD.get(aid, True) else 600)
            continue
        # o acesso vence a cada hora: renova antes de usar (login morto ->
        # para de tentar nesse canal e avisa, em vez de errar em tudo)
        try:
            yt.ensure_fresh(svcs[aid])
        except yt.AuthError as e:
            _auth_lost(state, log, str(e), aid)
            svcs[aid] = None
            next_try[aid] = now + 3600
    return [aid for aid in ids if svcs.get(aid) is not None]


def run_forever(upload: bool = True):
    """Modo automático: roda até fechar a janela / desligar o PC / apertar
    "Desligar" na interface."""
    from . import control
    if not control.start("auto"):
        other = control.running_info() or {}
        print(f"\n[!] Já tem um piloto ligado neste PC (processo {other.get('pid', '?')}, aberto pela "
              "interface ou por outro atalho). Dois ao mesmo tempo bagunçariam a fila: este não vai ligar.")
        return
    try:
        _run_forever(upload)
    except KeyboardInterrupt:
        pass
    finally:
        control.finish()
        keep_awake(False)


def _run_forever(upload: bool = True):
    from . import discovery
    out_root = Path(getattr(config, "AUTOPILOT_OUTPUT_DIR", "output/autopiloto"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log = Log(DATA_DIR / "autopilot.log")
    state = State(DATA_DIR / "state.json")
    keep_awake(True)

    log("=" * 60)
    from .version import VERSION
    log(f"  PILOTO AUTOMÁTICO LIGADO ({VERSION}) -- pra parar: botão Desligar na interface "
        "(ou feche a janela do .bat)")
    log(f"  canais: {', '.join(getattr(config, 'AUTOPILOT_CHANNELS', [])) or '(nenhum)'}")
    log(f"  buscas: {', '.join(getattr(config, 'AUTOPILOT_SEARCHES', [])) or '(nenhuma)'}")
    from . import accounts
    accounts.migrate_state(state)  # antes da v46: um canal só
    accs = accounts.active() if upload else []
    log(f"  postagem: {'LIGADA' if upload else 'DESLIGADA (só gera os clipes)'} -- até "
        f"{daily_limit()}/dia{' em cada canal' if len(accs) > 1 else ''}, 1 a cada {_base_gap() / 60:.0f} min"
        f"{' no mesmo canal' if len(accs) > 1 else ''}, das {getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[0]}h "
        f"às {getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[1]}h")
    if len(accs) > 1:
        log(f"  canais do YouTube ({len(accs)}): {', '.join(label(a) for a in accs)}")
    log(f"  fila atual: {len(state.data['queue'])} clipe(s) | já postados: {len(state.data['posted'])}")
    for a in accs:
        b = reach.brake_of(state, a["id"])
        if b:
            log(f"  [alcance] FREIO LIGADO em \"{label(a)}\": os últimos Shorts tiveram "
                f"{', '.join(map(str, b.get('recent', [])))} views (antes ~{b.get('base', 0):.0f}) -- "
                f"1 post a cada {reach.brake_hours():.0f} h nesse canal até voltar.")
    log("=" * 60)

    from . import rights
    rights.migrate(state, log)  # banimentos permanentes antigos: liberados
    prune_blocked(state, log)
    prune_hook_clips(state, log)
    from . import manual_post
    manual_post.cleanup(log)
    _forgive_local_failures(state, log)
    from . import control
    if control.post_only():
        low_priority(True)
        log("  MODO SÓ POSTAR: não baixa nem edita nada -- só posta a fila, do maior score pro menor, "
            "conferindo no canal antes de cada post.")
    else:
        repair_models(log)
        update_ytdlp(state, log)
    from . import housekeeping
    housekeeping.clean_work_dir(log)
    housekeeping.cleanup(state, out_root, log)
    svcs, next_try = {}, {}  # canal -> serviço do YouTube (None = sem login agora)
    live = _sync_services(svcs, next_try, state, log, upload)
    if len(live) > 1:
        log(f"  {len(live)} canais conectados: {daily_limit()} Shorts por dia em cada, alternando "
            f"(1 post a cada {global_gap_seconds(len(live)) / 60:.0f} min no total).")
    cur = {"acc": None}  # canal em uso agora (erro de login inesperado = desse canal)
    failures = 0
    startup = True
    last_mode = None
    _AUTO["on"] = True  # só o modo automático para a edição ao trocar pra "só postar"
    from . import youtube_uploader as yt
    while True:
        try:
            live = _sync_services(svcs, next_try, state, log, upload)

            # 0) vídeos que subiram privados: confere a auditoria / solta aos poucos
            acted = False
            if live:
                from . import release
                for aid in live:
                    cur["acc"] = aid
                    try:
                        acted = release.release_step(state, svcs[aid], log, startup=startup, acc=aid) or acted
                    except yt.QuotaExceeded as e:
                        state.data["blocked_until"] = _next_pacific_midnight()
                        state.save()
                        log(f"  [!] {e}")
                        break
                startup = False
                if acted:
                    continue

            expire_queue(state, log)
            if live:
                # a nota de viralidade acerta? confere com as views reais (1x por dia;
                # views de vídeo público: qualquer login lê)
                from . import feedback
                cur["acc"] = live[0]
                feedback.refresh_views(state, svcs[live[0]], log)
                feedback.calibrate(state, log)
                # o vídeo postado está visível ou o Content ID bloqueou? (src/rights.py;
                # cada canal com o login dele)
                from . import rights
                for aid in live:
                    cur["acc"] = aid
                    rights.check_recent_posts(state, svcs[aid], log, aid)

            # modo SÓ POSTAR (src/control.py): lido a cada volta -- a interface
            # troca com o piloto ligado
            only_post = control.post_only()
            if only_post != last_mode:
                if last_mode is not None:
                    log("  >> Modo SÓ POSTAR: parei de buscar e editar vídeos; só posto a fila, do maior "
                        "score pro menor." if only_post else "  >> Modo completo: volto a buscar, editar e postar.")
                last_mode = only_post
            low_priority(only_post)

            # 1) postar, se estiver na hora de algum canal -- e se o banco de
            # clipes já tem opção de podcasts diferentes pra escolher o mais
            # viral (no só postar não chega clipe novo: posta o melhor que tiver)
            waiting_pool = False
            if live and state.data["queue"]:
                # o YouTube está entregando os últimos Shorts? (freio
                # automático de cada canal, src/reach.py -- confere a cada 2 h)
                for aid in live:
                    cur["acc"] = aid
                    reach.check(svcs[aid], state, log, acc=aid, name=label(aid) if len(live) > 1 else "")
                aid, nxt = next_slot(state, live)
                ok = aid is not None and nxt <= time.time()
                waiting_pool = ok and not only_post and not pool_ready(state, log)
                if ok and not waiting_pool:
                    cur["acc"] = aid
                    try:
                        post_next(state, svcs[aid], log, strict=only_post, acc=aid)
                    except yt.AuthError as e:
                        _auth_lost(state, log, str(e), aid)
                        svcs[aid] = None
                        next_try[aid] = time.time() + 3600
                    housekeeping.cleanup(state, out_root, log)
                    continue
            cur["acc"] = None

            # 2) produzir, se a fila estiver curta (e o YouTube não estiver
            # bloqueando downloads -- ver YoutubeBlocked). Com vários canais a
            # fila precisa de mais clipes (até 1 dia de posts).
            target = max(getattr(config, "AUTOPILOT_QUEUE_TARGET", 8), total_per_day(len(live)) if live else 0)
            local_until = state.data.get("local_fail_until", 0)
            blocked_until = max(state.data.get("download_blocked_until", 0), local_until)
            if (not only_post and len(state.data["queue"]) < target and time.time() >= blocked_until
                    and make_room(state, out_root, log) and housekeeping.has_room(state, out_root, log)):
                repair_models(log)
                update_ytdlp(state, log)
                retry = state.data.get("retry_source")
                if retry and time.time() - retry.get("at", 0) < 24 * 3600 \
                        and retry["source"]["id"] not in state.seen_ids():
                    # o download foi bloqueado: tenta o MESMO vídeo de novo, sem
                    # refazer a busca (cada busca são dezenas de consultas ao YouTube)
                    source = retry["source"]
                    log(f"  >> Tentando de novo o download de \"{source.get('title', source['id'])}\"...")
                else:
                    log("  >> Procurando vídeo bombando pra cortar...")
                    # variedade: canal que já tem clipe esperando na fila também
                    # perde prioridade (o banco precisa de podcasts diferentes)
                    recent = state.recent_channels()
                    for ch in {_channel_of(state, c) for c in state.data["queue"]}:
                        recent[ch] = recent.get(ch, 0) + 1
                    source = discovery.pick_source(state.seen_ids(), recent, log=log,
                                                   avoid_channels=rights.avoid_channels(state))
                state.data.pop("retry_source", None)
                if source:
                    try:
                        produce(state, source, out_root, log)
                        state.data["download_blocks"] = 0
                        state.save()
                        housekeeping.clean_work_dir(log)
                        housekeeping.cleanup(state, out_root, log)
                    except Interrupted:
                        housekeeping.clean_work_dir(log)
                        continue  # o vídeo não foi marcado: volta pra lista
                    except LocalFailure as e:
                        state.data["local_fail_until"] = time.time() + 30 * 60
                        state.save()
                        housekeeping.clean_work_dir(log)
                        _REPAIR["at"] = 0.0  # confere os modelos de novo já
                        repair_models(log)
                        log(f"  [!] Problema no PC: {e}. O vídeo não foi descartado; "
                            "tento de novo em 30 min (a postagem continua).")
                    except YoutubeBlocked as e:
                        state.data["retry_source"] = {"source": source, "at": time.time()}
                        n = state.data.get("download_blocks", 0) + 1
                        # 1ª pausa curta: no PC do usuário o bloqueio passou em ~10 min
                        wait = min(15 * 60 * 2 ** (n - 1), 6 * 3600)
                        state.data["download_blocks"] = n
                        state.data["download_blocked_until"] = time.time() + wait
                        state.save()
                        from .downloader import cookies_file
                        log(f"  [!] O YouTube bloqueou o download ({e}).")
                        log(f"      Pausando downloads por {wait // 60:.0f} min (a postagem continua). "
                            + ("Os cookies em credentials/youtube_cookies.txt podem ter vencido -- exporte de novo."
                               if cookies_file() else
                               "Resolve de vez com cookies de uma conta logada: veja 'Bloqueio do YouTube' no README."))
                    failures = 0
                    continue
                wake = time.time() + getattr(config, "AUTOPILOT_IDLE_MINUTES", 30) * 60
                score_idle(state, log, wake)
                _sleep_until(wake, log, "nada novo agora, procuro de novo depois")
                continue

            # 3) fila cheia (ou downloads pausados): espera a próxima postagem
            if only_post and not live:
                _sleep_until(time.time() + 1800, log, "só postar: postagem indisponível (confira os canais do "
                             "YouTube na interface)")
            elif only_post and not state.data["queue"]:
                _sleep_until(time.time() + 1800, log, "só postar: a fila acabou, nada pra postar (ligue um "
                             "modo de GPU pra fazer mais clipes)")
            elif time.time() < blocked_until and not (live and state.data["queue"]):
                _sleep_until(blocked_until, log, "edição pausada (problema no PC)" if blocked_until == local_until
                             else "downloads pausados pelo bloqueio do YouTube")
            elif live and state.data["queue"]:
                nxt_acc, nxt = next_slot(state, live)
                if waiting_pool:
                    nxt = _pool_deadline(state)  # posta o melhor que tiver nessa hora
                wake = min(nxt, time.time() + 3600)
                if time.time() < blocked_until and not only_post:
                    wake = min(wake, blocked_until)  # volta a baixar assim que a pausa acabar
                if not only_post:
                    score_idle(state, log, wake)
                where = f" no canal \"{label(nxt_acc)}\"" if len(live) > 1 and nxt_acc else ""
                _sleep_until(wake, log,
                             f"fila com {len(state.data['queue'])} clipe(s); próxima postagem às "
                             f"{datetime.fromtimestamp(max(nxt, time.time())):%H:%M}{where}")
            else:
                _sleep_until(time.time() + 1800, log,
                             "fila cheia e postagem indisponível (confira os canais do YouTube na interface)")
        except KeyboardInterrupt:
            from . import control
            log("  Piloto automático desligado" + (" (botão Desligar)." if control.stop_requested() else " (Ctrl+C)."))
            keep_awake(False)
            return
        except Exception as e:  # nada derruba o loop
            aid = cur["acc"]
            if aid and svcs.get(aid) is not None and yt.dead_login(e):
                _auth_lost(state, log, yt.DEAD_LOGIN_HINT, aid)
                svcs[aid] = None
                next_try[aid] = time.time() + 3600
                continue
            failures += 1
            wait = min(60 * 2 ** failures, 3600)
            log(f"  [!] erro inesperado: {e.__class__.__name__}: {e} -- tentando de novo em {wait // 60:.0f} min")
            time.sleep(wait)


def run_once(url: str, n_clips: int, upload: bool = True):
    """Modo manual: um link -> edita -> posta os clipes na hora."""
    from . import control
    if not control.start("link"):
        print("\n[!] O piloto automático está ligado: desligue ele antes de cortar um link "
              "(os dois mexem na mesma fila).")
        return
    try:
        _run_once(url, n_clips, upload)
    except KeyboardInterrupt:
        Log(DATA_DIR / "autopilot.log")("  Corte do link interrompido.")
    finally:
        control.finish()
        keep_awake(False)


def _run_once(url: str, n_clips: int, upload: bool = True):
    out_root = Path(getattr(config, "AUTOPILOT_OUTPUT_DIR", "output/autopiloto"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    log = Log(DATA_DIR / "autopilot.log")
    state = State(DATA_DIR / "state.json")
    keep_awake(True)
    svcs, live = {}, []
    if upload:
        from . import accounts
        accounts.migrate_state(state)
        if not accounts.load():  # nenhum canal ainda: conecta o primeiro agora
            try:
                accounts.login(out=log)
            except Exception as e:
                log(f"  [!] YouTube: {e}")
        live = _sync_services(svcs, {}, state, log, upload)
        if not live:
            log("  Vou gerar os clipes mesmo assim; eles ficam na fila pra postar depois.")
    vid = _video_id(url) or f"manual_{int(time.time())}"
    source = {"id": vid, "url": url, "title": url}
    before = len(state.data["queue"])
    try:
        made = produce(state, source, out_root, log, n_clips=n_clips)
    except (LocalFailure, YoutubeBlocked) as e:
        log(f"  [!] {e}")
        keep_awake(False)
        return
    if not made or not live:
        keep_awake(False)
        return
    # posta os clipes deste vídeo agora (o resto da fila continua esperando),
    # um em cada canal, começando pelo que postou há mais tempo
    mine = [c for c in state.data["queue"][before:]]
    for c in mine:
        ok, nxt = can_post_now(state, ignore_schedule=True)
        if not ok:
            log(f"  Limite diário do YouTube atingido; os outros clipes ficam na fila "
                f"(o modo automático posta depois de {datetime.fromtimestamp(nxt):%d/%m %H:%M}).")
            break
        free = [a for a in live if sub(state, a).get("blocked_until", 0) <= time.time()]
        if not free:
            log("  Nenhum canal pode receber agora; os outros clipes ficam na fila.")
            break
        aid = min(free, key=lambda a: sub(state, a).get("last_post_at", 0))
        state.data["queue"].remove(c)
        state.data["queue"].insert(0, dict(c, priority=2))  # este primeiro
        post_next(state, svcs[aid], log, ignore_schedule=True, acc=aid)
    keep_awake(False)


def _video_id(url: str) -> Optional[str]:
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
    return m.group(1) if m else None
