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
  - respeita a cota diária da API do YouTube (6 uploads/dia no padrão);
    acabou a cota, espera zerar (meia-noite do horário do Pacífico).
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
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)
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
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)  # troca atômica: nunca fica um arquivo pela metade

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
def run_worker(url: str, n_clips: int, out_dir: Path, log: Log) -> list:
    """Roda main.py num processo separado pra editar um vídeo. Devolve a
    lista de clipes gerados ([] se falhou)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "resultado.json"
    if results.exists():
        results.unlink()
    cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--url", url, "--clips", str(n_clips),
           "--out", str(out_dir), "--results", str(results)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env, **kw)

    def _pump():
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line.strip():
                log(f"    | {line}")
    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    timeout = getattr(config, "AUTOPILOT_WORKER_TIMEOUT_MINUTES", 150) * 60
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        log(f"    [!] edição passou de {timeout // 60:.0f} min -- encerrando esse vídeo e seguindo.")
        _kill_tree(proc)
    reader.join(timeout=10)
    if not results.exists():
        log(f"    [!] a edição terminou sem gerar clipes (código {proc.returncode}).")
        return []
    try:
        clips = json.loads(results.read_text(encoding="utf-8")).get("clips", [])
    except ValueError:
        return []
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
def _in_post_window() -> bool:
    start, end = getattr(config, "AUTOPILOT_POST_HOURS", (0, 24))
    h = datetime.now().hour
    return start <= h < end if start <= end else (h >= start or h < end)


def daily_limit() -> int:
    """Uploads por dia: o menor entre o pedido (AUTOPILOT_POSTS_PER_DAY) e o
    que a cota da API comporta (1.600 por upload, reservando um pouco pra
    liberação dos vídeos privados -- src/release.py)."""
    reserve = getattr(config, "AUTOPILOT_RELEASE_PER_DAY", 3) * 50 + 100
    by_quota = max(getattr(config, "YOUTUBE_DAILY_QUOTA", 10000) - reserve, 0) // 1600
    return max(min(getattr(config, "AUTOPILOT_POSTS_PER_DAY", 6), by_quota), 0)


def _window_hours() -> float:
    start, end = getattr(config, "AUTOPILOT_POST_HOURS", (0, 24))
    return float((end - start) % 24 or 24)


def post_gap_seconds() -> float:
    """Intervalo entre posts: espalha o limite do dia pela janela de
    postagem inteira (com 6/dia em 24h: 1 a cada 4h; com 24/dia: 1 por
    hora), nunca menos que AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS. Sem isso,
    "1 por hora" com cota de 6 postava tudo de madrugada e parava."""
    minimum = getattr(config, "AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS", 60) * 60
    limit = daily_limit()
    spread = _window_hours() * 3600 / limit if limit else minimum
    return max(minimum, spread)


def can_post_now(state: State, ignore_schedule: bool = False) -> tuple:
    """(pode?, motivo/hora da próxima tentativa)."""
    now = time.time()
    if now < state.data.get("blocked_until", 0):
        return False, state.data["blocked_until"]
    if state.uploads_today() >= daily_limit():
        return False, _next_pacific_midnight()
    if ignore_schedule:
        return True, now
    nxt = state.data.get("last_post_at", 0) + post_gap_seconds()
    if now < nxt:
        return False, nxt
    if not _in_post_window():
        return False, now + 15 * 60
    return True, now


def post_next(state: State, service, log: Log, ignore_schedule: bool = False) -> bool:
    """Posta o melhor clipe da fila. True se postou."""
    from . import youtube_uploader as yt
    queue = state.data["queue"]
    if not queue:
        return False
    # melhor pontuação primeiro; entre parecidos, o mais antigo na fila
    queue.sort(key=lambda c: (-round(c.get("score", 0)), c.get("added", 0)))
    item = queue[0]
    if not Path(item["video"]).exists():
        log(f"    arquivo sumiu, tirando da fila: {item['video']}")
        queue.pop(0)
        state.save()
        return False
    meta = json.loads(Path(item["meta"]).read_text(encoding="utf-8"))
    log(f"  >> Postando no YouTube: \"{meta['title']}\"")
    try:
        vid = yt.upload_video(service, item["video"], meta, progress=log)
    except yt.QuotaExceeded as e:
        until = _next_pacific_midnight()
        state.data["blocked_until"] = until
        state.save()
        log(f"    [!] {e} -- volto a postar em {datetime.fromtimestamp(until):%d/%m %H:%M}.")
        return False
    except yt.AuthError:
        raise
    except yt.UploadError as e:
        item["attempts"] = item.get("attempts", 0) + 1
        log(f"    [!] falhou ({e}); tentativa {item['attempts']}/3")
        if item["attempts"] >= 3:
            queue.pop(0)
            log("    desisti desse clipe.")
        state.data["blocked_until"] = time.time() + 15 * 60
        state.save()
        return False
    queue.pop(0)
    try:
        from .release import privacy_of
        privacy = privacy_of(service, [vid]).get(vid) or "?"
    except Exception:
        privacy = "?"
    entry = {"youtube_id": vid, "title": meta["title"], "at": time.time(),
             "source_id": item.get("source_id"), "video": item["video"],
             "video_sem_musica": item.get("video_sem_musica"), "privacy": privacy}
    if item.get("replaces"):
        entry["replaces"] = item["replaces"]
    if privacy == "?":
        del entry["privacy"]  # confere depois
    state.data["posted"].append(entry)
    state.count_upload()
    note = "" if privacy in ("public", "?") else f" [{privacy} -- libera sozinho quando a auditoria sair]"
    log(f"    postado: https://youtube.com/shorts/{vid}  "
        f"({state.uploads_today()}/{daily_limit()} hoje){note}")
    if getattr(config, "AUTOPILOT_DELETE_POSTED_FILES", False):
        for f in (item["video"], item.get("video_sem_musica")):
            if f and Path(f).exists():
                Path(f).unlink()
    return True


# ------------------------------------------------------------ produzir ---
def produce(state: State, source: dict, out_root: Path, log: Log, n_clips: Optional[int] = None) -> int:
    vid = source["id"]
    n_clips = n_clips or getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3)
    log(f"  >> Editando: \"{source.get('title', vid)}\" -- {source['url']}")
    t0 = time.time()
    clips = run_worker(source["url"], n_clips, out_root / vid, log)
    if not clips:
        state.mark_source(vid, "failed", title=source.get("title"), channel=source.get("channel"))
        return 0
    for c in clips:
        no_music = Path(c["video"]).with_name(Path(c["video"]).stem + "_sem_musica.mp4")
        state.data["queue"].append({"video": c["video"], "meta": c["meta"], "score": c.get("score", 0),
                                    "source_id": vid, "added": time.time(),
                                    "video_sem_musica": str(no_music) if no_music.exists() else None})
    state.mark_source(vid, "done", title=source.get("title"), channel=source.get("channel"),
                      clips=len(clips))
    log(f"    {len(clips)} clipe(s) prontos em {(time.time() - t0) / 60:.0f} min -- "
        f"fila: {len(state.data['queue'])}")
    return len(clips)


# ------------------------------------------------------------ loop ---
def _sleep_until(ts: float, log: Log, why: str):
    wait = max(ts - time.time(), 0)
    if wait <= 0:
        return
    if wait > 120:
        log(f"  .. {why} -- próxima ação às {datetime.fromtimestamp(ts):%H:%M}")
    end = time.time() + wait
    while time.time() < end:
        time.sleep(min(30, end - time.time()))


def _get_service(log: Log, upload: bool):
    if not upload:
        return None
    from . import youtube_uploader as yt
    try:
        svc = yt.get_service(interactive=False)
        name, problem = yt.channel_check(svc)
        if problem:
            log(f"  [!] YouTube: {problem}")
            return None
        log(f"  YouTube conectado: canal \"{name}\"")
        return svc
    except yt.UploadError as e:
        log(f"  [!] YouTube: {e}")
        return None


def run_forever(upload: bool = True):
    """Modo automático: roda até fechar a janela / desligar o PC."""
    from . import discovery
    out_root = Path(getattr(config, "AUTOPILOT_OUTPUT_DIR", "output/autopiloto"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    log = Log(DATA_DIR / "autopilot.log")
    state = State(DATA_DIR / "state.json")
    keep_awake(True)

    log("=" * 60)
    log("  PILOTO AUTOMÁTICO LIGADO -- feche esta janela pra parar")
    log(f"  canais: {', '.join(getattr(config, 'AUTOPILOT_CHANNELS', [])) or '(nenhum)'}")
    log(f"  buscas: {', '.join(getattr(config, 'AUTOPILOT_SEARCHES', [])) or '(nenhuma)'}")
    log(f"  postagem: {'LIGADA' if upload else 'DESLIGADA (só gera os clipes)'} -- até "
        f"{daily_limit()}/dia, 1 a cada {post_gap_seconds() / 60:.0f} min, "
        f"das {getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[0]}h às "
        f"{getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[1]}h")
    log(f"  fila atual: {len(state.data['queue'])} clipe(s) | já postados: {len(state.data['posted'])}")
    log("=" * 60)

    service = _get_service(log, upload)
    next_auth_try = time.time() + 3600
    failures = 0
    startup = True
    while True:
        try:
            if upload and service is None and time.time() >= next_auth_try:
                service = _get_service(log, upload)
                next_auth_try = time.time() + 3600

            # 0) vídeos que subiram privados: confere a auditoria / solta aos poucos
            if service is not None:
                from . import release
                from . import youtube_uploader as yt
                try:
                    acted = release.release_step(state, service, log, startup=startup)
                except yt.QuotaExceeded as e:
                    state.data["blocked_until"] = _next_pacific_midnight()
                    state.save()
                    log(f"  [!] {e}")
                    acted = False
                startup = False
                if acted:
                    continue

            # 1) postar, se estiver na hora
            if service is not None and state.data["queue"]:
                ok, _ = can_post_now(state)
                if ok:
                    from . import youtube_uploader as yt
                    try:
                        post_next(state, service, log)
                    except yt.AuthError as e:
                        log(f"  [!] {e}")
                        service = None
                    continue

            # 2) produzir, se a fila estiver curta
            target = getattr(config, "AUTOPILOT_QUEUE_TARGET", 8)
            if len(state.data["queue"]) < target and make_room(state, out_root, log):
                log("  >> Procurando vídeo bombando pra cortar...")
                source = discovery.pick_source(state.seen_ids(), state.recent_channels(), log=log)
                if source:
                    produce(state, source, out_root, log)
                    failures = 0
                    continue
                _sleep_until(time.time() + getattr(config, "AUTOPILOT_IDLE_MINUTES", 30) * 60, log,
                             "nada novo agora, procuro de novo depois")
                continue

            # 3) fila cheia: espera a próxima janela de postagem
            if service is not None and state.data["queue"]:
                _, nxt = can_post_now(state)
                _sleep_until(min(nxt, time.time() + 3600), log,
                             f"fila com {len(state.data['queue'])} clipe(s), esperando a hora de postar")
            else:
                _sleep_until(time.time() + 1800, log,
                             "fila cheia e postagem indisponível (confira o login do YouTube)")
        except KeyboardInterrupt:
            log("  Piloto automático desligado.")
            keep_awake(False)
            return
        except Exception as e:  # nada derruba o loop
            failures += 1
            wait = min(60 * 2 ** failures, 3600)
            log(f"  [!] erro inesperado: {e.__class__.__name__}: {e} -- tentando de novo em {wait // 60:.0f} min")
            time.sleep(wait)


def run_once(url: str, n_clips: int, upload: bool = True):
    """Modo manual: um link -> edita -> posta os clipes na hora."""
    out_root = Path(getattr(config, "AUTOPILOT_OUTPUT_DIR", "output/autopiloto"))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    log = Log(DATA_DIR / "autopilot.log")
    state = State(DATA_DIR / "state.json")
    keep_awake(True)
    service = None
    if upload:
        from . import youtube_uploader as yt
        try:
            service = yt.get_service(interactive=True)
            name, problem = yt.channel_check(service)
            if problem:
                raise yt.UploadError(problem)
            log(f"  YouTube conectado: canal \"{name}\"")
        except yt.UploadError as e:
            log(f"  [!] YouTube: {e}")
            log("  Vou gerar os clipes mesmo assim; eles ficam na fila pra postar depois.")
    vid = _video_id(url) or f"manual_{int(time.time())}"
    source = {"id": vid, "url": url, "title": url}
    before = len(state.data["queue"])
    made = produce(state, source, out_root, log, n_clips=n_clips)
    if not made or service is None:
        keep_awake(False)
        return
    # posta os clipes deste vídeo agora (o resto da fila continua esperando)
    mine = [c for c in state.data["queue"][before:]]
    for c in mine:
        ok, nxt = can_post_now(state, ignore_schedule=True)
        if not ok:
            log(f"  Limite diário do YouTube atingido; os outros clipes ficam na fila "
                f"(o modo automático posta depois de {datetime.fromtimestamp(nxt):%d/%m %H:%M}).")
            break
        state.data["queue"].remove(c)
        state.data["queue"].insert(0, dict(c, score=10 ** 6))  # este primeiro
        post_next(state, service, log, ignore_schedule=True)
    keep_awake(False)


def _video_id(url: str) -> Optional[str]:
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
    return m.group(1) if m else None
