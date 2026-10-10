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
class YoutubeBlocked(Exception):
    """O YouTube recusou o download (anti-robô / excesso de pedidos). Não é
    culpa do vídeo: o piloto pausa os downloads em vez de queimar a lista."""


_BLOCK_MARKERS = ("Sign in to confirm you", "HTTP Error 429", "Too Many Requests")


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
        while proc.poll() is None:
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
def _in_post_window() -> bool:
    start, end = getattr(config, "AUTOPILOT_POST_HOURS", (0, 24))
    h = datetime.now().hour
    return start <= h < end if start <= end else (h >= start or h < end)


def daily_limit() -> int:
    """Posts por dia (AUTOPILOT_POSTS_PER_DAY). Pedido do usuário: 20 por dia
    "independente da cota da API" -- o piloto tenta até o próprio YouTube
    recusar (se a cota for aumentada, já posta mais sem mudar nada) e o que
    passar vai pra pasta "postar à mão" (src/manual_post.py).
    AUTOPILOT_LIMIT_BY_QUOTA = True volta ao limite calculado pela cota
    (1.600 por upload)."""
    want = getattr(config, "AUTOPILOT_POSTS_PER_DAY", 20)
    if not getattr(config, "AUTOPILOT_LIMIT_BY_QUOTA", False):
        return max(want, 0)
    reserve = getattr(config, "AUTOPILOT_RELEASE_PER_DAY", 3) * 50 + 100
    by_quota = max(getattr(config, "YOUTUBE_DAILY_QUOTA", 10000) - reserve, 0) // 1600
    return max(min(want, by_quota), 0)


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


def _diversity_penalty(state: State, item: dict) -> float:
    """Não posta o mesmo vídeo/podcast em sequência (achado real: os 2
    últimos posts eram do mesmo vídeo do Market Makers)."""
    posted = state.data["posted"]
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


def post_next(state: State, service, log: Log, ignore_schedule: bool = False) -> bool:
    """Posta o melhor clipe da fila. True se postou."""
    from . import youtube_uploader as yt
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
                     key=lambda c: -c.get("quality", 50))[:3]
    for c in pending:
        score_viral(state, c, log)
    # prioridade primeiro (repostagem / link colado na mão); depois a maior
    # nota combinada (viralidade + qualidade)
    weights = _learned_weights(state)

    def key(c):
        return (-_prio(c), -(rank_key(c, weights) - _diversity_penalty(state, c)), c.get("added", 0))
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
    if not _prio(item):
        others = len({_channel_of(state, c) for c in queue})
        log(f"  >> Escolhido entre {len(queue)} clipe(s) de {others} podcast(s): nota final "
            f"{rank_key(item, weights):.0f}" + (f" (IA: {item['ai_compare']}/100)" if "ai_compare" in item else ""))
    log(f"  >> Postando no YouTube: \"{meta['title']}\"")
    try:
        vid = yt.upload_video(service, item["video"], meta, progress=log)
    except yt.QuotaExceeded as e:
        until = _next_pacific_midnight()
        state.data["blocked_until"] = until
        state.save()
        done = state.uploads_today()
        log(f"    [!] {e} -- {done} postado(s) hoje pela API; volto a postar em "
            f"{datetime.fromtimestamp(until):%d/%m %H:%M}.")
        day = _pacific_day()
        if state.data.get("quota_explained_day") != day:
            state.data["quota_explained_day"] = day
            state.save()
            if "uploadLimitExceeded" in str(e):
                log("        Esse é o limite de uploads do CANAL no dia (não da API). Canal verificado por "
                    "telefone tem limite maior: youtube.com/verify")
            else:
                log("        A cota padrão da API do YouTube (10.000 unidades) dá ~6 uploads por dia. Pra postar "
                    f"{daily_limit()} sozinho, peça o aumento de cota (é grátis): "
                    "https://support.google.com/youtube/contact/yt_api_form -- peça 40.000 unidades. "
                    "Quando aprovarem, o piloto já posta mais sem mudar nada.")
        from . import manual_post
        manual_post.export(state, daily_limit() - done, day, key, log)
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
             "video_sem_musica": item.get("video_sem_musica"), "privacy": privacy,
             "channel": _channel_of(state, item)}
    # a nota que ele teve, pra conferir com as views reais (src/feedback.py)
    for k in ("viral", "viral_parts", "quality", "ai_score", "ai_compare"):
        if k in item:
            entry[k] = item[k]
    state.data.pop("pool_wait_since", None)
    if item.get("replaces"):
        entry["replaces"] = item["replaces"]
    if privacy == "?":
        del entry["privacy"]  # confere depois
    state.data["posted"].append(entry)
    state.count_upload()
    note = "" if privacy in ("public", "?") else f" [{privacy} -- libera sozinho quando a auditoria sair]"
    log(f"    postado: https://youtube.com/shorts/{vid}  "
        f"({state.uploads_today()}/{daily_limit()} hoje){note}")
    if queue:
        _, nxt = can_post_now(state)
        log(f"    próxima postagem às {datetime.fromtimestamp(max(nxt, time.time())):%H:%M} "
            f"(1 a cada {post_gap_seconds() / 60:.0f} min, até {daily_limit()} por dia)")
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
    return done


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
    end = time.time() + wait
    while time.time() < end:
        # passos curtos: o botão "Desligar" da interface (src/control.py)
        # para o piloto em ~2 s mesmo no meio de uma espera longa
        time.sleep(max(min(2.0, end - time.time()), 0))


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
    log("  PILOTO AUTOMÁTICO LIGADO -- pra parar: botão Desligar na interface (ou feche a janela do .bat)")
    log(f"  canais: {', '.join(getattr(config, 'AUTOPILOT_CHANNELS', [])) or '(nenhum)'}")
    log(f"  buscas: {', '.join(getattr(config, 'AUTOPILOT_SEARCHES', [])) or '(nenhuma)'}")
    log(f"  postagem: {'LIGADA' if upload else 'DESLIGADA (só gera os clipes)'} -- até "
        f"{daily_limit()}/dia, 1 a cada {post_gap_seconds() / 60:.0f} min, "
        f"das {getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[0]}h às "
        f"{getattr(config, 'AUTOPILOT_POST_HOURS', (0, 24))[1]}h")
    log(f"  fila atual: {len(state.data['queue'])} clipe(s) | já postados: {len(state.data['posted'])}")
    log("=" * 60)

    prune_blocked(state, log)
    prune_hook_clips(state, log)
    from . import manual_post
    manual_post.cleanup(log)
    _forgive_local_failures(state, log)
    repair_models(log)
    update_ytdlp(state, log)
    from . import housekeeping
    housekeeping.clean_work_dir(log)
    housekeeping.cleanup(state, out_root, log)
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

            expire_queue(state, log)
            if service is not None:
                # a nota de viralidade acerta? confere com as views reais (1x por dia)
                from . import feedback
                feedback.refresh_views(state, service, log)
                feedback.calibrate(state, log)
                # o vídeo postado está visível ou o Content ID bloqueou? (src/rights.py)
                from . import rights
                rights.check_recent_posts(state, service, log)

            # 1) postar, se estiver na hora -- e se o banco de clipes já tem
            # opção de podcasts diferentes pra escolher o mais viral
            waiting_pool = False
            if service is not None and state.data["queue"]:
                ok, _ = can_post_now(state)
                waiting_pool = ok and not pool_ready(state, log)
                if ok and not waiting_pool:
                    from . import youtube_uploader as yt
                    try:
                        post_next(state, service, log)
                    except yt.AuthError as e:
                        log(f"  [!] {e}")
                        service = None
                    housekeeping.cleanup(state, out_root, log)
                    continue

            # 2) produzir, se a fila estiver curta (e o YouTube não estiver
            # bloqueando downloads -- ver YoutubeBlocked)
            target = getattr(config, "AUTOPILOT_QUEUE_TARGET", 8)
            local_until = state.data.get("local_fail_until", 0)
            blocked_until = max(state.data.get("download_blocked_until", 0), local_until)
            if (len(state.data["queue"]) < target and time.time() >= blocked_until
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
                                                   avoid_channels=set(state.data.get("bad_channels", {})))
                state.data.pop("retry_source", None)
                if source:
                    try:
                        produce(state, source, out_root, log)
                        state.data["download_blocks"] = 0
                        state.save()
                        housekeeping.clean_work_dir(log)
                        housekeeping.cleanup(state, out_root, log)
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
            if time.time() < blocked_until and not (service is not None and state.data["queue"]):
                _sleep_until(blocked_until, log, "edição pausada (problema no PC)" if blocked_until == local_until
                             else "downloads pausados pelo bloqueio do YouTube")
            elif service is not None and state.data["queue"]:
                _, nxt = can_post_now(state)
                if waiting_pool:
                    nxt = _pool_deadline(state)  # posta o melhor que tiver nessa hora
                wake = min(nxt, time.time() + 3600)
                if time.time() < blocked_until:
                    wake = min(wake, blocked_until)  # volta a baixar assim que a pausa acabar
                score_idle(state, log, wake)
                _sleep_until(wake, log,
                             f"fila com {len(state.data['queue'])} clipe(s); próxima postagem às "
                             f"{datetime.fromtimestamp(max(nxt, time.time())):%H:%M}")
            else:
                _sleep_until(time.time() + 1800, log,
                             "fila cheia e postagem indisponível (confira o login do YouTube)")
        except KeyboardInterrupt:
            from . import control
            log("  Piloto automático desligado" + (" (botão Desligar)." if control.stop_requested() else " (Ctrl+C)."))
            keep_awake(False)
            return
        except Exception as e:  # nada derruba o loop
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
    try:
        made = produce(state, source, out_root, log, n_clips=n_clips)
    except (LocalFailure, YoutubeBlocked) as e:
        log(f"  [!] {e}")
        keep_awake(False)
        return
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
        state.data["queue"].insert(0, dict(c, priority=2))  # este primeiro
        post_next(state, service, log, ignore_schedule=True)
    keep_awake(False)


def _video_id(url: str) -> Optional[str]:
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/|live/)([\w-]{11})", url)
    return m.group(1) if m else None
