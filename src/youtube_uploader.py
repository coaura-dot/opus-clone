"""
Postagem automática no YouTube pela API oficial (YouTube Data API v3).

Login: OAuth do Google, uma vez só. Na primeira vez abre o navegador pra
você autorizar o seu canal; o acesso fica salvo em
credentials/youtube_token.json e é renovado sozinho daí em diante.

O que precisa existir antes (passo a passo no README, "Postagem automática"):
  credentials/client_secret.json -- baixado do Google Cloud Console
  (projeto com a "YouTube Data API v3" ativada, credencial OAuth do tipo
  "App para computador").

Limites do YouTube que o programa respeita:
  - cota da API por projeto (zera à meia-noite do horário do Pacífico):
    upload tem um balde próprio, 100 por dia (desde dez/2025; antes cada
    upload gastava 1.600 das 10.000 unidades e dava ~6 por dia); as outras
    consultas gastam as 10.000 unidades. Vale somando TODOS os canais
    conectados (src/accounts.py): é o mesmo projeto;
  - projeto da API não verificado pelo Google: o YouTube trava os vídeos
    enviados como PRIVADOS até o projeto passar pela auditoria (formulário
    no README). Isso é regra do YouTube, não do programa.
"""
import json
import random
import time
from pathlib import Path
from typing import Optional

from . import config

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.readonly",
          # mudar a visibilidade dos próprios vídeos (privado -> público
          # depois da auditoria; ver src/release.py)
          "https://www.googleapis.com/auth/youtube"]
MANAGE_SCOPE = "https://www.googleapis.com/auth/youtube"
_RETRIABLE_STATUS = {500, 502, 503, 504}


def _safe(text: str) -> str:
    # o YouTube recusa título/descrição com "<" ou ">"
    return (text or "").replace("<", "‹").replace(">", "›")


class UploadError(Exception):
    pass


class QuotaExceeded(UploadError):
    """Cota diária da API (ou limite de uploads do canal) acabou."""


class AuthError(UploadError):
    """Login do Google inválido/expirado -- precisa autorizar de novo."""


# achado real (14/out): "invalid_grant: Token has been expired or revoked"
# no meio da madrugada -- com o app do Google Cloud em "Teste" o Google
# derruba o login a cada 7 dias. O piloto ficou 8 h repetindo "confiro de
# novo em 10 min" sem dizer que o problema era o login.
DEAD_LOGIN_HINT = ("o login do YouTube venceu ou foi revogado pelo Google. Na interface, em CANAIS DO "
                   "YOUTUBE, clique em 'Reconectar' nesse canal. Se isso se repete a cada 7 dias, o app do "
                   "Google Cloud está em 'Teste': publique o app (README, 'Postagem automática', passo 3) e "
                   "conecte de novo")


def dead_login(e: BaseException) -> bool:
    """O Google recusou o login salvo (venceu/revogado): só um login novo resolve."""
    t = f"{e.__class__.__name__} {e}"
    return "invalid_grant" in t or ("RefreshError" in t and "expired or revoked" in t)


def credentials_dir() -> Path:
    d = Path(getattr(config, "YOUTUBE_CREDENTIALS_DIR", "credentials"))
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    d.mkdir(parents=True, exist_ok=True)
    return d


def client_secret_path() -> Path:
    return credentials_dir() / "client_secret.json"


def token_path() -> Path:
    return credentials_dir() / "youtube_token.json"


def get_service(interactive: bool = True, token: Optional[Path] = None):
    """Serviço autenticado da API. `interactive=False` (modo automático
    rodando sozinho) nunca abre navegador: sem login válido, levanta
    AuthError. `token`: o login de um dos canais (src/accounts.py); sem ele,
    o do canal principal."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as e:
        raise UploadError("Bibliotecas do Google não instaladas. Rode: pip install -r requirements.txt") from e

    creds = None
    tok = Path(token) if token else token_path()
    granted = set()
    if tok.exists():
        try:
            granted = set(json.loads(tok.read_text(encoding="utf-8")).get("scopes") or [])
            # usa as permissões que o login realmente tem (um login antigo
            # não tem a de mudar visibilidade -- ver src/release.py)
            creds = Credentials.from_authorized_user_file(str(tok))
        except Exception:
            creds = None
        if creds is not None and interactive and not set(SCOPES) <= granted:
            print("    O login salvo é de uma versão antiga (falta permissão nova): vamos autorizar de novo.")
            creds = None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            if not interactive:
                if dead_login(e):
                    raise AuthError(DEAD_LOGIN_HINT) from e
                raise AuthError(f"não consegui renovar o login do YouTube ({e}). "
                                "Na interface, em CANAIS DO YOUTUBE, clique em 'Reconectar' nesse canal.") from e
            creds = None
    if not creds or not creds.valid:
        if not interactive:
            raise AuthError("sem login do YouTube. Na interface, em CANAIS DO YOUTUBE, clique em "
                            "'Reconectar' nesse canal (ou '+ Adicionar canal').")
        creds = new_login(pick=False)
    tok.write_text(creds.to_json(), encoding="utf-8")
    return service_from(creds, granted)


def new_login(pick: bool = True):
    """Abre o navegador pra autorizar um canal. `pick`: o Google mostra a
    lista de contas/canais (pra adicionar um canal diferente do que já está
    logado no navegador)."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise UploadError("Bibliotecas do Google não instaladas. Rode: pip install -r requirements.txt") from e
    secret = client_secret_path()
    if not secret.exists():
        raise AuthError(f"falta o arquivo {secret} (credencial OAuth do Google Cloud). "
                        "Veja o passo a passo no README, seção 'Postagem automática'.")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    print("    Abrindo o navegador pra você autorizar o seu canal do YouTube...")
    extra = {"prompt": "select_account consent"} if pick else {}
    return flow.run_local_server(port=0, open_browser=True, authorization_prompt_message="",
                                 success_message="Pronto! Pode fechar esta aba e voltar pro programa.", **extra)


def service_from(creds, granted=None):
    from googleapiclient.discovery import build
    service = build("youtube", "v3", credentials=creds, cache_discovery=False)
    scopes = set(getattr(creds, "granted_scopes", None) or creds.scopes or granted or [])
    service.autoclipper_can_manage = MANAGE_SCOPE in scopes
    service.autoclipper_creds = creds
    return service


def ensure_fresh(service) -> None:
    """Renova o acesso (vale 1 h) antes de usar o serviço. Login morto ->
    AuthError (o piloto para de tentar e avisa na interface); sem rede, deixa
    passar (as chamadas tentam de novo)."""
    creds = getattr(service, "autoclipper_creds", None)
    if creds is None or creds.valid:
        return
    from google.auth.transport.requests import Request
    try:
        creds.refresh(Request())
    except Exception as e:
        if dead_login(e):
            raise AuthError(DEAD_LOGIN_HINT) from e


def channel_check(service) -> tuple:
    """(nome do canal, problema). Sem canal ou sem acesso, `problema` diz o
    que fazer (achado real: login OK mas "não consegui ler o nome")."""
    _, name, problem = channel_info(service)
    return name, problem


def channel_info(service) -> tuple:
    """(id do canal, nome, problema)."""
    try:
        r = service.channels().list(part="snippet", mine=True).execute()
    except Exception as e:
        reason = _error_reason(e) if hasattr(e, "content") else ""
        if reason in ("accessNotConfigured", "SERVICE_DISABLED") or "has not been used" in str(e):
            return None, None, ("a YouTube Data API v3 não está ativada no seu projeto do Google Cloud: "
                                "busque 'YouTube Data API v3' no console e clique em Enable (ou Ativar)")
        if reason in ("insufficientPermissions", "forbidden", "PERMISSION_DENIED"):
            return None, None, ("faltou permissão no login: conecte o canal de novo (interface: CANAIS DO "
                                "YOUTUBE -> Reconectar) e marque TODAS as caixas na tela do Google")
        if dead_login(e):
            return None, None, DEAD_LOGIN_HINT
        return None, None, f"erro ao consultar o canal ({reason or e.__class__.__name__}): {str(e)[:200]}"
    items = r.get("items") or []
    if not items:
        return None, None, ("essa conta do Google não tem canal no YouTube. Crie em "
                            "https://www.youtube.com/create_channel (ou, se o canal é de uma conta de marca, "
                            "conecte de novo escolhendo o CANAL na lista, não a conta pessoal)")
    return items[0].get("id"), items[0]["snippet"]["title"], None


def channel_name(service) -> Optional[str]:
    return channel_check(service)[0]


def _error_reason(err) -> str:
    try:
        data = json.loads(err.content.decode("utf-8"))
        errors = data.get("error", {}).get("errors") or [{}]
        return errors[0].get("reason", "") or data.get("error", {}).get("status", "")
    except Exception:
        return ""


def upload_video(service, video_path: str, meta: dict, privacy: Optional[str] = None,
                 publish_at: Optional[str] = None, max_retries: int = 8,
                 progress=print) -> str:
    """Envia o vídeo (upload em partes, retoma sozinho em falha de rede) e
    devolve o ID do vídeo no YouTube."""
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    privacy = privacy or getattr(config, "YOUTUBE_PRIVACY", "public")
    body = {
        "snippet": {
            "title": _safe(meta["title"])[:100],
            "description": _safe(meta.get("description", "")),
            "tags": meta.get("tags", []),
            "categoryId": str(getattr(config, "YOUTUBE_CATEGORY_ID", "24")),
            "defaultLanguage": getattr(config, "YOUTUBE_LANGUAGE", "pt-BR"),
            "defaultAudioLanguage": getattr(config, "YOUTUBE_LANGUAGE", "pt-BR"),
        },
        "status": {
            "privacyStatus": "private" if publish_at else privacy,
            "selfDeclaredMadeForKids": False,
            "embeddable": True,
        },
    }
    if publish_at:
        body["status"]["publishAt"] = publish_at  # RFC 3339; exige privado até a hora

    media = MediaFileUpload(video_path, mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    request = service.videos().insert(part="snippet,status", body=body, media_body=media,
                                      notifySubscribers=bool(getattr(config, "YOUTUBE_NOTIFY_SUBSCRIBERS", True)))
    response, retry, last_pct = None, 0, -1
    while response is None:
        try:
            status, response = request.next_chunk()
            if status is not None:
                pct = min(int(status.progress() * 100), 100)
                if pct >= last_pct + 25:
                    progress(f"      enviando... {pct}%")
                    last_pct = pct
        except HttpError as e:
            reason = _error_reason(e)
            code = int(getattr(e.resp, "status", 0) or 0)
            if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "uploadLimitExceeded"):
                raise QuotaExceeded(f"limite do YouTube atingido ({reason})") from e
            if code == 401 or reason in ("authError", "unauthorized"):
                raise AuthError(f"login do YouTube recusado ({reason or code})") from e
            if code not in _RETRIABLE_STATUS:
                raise UploadError(f"YouTube recusou o vídeo ({code} {reason}): {e}") from e
            retry = _backoff(retry, max_retries, f"erro {code} do YouTube", progress)
        except (OSError, ConnectionError, TimeoutError) as e:
            retry = _backoff(retry, max_retries, f"falha de rede ({e.__class__.__name__})", progress)
    video_id = response.get("id")
    if not video_id:
        raise UploadError(f"resposta inesperada do YouTube: {response}")
    return video_id


def _backoff(retry: int, max_retries: int, why: str, progress) -> int:
    retry += 1
    if retry > max_retries:
        raise UploadError(f"desisti do envio depois de {max_retries} tentativas ({why})")
    wait = min(2 ** retry, 120) + random.random()
    progress(f"      {why} -- tentando de novo em {wait:.0f}s")
    time.sleep(wait)
    return retry
