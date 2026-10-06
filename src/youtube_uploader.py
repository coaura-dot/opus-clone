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
  - cota da API: 10.000 unidades/dia por projeto; cada upload gasta 1.600
    -> no máximo 6 vídeos por dia (zera à meia-noite do horário do Pacífico);
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
          "https://www.googleapis.com/auth/youtube.readonly"]
UPLOAD_QUOTA_COST = 1600
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


def get_service(interactive: bool = True):
    """Serviço autenticado da API. `interactive=False` (modo automático
    rodando sozinho) nunca abre navegador: sem login válido, levanta
    AuthError."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as e:
        raise UploadError("Bibliotecas do Google não instaladas. Rode: pip install -r requirements.txt") from e

    creds = None
    tok = token_path()
    if tok.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(tok), SCOPES)
        except Exception:
            creds = None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as e:
            if not interactive:
                raise AuthError(f"não consegui renovar o login do YouTube ({e}). "
                                "Rode 'python autopilot.py --login' pra autorizar de novo.") from e
            creds = None
    if not creds or not creds.valid:
        if not interactive:
            raise AuthError("sem login do YouTube. Rode 'python autopilot.py --login' uma vez.")
        secret = client_secret_path()
        if not secret.exists():
            raise AuthError(f"falta o arquivo {secret} (credencial OAuth do Google Cloud). "
                            "Veja o passo a passo no README, seção 'Postagem automática'.")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
        print("    Abrindo o navegador pra você autorizar o seu canal do YouTube...")
        creds = flow.run_local_server(port=0, open_browser=True,
                                      authorization_prompt_message="",
                                      success_message="Pronto! Pode fechar esta aba e voltar pro programa.")
    tok.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def channel_name(service) -> Optional[str]:
    try:
        r = service.channels().list(part="snippet", mine=True).execute()
        items = r.get("items") or []
        return items[0]["snippet"]["title"] if items else None
    except Exception:
        return None


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
