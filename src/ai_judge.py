"""
Juiz de cortes com IA (Claude, API da Anthropic) -- opcional.

O seletor de cortes (clip_selector.py) acha os trechos candidatos por regras
(gancho, assunto que começa e termina, energia da voz). Regra não entende se
o assunto é interessante, se a fala é inteligente ou se o trecho se sustenta
sozinho -- pedido do usuário: "faça um bom seletor de clips, por hora está
bem peba". Com uma chave da API configurada, os candidatos de cada vídeo vão
numa chamada só pro Claude, que dá uma nota de 0 a 100 a cada um, explica em
uma frase e escreve um título melhor. A escolha final usa essa nota.

Chave: arquivo credentials/claude_api_key.txt (recomendado), CLAUDE_API_KEY
em config.py ou a variável ANTHROPIC_API_KEY. Sem chave, o programa funciona
igual, só com as regras.

Custo estimado (Claude Opus 5.5, US$4 / US$20 por milhão de tokens de
entrada / saída): ~8-12 mil tokens por vídeo (até 12 trechos de 1-3 min),
uns US$0,05-0,10 por vídeo.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from . import config

_CTX = {"title": None}
_STATE = {"disabled_reason": None, "warned": False}

_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "title", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}

_SYSTEM = """Você é o editor-chefe de um canal de cortes (YouTube Shorts) que só publica \
conteúdo de qualidade: conversas inteligentes, ideias que fazem pensar, histórias \
reais bem contadas, explicações claras, opiniões fortes bem argumentadas, momentos \
genuinamente engraçados de reacts bons. O público é brasileiro e adulto.

Você recebe trechos candidatos de UM vídeo (transcrição automática, pode ter erros \
de palavra) e dá a cada um uma nota de 0 a 100 de "vale postar como corte".

Nota alta:
- um insight, dado, história ou argumento que prende sozinho, sem precisar do resto do episódio;
- os primeiros segundos já criam curiosidade (pergunta, afirmação forte, começo de história);
- tem desfecho: a ideia fecha, a história termina, a piada tem punchline;
- quem assiste sai sabendo/sentindo algo.

Nota baixa:
- zoeira sem conteúdo, gritaria, piada interna, risada sem contexto;
- recado do canal, patrocínio, abertura/encerramento, "segue a gente";
- trecho que depende do que veio antes pra fazer sentido, ou que termina no meio da ideia;
- conversa morna, enrolação, assunto genérico.

Seja exigente: a maioria dos trechos de um episódio não merece virar corte. \
Use a escala toda (0-100) e não dê a mesma nota pra todos.

Para cada trecho escreva também:
- title: título do corte no idioma da fala, até 70 caracteres, que gere curiosidade \
SEM mentir sobre o conteúdo; sem hashtag, sem emoji, sem aspas, sem caixa alta inteira;
- reason: uma frase curta dizendo por que a nota."""


def set_context(source_title: Optional[str]) -> None:
    _CTX["title"] = source_title


def _key_file() -> Path:
    d = Path(getattr(config, "YOUTUBE_CREDENTIALS_DIR", "credentials"))
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    return d / "claude_api_key.txt"


def _api_key() -> Optional[str]:
    """Chave da API: credentials/claude_api_key.txt (recomendado -- a pasta
    credentials não é sobrescrita nas atualizações), CLAUDE_API_KEY em
    config.py ou a variável ANTHROPIC_API_KEY."""
    try:
        key = _key_file().read_text(encoding="utf-8").strip()
        if key:
            return key
    except OSError:
        pass
    return getattr(config, "CLAUDE_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY") or None


def enabled() -> bool:
    if not getattr(config, "AI_JUDGE_ENABLED", True) or _STATE["disabled_reason"]:
        return False
    if not _api_key():
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        _disable("biblioteca 'anthropic' não instalada (rode: pip install anthropic)")
        return False
    return True


def _disable(reason: str) -> None:
    _STATE["disabled_reason"] = reason
    print(f"    [IA] juiz de cortes desligado nesta rodada: {reason}")


def rank(candidates: list) -> Optional[List[dict]]:
    """Notas da IA pros candidatos (mesma ordem da lista): [{score, title,
    reason}] ou None se a IA não está disponível / falhou (aí vale a regra)."""
    if not candidates or not enabled():
        return None
    import anthropic

    lang = getattr(config, "CONTENT_LANGUAGE", None) or getattr(config, "WHISPERCPP_LANGUAGE", "pt")
    parts = [f"Vídeo de origem: {_CTX['title'] or '(sem título)'}",
             f"Idioma da fala: {lang}", ""]
    for i, c in enumerate(candidates):
        parts.append(f"<trecho id=\"{i}\" duracao=\"{c.end - c.start:.0f}s\">\n{c.text.strip()}\n</trecho>")
    parts.append("\nDê nota a TODOS os trechos acima, um item por id.")

    client = anthropic.Anthropic(api_key=_api_key(), timeout=180.0, max_retries=2)
    model = getattr(config, "AI_JUDGE_MODEL", "claude-opus-5-5")
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=_SYSTEM,
            output_config={"effort": getattr(config, "AI_JUDGE_EFFORT", "medium"),
                           "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": "\n".join(parts)}],
        )
    except anthropic.AuthenticationError:
        _disable("chave da API recusada (confira credentials/claude_api_key.txt)")
        return None
    except anthropic.PermissionDeniedError as e:
        _disable(f"sem permissão na API ({e.message})")
        return None
    except anthropic.BadRequestError as e:
        print(f"    [IA] pedido recusado ({e.message}); usando só as regras neste vídeo")
        return None
    except anthropic.RateLimitError:
        print("    [IA] limite da API atingido agora; usando só as regras neste vídeo")
        return None
    except anthropic.APIStatusError as e:
        print(f"    [IA] erro da API ({e.status_code}); usando só as regras neste vídeo")
        return None
    except anthropic.APIConnectionError:
        print("    [IA] sem conexão com a API; usando só as regras neste vídeo")
        return None

    if response.stop_reason == "refusal":
        print("    [IA] a IA recusou avaliar estes trechos; usando só as regras neste vídeo")
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        items = json.loads(text)["clips"]
    except (ValueError, KeyError, TypeError):
        print("    [IA] resposta fora do formato; usando só as regras neste vídeo")
        return None

    out: List[Optional[dict]] = [None] * len(candidates)
    for it in items:
        i = it.get("id")
        if isinstance(i, int) and 0 <= i < len(candidates) and out[i] is None:
            out[i] = {"score": int(max(0, min(100, it.get("score", 0)))),
                      "title": (it.get("title") or "").strip().strip('"'),
                      "reason": (it.get("reason") or "").strip()}
    if any(o is None for o in out):
        print("    [IA] a IA pulou algum trecho; usando só as regras neste vídeo")
        return None
    u = response.usage
    cost = (u.input_tokens * 4 + u.output_tokens * 20) / 1e6
    print(f"    [IA] {len(candidates)} trechos avaliados ({u.input_tokens}+{u.output_tokens} tokens, "
          f"~US${cost:.3f})")
    return out
