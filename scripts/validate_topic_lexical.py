#!/usr/bin/env python3
"""
Testa/calibra o detector de fim-de-assunto por coesão lexical (TextTiling)
descrito no RELATORIO_PROXIMOS_PASSOS.txt, sem precisar rodar o pipeline
inteiro (nem whisper, nem vídeo) — só texto.

Uso:
    python3 scripts/validate_topic_lexical.py transcricao.txt

O arquivo de entrada é TEXTO PLANO, uma FRASE por linha (cole a
transcrição do seu vídeo, uma frase/fala por linha — não precisa de
timestamp real, o script fabrica timestamps sequenciais só pra reusar a
mesma função do pipeline). Linhas em branco são ignoradas.

Pra cada ponto de corte (entre a frase N e a frase N+1), o script imprime
o score de coesão lexical (0 = claramente o mesmo assunto, 1 = fronteira
de tópico forte) e marca quais cortes ULTRAPASSAM o limiar atual
(TOPIC_LEXICAL_BOUNDARY_MIN em config.py) — ou seja, quais pontos o
seletor de clipes passaria a tratar como fim de assunto válido por causa
só do vocabulário, mesmo sem pausa nem palavra-chave.

Use isso pra conferir, com transcrições REAIS suas (cole trechos de
vídeos diferentes, de estilos de fala diferentes), se o limiar padrão
(0.55) está pegando as viradas de assunto que você esperava e NÃO
disparando no meio de um assunto só. Ajuste TOPIC_LEXICAL_BOUNDARY_MIN /
TOPIC_LEXICAL_TARGET_WORDS em src/config.py e rode de novo pra iterar
rápido, sem precisar gerar clipe nenhum.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clip_selector import _lexical_boundary_scores  # noqa: E402
from src import config  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Uso: python3 {sys.argv[0]} transcricao.txt")

    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"Arquivo não encontrado: {path}")

    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 3:
        sys.exit("Preciso de pelo menos 3 frases (linhas não-vazias) pra ter algo pra comparar.")

    # timestamps sintéticos sequenciais -- só a ORDEM e o TEXTO importam
    # pra essa função (ela não usa `start`/`end` pra nada além de existir
    # no dicionário); durações/pausas reais não afetam o score lexical.
    sentences = []
    t = 0.0
    for ln in lines:
        dur = max(len(ln.split()) * 0.35, 0.5)
        sentences.append({"start": t, "end": t + dur, "text": ln})
        t += dur + 0.3

    target_words = getattr(config, "TOPIC_LEXICAL_TARGET_WORDS", 40)
    min_words = getattr(config, "TOPIC_LEXICAL_MIN_WORDS", 8)
    threshold = getattr(config, "TOPIC_LEXICAL_BOUNDARY_MIN", 0.55)

    scores = _lexical_boundary_scores(sentences, target_words=target_words, min_words=min_words)

    print(f"(target_words={target_words}, min_words={min_words}, limiar={threshold})\n")
    for i, s in enumerate(sentences):
        print(f"  [{i:3d}] {s['text']}")
        if i < len(scores):
            fires = scores[i] >= threshold
            marker = "  <=== FIM DE ASSUNTO (lexical)" if fires else ""
            print(f"        corte {i:3d}: score={scores[i]:.3f}{marker}")

    n_fired = sum(1 for s in scores if s >= threshold)
    print(f"\n{n_fired} de {len(scores)} cortes ultrapassaram o limiar {threshold}.")
    print("Se algum desses NÃO for uma virada de assunto de verdade no seu texto,")
    print("suba TOPIC_LEXICAL_BOUNDARY_MIN em config.py. Se uma virada óbvia não")
    print("apareceu marcada, desça o limiar (ou confira se as frases ao redor têm")
    print(f"pelo menos {min_words} palavras de conteúdo cada lado).")


if __name__ == "__main__":
    main()
