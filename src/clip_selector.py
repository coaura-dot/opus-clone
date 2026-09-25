"""
Motor de seleção automática de cortes "virais".

Combina sinais de texto (ganchos, perguntas, números, palavras de forte
carga emocional) com sinais de áudio (picos de energia/entusiasmo na voz)
para pontuar janelas candidatas de corte e escolher as N melhores, sem
sobreposição e respeitando limites de frase (nunca corta no meio de uma
frase).
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .transcriber import Transcript, Word
from .audio import audio_energy
from .opener import opening_depends_on_context
from . import config


HOOK_PATTERNS_PT = [
    r"\bvocê (sabia|acredita|imagina)\b", r"\bninguém (te|fala|conta|ensina|mostra)\b",
    r"\bsegredo\b", r"\bo (maior|pior|melhor) erro\b", r"\bisso mudou\b",
    r"\bpor que\b", r"\bcomo (eu|fazer|fiz|consigo)\b",
    r"\b\d+ (dicas|passos|motivos|razões|formas|maneiras|coisas|erros)\b",
    r"\bo que ninguém\b",
    r"\bvocê está fazendo (errado|isso errado)\b",
    r"\bimagina (se|só)\b", r"\bisso (é|foi) loucura\b",
    # PT-BR coloquial / viral
    r"\bminha gente\b", r"\bboa notícia\b", r"\bnotícia (boa|ruim|incrível)\b",
    r"\bmuita gente (não|faz|sabe|percebe)\b", r"\bnão faz isso\b",
    r"\bperdendo (dinheiro|tempo|oportunidade)\b", r"\bvai mudar\b",
    r"\bse você (faz|fizer|está)\b", r"\bpresta atenção\b",
    r"\bcalma (que|aí)\b", r"\bespera (aí|só)\b", r"\bsó (uma|um) (coisa|momento)\b",
    r"\bfato (é|curioso)\b",
    r"\bfui (eu|buscar|fazer|descobrir)\b", r"\bestou aqui\b",
    r"\bvou (te|contar|revelar|mostrar)\b", r"\btem (gente|pessoas) (que|fazendo)\b",
]
HOOK_PATTERNS_EN = [
    r"\bdid you know\b", r"\bnobody (tells|talks about)\b", r"\bsecret\b",
    r"\bbiggest mistake\b", r"\btruth (is|about)\b", r"\bthis changed\b",
    r"\bnever\b", r"\balways\b", r"\bwhy\b", r"\bhow to\b",
    r"\b\d+ (tips|steps|reasons|ways|things)\b", r"\bimportant\b",
    r"\bwhat nobody\b", r"you('|)re doing (it |this )?wrong\b",
]
EMOTION_WORDS = [
    # PT-BR
    "incrível", "chocante", "impressionante", "louco", "absurdo", "surreal",
    "bizarro", "hilário", "devastador", "épico", "brutal", "insano", "polêmico",
    "histórico", "revolucionário", "viral", "bombou", "explodiu", "destruiu",
    "perturbador", "assustador", "fantástico", "extraordinário", "ridículo",
    "impossível", "absurdo", "inacreditável",
    # EN
    "amazing", "shocking", "insane", "crazy", "unbelievable", "wild",
    "disgusting", "incredible", "hilarious", "devastating", "epic", "brutal",
]
FILLERS = ["né", "tipo", "então assim", " um ", " uh ", " hmm ", "like", "you know"]

# --- Desfecho de pergunta-gancho/piada (ver RELATORIO_PROXIMOS_PASSOS.txt,
# item 3b) ---
# Achado real: um clipe cortou em cima de "Sabe qual é o durão?" sem a
# resposta — a janela vencedora terminava numa pergunta-gancho porque isso
# pontuava bem em _text_score (HOOK_PATTERNS + "?"), mas ninguém verificava
# se havia uma frase seguinte que RESOLVIA a pergunta. Em vez de mudar a
# fronteira de frase (já é respeitada — nunca corta no meio de uma frase),
# aplicamos uma PENALIDADE quando a ÚLTIMA frase da janela termina em
# pergunta/gancho E existe uma próxima frase no vídeo que poderia ter sido
# incluída — isso faz o loop de janelas (que já testa duração+1 frase,
# +2 frases etc.) preferir naturalmente a versão que inclui a resposta,
# sem precisar de nenhuma lógica de busca nova.
SETUP_ENDING_PATTERNS_PT = [
    r"\bsabe (qual|quem|o que|como|por que)\b", r"\bvocê sabe\b",
    r"\badivinha\b", r"\bvocê acha que\b", r"\bo que (você )?acha\b",
    r"\bquer saber\b", r"\bsabe o que (aconteceu|rolou)\b",
]
SETUP_ENDING_PATTERNS_EN = [
    r"\bguess what\b", r"\bdo you know\b", r"\bwant to know\b",
    r"\bwhat do you think\b", r"\byou know what\b",
]

# --- Fim de ASSUNTO/tópico (ver config.py, TOPIC_BOUNDARY_*) ---
# Dois grupos de marcador textual usados como sinal de "esse assunto
# terminou aqui" (além do sinal de pausa longa, medido diretamente dos
# timestamps de frase em select_clips, não por regex):
#   CONCLUSIVE_END_*: a ÚLTIMA frase da janela soa como um fechamento/
#   conclusão do que estava sendo dito.
#   TOPIC_SHIFT_START_*: a PRÓXIMA frase (fora da janela) começa puxando
#   assunto novo — se a próxima frase já é outro assunto, a janela atual
#   terminou no lugar certo.
CONCLUSIVE_END_PATTERNS_PT = [
    r"\be (foi|é) isso\b", r"\be pronto\b", r"\bno final das contas\b",
    r"\bresumindo\b", r"\bé (basicamente|praticamente) isso\b",
    r"\bentão (é|foi) isso\b", r"\bdito isso\b",
    # fechamento no FIM da frase ("A gente nasce perdendo, é isso.") —
    # achado real: o clipe cortava logo antes dessa frase-conclusão
    r"\b(é|foi) isso( aí)?[.!]*\s*$", r"\bsimples assim\b", r"\bacabou[.!]*\s*$",
    r"\bponto final\b",
]
CONCLUSIVE_END_PATTERNS_EN = [
    r"\bthat'?s (basically |pretty much )?it\b", r"\bat the end of the day\b",
    r"\blong story short\b", r"\bin the end\b", r"\bthat'?s the (point|gist)\b",
    r"\bto sum (it|this) up\b",
]
TOPIC_SHIFT_START_PATTERNS_PT = [
    r"\bmudando de assunto\b", r"\bfalando nisso\b", r"\baliás\b",
    r"\bvoltando (ao|pro|para o)\b", r"\benfim\b",
    r"\bmas (voltando|enfim)\b", r"\bagora (falando|mudando|indo) (de|para)\b",
    r"\bpra (fechar|terminar) (esse|este) (assunto|ponto)\b",
]
TOPIC_SHIFT_START_PATTERNS_EN = [
    r"\bmoving on\b", r"\bspeaking of\b", r"\banyway\b", r"\bso anyway\b",
    r"\bto wrap (this|it) up\b", r"\bon a different note\b",
    r"\bswitching gears\b",
]

# --- Coesão lexical (TextTiling) — pega troca de assunto SEM pausa nem
# palavra-chave (ver config.py, TOPIC_LEXICAL_*) ---
# Os sinais acima (CONCLUSIVE_END_*, TOPIC_SHIFT_START_*, pausa em
# select_clips) são heurística de SUPERFÍCIE: cobrem bem quando existe uma
# frase de transição prevista ou uma pausa acima do normal, mas ficam CEGOS
# pra uma virada de assunto "silenciosa" — a pessoa só começa a falar de
# outra coisa, sem pausar nem usar nenhuma das frases previstas. Foi
# exatamente essa a cobrança: "o que resolve é ele ler e entender o
# contexto das frases". Sem LLM (custo/API/nuvem num projeto pensado pra
# rodar 100% local/offline — ver justificativa completa no relatório) e
# sem GPU (roda em CPU pura, texto puro), a técnica clássica pra isso é
# TextTiling (Hearst, 1997): medir o quanto o VOCABULÁRIO de conteúdo muda
# ao redor de cada ponto de corte, comparando uma janela de frases ANTES
# contra uma janela DEPOIS — um "vale" bem cavado de similaridade (baixo
# ali, cercado de picos de similaridade dos dois lados) indica troca de
# assunto de verdade, mesmo sem nenhum marcador textual ou pausa.
_STOPWORDS_PT = {
    "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
    "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele",
    "deles", "depois", "do", "dos", "e", "ela", "elas", "ele", "eles",
    "em", "entre", "era", "essa", "essas", "esse", "esses", "esta",
    "estas", "este", "estes", "eu", "foi", "for", "há", "isso", "isto",
    "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "meus",
    "minha", "minhas", "muito", "na", "não", "nas", "nem", "no", "nos",
    "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou",
    "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando",
    "que", "quem", "se", "sem", "ser", "seu", "seus", "só", "sua", "suas",
    "também", "te", "tem", "ter", "teu", "teus", "tu", "tua", "tuas",
    "um", "uma", "você", "vocês", "vou", "são", "está", "estão", "essa",
    "então", "assim", "aí", "ali", "lá", "cá", "né", "tipo", "coisa",
    "coisas", "vai", "vamos", "pra", "pro", "aqui", "onde",
}
_STOPWORDS_EN = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as",
    "by", "that", "this", "these", "those", "it", "its", "i", "you", "he",
    "she", "we", "they", "them", "his", "her", "our", "your", "their",
    "not", "no", "so", "if", "then", "than", "just", "like", "well",
    "really", "very", "there", "here", "what", "when", "where", "who",
    "how", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "about", "into", "out", "up", "down",
}
_LEXICAL_STOPWORDS = _STOPWORDS_PT | _STOPWORDS_EN
_WORD_RE = re.compile(r"[a-zà-öø-ÿ0-9]+", re.IGNORECASE)


def _content_words(text: str) -> List[str]:
    """Palavras de conteúdo de uma frase: minúsculas, sem pontuação, sem
    stopwords (artigos, preposições, pronomes etc.) nem palavras muito
    curtas (2 letras ou menos) — o que sobra tende a ser substantivo,
    verbo, nome próprio, número: o vocabulário que de fato caracteriza do
    que aquele trecho está falando."""
    return [w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _LEXICAL_STOPWORDS]


def _cosine_sim(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_boundary_scores(sentences: List[dict], target_words: int = 40,
                              min_words: int = 8) -> List[float]:
    """Para cada ponto de corte possível (entre a frase i e i+1), mede o
    quanto o vocabulário de conteúdo muda ali — técnica TextTiling
    adaptada a fronteiras de frase (em vez dos blocos de N palavras do
    algoritmo original), sem nenhuma dependência nova (só regex +
    contagem, como o resto do arquivo já faz).

    Como funciona:
    1. Pra cada ponto de corte, acumula frases pra ESQUERDA e pra DIREITA
       até juntar ~`target_words` palavras de conteúdo cada lado (não um
       número fixo de FRASES — frases variam muito de tamanho em fala
       real, de "Verdade." a um parágrafo inteiro sem pontuação; um
       número fixo de palavras dá uma janela de conteúdo mais consistente
       dos dois lados).
    2. Similaridade de cosseno entre os dois "sacos de palavras" —
       vocabulário parecido dos dois lados = mesma assunto ainda rolando;
       vocabulário bem diferente = provável troca.
    3. "Profundidade": cada ponto é comparado com o pico de similaridade
       mais próximo à esquerda e à direita (subindo a partir dali até a
       curva parar de crescer) — um "vale" bem cavado dos DOIS lados é uma
       fronteira de tópico real, não só ruído local da curva de
       similaridade (é a definição clássica de boundary do TextTiling).
    4. Normaliza pelo vale mais forte deste vídeo/bloco específico (0..1)
       — adaptativo, no mesmo espírito de `_typical_pause` acima: o que
       conta como "vale forte" varia muito entre um papo com vocabulário
       repetitivo e uma entrevista com muitos assuntos diferentes.

    Pontos onde não há palavra de conteúdo suficiente dos dois lados (frase
    curta demais, tipo uma interjeição isolada, ou perto demais da borda do
    transcript) são marcados como SEM DADO (não um valor arbitrário) e
    ficam de fora inteiramente da busca de pico/vale — tratá-los como
    "similaridade 1.0" (uma versão anterior desta função fazia isso) criava
    um pico artificial gigante bem na borda do transcript, inflando a
    profundidade de qualquer corte real perto dali. Além disso, só um corte
    que é um MÍNIMO LOCAL de verdade da curva de similaridade (mais baixo
    que os dois vizinhos válidos mais próximos) é candidato a fronteira —
    um ponto no meio de uma encosta descendo suavemente até um vale
    genuíno (comum mesmo DENTRO do mesmo assunto, quando as frases usam
    palavras diferentes entre si) não conta sozinho, só o fundo do vale
    conta. Confirmado com teste sintético: sem esses dois cuidados, pontos
    a caminho do vale real e um corte na borda do transcript apareciam com
    profundidade quase tão alta quanto a fronteira de tópico de verdade.

    Retorna uma lista com len(sentences)-1 valores (um por ponto de corte
    entre frases consecutivas), 0 (claramente mesmo assunto, ou sem dado
    suficiente) a 1 (fronteira de tópico forte)."""
    n = len(sentences)
    if n < 3:
        return [0.0] * max(n - 1, 0)

    tokenized = [_content_words(s["text"]) for s in sentences]

    def window_counts(start_idx: int, direction: int):
        counts: dict = {}
        total = 0
        idx = start_idx
        while 0 <= idx < n and total < target_words:
            for w in tokenized[idx]:
                counts[w] = counts.get(w, 0) + 1
                total += 1
            idx += direction
        return counts, total

    # None = dado insuficiente nesse corte -- fica de fora da busca de
    # pico/vale (ver docstring acima), não conta como candidato nem serve
    # de referência de pico pros vizinhos.
    similarities: List[Optional[float]] = []
    for i in range(n - 1):
        left, lc = window_counts(i, -1)
        right, rc = window_counts(i + 1, 1)
        if lc < min_words or rc < min_words:
            similarities.append(None)
        else:
            similarities.append(_cosine_sim(left, right))

    valid_positions = [i for i, s in enumerate(similarities) if s is not None]
    depths = [0.0] * len(similarities)
    for pos, i in enumerate(valid_positions):
        sim = similarities[i]
        # só é candidato a fronteira se for um mínimo local de verdade
        # (mais baixo que o vizinho válido de cada lado) -- um ponto numa
        # encosta descendo suavemente até o vale real fica com depth=0.
        prev_i = valid_positions[pos - 1] if pos > 0 else None
        next_i = valid_positions[pos + 1] if pos + 1 < len(valid_positions) else None
        is_local_min = (
            (prev_i is None or sim <= similarities[prev_i])
            and (next_i is None or sim <= similarities[next_i])
        )
        if not is_local_min:
            continue

        peak_left = sim
        for k in range(pos - 1, -1, -1):
            j = valid_positions[k]
            if similarities[j] >= peak_left:
                peak_left = similarities[j]
            else:
                break
        peak_right = sim
        for k in range(pos + 1, len(valid_positions)):
            j = valid_positions[k]
            if similarities[j] >= peak_right:
                peak_right = similarities[j]
            else:
                break
        depths[i] = max((peak_left - sim) + (peak_right - sim), 0.0)

    max_depth = max(depths) if depths else 0.0
    if max_depth <= 1e-6:
        return [0.0] * len(depths)
    return [d / max_depth for d in depths]


def _ends_on_topic_conclusion(last_sentence_text: str) -> bool:
    """A ÚLTIMA frase da janela soa como um fechamento explícito do que
    estava sendo dito (ex.: "e é isso", "resumindo")."""
    tl = f" {last_sentence_text.strip().lower()} "
    for pat in CONCLUSIVE_END_PATTERNS_PT + CONCLUSIVE_END_PATTERNS_EN:
        if re.search(pat, tl):
            return True
    return False


def _starts_new_topic(next_sentence_text: str) -> bool:
    """A frase seguinte (fora da janela) já começa puxando outro assunto."""
    tl = f" {next_sentence_text.strip().lower()} "
    for pat in TOPIC_SHIFT_START_PATTERNS_PT + TOPIC_SHIFT_START_PATTERNS_EN:
        if re.search(pat, tl):
            return True
    return False


def _ends_on_open_question(last_sentence_text: str) -> bool:
    """A ÚLTIMA frase da janela parece uma pergunta-gancho sem resposta
    (termina com '?' ou casa um padrão de 'setup' de piada/pergunta)."""
    t = last_sentence_text.strip()
    if t.endswith("?"):
        return True
    tl = f" {t.lower()} "
    for pat in SETUP_ENDING_PATTERNS_PT + SETUP_ENDING_PATTERNS_EN:
        if re.search(pat, tl):
            return True
    return False


# --- Início de frase que pressupõe contexto externo (achado real: ver
# clipe do usuário "Mas ao fazer todas essas medidas eu vou abrir de...")
# ---
# `is_topic_boundary`/`starts_clean` (abaixo) decidem se um início é "limpo"
# olhando a PAUSA/marcador ANTES da frase — mas um orador frequentemente faz
# uma pausa retórica de efeito bem ali ANTES de "Mas..." (pra dar ênfase ao
# contraste que vem a seguir), o que empurra a pausa pra cima do limiar de
# fim-de-assunto mesmo sem ter havido troca de assunto nenhuma. Resultado
# visto na prática: um clipe inteiro começando com "Mas ao fazer..." — a
# palavra "Mas" sozinha já avisa que o que vem a seguir é a continuação/
# contraste de uma ideia anterior, e cortar bem ali joga fora exatamente a
# ideia que dá sentido ao "Mas". Isso é um sinal de CONTEÚDO da própria
# frase que abre a janela, independente de quão longa foi a pausa antes
# dela — por isso mora aqui como uma checagem separada, não como mais um
# ajuste de limiar de pausa (que já está no ponto certo pra outros casos).
# Só os conectivos FORTES de contraste/continuação entram aqui (não "e"
# nem "então" sozinhos — usados demais como muleta de fala em PT-BR
# coloquial pra servirem de sinal confiável sem gerar falso positivo
# demais e recusar começos legítimos).
STARTS_MID_THOUGHT_PATTERNS_PT = [
    r"^mas\b", r"^só que\b", r"^porém\b", r"^contudo\b", r"^entretanto\b",
    r"^no entanto\b", r"^apesar disso\b", r"^mesmo assim\b",
]
# pergunta-muleta no fim da frase ("..., né?", "tá ligado?"): não conta
# como pergunta-gancho
_TAG_QUESTION_RE = re.compile(
    r"[,\s]*\b(né|não é|tá ligado|entendeu|sabe|certo|cara|mano|right|you know)\?+\s*$",
    re.IGNORECASE)
STARTS_MID_THOUGHT_PATTERNS_EN = [
    r"^but\b", r"^however\b", r"^yet\b", r"^even so\b", r"^that said\b",
]


def _starts_mid_thought(first_sentence_text: str) -> bool:
    """A PRIMEIRA frase da janela abre com um conectivo forte de contraste/
    continuação (ver comentário acima) — sinal de que o clipe começa em
    cima de uma ideia que só faz sentido com o que veio antes."""
    t = first_sentence_text.strip().lower()
    for pat in STARTS_MID_THOUGHT_PATTERNS_PT + STARTS_MID_THOUGHT_PATTERNS_EN:
        if re.match(pat, t):
            return True
    return False


@dataclass
class ClipCandidate:
    start: float
    end: float
    text: str
    score: float
    title: str
    # Lista de palavras (com timestamp ABSOLUTO no vídeo de origem) a usar
    # para gerar a legenda deste candidato especificamente. Vazio = usa o
    # transcript.words completo passado separadamente para build_clip (uso
    # normal, vídeo transcrito de uma vez só). Preenchido pelo pipeline de
    # vídeo longo (src/long_video.py), onde cada bloco tem sua própria
    # transcrição parcial e não existe um transcript.words único do vídeo
    # inteiro para passar adiante.
    words: List[Word] = field(default_factory=list)
    # frase-gancho que originou o clipe (vira o título / balão do topo)
    hook_text: str = ""

    @property
    def duration(self):
        return self.end - self.start


# palavras de gancho GENÉRICAS demais pra valer como um padrão de gancho
# inteiro: achado real — "Nunca vi, eu preciso ver, vamos ver!" virava o
# melhor gancho do vídeo só por ter "nunca"
WEAK_HOOK_PATTERNS = [
    r"\bnunca\b", r"\bsempre\b", r"\bverdade\b", r"\bimportante\b",
    r"\bna real\b", r"\bé sério\b",
]


def _text_score(text: str) -> float:
    t = f" {text.lower()} "
    score = 0.0
    for pat in HOOK_PATTERNS_PT + HOOK_PATTERNS_EN:
        if re.search(pat, t):
            score += 3.0
    for pat in WEAK_HOOK_PATTERNS:
        if re.search(pat, t):
            score += 1.0
    for w in EMOTION_WORDS:
        score += 2.0 * t.count(w)
    score += 1.5 * t.count("?")
    score += 1.0 * t.count("!")
    score += 0.5 * len(re.findall(r"\b\d+\b", t))
    for f in FILLERS:
        score -= 0.3 * t.count(f)
    return max(score, 0.0)


def _energy_score(energies, start: float, end: float, hop: float = 0.5) -> float:
    i0 = int(start / hop)
    i1 = max(int(end / hop), i0 + 1)
    window = energies[i0:i1]
    if len(window) == 0:
        return 0.0
    # combina energia média (entusiasmo geral) com variância (picos de ênfase)
    return float(window.mean() * 4.0 + window.std() * 6.0)


# --- Frase "fantasma" cortada no meio por pausa de respiração (achado
# real: clipe do usuário terminando em "...bilhões que a gente") ---
# `_build_sentences` (abaixo) fecha uma frase quando acha pontuação OU uma
# pausa > max_pause — mas gente pausa pra RESPIRAR ou pensar no meio de uma
# oração o tempo todo, sem ter terminado o pensamento (comum em fala
# política/argumentativa, cheia de intercalações). Se a pausa cai logo
# depois de uma preposição/conjunção/artigo, é um sinal forte de que a
# oração NÃO terminou ali de verdade — nenhuma dessas palavras encerra uma
# frase completa em português ou inglês. Nesse caso, a pausa sozinha não
# fecha a frase (só pontuação, ou uma pausa MUITO maior — ver
# `_INCOMPLETE_CLAUSE_HARD_PAUSE` abaixo — força o fechamento de qualquer
# jeito, pra não acumular uma "frase" infinita se a transcrição nunca vier
# com pontuação de verdade).
_INCOMPLETE_CLAUSE_ENDINGS_PT = {
    "que", "e", "mas", "ou", "se", "a", "o", "os", "as", "um", "uma", "uns",
    "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
    "para", "pra", "com", "por", "pelo", "pela", "ao", "aos", "à", "às",
    "num", "numa", "meu", "minha", "seu", "sua", "nosso", "nossa", "sem",
}
_INCOMPLETE_CLAUSE_ENDINGS_EN = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by",
    "and", "or", "but", "if", "that", "as", "from", "into", "than",
}
_INCOMPLETE_CLAUSE_ENDINGS = _INCOMPLETE_CLAUSE_ENDINGS_PT | _INCOMPLETE_CLAUSE_ENDINGS_EN
_INCOMPLETE_CLAUSE_HARD_PAUSE = 2.5  # segundos — acima disso, fecha a frase de
                                     # qualquer forma, mesmo terminando numa
                                     # dessas palavras (rede de segurança)


def _build_sentences(transcript: Transcript, max_pause: float = 0.6) -> List[dict]:
    """Agrupa palavras em 'sentenças' usando pausas e pontuação como limites,
    para garantir que os cortes nunca comecem/terminem no meio de uma frase."""
    sentences = []
    cur_words: List[Word] = []

    def flush():
        if cur_words:
            sentences.append({
                "start": cur_words[0].start,
                "end": cur_words[-1].end,
                "text": " ".join(w.text for w in cur_words),
            })

    all_words = transcript.words
    for i, w in enumerate(all_words):
        cur_words.append(w)
        ends_sentence = bool(re.search(r"[.!?…]$", w.text))
        pause_after = 0.0
        if i + 1 < len(all_words):
            pause_after = all_words[i + 1].start - w.end
        last_word_bare = re.sub(r"[^\wà-öø-ÿ]", "", w.text.lower())
        looks_incomplete = (
            not ends_sentence
            and last_word_bare in _INCOMPLETE_CLAUSE_ENDINGS
            and pause_after < _INCOMPLETE_CLAUSE_HARD_PAUSE
        )
        if (ends_sentence or pause_after > max_pause) and not looks_incomplete:
            flush()
            cur_words = []
    flush()
    return sentences


def _make_title(text: str, max_words: int = 10) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    title = " ".join(words[:max_words])
    if len(words) > max_words:
        title += "..."
    return title.capitalize() if title else "Clip"


def _boundary_confirmed(sentences: List[dict], tokenized: List[List[str]], end_idx: int,
                         confirm_words: int, max_similarity: float, min_words: int,
                         pause_threshold: float) -> bool:
    """Pedido explícito do usuário: em vez de aceitar o PRIMEIRO sinal de
    fim de assunto (pausa, frase de fechamento, pico do TextTiling) que
    aparecer, confirma que o assunto genuinamente terminou olhando MAIS
    ALÉM do que a janela padrão de detecção enxerga (`confirm_words` >
    `TOPIC_LEXICAL_TARGET_WORDS`) — pega o caso de uma pausa ou comentário
    passageiro que parece fim de assunto bem de perto, mas o mesmo assunto
    volta a aparecer um pouco mais adiante (o orador faz uma pausa, um
    parêntese curto, e RETOMA o mesmo papo).

    Como isso "empurra o corte mais pra frente" (nas palavras do usuário):
    quando a confirmação falha aqui, a função só devolve False — o loop
    principal em select_clips() já escaneia end_idx em ORDEM CRESCENTE e
    escolhe o candidato válido mais próximo da duração ideal; ele
    naturalmente ignora esse ponto rejeitado e considera o PRÓXIMO ponto
    candidato adiante sozinho. Não precisa de um loop de "empurrar"
    separado — a rejeição aqui já basta pra ter esse efeito.

    CORREÇÃO (achado por revisão, não por teste): a janela de confirmação
    (até `confirm_words` palavras de conteúdo) podia "vazar" bem além do
    assunto imediatamente seguinte, se ele fosse curto — comparando o
    assunto atual contra uma MISTURA de dois ou mais assuntos diferentes
    lá na frente, em vez de só o próximo. Isso arrisca uma falsa
    "semelhança" (e uma rejeição indevida do fim de assunto) só por
    coincidência de vocabulário com um assunto ainda mais adiante, que nem
    é o candidato a "voltar" que a confirmação deveria estar checando.
    Correção: para de acumular cada lado assim que cruzar OUTRA pausa
    grande o bastante pra ser ela mesma um sinal de troca de assunto
    (`pause_threshold`, o mesmo limiar adaptativo já usado pra detectar
    fronteira — não é circular, porque aqui é usado só como um "freio" na
    largura da janela, não pra decidir se ESTE ponto é fronteira) — mesmo
    que isso feche a janela antes de `confirm_words` palavras. Nesse caso,
    o guard de `min_words` abaixo aceita por padrão em vez de arriscar uma
    comparação com material insuficiente."""
    n = len(tokenized)
    if end_idx + 1 >= n:
        return True  # não sobra mais nada depois pra desconfirmar contra
    left_counts: dict = {}
    left_total = 0
    idx = end_idx
    while idx >= 0 and left_total < confirm_words:
        for w in tokenized[idx]:
            left_counts[w] = left_counts.get(w, 0) + 1
            left_total += 1
        if idx > 0 and (sentences[idx]["start"] - sentences[idx - 1]["end"]) >= pause_threshold:
            break  # já cruzou outro assunto pra trás -- não acumula mais
        idx -= 1
    right_counts: dict = {}
    right_total = 0
    idx = end_idx + 1
    while idx < n and right_total < confirm_words:
        for w in tokenized[idx]:
            right_counts[w] = right_counts.get(w, 0) + 1
            right_total += 1
        if idx + 1 < n and (sentences[idx + 1]["start"] - sentences[idx]["end"]) >= pause_threshold:
            break  # já cruzou outro assunto pra frente -- não acumula mais
        idx += 1
    if left_total < min_words or right_total < min_words:
        return True  # material insuficiente pra desconfirmar -- aceita por padrão
    return _cosine_sim(left_counts, right_counts) < max_similarity


def _typical_pause(sentences: List[dict]) -> float:
    """Pausa 'normal' entre frases NESSE vídeo específico (mediana dos
    intervalos positivos entre o fim de uma frase e o início da próxima).
    Usado para detectar pausa de troca de assunto de forma ADAPTATIVA em
    vez de um limiar fixo — um podcast de fala rápida e um de fala pausada
    têm ritmos de pausa completamente diferentes; um limiar fixo (só)
    calibrado para um dos dois erra no outro."""
    gaps = []
    for i in range(len(sentences) - 1):
        g = sentences[i + 1]["start"] - sentences[i]["end"]
        if g > 0:
            gaps.append(g)
    if not gaps:
        return 0.0
    gaps.sort()
    mid = len(gaps) // 2
    if len(gaps) % 2 == 0:
        return (gaps[mid - 1] + gaps[mid]) / 2
    return gaps[mid]


def select_clips(transcript: Transcript, audio_path: str, total_duration: float,
                  n_clips: int, is_first_segment: bool = True,
                  is_last_segment: bool = True) -> List[ClipCandidate]:
    """`is_first_segment`/`is_last_segment`: True (padrão) quando `transcript`
    cobre um vídeo INTEIRO — nesse caso, a borda do transcript (primeira/
    última frase vistas) é tratada como um começo/fim de assunto válido por
    definição, já que não existe nada além dela.

    Passe False quando `transcript` é só um TRECHO (bloco) de um vídeo maior
    -- ver long_video.py -- pra evitar tratar um corte arbitrário de bloco
    como se fosse um início/fim de assunto de verdade. Ver V7 abaixo."""
    print(f"[3/6] Analisando transcrição e áudio para encontrar os {n_clips} melhores cortes...")

    sentences = _build_sentences(transcript)
    if not sentences:
        raise RuntimeError("Não foi possível segmentar a transcrição em frases.")

    energies = audio_energy(audio_path, total_duration)
    typical_pause = _typical_pause(sentences)

    # V3 — CORREÇÃO DA CAUSA RAIZ DE VERDADE (v1 e v2 tratavam sintoma, não
    # causa; ver histórico completo em RELATORIO_PROXIMOS_PASSOS.txt itens 9
    # e 12). Diagnóstico final: mesmo com a v2 (sem gate bloqueando janela
    # longa, bônus de fim-de-assunto), o problema continuava porque
    # _text_score soma pontos sobre o texto ACUMULADO da janela — quase
    # sempre CRESCE com mais frases (mais chance de bater algum HOOK_PATTERN,
    # "?", "!", número), e a única coisa que se opunha a isso
    # (abs(dur-IDEAL)*0.03) era fraca demais: 100s a mais custavam só 3
    # pontos, contra dezenas de pontos possíveis de ganho de texto. Resultado:
    # entre vários limites de assunto válidos, o mais LONGO quase sempre
    # vencia por score — o mesmo sintoma ("sempre bate o teto") com outra
    # cara.
    #
    # A CORREÇÃO NÃO é aumentar o peso da duração dentro da mesma fórmula (é
    # só adiar o mesmo problema pra outro número) — é SEPARAR as duas
    # decisões, como devem ser:
    #   (1) "onde ESSE assunto/trecho termina" -> decidido primariamente
    #       pela PROXIMIDADE de IDEAL_CLIP_DURATION entre os limites de
    #       assunto DISPONÍVEIS (pausa/marcador/fim-de-vídeo) a partir de
    #       cada início — não mais uma corrida de pontuação de texto sem
    #       teto contra uma penalidade fraca.
    #   (2) "qual TRECHO do vídeo (entre vários pontos de início possíveis)
    #       é o melhor pra virar clipe" -> aí sim decidido pelo conteúdo
    #       (gancho/emoção/energia de áudio), comparando as janelas já
    #       fechadas na etapa (1) entre si.
    # Ou seja: agora existe NO MÁXIMO uma janela candidata por ponto de
    # início (a que melhor fecha o assunto perto da duração alvo), em vez de
    # uma pra cada duração possível competindo por pontuação de texto sem
    # limite.
    topic_grace = getattr(config, "TOPIC_BOUNDARY_GRACE_SECONDS", 0.0)
    hard_max_duration = config.MAX_CLIP_DURATION + topic_grace
    duration_fit_weight = getattr(config, "TOPIC_DURATION_FIT_WEIGHT", 0.3)
    pause_threshold = max(
        config.TOPIC_BOUNDARY_PAUSE_SECONDS,
        typical_pause * getattr(config, "TOPIC_BOUNDARY_PAUSE_RELATIVE_MULT", 1.0),
    )
    # Coesão lexical (TextTiling) — ver _lexical_boundary_scores acima.
    # Terceiro sinal de fim-de-assunto, além de pausa/palavra-chave: pega
    # troca de assunto "silenciosa" (sem pausa nem frase de transição)
    # medindo o vocabulário em vez de procurar um marcador específico.
    lexical_min = getattr(config, "TOPIC_LEXICAL_BOUNDARY_MIN", 0.55)
    lexical_scores = _lexical_boundary_scores(
        sentences,
        target_words=getattr(config, "TOPIC_LEXICAL_TARGET_WORDS", 40),
        min_words=getattr(config, "TOPIC_LEXICAL_MIN_WORDS", 8),
    )

    # V9 — CONFIRMAÇÃO POR LOOKAHEAD (pedido explícito do usuário: "ele
    # analisa as frases seguintes, e, caso elas continuem o assunto, ele
    # retira o ponto de fim de vídeo e joga mais pra frente, e vai fazendo
    # isso até achar o ponto onde o assunto acabou"). `_boundary_confirmed`
    # (definida acima) já implementa exatamente essa ideia — compara o
    # vocabulário numa janela BEM MAIOR que a de detecção padrão
    # (TOPIC_LEXICAL_CONFIRM_WORDS > TOPIC_LEXICAL_TARGET_WORDS) dos dois
    # lados de cada candidato a fronteira, pra pegar o caso de o assunto
    # "voltar" um pouco mais adiante depois de uma pausa/comentário que só
    # PARECIA fim de assunto de perto. Existia no arquivo mas nunca era
    # chamada em lugar nenhum — a confirmação nunca rodava de verdade.
    # Conectando aqui: só os sinais INDIRETOS (pausa acima do normal, vale
    # do TextTiling) passam por essa confirmação antes de virar fronteira
    # de verdade — quando ela discorda, is_topic_boundary devolve False pra
    # esse ponto, e o loop principal (que já escaneia end_idx em ordem
    # crescente) naturalmente considera o PRÓXIMO ponto adiante, sem
    # precisar de nenhum loop novo. Os sinais EXPLÍCITOS (frase de
    # fechamento tipo "e é isso", ou a próxima frase já abrindo com "mudando
    # de assunto") continuam sendo aceitos direto, sem confirmação — ver o
    # porquê no docstring de is_topic_boundary logo abaixo (resumo: são uma
    # declaração direta do próprio orador, e a confirmação por vocabulário
    # poderia rejeitar um fechamento genuíno só por coincidência léxica com
    # o assunto seguinte).
    tokenized_for_confirm = [_content_words(s["text"]) for s in sentences]
    confirm_words = getattr(config, "TOPIC_LEXICAL_CONFIRM_WORDS", 90)
    confirm_max_similarity = getattr(config, "TOPIC_LEXICAL_CONFIRM_MAX_SIMILARITY", 0.22)
    confirm_min_words = getattr(config, "TOPIC_LEXICAL_MIN_WORDS", 8)

    def is_topic_boundary(end_idx: int) -> bool:
        """Esse fim de FRASE também é um fim de ASSUNTO? (pausa bem acima do
        normal desse vídeo, frase de fechamento, próxima frase já muda de
        assunto, vocabulário mudando de forma significativa (TextTiling —
        pega virada "silenciosa", sem pausa nem palavra-chave), ou é a
        última frase do vídeo DE VERDADE — não só a última frase deste
        TRECHO, se for um bloco de um vídeo maior; ver V7).

        Sinais "moles" (pausa acima do normal, ou vale do TextTiling) ainda
        passam pela confirmação por lookahead (V9) antes de virar fronteira
        de verdade — são inferências indiretas, então vale checar se o
        assunto não volta logo depois.

        Sinais EXPLÍCITOS (o orador literalmente diz algo tipo "e é isso"/
        "resumindo", ou a próxima frase já abre com "mudando de assunto"/
        "falando nisso") NÃO passam pela confirmação — são uma declaração
        direta da própria pessoa falando de que o assunto acabou ali, e
        pedir uma segunda opinião do vocabulário por cima disso poderia
        REJEITAR um fechamento genuíno só porque o próximo assunto
        compartilha alguma palavra em comum por coincidência (ex.: os dois
        assuntos mencionam "governo" ou "brasil" de passagem). Confiar no
        marcador explícito sem confirmação é o comportamento mais seguro
        aqui — é exatamente o que o código já fazia ANTES do V9 (a
        confirmação é uma camada A MAIS só para os sinais indiretos, não
        uma substituição da confiança nos marcadores diretos)."""
        if end_idx + 1 >= len(sentences):
            return is_last_segment
        explicit_marker = (
            _ends_on_topic_conclusion(sentences[end_idx]["text"])
            or _starts_new_topic(sentences[end_idx + 1]["text"])
        )
        if explicit_marker:
            return True
        pause_after = sentences[end_idx + 1]["start"] - sentences[end_idx]["end"]
        signal = (
            pause_after >= pause_threshold
            or lexical_scores[end_idx] >= lexical_min
        )
        if not signal:
            return False
        return _boundary_confirmed(
            sentences, tokenized_for_confirm, end_idx,
            confirm_words=confirm_words,
            max_similarity=confirm_max_similarity,
            min_words=confirm_min_words,
            pause_threshold=pause_threshold,
        )

    # V7 — is_topic_boundary (acima) e o "starts_clean" (abaixo) tratavam
    # a BORDA do transcript disponível (início/fim da lista de `sentences`)
    # como automaticamente um começo/fim de assunto válido. Isso é certo
    # pra um vídeo processado inteiro (não existe nada antes/depois pra
    # comparar mesmo). Mas em vídeo longo (long_video.py), cada bloco de
    # CHUNK_DURATION_SECONDS é transcrito e passa por select_clips()
    # ISOLADAMENTE — a "borda do transcript" de um bloco do MEIO do vídeo é
    # só um corte de tempo arbitrário, não tem nada a ver com onde um
    # assunto de verdade começa ou termina. Sintoma real visto pelo
    # usuário: clipe começando com letra minúscula ("minha produtividade,
    # faço mais...") -- claramente uma frase que já estava em andamento
    # quando o BLOCO começou, tratada como início limpo só por ser a
    # primeira frase que o transcript daquele bloco conseguia ver.
    #
    # Correção: quem chama select_clips() agora informa explicitamente se
    # este transcript É o vídeo inteiro (comportamento antigo, é o padrão)
    # ou só um bloco no meio/início/fim de algo maior — nesse caso as
    # bordas deixam de contar como início/fim de assunto automático, PASSAM
    # A EXIGIR o mesmo sinal real (pausa longa, frase de fechamento, troca
    # de assunto) que qualquer fronteira NO MEIO do transcript já exigia.
    boundary_cache: dict = {}

    def boundary(idx: int) -> bool:
        if idx not in boundary_cache:
            boundary_cache[idx] = is_topic_boundary(idx)
        return boundary_cache[idx]

    def starts_clean(start_idx: int) -> bool:
        if _starts_mid_thought(sentences[start_idx]["text"]):
            return False
        if start_idx == 0:
            return is_first_segment
        return boundary(start_idx - 1)

    # V10 — SELEÇÃO POR GANCHO (pedido do usuário: "procura gancho, pega
    # contexto, faz o clip de 40 segundos a 3 minutos, pode ser qualquer
    # tamanho dentro dessa faixa"). Antes, cada início de frase virava uma
    # janela fechada no fim de assunto MAIS PRÓXIMO de IDEAL_CLIP_DURATION —
    # todos os clipes saíam com a mesma cara de duração (curtos), e o gancho
    # podia nem estar neles. Agora a ordem é a de um editor:
    #   1. acha as frases-GANCHO (padrão de gancho, pergunta, número,
    #      palavra de emoção, pico de energia na voz);
    #   2. volta até o COMEÇO DO ASSUNTO em que o gancho está (o contexto
    #      que faz ele fazer sentido), até HOOK_CONTEXT_MAX_SECONDS antes;
    #   3. vai até o PRIMEIRO fim de assunto confirmado depois do gancho
    #      (com pelo menos HOOK_PAYOFF_MIN_SECONDS de "resposta" depois
    #      dele) — sem alvo de duração: o assunto é que decide, dentro de
    #      [MIN_CLIP_DURATION, MAX_CLIP_DURATION];
    #   4. ranqueia pela força do gancho + densidade de conteúdo + energia,
    #      com prêmio pra começo/fim limpos.
    clean_cache: dict = {}
    opener_cache: dict = {}

    def opens_dependent(idx: int) -> bool:
        """As duas primeiras frases a partir de `idx` dependem do que veio
        antes (ver src/opener.py)?"""
        if idx not in opener_cache:
            nxt = sentences[idx + 1]["text"] if idx + 1 < len(sentences) else ""
            opener_cache[idx] = opening_depends_on_context(sentences[idx]["text"], nxt)
        return opener_cache[idx]

    def clean_start_at(idx: int) -> bool:
        if idx not in clean_cache:
            clean_cache[idx] = starts_clean(idx)
        return clean_cache[idx]

    hop = 0.5  # resolução de `energies` (audio_energy padrão)
    e_mean = float(energies.mean()) if len(energies) else 0.0
    e_std = float(energies.std()) if len(energies) else 0.0
    energy_weight = getattr(config, "HOOK_ENERGY_WEIGHT", 1.5)

    def hook_score(idx: int) -> float:
        sent = sentences[idx]
        text = sent["text"].strip()
        # pergunta-muleta no fim ("..., né?", "tá ligado?") não é pergunta
        core = _TAG_QUESTION_RE.sub("", text)
        if len(core.split()) < 6 or len(_content_words(core)) < 3:
            return 0.0  # "É muito louco né?", "Tu falou a língua?" — curto demais pra gancho
        score = _text_score(core)
        # nomes próprios (palavra capitalizada fora do início de frase):
        # gancho que cita alguém/algum lugar/marca é mais concreto
        proper = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-ZÀ-Ý][a-zà-ÿ]{2,}", core)
        score += min(len(proper), 3) * 0.8
        if text.endswith(("...", "…")):
            score -= 2.5  # frase que morre no meio não segura ninguém
        window = energies[int(sent["start"] / hop):int(sent["end"] / hop) + 1]
        if len(window) and e_std > 1e-6:
            score += float(np.clip((window.mean() - e_mean) / e_std, 0.0, 3.0)) * energy_weight
        return score

    min_dur, max_dur = config.MIN_CLIP_DURATION, config.MAX_CLIP_DURATION
    context_max = min(getattr(config, "HOOK_CONTEXT_MAX_SECONDS", 30.0), max_dur * 0.4)
    payoff_min = min(getattr(config, "HOOK_PAYOFF_MIN_SECONDS", 12.0), min_dur * 0.5)

    # Força de cada fim de assunto: marcador explícito ("enfim", "mudando de
    # assunto") vale mais; pausa longa pro ritmo do vídeo e vale forte do
    # TextTiling somam. Medido num podcast real: o detector confirma uma
    # troca a cada ~24s (qualquer pausa um pouco maior) — fechar o clipe na
    # primeira delas depois do mínimo dava sempre clipes curtos. Pra FECHAR
    # um clipe só contam as trocas FORTES, calibradas por vídeo:
    # ~STRONG_TOPIC_CHANGES_PER_MINUTE delas (as mais fortes do vídeo).
    def strength(idx: int) -> float:
        if idx + 1 >= len(sentences):
            return 3.0 if is_last_segment else 0.0
        st = 0.0
        if (_ends_on_topic_conclusion(sentences[idx]["text"])
                or _starts_new_topic(sentences[idx + 1]["text"])):
            st += 2.0
        pause = sentences[idx + 1]["start"] - sentences[idx]["end"]
        if pause >= pause_threshold:
            st += min(pause / max(pause_threshold, 1e-3), 3.0) * 0.5
        return st + lexical_scores[idx] * 1.5

    all_strengths = sorted((strength(i) for i in range(len(sentences)) if boundary(i)),
                           reverse=True)
    minutes = max((sentences[-1]["end"] - sentences[0]["start"]) / 60.0, 1.0)
    n_strong = max(int(round(minutes * getattr(config, "STRONG_TOPIC_CHANGES_PER_MINUTE", 0.6))), 1)
    strong_min = (all_strengths[min(n_strong, len(all_strengths)) - 1]
                  if all_strengths else float("inf"))

    ranked_hooks = sorted(((hook_score(i), i) for i in range(len(sentences))), reverse=True)
    hooks = [h for h in ranked_hooks if h[0] >= getattr(config, "HOOK_MIN_SCORE", 2.0)]
    if len(hooks) < n_clips:
        hooks = ranked_hooks  # vídeo sem gancho claro: usa os melhores que houver
    hooks = hooks[:max(n_clips * 8, 24)]

    candidates: List[ClipCandidate] = []
    for h_score, h in hooks:
        hook_t = sentences[h]["start"]

        # ETAPA 1: contexto — começo do assunto em que o gancho está
        start_idx, clean_start = h, clean_start_at(h)
        j = h
        while not clean_start and j > 0 and hook_t - sentences[j - 1]["start"] <= context_max:
            j -= 1
            if clean_start_at(j):
                start_idx, clean_start = j, True
        if not clean_start:
            # nenhum começo de assunto ao alcance: começa no próprio gancho,
            # recuando só enquanto a frase abre com conectivo ("Mas...")
            start_idx = h
            while (start_idx > 0 and _starts_mid_thought(sentences[start_idx]["text"])
                   and hook_t - sentences[start_idx - 1]["start"] <= context_max):
                start_idx -= 1
        # a ABERTURA tem que se sustentar sozinha (ver src/opener.py) —
        # achado real: "Fez uma lavagem cerebral, eles viraram louco..." (quem
        # fez? quem são eles?) passava como começo de assunto. Se depende do
        # que veio antes, avança até a primeira frase que se sustenta antes
        # do gancho; sem nenhuma, começa no próprio gancho.
        if opens_dependent(start_idx):
            for j in range(start_idx + 1, h + 1):
                if not opens_dependent(j):
                    start_idx = j
                    break
            else:
                start_idx = h
            clean_start = clean_start_at(start_idx)
        self_contained = not opens_dependent(start_idx)
        start_t = sentences[start_idx]["start"]

        # ETAPA 2: fim — primeira troca de assunto FORTE depois da
        # "resposta" ao gancho; sem nenhuma ao alcance, a troca de assunto
        # mais forte dentro da faixa; sem nenhuma, o fim de frase com o
        # sinal mais forte (pausa + vale lexical)
        end_idx, end_is_boundary, fallback, weak_best = None, False, None, None
        for k in range(h, len(sentences)):
            end_t = sentences[k]["end"]
            dur = end_t - start_t
            if dur > max_dur:
                break
            if dur < min_dur or end_t < sentences[h]["end"] + payoff_min:
                continue
            has_next = k + 1 < len(sentences)
            more_content_exists = has_next or not is_last_segment
            dangling = ((more_content_exists and _ends_on_open_question(sentences[k]["text"]))
                        or (has_next and _starts_mid_thought(sentences[k + 1]["text"])))
            if dangling:
                continue  # pergunta sem resposta / próxima frase é "Mas..."
            if boundary(k):
                if strength(k) >= strong_min:
                    end_idx, end_is_boundary = k, True
                    break
                if weak_best is None or strength(k) > weak_best[0]:
                    weak_best = (strength(k), k)
                continue
            sig = (sentences[k + 1]["start"] - end_t if has_next else 0.0) + lexical_scores[k]
            if fallback is None or sig > fallback[0]:
                fallback = (sig, k)
        if end_idx is None and weak_best is not None:
            end_idx, end_is_boundary = weak_best[1], True
        if end_idx is None and fallback is not None:
            end_idx = fallback[1]
        if end_idx is None:
            continue  # nem a duração mínima coube a partir deste gancho

        # a conclusão do raciocínio às vezes vem logo DEPOIS do fim de
        # assunto detectado ("...não tem vitória." + "A gente nasce
        # perdendo, é isso.") — estica até ela se estiver a 1-2 frases
        for k in range(end_idx + 1, min(end_idx + 3, len(sentences))):
            if (sentences[k]["end"] - start_t > max_dur
                    or sentences[k]["end"] - sentences[end_idx]["end"] > 12.0):
                break
            if _ends_on_topic_conclusion(sentences[k]["text"]):
                end_idx = k
                break

        # ETAPA 3: pontuação
        end_t = sentences[end_idx]["end"]
        dur = end_t - start_t
        text = " ".join(sentences[i]["text"] for i in range(start_idx, end_idx + 1))
        density = _text_score(text) / max(dur / 60.0, 0.5)  # pontos por minuto
        score = 2.0 * h_score + 0.5 * density + _energy_score(energies, start_t, end_t)
        # gancho enterrado no meio do clipe perde força (quem rola o feed
        # decide nos primeiros segundos)
        score -= max(hook_t - start_t - 10.0, 0.0) * 0.08
        # começo: prêmio se abre um assunto E se sustenta sozinho; penalidade
        # se nem isso deu pra garantir (o gancho em si depende do contexto)
        if not self_contained:
            score -= getattr(config, "TOPIC_START_PENALTY", 6.0)
        elif clean_start:
            score += config.TOPIC_BOUNDARY_BONUS * 0.5
        if end_is_boundary:
            score += config.TOPIC_BOUNDARY_BONUS * 0.5

        candidates.append(ClipCandidate(start=start_t, end=end_t, text=text,
                                         score=score, title="",
                                         hook_text=sentences[h]["text"]))

    if not candidates:
        raise RuntimeError(
            "Nenhum candidato de corte válido foi gerado. "
            "O vídeo pode ser curto demais para a duração mínima configurada."
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    chosen: List[ClipCandidate] = []
    for c in candidates:
        if len(chosen) >= n_clips:
            break
        overlaps = any(
            not (c.end + config.MIN_GAP_BETWEEN_CLIPS <= o.start or
                 c.start >= o.end + config.MIN_GAP_BETWEEN_CLIPS)
            for o in chosen
        )
        if not overlaps:
            c.title = _make_title(c.hook_text or c.text)
            chosen.append(c)

    if not chosen:
        raise RuntimeError("Não foi possível selecionar clipes sem sobreposição.")

    chosen.sort(key=lambda c: c.start)

    if len(chosen) < n_clips:
        print(f"    [aviso] só foi possível encontrar {len(chosen)} corte(s) "
              f"sem sobreposição (pedidos: {n_clips}).")

    for i, c in enumerate(chosen, 1):
        print(f"    -> Clip {i}: {c.start:6.1f}s - {c.end:6.1f}s "
              f"({c.duration:4.1f}s) score={c.score:.1f}  \"{c.title}\"")

    return chosen
