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

from .transcriber import Transcript, Word
from .audio import audio_energy
from . import config


HOOK_PATTERNS_PT = [
    r"\bvocê (sabia|acredita|imagina)\b", r"\bninguém (te|fala|conta|ensina|mostra)\b",
    r"\bsegredo\b", r"\bo (maior|pior|melhor) erro\b", r"\bverdade\b", r"\bisso mudou\b",
    r"\bnunca\b", r"\bsempre\b", r"\bpor que\b", r"\bcomo (eu|fazer|fiz|consigo)\b",
    r"\b\d+ (dicas|passos|motivos|razões|formas|maneiras|coisas|erros)\b",
    r"\bimportante\b", r"\bo que ninguém\b",
    r"\bvocê está fazendo (errado|isso errado)\b",
    r"\bimagina (se|só)\b", r"\bisso (é|foi) loucura\b",
    # PT-BR coloquial / viral
    r"\bminha gente\b", r"\bboa notícia\b", r"\bnotícia (boa|ruim|incrível)\b",
    r"\bmuita gente (não|faz|sabe|percebe)\b", r"\bnão faz isso\b",
    r"\bperdendo (dinheiro|tempo|oportunidade)\b", r"\bvai mudar\b",
    r"\bse você (faz|fizer|está)\b", r"\bpresta atenção\b",
    r"\bcalma (que|aí)\b", r"\bespera (aí|só)\b", r"\bsó (uma|um) (coisa|momento)\b",
    r"\bfato (é|curioso)\b", r"\bna real\b", r"\bé sério\b",
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

    @property
    def duration(self):
        return self.end - self.start


def _text_score(text: str) -> float:
    t = f" {text.lower()} "
    score = 0.0
    for pat in HOOK_PATTERNS_PT + HOOK_PATTERNS_EN:
        if re.search(pat, t):
            score += 3.0
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
    def starts_clean(start_idx: int) -> bool:
        if _starts_mid_thought(sentences[start_idx]["text"]):
            return False
        if start_idx == 0:
            return is_first_segment
        return is_topic_boundary(start_idx - 1)

    # Histórico de por que início E fim de assunto são checados (V4/V5) e
    # por que a checagem virou pontuação em vez de filtro rígido (V5) está
    # documentado em starts_clean()/is_topic_boundary() acima.
    candidates: List[ClipCandidate] = []
    for start_idx in range(len(sentences)):
        start_t = sentences[start_idx]["start"]
        clean_start = starts_clean(start_idx)

        # ETAPA 1: escolhe QUAL fim de assunto usar para este início — a
        # duração (proximidade de IDEAL_CLIP_DURATION) é o critério
        # DOMINANTE aqui; um pequeno ajuste de conteúdo só desempata entre
        # limites de assunto de duração parecida (prefere uma conclusão
        # explícita a uma pergunta-gancho sem resposta, por exemplo) — nunca
        # o suficiente pra justificar pular pra um limite muito mais distante
        # só por ter mais texto-gancho acumulado.
        best_end_idx = None
        best_fit_score = None
        best_is_boundary = False
        fallback_end_idx = None  # melhor fim de FRASE simples (sem exigir
                                  # fim de assunto), só usado se nenhum fim
                                  # de assunto existir dentro do alcance
        fallback_fit = None

        for end_idx in range(start_idx, len(sentences)):
            end_t = sentences[end_idx]["end"]
            dur = end_t - start_t
            if dur < config.MIN_CLIP_DURATION:
                continue
            if dur > hard_max_duration:
                break

            dur_penalty = abs(dur - config.IDEAL_CLIP_DURATION) * duration_fit_weight
            has_next = end_idx + 1 < len(sentences)
            # Achado real (clipe do usuário terminando em "...que eu vou
            # fazer?", uma pergunta aberta): `has_next` só enxerga a lista
            # de frases DESTE bloco/chunk (ver long_video.py) — quando o
            # vídeo é longo e processado em blocos de ~20min, a pergunta
            # caiu bem na ÚLTIMA frase TRANSCRITA daquele bloco. `has_next`
            # dava False ali (não achou mais nada na lista local), então a
            # penalidade de pergunta-sem-resposta nunca disparava — mesmo
            # com `is_last_segment=False` já sabendo corretamente que o
            # vídeo de origem CONTINUA depois daquele bloco (a resposta
            # bem provavelmente está a poucos segundos dali, só que no
            # próximo bloco, que ainda nem foi transcrito). "Não achei mais
            # frase na minha lista" e "não existe mais vídeo depois disso"
            # são coisas DIFERENTES, e o código tratava as duas como se
            # fossem a mesma. `is_last_segment` já existe e já é usado pra
            # esse exato propósito em outro ponto (ver linha ~559) — só não
            # tinha sido conectado aqui.
            # Duas variantes: pra pergunta-sem-resposta (só olha a frase
            # ATUAL, `sentences[end_idx]`) basta saber que existe algo
            # depois, mesmo sem ter o texto — `more_content_exists` cobre
            # isso. Já a checagem de "próxima frase começa com Mas..."
            # PRECISA do texto de verdade da próxima frase pra funcionar,
            # que só existe se ela estiver na nossa lista local — por isso
            # continua exigindo `has_next` (índice válido), não o mais
            # amplo `more_content_exists` (evita também IndexError).
            more_content_exists = has_next or not is_last_segment
            effective_penalty = dur_penalty
            if more_content_exists and _ends_on_open_question(sentences[end_idx]["text"]):
                effective_penalty += config.DANGLING_QUESTION_PENALTY
            # Simétrico ao caso de "Mas ao fazer..." corrigido no início da
            # janela (ver item 17 do RELATORIO_PROXIMOS_PASSOS.txt): lá, o
            # problema era o clipe COMEÇAR em cima de um conectivo de
            # contraste/continuação. Aqui é o espelho — o clipe TERMINA bem
            # antes de uma frase (excluída, fora do clipe) que abre com um
            # desses conectivos. Ex.: corta em "...e conseguimos reduzir o
            # déficit." e a próxima frase (fora do clipe) é "Mas isso trouxe
            # um problema seríssimo pro emprego." — o espectador fica com uma
            # versão só otimista/só pessimista de um argumento que na
            # gravação original tinha as duas pontas. Reaproveita
            # `_starts_mid_thought` (mesma lista de conectivos fortes) em vez
            # de duplicar a lista — o sinal é o mesmo, só o lado que muda.
            if has_next and _starts_mid_thought(sentences[end_idx + 1]["text"]):
                effective_penalty += config.DANGLING_QUESTION_PENALTY

            if dur <= config.MAX_CLIP_DURATION and (fallback_fit is None or effective_penalty < fallback_fit):
                fallback_fit, fallback_end_idx = effective_penalty, end_idx

            if not is_topic_boundary(end_idx):
                continue

            fit_score = -dur_penalty
            if has_next and _ends_on_topic_conclusion(sentences[end_idx]["text"]):
                fit_score += config.TOPIC_BOUNDARY_BONUS
            if more_content_exists and _ends_on_open_question(sentences[end_idx]["text"]):
                fit_score -= config.DANGLING_QUESTION_PENALTY
            if has_next and _starts_mid_thought(sentences[end_idx + 1]["text"]):
                fit_score -= config.DANGLING_QUESTION_PENALTY
            # pequeno desempate: entre limites de assunto de duração
            # parecida, inclina levemente pro que tem um "vale" lexical
            # mais forte (mais confiança de que é troca de assunto de
            # verdade, não só uma pausa comum) — não decide sozinho
            # (peso pequeno), só ajuda a desempatar.
            if has_next:
                fit_score += lexical_scores[end_idx] * getattr(
                    config, "TOPIC_LEXICAL_FIT_BONUS", 1.0)

            if best_fit_score is None or fit_score > best_fit_score:
                best_fit_score, best_end_idx, best_is_boundary = fit_score, end_idx, True

        if best_end_idx is None:
            # nenhum fim de ASSUNTO detectável dentro do alcance permitido —
            # cai pro fim de FRASE mais próximo da duração ideal, sem
            # esticar até a folga (não há sinal nenhum que justifique
            # esticar; ver honestidade no comentário de config.py).
            best_end_idx = fallback_end_idx

        if best_end_idx is None:
            continue  # nem a duração mínima coube a partir deste início

        # ETAPA 2: com a janela já fechada, o conteúdo decide o quão boa ela
        # é (usado só pra comparar ESTE início contra os outros, não pra
        # escolher o próprio fim — isso já foi decidido acima).
        end_t = sentences[best_end_idx]["end"]
        text = " ".join(sentences[i]["text"] for i in range(start_idx, best_end_idx + 1))
        score = _text_score(text) + _energy_score(energies, start_t, end_t)
        # normaliza pelo comprimento — sem isso clipes longos acumulam mais
        # hooks que clipes curtos e sempre ganham mesmo sendo piores pra Shorts
        dur_ratio = (end_t - start_t) / max(config.IDEAL_CLIP_DURATION, 1.0)
        if dur_ratio > 1.0:
            score /= dur_ratio ** 0.55
        if best_is_boundary:
            score += config.TOPIC_BOUNDARY_BONUS * 0.5  # pequeno prêmio por
                                                          # ser um fechamento
                                                          # de assunto de
                                                          # verdade, não só
                                                          # o encaixe de
                                                          # duração do
                                                          # fallback
        if clean_start:
            score += config.TOPIC_BOUNDARY_BONUS * 0.5  # mesmo prêmio, agora
                                                          # simétrico pro início
        else:
            score -= getattr(config, "TOPIC_START_PENALTY", 6.0)  # começa no
                                                                    # MEIO de
                                                                    # um assunto
                                                                    # em andamento

        candidates.append(ClipCandidate(start=start_t, end=end_t, text=text,
                                         score=score, title=""))

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
            c.title = _make_title(c.text)
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
