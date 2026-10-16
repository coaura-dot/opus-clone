"""
A frase que ABRE um clipe se sustenta sozinha, ou depende do que veio
antes no vídeo?

Achado real (podcast): um clipe começava em "Fez uma lavagem cerebral,
eles viraram louco e começaram a fazer só as merda." — quem fez? quem são
"eles"? Quem cai no clipe pelo feed não viu o que veio antes, então esse
começo não faz sentido nenhum. Regex de pronoun no começo da frase não pega
isso (a frase começa com um verbo).

Com o spaCy (modelo pt_core_news_sm, análise gramatical local em CPU), uma
abertura depende do contexto quando:
  - o primeiro verbo conjugado está na 3ª pessoa e não tem sujeito
    explícito antes dele ("Fez...", "Começou a falar...", "Foi 99.") —
    em PT-BR o sujeito oculto de 3ª pessoa aponta pra alguém citado antes
    (o de 1ª pessoa, "Fui lá...", é o próprio falante e está ok);
  - o sujeito é um pronome de 3ª pessoa/demonstrativo ("Na verdade, ele é
    um jovem..." — quem?);
e NÃO depende quando é uma pergunta pra quem assiste ("Lembra do Felipe
Solari?", "Sabe o que...?"), um imperativo ("Imagina...", "Olha...",
"Assistam isso") ou uma frase existencial ("Tem um cara que...", "Existe...").

Sem o spaCy instalado, cai pra uma lista curta dos verbos mais comuns em
3ª pessoa no começo de frase falada.
"""
import re
from functools import lru_cache

_ANAPHORIC_PRONOUNS = {"ele", "ela", "eles", "elas", "isso", "isto", "aquilo",
                       "esse", "essa", "esses", "essas", "aquele", "aquela"}
# verbos que, sem sujeito, estão falando COM quem assiste (você) — perguntas
# e imperativos típicos de abertura de corte
_LISTENER_VERBS = {"imagina", "imaginem", "olha", "olhem", "pensa", "pensem",
                   "presta", "prestem", "escuta", "escutem", "repara", "reparem",
                   "veja", "vejam", "vê", "lembra", "lembram", "sabe", "sabem",
                   "conhece", "conhecem", "assiste", "assistam", "assistiu",
                   "segura", "espera", "calma", "bora", "vamos"}
# existenciais/impessoais: não apontam pra ninguém citado antes
_IMPERSONAL_LEMMAS = {"haver", "existir", "acontecer", "parecer", "chover"}
_IMPERSONAL_FORMS = {"tem", "têm", "tinha", "teve", "há", "existe", "existem",
                     "acontece", "aconteceu", "parece", "dá", "deu", "rola"}
# fallback sem spaCy: verbos de 3ª pessoa mais comuns abrindo frase falada
_FALLBACK_3RD_PERSON = {
    "fez", "foi", "era", "disse", "falou", "começou", "virou", "ficou", "veio",
    "chegou", "pegou", "mandou", "fica", "faz", "vai", "pode", "quer", "tava",
    "estava", "ia", "viu", "achou", "pensou", "perguntou", "respondeu", "saiu",
    "entrou", "voltou", "morreu", "nasceu", "ganhou", "perdeu", "levou",
    "trouxe", "botou", "colocou", "matou", "roubou", "mentiu", "fala", "falava",
    "gosta", "gostava", "acha", "achava", "sabia", "conhecia", "tinham", "eram",
    "foram", "fizeram", "viraram", "começaram", "disseram", "falaram",
}
_WEAK_START_RE = re.compile(
    r"^\s*(mas|só que|porém|contudo|entretanto|no entanto|e|aí|daí|também|tipo|"
    r"porque|pois|que|então|ele|ela|eles|elas|isso|isto|esse|essa|esses|essas|aquilo|"
    r"aquele|aquela|he|she|they|it|that|this|but|and|so)\b", re.IGNORECASE)


# oração de finalidade solta ("Pra falar que...", "Para mostrar que...") é o
# fim de uma frase anterior ("[fizeram isso] pra falar que..."). O spaCy erra
# o "pra" coloquial (marca como substantivo), então vai por regex; "Pra
# mim, ..." (opinião) não entra — o verbo tem que estar no infinitivo.
_PURPOSE_CLAUSE_RE = re.compile(r"^\s*(pra|para)\s+[a-zà-ÿ]+(ar|er|ir|or)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        return spacy.load("pt_core_news_sm", disable=["ner", "lemmatizer"])
    except Exception:
        return None


def _first_word(text: str) -> str:
    m = re.match(r"\W*([\wÀ-ÿ]+)", text)
    return m.group(1).lower() if m else ""


@lru_cache(maxsize=4096)
def depends_on_context(text: str) -> bool:
    """True se a frase, aberta sozinha no começo de um clipe, depende do que
    veio antes pra fazer sentido."""
    text = text.strip()
    if not text:
        return True
    if text[:1].islower() or _WEAK_START_RE.match(text):
        return True  # continuação da frase anterior / conectivo / pronome
    if _PURPOSE_CLAUSE_RE.match(text):
        return True
    if text.endswith(("...", "…")) and len(text.split()) <= 5:
        return True  # hesitação que morre no meio ("O 1...", "Mas como é...")
    if _unintroduced_person_pronoun(text):
        return True  # "Você viu ele falando...?" / "Na verdade, ele é..." — quem?
    first = _first_word(text)
    question = text.rstrip().endswith("?")
    if first in _LISTENER_VERBS or (question and first not in _ANAPHORIC_PRONOUNS):
        return False  # falando com quem assiste
    if first in _IMPERSONAL_FORMS:
        return False

    nlp = _nlp()
    if nlp is None:
        return first in _FALLBACK_3RD_PERSON

    doc = nlp(text)
    # só a primeira frase/oração interessa (o spaCy pode ver mais de uma)
    sent = next(iter(doc.sents), doc)
    for tok in sent:
        if tok.dep_.startswith("nsubj"):
            # tem sujeito explícito antes do verbo: depende se for "ele/isso"
            return tok.pos_ == "PRON" and tok.text.lower() in _ANAPHORIC_PRONOUNS
        if tok.pos_ in ("VERB", "AUX") and "Fin" in tok.morph.get("VerbForm"):
            if any(c.dep_.startswith("nsubj") for c in tok.children):
                subj = next(c for c in tok.children if c.dep_.startswith("nsubj"))
                return subj.pos_ == "PRON" and subj.text.lower() in _ANAPHORIC_PRONOUNS
            if tok.lemma_ in _IMPERSONAL_LEMMAS or tok.text.lower() in _IMPERSONAL_FORMS:
                return False
            return "3" in tok.morph.get("Person")  # sujeito oculto de 3ª pessoa
    return False


_PERSON_PRONOUNS = {"ele", "ela", "eles", "elas", "he", "she", "they"}


def _referent_tokens(doc):
    """Índices dos tokens que apresentam alguém/algo (nome próprio ou
    substantivo), ignorando interjeição/vocativo solto ("Puts, ...",
    "..., cara, ...") e "a gente" (= nós)."""
    idx = []
    for i, tok in enumerate(doc):
        if tok.pos_ not in ("PROPN", "NOUN"):
            continue
        after = doc[i + 1].text if i + 1 < len(doc) else ""
        before = doc[i - 1].text if i > 0 else ""
        if after == "," and (i == 0 or before == ","):
            continue
        if tok.text.lower() == "gente" and before.lower() == "a":
            continue
        idx.append(i)
    return idx


def _unintroduced_person_pronoun(text: str) -> bool:
    """"ele/ela/eles/elas" aparece antes de qualquer pessoa/coisa citada na
    própria frase?"""
    nlp = _nlp()
    if nlp is None:
        m = re.search(r"\b(ele|ela|eles|elas)\b", text, re.IGNORECASE)
        return bool(m) and not re.search(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}", text[1:m.start()])
    doc = nlp(text)
    first_ref = min(_referent_tokens(doc), default=len(doc))
    return any(tok.text.lower() in _PERSON_PRONOUNS and tok.pos_ == "PRON" and i < first_ref
               for i, tok in enumerate(doc))


def _has_anaphoric_subject(text: str) -> bool:
    """A frase tem "ele/ela/eles/elas" como sujeito ("Ah, não sei se ele
    é...", "Ele posta textos de política")?"""
    nlp = _nlp()
    if nlp is None:
        return bool(re.search(r"\b(ele|ela|eles|elas)\s+\w+", text, re.IGNORECASE))
    return any(tok.dep_.startswith("nsubj") and tok.text.lower() in _PERSON_PRONOUNS
               for tok in nlp(text))


def _introduces_referent(text: str) -> bool:
    """A frase cita alguém/algo que um "ele" logo depois possa retomar
    (nome próprio ou substantivo)?"""
    nlp = _nlp()
    if nlp is None:
        return bool(re.search(r"(?<!^)\b[A-ZÀ-Ý][a-zà-ÿ]{2,}", text.strip()))
    return bool(_referent_tokens(nlp(text)))


def opening_depends_on_context(first: str, second: str = "") -> bool:
    """Como `depends_on_context`, mas olhando as DUAS primeiras frases do
    clipe: uma abertura que se sustenta sozinha ("Puts, tu virar muito são,
    tu fica louco.") ainda depende do que veio antes se a frase seguinte
    fala de "ele" sem que a primeira tenha apresentado ninguém ("Ah, não sei
    se ele é..." -- quem?). Achado real no teste com podcast."""
    if depends_on_context(first):
        return True
    return bool(second) and _has_anaphoric_subject(second) and not _introduces_referent(first)
