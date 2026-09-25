"""
Palavra falada -> emoji (Twemoji, CC BY 4.0, PNGs 256px em assets/emoji/).

Chave = radical da palavra já normalizada (minúscula, sem acento): casa com
qualquer palavra que COMECE com ele ("dinheir" pega dinheiro/dinheirinho),
exceto as marcadas com "=" no fim, que exigem a palavra exata (palavras
curtas que seriam radical de muita coisa, ex. "rei=" não pode pegar
"reinaldo"/"reitor").

Valor = um GRUPO de emojis da mesma ideia: o plano de efeitos (fx.py)
alterna entre eles, pra o mesmo assunto não repetir sempre o mesmo emoji.
Frases de duas palavras ("meu deus", "sei la") ficam em EMOJI_PHRASES.
"""

_MONEY = ["1f4b0", "1f4b5", "1f911", "1f4b8"]
_RICH = ["1f911", "1f48e", "1f4b0"]
_POOR = ["1f62c", "1f4c9", "1f972"]
_WIN = ["1f3c6", "1f947", "1f389"]
_SUCCESS = ["1f680", "1f4c8", "1f525"]
_LAUGH = ["1f602", "1f923", "1f606"]
_SHOCK = ["1f631", "1f633", "1f92f", "1f62e"]
_MIND_BLOWN = ["1f92f", "1f635-200d-1f4ab", "1f631"]
_SAD = ["1f622", "1f62d", "1f494"]
_ANGRY = ["1f621", "1f624", "1f92c"]
_LOVE = ["2764", "1f60d", "1f970"]
_DEATH = ["1f480", "26b0", "1f47b"]
_FIRE = ["1f525", "1f4af", "26a1"]
_BRAIN = ["1f9e0", "1f4a1", "1f913"]
_LIE = ["1f925", "1f921", "274c"]
_TRUTH = ["2705", "1f4af", "1f44d"]
_WRONG = ["274c", "1f6ab", "1f645"]
_DANGER = ["26a0", "1f6a8", "1f4a5"]
_POLICE = ["1f694", "1f6a8", "1f46e"]
_CRIME = ["1f6a8", "1f977", "1f52a"]
_POLITICS = ["1f3db", "1f5f3", "1f4dc"]
_WAR = ["1f4a3", "1f4a5", "2694"]
_GOD = ["1f64f", "271d", "1f607"]
_DEVIL = ["1f608", "1f479", "1f525"]
_MUSIC = ["1f3b5", "1f3a4", "1f3b8", "1f3a7"]
_MOVIE = ["1f3ac", "1f37f", "1f3a5"]
_GAME = ["1f3ae", "1f579", "1f47e"]
_FOOD = ["1f354", "1f355", "1f35f", "1f356"]
_DRINK = ["1f37a", "1f37b", "1f942"]
_SLEEP = ["1f634", "1f4a4", "1f971"]
_SICK = ["1f912", "1f637", "1f915"]
_WORK = ["1f4bc", "1f4bb", "1f4c8"]
_STUDY = ["1f4da", "1f393", "270f"]
_PHONE = ["1f4f1", "1f4f2", "1f4ac"]
_INTERNET = ["1f310", "1f4bb", "1f4f1"]
_EYES = ["1f440", "1f441", "1f9d0"]
_THINK = ["1f914", "1f9d0", "2753"]
_SHRUG = ["1f937", "1f615", "1f643"]
_ALIEN = ["1f47d", "1f6f8", "1f47e"]
_SPACE = ["1f680", "1fa90", "1f30c"]
_BRAZIL = ["1f1e7-1f1f7", "1f49a", "26bd"]
_USA = ["1f1fa-1f1f8", "1f5fd", "1f985"]
_WORLD = ["1f30e", "1f30d", "1f30f"]
_CAR = ["1f697", "1f3ce", "1f698"]
_TRAVEL = ["2708", "1f30d", "1f9f3"]
_HOUSE = ["1f3e0", "1f3e1", "1f511"]
_SPORT = ["26bd", "1f3c0", "1f3c6"]
_GYM = ["1f4aa", "1f3cb", "1f3c3"]
_KID = ["1f476", "1f9d2", "1f37c"]
_FAMILY = ["1f46a", "1f468-200d-1f469-200d-1f467", "1f3e0"]
_WOMAN = ["1f469", "1f483", "1f484"]
_MAN = ["1f468", "1f9d4", "1f57a"]
_DOG = ["1f436", "1f415", "1f9b4"]
_CAT = ["1f431", "1f408", "1f63a"]
_TIME = ["23f0", "231b", "1f570"]
_SECRET = ["1f92b", "1f510", "1f5dd"]
_HOSPITAL = ["1f3e5", "1fa7a", "1f489"]
_TECH = ["1f916", "1f4bb", "2699"]
_PHOTO = ["1f4f8", "1f5bc", "1f933"]
_WEDDING = ["1f48d", "1f470", "1f492"]
_PARTY = ["1f389", "1f973", "1f38a"]
_COOL = ["1f60e", "1f525", "1f918"]
_CRAZY = ["1f92a", "1f92f", "1f300"]
_FEAR = ["1f631", "1f628", "1f47b"]
_DISGUST = ["1f922", "1f92e", "1f644"]
_SHAME = ["1f633", "1f648", "1f605"]
_PEACE = ["262e", "1f54a", "1f91d"]
_TRAP = ["1f6a9", "26a0", "1f440"]

EMOJI_KEYWORDS = {
    # dinheiro / sucesso
    "dinheir": _MONEY, "grana=": _MONEY, "reais=": _MONEY, "milhao": _MONEY, "milhoes": _MONEY,
    "bilhao": _MONEY, "bilhoes": _MONEY, "dolar": _MONEY, "dolares": _MONEY, "salario": _MONEY,
    "pix=": _MONEY, "pagar": _MONEY, "pagou": _MONEY, "pagando": _MONEY, "caro=": _MONEY,
    "rico=": _RICH, "ricos=": _RICH, "riqueza": _RICH, "milionario": _RICH, "bilionario": _RICH,
    "pobre": _POOR, "pobreza": _POOR, "quebrado": _POOR, "divida": _POOR, "faliu": _POOR,
    "imposto": ["1f9fe", "1f4b8", "1f3db"], "bitcoin": ["1fa99", "1f4b0", "1f4c8"],
    "cripto": ["1fa99", "1f4c8", "1f4b0"], "investi": ["1f4c8", "1f4b9", "1f4b0"],
    "empresa": ["1f3e2", "1f4bc", "1f4c8"], "negocio": _WORK, "trabalh": _WORK, "emprego": _WORK,
    "chefe=": ["1f454", "1f4bc", "1f624"], "sucesso": _SUCCESS, "crescer": _SUCCESS,
    "cresceu": _SUCCESS, "lucro": _SUCCESS, "prejuizo": _POOR, "venceu": _WIN, "vitoria": _WIN,
    "ganhou": _WIN, "ganhei": _WIN, "campeao": _WIN, "perdeu": ["1f614", "1f4c9", "1f62d"],
    "perdi=": ["1f614", "1f4c9", "1f62d"],
    # emoções / reações
    "kkk": _LAUGH, "hahah": _LAUGH, "risada": _LAUGH, "engracad": _LAUGH, "piada": _LAUGH,
    "zoeira": _LAUGH, "zuera": _LAUGH, "comedia": _LAUGH, "loucura": _CRAZY, "louco=": _CRAZY,
    "louca=": _CRAZY, "maluco": _CRAZY, "insano": _MIND_BLOWN, "absurdo": _SHOCK, "surreal": _MIND_BLOWN,
    "chocad": _SHOCK, "chocante": _SHOCK, "caralho": _SHOCK, "caramba": _SHOCK, "nossa=": _SHOCK,
    "porra=": _SHOCK, "bizarro": _SHOCK, "medo=": _FEAR, "assustador": _FEAR, "terror": _FEAR,
    "chorar": _SAD, "chorei": _SAD, "chorou": _SAD, "triste": _SAD, "tristeza": _SAD,
    "depressao": _SAD, "saudade": _SAD, "raiva": _ANGRY, "odio": _ANGRY, "puto=": _ANGRY,
    "irritad": _ANGRY, "nojo": _DISGUST, "nojento": _DISGUST, "vergonha": _SHAME,
    "amor=": _LOVE, "apaixon": _LOVE, "lindo=": _LOVE, "linda=": _LOVE, "bonit": _LOVE,
    "beijo": ["1f618", "1f48b", "1f60d"], "beijei": ["1f618", "1f48b", "1f60d"],
    "genial": _BRAIN, "inteligente": _BRAIN, "burro": ["1f921", "1f9e0", "1f926"],
    "ideia": ["1f4a1", "1f9e0", "2728"], "segredo": _SECRET, "mentira": _LIE, "mentiu": _LIE,
    "mentiroso": _LIE, "fake=": _LIE, "verdade=": _TRUTH, "certeza": _TRUTH, "exatamente": _TRUTH,
    "errado": _WRONG, "proibido": _WRONG, "obrigatorio": ["26a0", "1f4dc", "2757"],
    "problema": _DANGER, "perigo": _DANGER, "cuidado": _DANGER, "golpe": _TRAP, "armadilha": _TRAP,
    "fogo=": _FIRE, "incrivel": _FIRE, "brabo": _FIRE, "pesado": _FIRE, "sinistro": _FIRE,
    "foda=": _COOL, "estiloso": _COOL, "festa": _PARTY, "balada": _PARTY, "aniversario": _PARTY,
    "bebado": ["1f974", "1f37a", "1f943"], "bebi=": _DRINK, "beber": _DRINK, "cerveja": _DRINK,
    "cachaca": ["1f943", "1f37a", "1f974"], "vinho": ["1f377", "1f942", "1f347"],
    "sono=": _SLEEP, "dormir": _SLEEP, "dormi=": _SLEEP, "cansad": ["1f629", "1f62b", "1f634"],
    "doente": _SICK, "doenca": _SICK, "virus": _SICK, "morte": _DEATH, "morreu": _DEATH,
    "morreram": _DEATH, "morrer": _DEATH, "matar": _DEATH, "matou": _DEATH, "morto=": _DEATH,
    "paz=": _PEACE,
    # pessoas / lugares / coisas
    "deus=": _GOD, "igreja": ["26ea", "1f64f", "271d"], "jesus": _GOD, "reza": _GOD, "orar": _GOD,
    "diabo": _DEVIL, "satanas": _DEVIL, "inferno": _DEVIL, "policia": _POLICE, "policial": _POLICE,
    "preso=": ["1f6a8", "26d3", "1f46e"], "presos=": ["1f6a8", "26d3", "1f46e"],
    "prisao": ["26d3", "1f6a8", "1f46e"], "cadeia": ["26d3", "1f6a8", "1f46e"], "crime": _CRIME,
    "bandido": _CRIME, "ladrao": _CRIME, "roubo": _CRIME, "roubou": _CRIME, "assalto": _CRIME,
    "arma=": ["1f52b", "1f6a8", "26a0"], "armas=": ["1f52b", "1f6a8", "26a0"], "guerra": _WAR,
    "bomba": _WAR, "exercito": _WAR, "brasil": _BRAZIL, "brasileiro": _BRAZIL, "eua=": _USA,
    "americano": _USA, "america=": _USA, "mundo=": _WORLD, "planeta": _WORLD,
    "politica": _POLITICS, "politico": _POLITICS, "presidente": _POLITICS, "governo": _POLITICS,
    "eleicao": ["1f5f3", "1f3db", "1f4dc"], "votar": ["1f5f3", "1f3db", "1f4dc"],
    "votou": ["1f5f3", "1f3db", "1f4dc"], "bolsonaro": _POLITICS, "lula=": _POLITICS,
    "direita": _POLITICS, "esquerda": _POLITICS, "lei=": ["1f4dc", "2696", "1f3db"],
    "justica": ["2696", "1f3db", "1f4dc"], "juiz=": ["2696", "1f468-200d-2696", "1f3db"],
    "escola": ["1f3eb", "1f4da", "270f"], "faculdade": _STUDY, "estudar": _STUDY,
    "estudei": _STUDY, "estudou": _STUDY, "estudante": _STUDY, "livro": _STUDY, "professor": _STUDY,
    "musica": _MUSIC, "cantar": _MUSIC, "cantando": _MUSIC, "cantor": _MUSIC, "banda=": _MUSIC,
    "show=": ["1f3a4", "1f3b8", "1f3b6"], "beatles": ["1f3b8", "1f3b5", "1f1ec-1f1e7"],
    "filme": _MOVIE, "cinema": _MOVIE, "serie=": ["1f4fa", "1f37f", "1f3ac"], "matrix": ["1f48a", "1f576", "1f4bb"],
    "video": ["1f3a5", "1f4f9", "25b6"], "entrevista": ["1f399", "1f5e3", "1f3a4"],
    "podcast": ["1f399", "1f3a7", "1f5e3"], "programa": ["1f4fa", "1f3ac", "1f399"],
    "televisao": ["1f4fa", "1f4e1", "1f3ac"], "jogo=": _GAME, "jogar": _GAME, "videogame": _GAME,
    "starcraft": _GAME, "futebol": _SPORT, "gol=": ["26bd", "1f945", "1f3c6"],
    "academia": _GYM, "treino": _GYM, "treinar": _GYM, "forte=": _GYM, "musculo": _GYM,
    "comida": _FOOD, "comer": _FOOD, "comi=": _FOOD, "fome": ["1f37d", "1f924", "1f354"],
    "pizza": ["1f355", "1f924", "1f37d"], "hamburguer": ["1f354", "1f35f", "1f924"],
    "churrasco": ["1f356", "1f525", "1f37a"], "cafe=": ["2615", "1f950", "1f60c"],
    "carro": _CAR, "moto=": ["1f3cd", "1f3ce", "1f4a8"], "aviao": _TRAVEL, "viagem": _TRAVEL,
    "viajar": _TRAVEL, "praia": ["1f3d6", "1f30a", "2600"], "casa=": _HOUSE, "apartamento": _HOUSE,
    "celular": _PHONE, "whatsapp": _PHONE, "mensagem": _PHONE, "internet": _INTERNET,
    "instagram": ["1f4f8", "1f4f1", "2764"], "tiktok": ["1f3b5", "1f4f1", "1f525"],
    "youtube": ["25b6", "1f3a5", "1f4fa"], "twitter": ["1f426", "1f4ac", "1f4f1"],
    "facebook": ["1f4d8", "1f4f1", "1f44d"], "foto=": _PHOTO, "fotos=": _PHOTO,
    "cachorro": _DOG, "gato=": _CAT, "crianca": _KID, "criancas": _KID, "bebe=": _KID,
    "filho=": _FAMILY, "filhos=": _FAMILY, "familia": _FAMILY, "pai=": _FAMILY, "mae=": _FAMILY,
    "mulher": _WOMAN, "homem": _MAN, "homens=": _MAN, "namorad": ["1f491", "2764", "1f60d"],
    "casamento": _WEDDING, "casou": _WEDDING, "casar": _WEDDING, "divorcio": ["1f494", "1f62d", "274c"],
    "hospital": _HOSPITAL, "medico": _HOSPITAL, "remedio": ["1f48a", "1f489", "1fa7a"],
    "tempo=": _TIME, "relogio": _TIME, "atrasado": _TIME, "ovni": _ALIEN, "alien": _ALIEN,
    "extraterrestre": _ALIEN, "identificad": _ALIEN, "nave=": _SPACE, "espaco=": _SPACE,
    "foguete": _SPACE, "lua=": ["1f315", "1f680", "1f30c"], "robo=": _TECH, "tecnologia": _TECH,
    "computador": _TECH, "inteligencia": _BRAIN, "olha=": _EYES, "olhar": _EYES, "olhando": _EYES,
    "vi=": _EYES, "viu=": _EYES, "pergunta": _THINK, "misterio": ["1f575", "1f50e", "2753"],
    "teoria": ["1f9d0", "1f4a1", "1f50e"], "conspiracao": ["1f441", "1f50e", "1f47d"],
    # inglês
    "money": _MONEY, "rich=": _RICH, "crazy": _CRAZY, "insane": _MIND_BLOWN, "love=": _LOVE,
    "death": _DEATH, "fire=": _FIRE, "police": _POLICE, "war=": _WAR, "secret": _SECRET,
    "lie=": _LIE, "truth": _TRUTH, "god=": _GOD, "true=": _TRUTH,
}

# expressões de 2+ palavras (normalizadas, sem acento) — conferidas antes
# das palavras soltas; o emoji entra na ÚLTIMA palavra da expressão
EMOJI_PHRASES = {
    "meu deus": _SHOCK, "puta merda": _SHOCK, "que isso": _SHOCK, "nao acredito": _SHOCK,
    "sei la": _SHRUG, "tanto faz": _SHRUG, "presta atencao": _EYES,  # "tá ligado" não: muleta de fala
    "sera que": _THINK, "por que": _THINK, "faz sentido": _BRAIN, "vai dar certo": _SUCCESS,
    "deu ruim": _DANGER, "deu merda": _DANGER, "que vergonha": _SHAME, "morri de rir": _LAUGH,
    "chorei de rir": _LAUGH, "todo mundo": [], "de verdade": _TRUTH, "com certeza": _TRUTH,
}
