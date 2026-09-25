"""
Palavra falada -> emoji (Twemoji, CC BY 4.0, PNGs 256px em assets/emoji/).

Chave = radical da palavra já normalizada (minúscula, sem acento): casa com
qualquer palavra que COMECE com ele ("dinheir" pega dinheiro/dinheirinho),
exceto as marcadas com "=" no fim, que exigem a palavra exata (palavras
curtas que seriam radical de muita coisa, ex. "rei=" não pode pegar
"reinaldo"/"reitor").
"""

EMOJI_KEYWORDS = {
    # dinheiro / sucesso
    "dinheir": "1f4b0", "grana=": "1f4b0", "reais=": "1f4b8",  # "real" não: é adjetivo demais
    "milhao": "1f4b8", "milhoes": "1f4b8", "bilhao": "1f4b8", "bilhoes": "1f4b8",
    "dolar": "1f4b5", "dolares": "1f4b5", "salario": "1f4b5", "rico=": "1f911", "ricos=": "1f911",
    "riqueza": "1f911", "pobre": "1f62c", "imposto": "1f9fe", "bitcoin": "1fa99",
    "empresa": "1f3e2", "negocio": "1f4bc", "trabalh": "1f4bc", "emprego": "1f4bc",
    "sucesso": "1f680", "crescer": "1f4c8", "cresceu": "1f4c8", "lucro": "1f4c8",
    "prejuizo": "1f4c9", "faliu": "1f4c9", "venceu": "1f3c6", "vitoria": "1f3c6",
    "ganhou": "1f3c6", "campeao": "1f3c6", "perdeu": "1f614",
    # emoções / reações
    "kkk": "1f602", "risada": "1f602", "engracad": "1f602", "piada": "1f602",
    "loucura": "1f92f", "louco=": "1f92f", "maluco": "1f92f", "insano": "1f92f",
    "absurdo": "1f633", "surreal": "1f92f", "chocad": "1f631", "medo": "1f631",
    "assustador": "1f631", "chorar": "1f62d", "chorei": "1f62d", "chorou": "1f62d",
    "triste": "1f622", "raiva": "1f621", "odio": "1f621", "puto=": "1f621",
    "nojo": "1f922", "vergonha": "1f633", "amor=": "2764", "apaixon": "1f60d",
    "lindo": "1f60d", "linda": "1f60d", "bonit": "1f60d", "beijo": "1f618",
    "genial": "1f9e0", "inteligente": "1f9e0", "burro": "1f9e0", "ideia": "1f4a1",
    "segredo": "1f92b", "mentira": "1f925", "mentiu": "1f925", "mentiroso": "1f925",
    "verdade=": "2705", "certeza": "2705", "errado": "274c", "proibido": "1f6ab",
    "obrigatorio": "26a0", "problema": "26a0", "perigo": "26a0", "cuidado": "26a0",
    "fogo=": "1f525", "incrivel": "1f525", "brabo": "1f525", "pesado": "1f525",
    "bebado": "1f974", "bebi=": "1f37a", "cerveja": "1f37a", "cachaca": "1f943",
    "sono=": "1f634", "dormir": "1f634", "cansad": "1f629", "doente": "1f912",
    "morte": "1f480", "morreu": "1f480", "morreram": "1f480", "morrer": "1f480", "matar": "1f480",
    # pessoas / lugares / coisas
    "deus=": "1f64f", "igreja": "26ea", "diabo": "1f608", "satanas": "1f608",
    "policia": "1f694", "preso=": "1f6a8", "presos=": "1f6a8", "prisao": "1f6a8",
    "cadeia": "1f6a8", "crime": "1f6a8", "bandido": "1f977", "roubo": "1f977",
    "roubou": "1f977", "arma=": "1f52b", "armas=": "1f52b", "guerra": "1f4a3", "bomba": "1f4a3",
    "brasil": "1f1e7-1f1f7", "eua=": "1f1fa-1f1f8", "americano": "1f1fa-1f1f8",
    "mundo=": "1f30e", "planeta": "1f30e",
    "politica": "1f3db", "presidente": "1f3db", "governo": "1f3db", "eleicao": "1f5f3",
    "vota": "1f5f3", "bolsonaro": "1f3db", "lula=": "1f3db",
    "escola": "1f3eb", "faculdade": "1f393", "estudar": "1f4da", "estudei": "1f4da",
    "estudou": "1f4da", "estudante": "1f4da", "livro": "1f4da",
    "musica": "1f3b5", "cantar": "1f3a4", "cantando": "1f3a4", "show=": "1f3a4",
    "filme": "1f3ac", "cinema": "1f3ac", "matrix": "1f48a", "video": "1f3a5",
    "entrevista": "1f399", "podcast": "1f399", "programa": "1f4fa", "televisao": "1f4fa",
    "jogo=": "1f3ae", "jogar": "1f3ae", "videogame": "1f3ae", "futebol": "26bd",
    "academia": "1f4aa", "treino": "1f4aa", "forte=": "1f4aa",
    "comida": "1f354", "comer": "1f354", "fome": "1f37d", "carro": "1f697",
    "aviao": "2708", "viagem": "2708", "casa=": "1f3e0", "celular": "1f4f1",
    "internet": "1f310", "instagram": "1f4f1", "tiktok": "1f4f1", "youtube": "1f3a5",
    "foto=": "1f4f8", "fotos=": "1f4f8", "cachorro": "1f436", "gato=": "1f431",
    "crianca": "1f476", "bebe=": "1f476", "mulher": "1f469", "namorad": "1f491",
    "casamento": "1f48d", "casou": "1f48d", "hospital": "1f3e5", "medico": "1fa7a",
    "tempo=": "23f0", "relogio": "23f0", "ovni": "1f6f8", "alien": "1f47d",
    "extraterrestre": "1f47d", "nave=": "1f6f8", "espaco=": "1f680", "foguete": "1f680",
    "robo=": "1f916", "tecnologia": "1f916", "computador": "1f4bb", "olha=": "1f440",
    "olhar": "1f440", "pergunta": "2753", "misterio": "1f575",
    # inglês
    "money": "1f4b0", "rich=": "1f911", "crazy": "1f92f", "insane": "1f92f",
    "love=": "2764", "death": "1f480", "fire=": "1f525", "police": "1f694",
    "war=": "1f4a3", "secret": "1f92b", "lie=": "1f925", "truth": "2705",
}

# (1f1e7-1f1f7 etc. = bandeiras: dois códigos)
