"""
Configurações globais do Auto Clipper.
Ajuste estes valores conforme necessário.
"""
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Resolução de saída (formato vertical padrão Shorts/Reels/TikTok) ---
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# --- Duração dos clipes (segundos) ---
MIN_CLIP_DURATION = 15
MAX_CLIP_DURATION = 75
IDEAL_CLIP_DURATION = 45

# --- Whisper (transcrição) ---
WHISPER_MODEL_SIZE = "small"       # tiny, base, small, medium, large-v3
WHISPER_DEVICE = "auto"            # "cpu", "cuda" ou "auto"
WHISPER_COMPUTE_TYPE = "auto"
WHISPER_CPU_THREADS = 0             # 0 = detecta automaticamente (núcleos físicos da CPU)
WHISPER_NUM_WORKERS = 1             # >1 só ajuda se transcrever vários áudios em paralelo
WHISPER_BEAM_SIZE = 1               # 1 = busca gulosa (rápido); 5 = padrão do Whisper (mais
                                     # lento, ganho de precisão pequeno pra este uso)
WHISPER_BATCHED = True              # usa BatchedInferencePipeline (processa vários trechos
                                     # de fala em paralelo) — desative só se causar erro
WHISPER_BATCH_SIZE = 8              # nº de trechos processados por vez. Suba se tiver muita
                                     # RAM livre e núcleos sobrando (teste 12-16 no seu Xeon);
                                     # desça se estourar memória

# --- Motor de transcrição ---
# "faster-whisper" (padrão) = CPU ou GPU NVIDIA (CUDA) via CTranslate2.
#   Não existe backend AMD para esse motor em NENHUM sistema operacional —
#   a RX 580 nunca vai ser usada aqui, mesmo no Windows.
# "whispercpp" = CPU ou GPU AMD/Intel/NVIDIA via Vulkan (motor whisper.cpp,
#   compilado à parte). É o único caminho que de fato usa a RX 580 pra
#   transcrever mais rápido. Veja o README.md ("Transcrição via GPU/Vulkan")
#   pra como baixar/compilar o whisper-cli.exe com Vulkan e o modelo .bin.
TRANSCRIBE_ENGINE = "whispercpp"

# --- Configuração do motor "whispercpp" (só usada se TRANSCRIBE_ENGINE acima
# estiver como "whispercpp") ---
# Caminho completo pro executável, ex: r"C:\whispercpp\whisper-cli.exe"
# (ou só "whisper-cli" se você colocou a pasta no PATH do Windows).
WHISPERCPP_BIN = r"C:\Users\igo\Desktop\opus2\opus-clip-clone\whisper.cpp\build\bin\Release\whisper-cli.exe"
# Caminho completo pro modelo .bin (formato ggml), ex:
# r"C:\whispercpp\models\ggml-small.bin". Baixe manualmente — veja README.md.
WHISPERCPP_MODEL = r"C:\Users\igo\Desktop\opus2\opus-clip-clone\whisper.cpp\models\ggml-small.bin"
WHISPERCPP_LANGUAGE = "pt"          # "auto" pra detectar sozinho, ou o código do idioma
WHISPERCPP_THREADS = 0              # 0 = deixa o whisper.cpp decidir (usa os núcleos da CPU
                                     # nas partes que não rodam na GPU, ex: carregar áudio)
WHISPERCPP_USE_GPU = True           # False força CPU mesmo com um build com Vulkan/CUDA
WHISPERCPP_GPU_DEVICE = 0           # índice da GPU, se você tiver mais de uma

# --- Encode de vídeo (GPU/CPU) ---
# "auto"  = detecta e testa sozinho o melhor disponível (recomendado)
# "amf"   = força GPU AMD via AMF (Windows) — caminho usado pela RX 580
# "nvenc" = força GPU NVIDIA
# "vaapi" = força GPU AMD/Intel via VAAPI (Linux)
# "cpu"   = força libx264 (CPU), útil para depuração
ENCODER_MODE = "auto"
VAAPI_DEVICE = "/dev/dri/renderD128"  # só usado no Linux (VAAPI); ignorado no Windows

# --- Processamento paralelo de clipes ---
# "auto" = decide sozinho com base nos núcleos de CPU e no tipo de encode
# ou um número fixo de clipes a processar simultaneamente
PARALLEL_CLIPS = "auto"
MAX_PARALLEL_CLIPS_HW = 2            # limite quando o encode é via GPU (evita disputa pelo chip de vídeo)
MAX_PARALLEL_CLIPS_CPU = 4           # limite quando o encode é via CPU (cada clipe já usa vários threads)

# --- Detecção de corte de câmera (scene cut) ---
# Quando o podcast corta de plano (ex.: de close-up pro dois-shot ou pra
# outro ângulo), o tracker normalmente esperaria o próximo ciclo normal de
# detecção (a cada FACE_DETECT_EVERY_N_FRAMES) antes de achar o rosto no
# novo plano — causando o delay de enquadramento que o usuário vê. Com
# SCENE_CUT_DETECT_ENABLED, o programa mede a diferença de pixel entre
# frames consecutivos (em baixa resolução, barato) e, quando detecta uma
# mudança grande o suficiente (SCENE_CUT_THRESHOLD), dispara detecção de
# rosto imediatamente naquele frame + roda em modo "burst" (detecção a
# cada frame) pelos próximos SCENE_CUT_BURST_FRAMES frames — garantindo
# que o primeiro rosto visível no novo plano seja encontrado sem delay.
# Também zera o mode_hold_remaining para não ficar travado no modo do
# plano anterior depois de um corte real.
SCENE_CUT_DETECT_ENABLED = True
SCENE_CUT_THRESHOLD = 28.0        # diff médio de pixel (0-255) para considerar corte
SCENE_CUT_DETECT_WIDTH = 160      # largura do frame downscalado para a comparação (barato)
SCENE_CUT_BURST_FRAMES = 20       # frames em modo "detecta todo frame" após corte detectado

# --- Reenquadramento (crop 9:16 seguindo o rosto) ---
FACE_DETECT_EVERY_N_FRAMES = 4
# Segundo passe de detecção (menos downscale), só tentado quando o passe
# rápido (480px, embutido em reframer.py) não encontra NENHUM rosto —
# achado real: num plano aberto (2 pessoas visíveis, mas cada rosto
# ocupando pouco da tela, ex.: corte de câmera pro "2-shot" de um
# podcast), o rosto perde detalhe de pixel suficiente a 480px pro Haar
# reconhecer — não é questão de minSize (testado de 60 até 15px, nenhum
# encontrou nada), é resolução insuficiente mesmo. Sem esse segundo
# passe, perder o rosto nesses momentos derruba a confiança do
# rastreamento e troca pro modo "plano aberto" (ninguém em destaque) —
# exatamente o oposto de "seguir quem fala mesmo depois de um corte de
# câmera". Confirmado com um frame real: 0 rostos em qualquer minSize a
# 480px, 2 rostos encontrados de cara a 960px.
FACE_DETECT_WIDTH_FALLBACK = 960
# Fração máxima de altura do frame onde o centro de um rosto detectado pode
# estar. Detecções abaixo desse limite (ex.: logos, brinquedos, placas de mesa
# num plano aberto de podcast) são descartadas como falsos positivos — rostos
# reais de apresentadores sentados ficam sempre na metade/centro superior.
FACE_MAX_Y_FRAC = 0.80
# Limites mais agressivos usados SOMENTE na busca escalona da do primeiro frame
# após um corte de câmera (_detect_faces_on_cut). Nesse momento o custo de um
# falso positivo é máximo (snap imediato), então exigimos que o rosto esteja
# no terço/metade superior.
CUT_FACE_MAX_Y_FRAC = 0.55   # frame inteiro: apenas metade superior
CUT_HALF_FACE_MAX_Y_FRAC = 0.70  # busca por metades: um pouco mais relaxado
FACE_SMOOTHING_ALPHA = 0.18         # menor = mais suave/lento, maior = mais responsivo
HEADROOM_RATIO = 0.38               # posição vertical do rosto no quadro (0=topo, 1=base)
# 1.0 = sem zoom extra: usa o maior crop 9:16 possível preservando a altura
# toda da fonte, só centralizado no rosto rastreado (era 1.18 — apertava o
# enquadramento além do natural, cortando testa/queixo desnecessariamente
# como se estivesse sempre dando zoom no rosto).
FACE_CROP_ZOOM = 1.0

# --- Zoom dinâmico por tamanho de rosto ---
# Quando a câmera original do podcast está num plano mais aberto (speaker
# visível até a cintura), o rosto detectado ocupa uma fração pequena do
# frame, e o crop mostra muita mesa/mãos abaixo. Com FACE_DYNAMIC_ZOOM_ENABLED,
# o crop ENCOLHE ao redor do rosto para que ele ocupe ~FACE_DYNAMIC_ZOOM_TARGET_FRAC
# do crop height (ex: 0.25 = 25%). Em close-up, a fórmula naturalmente
# retorna o crop_h original (sem mudança). Em plano aberto, o crop shrinks
# dando um zoom suave no rosto — EMA-suavizado para a transição ser fluida.
FACE_DYNAMIC_ZOOM_ENABLED = True
FACE_DYNAMIC_ZOOM_TARGET_FRAC = 0.25   # fração-alvo do crop height que o rosto deve ocupar
FACE_DYNAMIC_ZOOM_MIN_CROP_FRAC = 0.40 # crop_h mínimo como fração de src_h (evita zoom excessivo)

# --- Priorizar quem está FALANDO (ou reagindo) quando há vários rostos ---
# Com mais de uma pessoa em quadro, o programa rastreia TODOS os rostos
# visíveis e mede, a cada frame, o quanto a região da boca/expressão de
# cada um está se mexendo — isso funciona tanto para detectar fala quanto
# reações exageradas (rir, se surpreender), já que ambas envolvem bastante
# movimento facial. O rosto com mais atividade acumulada é quem a câmera
# segue. SPEAKING_PRIORITY_ENABLED=False volta ao comportamento antigo
# (sempre segue o maior rosto detectado, ignorando quem está falando).
SPEAKING_PRIORITY_ENABLED = True
FACE_ACTIVITY_DECAY = 0.85          # suavização (EMA) do placar de atividade por frame
# Só troca de rosto ativo quando o desafiante tem MUITO mais atividade que
# o atual E isso se sustenta por FACE_SWITCH_MIN_SECONDS — essa histerese é
# o que evita ficar pulando de um rosto pro outro a cada pequena flutuação
# (ex.: quem ouve dá uma risadinha rápida).
# Valores subidos nesta revisão (margem 1.4->1.7, tempo 0.5->0.7s):
# processamento mais lento não é problema (é whisper.cpp que domina o
# tempo total, não o reenquadramento), então vale trocar um pouco de
# responsividade da câmera por mais certeza antes de decidir trocar de
# rosto — o pedido explícito foi "capriche mais no enquadramento, mesmo
# que demore um pouco mais". Combinado com os filtros de oclusão/
# sustentação em reframer.py (que já reduzem picos falsos na origem),
# isso reduz bastante a chance de a câmera pular pra quem só fez um
# gesto rápido perto do rosto.
FACE_SWITCH_MARGIN = 1.7
FACE_SWITCH_MIN_SECONDS = 0.7
# Quantas CHECAGENS (não segundos) um slot de rosto recém-criado precisa
# sobreviver antes de poder competir em pé de igualdade pelo foco da
# câmera -- ver comentário completo em reframer.py
# (_FaceActivityTracker._trust). Achado num clipe REACT real: um rosto
# aparecendo no vídeo de FUNDO reagido (ex.: uma entrevista) é maior em
# tela que a facecam pequena do streamer, e sem essa "rampa de confiança"
# ele podia roubar o foco assim que aparecesse. Baixe se um rosto novo
# genuíno (alguém entrando em quadro) estiver demorando demais pra ganhar
# foco; suba se ainda estiver pegando rosto de vídeo de fundo.
FACE_AGE_RAMP_CHECKS = 6
# quanto tempo um rosto que sumiu de quadro continua "reservado" (mantendo
# seu placar de atividade E sua última posição conhecida) antes de ser
# esquecido — cobre perdas momentâneas de detecção sem resetar o histórico
# de quem estava falando. Subido de 1.2 -> 2.0s: esse é o parâmetro que
# mais importava no caso real testado — quem fala muito animado
# (gesticulando/apontando perto do próprio rosto, como no clipe testado)
# tende a tampar o próprio rosto com a própria mão ou dedo por instantes, e
# o classificador Haar simplesmente para de encontrá-lo por 1-1.5s. Com um
# timeout curto, isso apagava o slot dele e forçava uma re-escolha IMEDIATA
# (sem NENHUMA histerese — ver o ramo "active_id not in live" em
# choose_active) para quem quer que estivesse visível naquele instante,
# mesmo que fosse só por uma fração de segundo, mesmo que a pessoa que
# sumiu continuasse sendo quem estava de fato falando. Enquanto o slot
# continua "vivo" (dentro do timeout), a câmera simplesmente CONGELA na
# última posição conhecida em vez de saltar — muito mais estável.
FACE_SLOT_TIMEOUT_SECONDS = 4.0
# O placar de "atividade" (movimento perto da boca) sozinho não distingue
# fala de qualquer outro movimento facial (rir sem falar, se ajeitar,
# reagir com a cabeça) — por isso a câmera podia grudar em quem só está
# "agitado" no vídeo, mesmo quando é a OUTRA pessoa quem está falando.
# FACE_AUDIO_GATE_ENABLED pondera esse movimento pela energia real do
# áudio no mesmo instante (RMS normalizado 0..1): pouco movimento com
# áudio alto conta mais como "falando" do que muito movimento no silêncio.
# FACE_AUDIO_GATE_HOP é a resolução temporal dessa checagem de energia,
# em segundos (menor = mais preciso, mais barato de calcular do que
# parece, já que é só RMS sobre o áudio já extraído do clipe).
# FACE_AUDIO_GATE_FLOOR evita que a atividade zere completamente durante
# uma pausa curta de respiração/silêncio natural no meio de uma fala.
FACE_AUDIO_GATE_ENABLED = True
FACE_AUDIO_GATE_HOP = 0.1
FACE_AUDIO_GATE_FLOOR = 0.12

# --- Distinguir fala real de "qualquer coisa mexendo perto da boca" ---
# Achado num clipe real: uma pessoa ouvindo leva a mão ao queixo/boca (gesto
# comum de quem está pensando) e essa mudança de pixel é tão grande quanto
# ou maior que a boca de quem fala de verdade abrindo e fechando — a câmera
# pulava pra ela por um instante mesmo com a outra pessoa no meio da mesma
# frase. Dois filtros novos em reframer.py (`_FaceActivityTracker.
# sample_motion`) atacam isso ANTES de chegar no placar de atividade:
#
# FACE_OCCLUSION_CONTRAST_MIN: uma boca de verdade (mesmo fechada) tem
# bordas de lábio/dentes/sombra — contraste (desvio padrão de pixel) mais
# alto que uma mão ou outra superfície de pele cobrindo a mesma região.
# Recortes com contraste abaixo deste valor têm o score de movimento
# descontado (não zerado) antes de entrar no placar. Baixe este valor se
# rostos legítimos estiverem sendo descontados demais (pele muito lisa,
# barba pouco texturizada); suba se ainda estiver pegando mão/objeto.
#
# RESOLVIDO (ver RELATORIO_PROXIMOS_PASSOS.txt, item 2): esse filtro em
# escala de cinza discriminava mal (~1.11x mão-vs-boca — quase nada) e foi
# substituído pelo modo hue abaixo (FACE_OCCLUSION_USE_HUE=True, padrão).
# Mantido aqui só como fallback de comparação/depuração
# (FACE_OCCLUSION_USE_HUE=False volta a usar este valor).
FACE_OCCLUSION_CONTRAST_MIN = 9.0

# --- Filtro de oclusão por MATIZ (hue) — substitui o contraste acima ---
# Métrica principal agora: desvio-padrão do MATIZ (canal H do HSV, 0-179 no
# OpenCV) do mesmo recorte perto da boca, não do contraste de luminância. Uma
# boca real tem cores variadas (vermelho do lábio, branco/creme do dente,
# sombra escura) -> hue_std alto. Pele cobrindo (mão, dedo) é essencialmente
# um tom só -> hue_std baixo, mesmo em recortes com bastante TEXTURA de
# luminância (dedos, sombra entre eles) que enganava o filtro antigo.
# Validado com vídeo real: ~2.2x de discriminação mão-vs-boca (contra ~1.11x
# do contraste em cinza). FACE_OCCLUSION_USE_HUE=False volta ao modo antigo.
#
# ATENÇÃO: os valores abaixo são um ponto de partida razoável, não foram
# recalibrados nos MESMOS dados do relatório (não tenho acesso ao clipe de
# teste original neste ambiente). Rode scripts/validate_occlusion_hue.py
# (novo, incluído neste zip) com o mesmo clipe e as mesmas janelas do
# relatório antes de considerar a calibração final — ajuste
# FACE_OCCLUSION_HUE_STD_MIN pra baixo se rostos legítimos estiverem sendo
# descontados demais, ou pra cima se ainda estiver pegando mão/objeto.
FACE_OCCLUSION_USE_HUE = True
FACE_OCCLUSION_HUE_STD_MIN = 10.0
# saturação média (canal S do HSV, 0-255) abaixo da qual o hue é tratado
# como pouco confiável (tons quase acinzentados/estourados de brilho) e o
# desconto de oclusão fica mais conservador em vez de confiar no hue_std cru.
FACE_OCCLUSION_SAT_FLOOR = 40.0

# FACE_ACTIVITY_BURST_MIN_FRACTION: fala real oscila (a boca abre/fecha
# várias vezes por segundo) — no histórico recente de scores brutos, uma
# fração razoável das amostras fica acima do "ruído de fundo" daquele
# slot. Um evento único (mão chegando e parando, cabeça virando rápido)
# aparece como UM pico isolado seguido de quase nada — fração baixa. Só
# picos que se repetem no tempo contam em cheio no placar; picos isolados
# são descontados proporcionalmente. Suba para exigir mais sustentação
# (mais rigoroso contra falsos positivos, mas pode demorar mais pra
# reconhecer uma fala genuína); desça para reagir mais rápido.
FACE_ACTIVITY_BURST_MIN_FRACTION = 0.35

# --- Enquadramento "dois rostos" (two-shot) ---
# Quando dois rostos estão visíveis com atividade parecida (ninguém
# dominando claramente), a câmera centraliza no ponto MÉDIO entre os dois
# em vez de ficar presa num só e cortar o outro.
# FACE_TWO_SHOT_MAX_RATIO: se o 2º rosto tiver atividade > (1º / ratio),
# os dois entram no enquadramento. Suba para exigir mais desequilíbrio
# antes de centralizar num só; desça para centralizar os dois com mais frequência.
FACE_TWO_SHOT_ENABLED = False        # desligado: em podcast duas-pessoas lado-a-lado
                                      # o ponto médio cai no espaço entre eles,
                                      # não nos rostos — resultado é enquadramento torto
FACE_TWO_SHOT_MAX_RATIO = 2.5
# Tempo mínimo que um modo (face/wide) deve ser mantido antes de poder trocar.
# Evita que detecção esporádica de rosto cause flicker face↔wide a cada poucos frames.
MODE_MIN_HOLD_SECONDS = 2.5

# FACE_ACTIVITY_OUTLIER_CAP_MULT: mesmo depois dos dois filtros acima, um
# único frame não pode sozinho empurrar o placar (EMA) muito acima da
# própria média recente do slot — evita que um pico remanescente grude no
# placar por vários frames só por causa da memória longa do EMA
# (FACE_ACTIVITY_DECAY alto). O teto por frame é
# `FACE_ACTIVITY_OUTLIER_CAP_MULT * atividade_atual_do_slot + 4.0`.
FACE_ACTIVITY_OUTLIER_CAP_MULT = 3.0

# Quando nenhum rosto é detectado por muito tempo (plano aberto, corte de
# câmera, duas pessoas afastadas etc.), em vez de cravar o crop apertado no
# centro da imagem (o que costuma mostrar só mesa/objetos, sem ninguém em
# quadro), o programa troca para um modo "plano aberto": mostra o frame
# inteiro (sem cortar ninguém de fora) encaixado no centro, com um fundo
# desfocado/escurecido preenchendo o resto do quadro vertical — a mesma
# técnica usada por Opus Clip/CapCut para planos largos.
#
# NO_FACE_FALLBACK_SECONDS precisa ser generoso: o detector de rosto (Haar
# cascade) é ruidoso e perde o rosto por ~1-1.5s toda vez que a pessoa vira
# a cabeça pra falar com alguém, gesticula na frente do rosto ou pisca —
# coisa que acontece o tempo todo numa conversa normal. Com um limiar curto
# (era 0.8s), cada uma dessas perdas momentâneas contava como "ninguém em
# quadro" e trocava pro plano aberto (bem mais afastado) e de volta, um
# "zoom out e zoom in" falso a cada poucos segundos — incômodo de assistir
# e sem nenhum corte de câmera real acontecendo. 2.5s dá tempo de sobra pro
# rastreamento se recuperar sozinho nesses casos comuns, e só troca de modo
# quando o rosto some por tempo grande o bastante pra ser mesmo um plano
# largo ou um corte de câmera.
NO_FACE_FALLBACK_SECONDS = 5.0      # tempo sem detectar rosto até trocar de modo
# Quando em modo "wide", se ainda há um rosto recente em memória (face_confidence > 0.05),
# usa um crop MÉDIO (mais largo que o close-up, mas ancorado na última posição conhecida)
# em vez de mostrar o frame inteiro letterboxado — evita o salto visual brutal de
# "close-up apertado → vídeo minúsculo no centro". Só vai pro letterbox completo quando
# a confiança cair a zero (rosto perdido há muito tempo).
WIDE_MODE_MEDIUM_ENABLED = False
WIDE_MODE_MEDIUM_CROP_SCALE = 2.5   # multiplica o crop height por este fator (2.5x = muito mais aberto)
# Duração da transição suave (crossfade) entre os dois modos, em SEGUNDOS
# (não em frames — um valor fixo de frames dava uma transição 2x mais curta
# em vídeo de 60fps do que em 30fps, e mais importante: 6 frames a 60fps são
# só 0.1s, um corte quase instantâneo, que por si só já parecia um "pulo" de
# zoom mesmo nas trocas de modo genuínas). 0.6s é suave o bastante pra não
# chamar atenção quando o modo realmente precisa mudar.
MODE_BLEND_SECONDS = 1.5
FALLBACK_BG_BLUR_SIGMA = 25.0       # intensidade do desfoque do fundo no modo "plano aberto"
FALLBACK_BG_DARKEN = 0.55           # 0-1: quanto o fundo desfocado é escurecido
WIDE_FIT_ZOOM = 1.12                # layout fit: amplia o quadro central 12% (corta só as bordas laterais)

# Detecção de CORTE DE CÂMERA real (o vídeo de origem alterna para outra
# pessoa/ângulo — comum em podcast de duas câmeras). Um salto de posição
# maior que CUT_JUMP_RATIO * crop_w só é tratado como corte real (câmera
# PULA direto pro novo enquadramento) quando confirmado por DUAS checagens
# de rosto seguidas concordando sobre a nova posição (dentro de
# CUT_CONFIRM_RATIO * crop_w uma da outra) — isso evita reagir a uma
# detecção Haar ruidosa isolada (falso positivo em mão, objeto, reflexo).
CUT_JUMP_RATIO = 0.35
CUT_CONFIRM_RATIO = 0.15

# --- Vídeo dentro do vídeo (ver RELATORIO_PROXIMOS_PASSOS.txt, item 3a) ---
# "Se o youtuber estiver assistindo um vídeo, focar no vídeo (não no rosto)
# enquanto ele estiver rodando ou enquanto alguém estiver falando nele."
# Detecta um retângulo (celular, monitor, notebook) via bordas/contornos
# (Canny + approxPolyDP) e só considera "tela ativa" quando a mesma região
# aparece de forma ESTÁVEL por VIDEO_IN_VIDEO_STABLE_SECONDS E tem atividade
# de pixel interna sustentada (indício de vídeo tocando, não foto/quadro
# parado) acima de VIDEO_IN_VIDEO_ACTIVITY_MIN — nesse caso a câmera passa a
# seguir a tela em vez do rosto, com o mesmo crossfade suave usado na troca
# entre os outros modos.
#
# DESLIGADO POR PADRÃO: funcionalidade NOVA (não uma correção de algo já
# existente), não testada contra nenhum clipe real de "vídeo reagindo a
# vídeo" — a heurística geométrica (retângulo 4 vértices + proporção
# 16:9/4:3/1:1) pode confundir outros objetos retangulares de tela (janela,
# quadro na parede, porta) com uma tela de verdade. Ligue
# (VIDEO_IN_VIDEO_ENABLED=True) e teste num clipe real antes de usar em
# produção; ajuste VIDEO_IN_VIDEO_ACTIVITY_MIN pra cima se estiver pegando
# objetos parados como "tela", ou pra baixo se estiver perdendo uma tela
# real com conteúdo de baixo movimento.
VIDEO_IN_VIDEO_ENABLED = False
VIDEO_IN_VIDEO_STABLE_SECONDS = 1.0   # tempo de estabilidade antes de confirmar a região
VIDEO_IN_VIDEO_TIMEOUT_SECONDS = 1.0  # tempo sem detectar antes de esquecer a região
VIDEO_IN_VIDEO_ACTIVITY_MIN = 3.0     # piso de atividade de pixel pra considerar "tocando"

# --- Legendas estilo "karaokê" (efeito Opus Clip / CapCut) ---
CAPTION_WORDS_PER_GROUP = 3           # grupos curtos = leitura rápida (padrão de corte viral)
CAPTION_UPPERCASE = True
# "Anton" (Google Fonts, licença OFL livre) vem embutida em assets/fonts/ e é
# carregada via "fontsdir" no filtro `ass` do ffmpeg — funciona em qualquer
# sistema, sem precisar instalar a fonte no SO. O original usava
# "Arial Black", que normalmente não existe no Linux (a legenda cairia numa
# fonte genérica do sistema).
CAPTION_FONT = "Anton"
CAPTION_FONT_SIZE = 135              # pt de referência para vídeo de 1080px de largura
                                      # (era 22 — praticamente ilegível; depois 84, que
                                      # medido no vídeo final dava letras de ~44px e ocupava
                                      # só ~1/3 da largura — Opus Clip/CapCut usam legendas
                                      # bem maiores, ocupando boa parte da largura)
CAPTION_OUTLINE_WIDTH = 8            # pt de referência p/ 1080px; escala junto com a fonte
CAPTION_SHADOW = 4                   # sombra preta semitransparente atrás do contorno
CAPTION_LETTER_SPACING = 1
CAPTION_POP_START_SCALE = 70         # cada grupo entra crescendo de 70% -> 100% ("pop")
CAPTION_PRIMARY_BGR = "FFFFFF"      # branco (formato BGR usado pelo ASS)
CAPTION_HIGHLIGHT_BGR = "00D7FF"    # dourado/amarelo (BGR) — palavra sendo falada
CAPTION_EMPHASIS_BGR = "5BFF3C"     # verde (BGR) — números, dinheiro, palavras de impacto
CAPTION_OUTLINE_BGR = "000000"      # preto
# Zona segura do TikTok / Reels / Shorts: a interface do app cobre os ~25%
# de baixo do vídeo (descrição, nome da música, botões) e uma faixa de
# ~120px na direita (curtir/comentar/compartilhar). Com 320 a legenda ficava
# em 80-83% da altura — bem embaixo da descrição do post. 560 põe a legenda
# em ~68-71% da altura, acima dessa faixa, onde Opus Clip/CapCut também
# posicionam.
CAPTION_MARGIN_V = 560               # distância da legenda até a base do quadro
CAPTION_MARGIN_H = 120               # margem lateral (quebra de linha antes dos botões da direita)
CAPTION_HIGHLIGHT_SCALE = 115        # escala (%) da palavra destacada — "pop" ao ser falada

# Exporta também clip_XX_..._sem_musica.mp4 (mesma imagem, só a voz) pra
# postar com um som em alta escolhido na biblioteca do próprio TikTok /
# Instagram / YouTube -- o único jeito de usar música comercial viral sem
# reivindicação de direitos, e o algoritmo favorece vídeo com som em alta.
EXPORT_NO_MUSIC_VERSION = True

# --- Efeitos sonoros (src/sfx.py, sintetizados localmente) ---
SFX_ENABLED = True
SFX_WHOOSH_DB = -18.0                # volume do whoosh na entrada do título-gancho

# --- Título-gancho no topo (primeiros segundos de cada clipe) ---
# Balão branco com texto preto (visual clássico de TikTok/Reels) com o
# título do .post.txt, pra quem está rolando o feed entender em 1 segundo
# do que é o corte. Fica abaixo da faixa de abas do app no topo.
HOOK_ENABLED = True
HOOK_SECONDS = 3.2
HOOK_FONT = "Anton"
HOOK_FONT_SIZE = 86
HOOK_TEXT_BGR = "000000"
HOOK_BOX_BGR = "FFFFFF"
HOOK_BOX_PADDING = 20
HOOK_MARGIN_V = 250                  # distância do topo (a faixa de abas do app ocupa ~8%)
HOOK_MARGIN_H = 110
HOOK_MAX_LINE_CHARS = 22

# --- Jump cuts: corte das pausas de dentro do clipe (src/jumpcut.py) ---
# Pausa entre duas palavras maior que JUMPCUT_MIN_GAP_SECONDS vira corte,
# deixando JUMPCUT_PAD_AFTER depois da palavra anterior e JUMPCUT_PAD_BEFORE
# antes da próxima (respiro natural, não come sílaba). Só corta se o áudio
# da pausa estiver em silêncio de verdade (abaixo de JUMPCUT_SILENCE_RATIO x
# o volume típico da fala) -- risada/reação ficam. False = clipe corrido.
JUMPCUT_ENABLED = True
JUMPCUT_MIN_GAP_SECONDS = 0.40
JUMPCUT_PAD_AFTER_SECONDS = 0.10
JUMPCUT_PAD_BEFORE_SECONDS = 0.06
JUMPCUT_MIN_CUT_SECONDS = 0.15
JUMPCUT_TAIL_KEEP_SECONDS = 0.35
JUMPCUT_SILENCE_RATIO = 0.35

# --- Kit de postagem (arquivo .post.txt ao lado de cada clipe) ---
# Hashtags fixas do seu canal, somadas às de assunto (tiradas da fala do
# clipe) e às de cada rede (#shorts / #reels / #fyp #viral). Ex.:
# ["#cortes", "#podcast", "#nomedocanal"].
POST_EXTRA_HASHTAGS = ["#cortes"]

# --- Música de fundo ---
# Caminho ABSOLUTO (resolvido a partir da localização deste arquivo, igual
# ao FONTS_DIR em video_editor.py) — não relativo ao diretório onde o
# comando foi executado. Com um caminho relativo, rodar main.py a partir de
# outra pasta (atalho do Windows, agendador de tarefas, IDE com "working
# directory" diferente etc.) faz "assets/music" apontar para um lugar que
# não existe; list_usable_tracks() então retorna [] silenciosamente (sem
# erro nenhum) e TODO clipe cai no fallback de trilha ambiente sintetizada
# — que é sempre a mesma, o que parece "a música não está sendo sorteada"
# mesmo a lógica de sorteio (random.choice) estando correta.
MUSIC_DIR = str(_PROJECT_ROOT / "assets" / "music")
# Pedido: reduzir o volume da música em 60%, deixando 40% do volume ATUAL
# (não 40% de um "volume normal" hipotético — 40% do que já estava
# configurado). Valor anterior: -16.5dB, que em fator de amplitude
# (a mesma escala usada pelo filtro `volume` do ffmpeg) é
# 10^(-16.5/20) ≈ 0.1496 (~15% do volume original de antes daquele ajuste).
# Novo fator = 0.1496 * 0.40 ≈ 0.0598 -> em dB: 20*log10(0.0598) ≈ -24.46dB.
MUSIC_VOLUME_DB = -20.94  # +50% de ganho linear sobre o valor anterior
                          # (-24.46 dB). "+50%" em áudio = multiplicar a
                          # amplitude por 1.5x, não somar/subtrair dB
                          # diretamente (dB já é uma escala logarítmica) —
                          # 1.5x em amplitude equivale a +3.52 dB, por isso
                          # -24.46 + 3.52 ≈ -20.94. Se ainda estiver baixa
                          # (ou alta demais) pro seu gosto, é só esse
                          # número — mais perto de 0 = mais alto.
MUSIC_DUCKING = True
MUSIC_DUCKING_RATIO = 4              # compressão do ducking (era 8 = quase mutava a música
                                      # inteira enquanto havia qualquer fala)
MUSIC_DUCKING_THRESHOLD = 0.08

# --- Normalização de volume (padrão streaming: TikTok/YouTube/Instagram) ---
LOUDNESS_TARGET_LUFS = -14

# --- Efeitos ---
# Zoom punch com easing suave (sine bell): zoom entra e sai gradualmente em
# vez de cortar seco — muito menos incômodo que a versão anterior que fazia
# um snap binário. ZOOM_PUNCH_EASE_SECONDS controla a rampa de entrada/saída;
# ZOOM_PUNCH_HOLD é o tempo no pico máximo. Intensidade reduzida de 14%->7%
# para ficar discreto mas perceptível.
ZOOM_PUNCH_ENABLED = False           # desligado: em conversa/podcast o áudio
                                      # nunca para, gerando punches demais;
                                      # ligue só em vídeos de reação/highlights
ZOOM_PUNCH_INTENSITY = 0.07          # 7% de zoom (era 14% snap; agora é suave)
ZOOM_PUNCH_HOLD = 0.30               # segundos no pico (entrada + hold + saída)
ZOOM_PUNCH_EASE_SECONDS = 0.25       # rampa suave de entrada e saída (cosine)
ZOOM_PUNCH_MIN_GAP = 3.0             # mínimo entre punches (era 2.5s)

# --- Zoom contínuo por energia ("respiração") ---
# Acompanha a curva de energia RMS com suavização EMA alta (resposta lenta),
# produzindo um zoom sutil e gradual que dá vida ao enquadramento — diferente
# do punch (que reage a picos isolados, este acompanha o "nível" geral).
ENERGY_BREATHING_ENABLED = True
ENERGY_BREATHING_MAX = 0.012         # máximo de zoom adicional pela respiração (1.2%)
ENERGY_BREATHING_SMOOTHING = 0.97    # EMA alpha (próximo de 1 = resposta muito lenta)

# --- Correção de cor + vinheta (grade "cinematográfica" no encode final) ---
COLOR_GRADE_ENABLED = True
COLOR_EQ_CONTRAST = 1.08
COLOR_EQ_SATURATION = 1.18
COLOR_EQ_BRIGHTNESS = 0.015
COLOR_EQ_GAMMA = 1.02
VIGNETTE_ENABLED = True
VIGNETTE_ANGLE = "PI/5"              # maior = vinheta mais forte/fechada

# --- Qualidade/tamanho do encode final ---
# CRF pro caminho de CPU (libx264) e QP pro caminho de GPU (VAAPI/AMF —
# ambos usam a mesma escala 0-51, menor = melhor qualidade/arquivo maior).
# Era 20 fixo no código (não configurável) — arquivo saindo de ~110MB pra
# ~2:45 de clipe (1080x1920, ~5.3 Mbps de vídeo). CRF/QP 20 é uma qualidade
# bem alta (praticamente "sem perda visível") pra um vídeo que vai ser
# assistido no celular e provavelmente reprocessado de novo pela própria
# plataforma (TikTok/Reels/Shorts sempre recomprimem o upload) — 23 já é
# imperceptível pra esse uso e reduz bastante o tamanho do arquivo. Suba
# (número menor) se preferir mais qualidade/arquivo maior; desça (número
# maior) pra arquivos ainda menores.
ENCODE_CRF = 23
# Preset era "veryfast" fixo (não configurável). No caminho de CPU
# (libx264) isso realmente prioriza velocidade sobre compressão. No
# caminho de GPU AMD (AMF), porém, "veryfast"/"faster" mapeiam pro modo
# "-quality speed" do AMF (ver `_amf_quality_for_preset` em hwaccel.py) —
# que sacrifica eficiência de compressão (arquivo maior pro MESMO QP) para
# ganhar velocidade de encode. Como o encoder de GPU já é rápido o
# suficiente em QUALQUER modo pra esse tamanho de vídeo (a etapa que
# realmente demora no pipeline é o Whisper, não o encode), não faz sentido
# pagar esse preço — "fast" mapeia pro modo "balanced" do AMF (nem
# "speed" nem "quality"), com compressão bem melhor pelo mesmo QP e ainda
# rápido. Se estiver rodando só em CPU (sem GPU detectada) e quiser
# priorizar velocidade de fato, mude pra "veryfast" aqui.
ENCODE_PRESET = "fast"
# Teto de segurança de bitrate (Mbps) só pro caminho de GPU (AMF/VAAPI) —
# QP/CQP puro não tem limite superior de bitrate: em trechos raros de
# movimento/complexidade muito alta, o encoder pode gastar bits muito
# acima do necessário pra manter o QP alvo. -maxrate/-bufsize colocam um
# teto (o vídeo continua guiado por QP abaixo desse teto — isso NÃO reduz
# a qualidade do conteúdo normal, só evita um pico de bitrate fora do
# comum). 8 Mbps é generoso pra 1080x1920 (bem acima do que YouTube/TikTok
# recomendam pra essa resolução).
ENCODE_MAXRATE_MBPS = 8.0

# --- Seleção automática de clipes virais ---
MIN_GAP_BETWEEN_CLIPS = 8            # intervalo mínimo (s) entre clipes escolhidos

# Penalidade de pontuação quando a ÚLTIMA frase da janela candidata termina
# numa pergunta-gancho/piada (ex.: "Sabe qual é o durão?") sem incluir a
# frase seguinte que traria a resposta — ver RELATORIO_PROXIMOS_PASSOS.txt,
# item 3b. Precisa ser maior que o ganho de pontuação de terminar bem numa
# pergunta (HOOK_PATTERNS = +3.0, "?" = +1.5, total até +4.5) para que o
# algoritmo prefira de fato a versão com a resposta incluída, mesmo com o
# leve custo de se afastar de IDEAL_CLIP_DURATION (0.03/s — alguns segundos
# a mais custam bem menos que isso).
DANGLING_QUESTION_PENALTY = 6.0

# --- Encerrar o clipe quando o ASSUNTO termina, não num ponto arbitrário ---
# HISTÓRICO COMPLETO em RELATORIO_PROXIMOS_PASSOS.txt (itens 9 e 12) — aqui
# só o resumo de por que a arquitetura mudou de novo (v3).
#
# v1: só deixava passar de MAX_CLIP_DURATION quando havia um sinal exato de
# fim de assunto — na prática esse sinal quase nunca disparava em fala real,
# então NENHUM candidato longo era sequer criado -> sempre no teto (75s).
#
# v2: removeu esse bloqueio (toda janela até a folga era criada e pontuada),
# mas o sintoma voltou do mesmo jeito (usuário mediu 163.8s, quase o novo
# teto de 180s) por uma causa diferente: o texto ACUMULADO da janela quase
# sempre GANHA pontos com mais frases (mais chance de bater algum gancho/
# pergunta/número), e a penalidade de duração (0.03/s) era fraca demais pra
# competir — 100s a mais custavam só 3 pontos. Resultado: entre vários
# limites de assunto válidos, o mais LONGO sempre vencia a corrida de score.
#
# v3 (atual): em vez de tentar de novo achar o número "certo" pra fazer a
# duração pesar mais na MESMA fórmula (é só adiar o problema), a escolha foi
# separada em duas etapas com responsabilidades diferentes, em
# src/clip_selector.py:
#   (1) pra CADA início possível, o fim do assunto usado é o que fica mais
#       PRÓXIMO de IDEAL_CLIP_DURATION entre os fins de assunto disponíveis
#       (pausa/marcador/fim de vídeo) — TOPIC_DURATION_FIT_WEIGHT abaixo é o
#       peso dominante dessa escolha, não mais uma pontuação de texto sem
#       teto competindo com uma penalidade fraca;
#   (2) só DEPOIS de cada janela já estar fechada, o conteúdo (gancho/
#       emoção/energia) decide QUAL das janelas (entre vários inícios
#       possíveis) é a melhor pra virar clipe — nunca decide mais o TAMANHO
#       da janela em si.
# Efeito esperado: a maioria dos clipes deve ficar perto de
# IDEAL_CLIP_DURATION; só estica de verdade quando o assunto daquele trecho
# específico genuinamente não tem nenhum fim de assunto mais cedo — não mais
# "sempre no teto" por um desequilíbrio de pontuação.
#
# Heurística de "assunto terminou" (3 sinais independentes, qualquer um
# basta): (1) pausa bem acima do normal DESSE vídeo depois da última frase
# da janela; (2) a última frase da janela usa uma expressão de FECHAMENTO
# ("e é isso", "resumindo", "no final das contas"); (3) a PRÓXIMA frase
# (fora da janela) começa com uma expressão de TRANSIÇÃO de assunto
# ("mudando de assunto", "voltando", "anyway", "moving on").
#
# HONESTIDADE: isso continua sendo uma heurística de padrão de texto + pausa
# + proximidade de duração-alvo, não uma segmentação semântica de tópico de
# verdade (exigiria embeddings/um modelo de linguagem rodando por cima da
# transcrição, fora do escopo desta mudança). Quando NENHUM fim de assunto é
# detectável dentro do alcance permitido a partir de um início, o algoritmo
# cai pro fim de FRASE simples mais próximo da duração ideal (sem esticar
# até a folga — não há sinal nenhum que justifique isso). Testado com
# transcrição sintética reproduzindo os dois casos (texto que sempre ganha
# pontuação com mais frases E um assunto genuinamente longo sem pausas) —
# ver docstring de select_clips. Se um vídeo real ainda sair mais curto ou
# mais longo do que o esperado, os ajustes diretos são
# IDEAL_CLIP_DURATION (o alvo) e/ou TOPIC_DURATION_FIT_WEIGHT (quão forte a
# duração domina sobre o pequeno ajuste de conteúdo entre limites parecidos).
TOPIC_BOUNDARY_PAUSE_SECONDS = 1.1      # piso ABSOLUTO (s) — usado só quando
                                         # a pausa típica do vídeo (calculada
                                         # automaticamente) for menor que isso
TOPIC_BOUNDARY_PAUSE_RELATIVE_MULT = 1.8  # pausa >= (pausa típica do vídeo) *
                                           # este valor já conta como sinal de
                                           # troca de assunto -- adapta ao
                                           # ritmo de fala de cada vídeo
TOPIC_BOUNDARY_BONUS = 4.0              # pequeno prêmio (metade disso, na
                                         # etapa 2) por o fim escolhido ser um
                                         # fim de assunto de verdade, e bônus
                                         # cheio (etapa 1) por terminar numa
                                         # frase de fechamento explícita ao
                                         # desempatar entre limites próximos
TOPIC_START_PENALTY = 6.0               # penalidade (não exclusão -- ver
                                         # comentário "V5" em clip_selector.py)
                                         # quando o INÍCIO escolhido cai no
                                         # meio de um assunto já em andamento,
                                         # em vez de logo após uma pausa/
                                         # conclusão/troca de assunto. Mesma
                                         # ordem de grandeza de
                                         # DANGLING_QUESTION_PENALTY -- forte
                                         # o bastante pra mudar o ranking na
                                         # maioria dos casos, sem impedir por
                                         # completo um início "no meio" quando
                                         # não existe alternativa melhor no
                                         # vídeo inteiro.
TOPIC_BOUNDARY_GRACE_SECONDS = 20.0     # quanto além de MAX_CLIP_DURATION um
                                         # fim de assunto ainda pode ser
                                         # considerado -- MAX_CLIP_DURATION
                                         # (75s) + esta folga (20s) = teto
                                         # real de 95s. Era 105s → teto de
                                         # 180s, clips de 3min ganhavam por
                                         # acumular mais hooks que clipes
                                         # curtos; reduzido pra Shorts.
                                         # antes disso (etapa 1 decide).
TOPIC_DURATION_FIT_WEIGHT = 0.3         # peso (pontos por segundo de
                                         # distância de IDEAL_CLIP_DURATION)
                                         # usado na ETAPA 1 (escolher qual fim
                                         # de assunto usar) — dominante de
                                         # propósito, pra duração realmente
                                         # decidir isso em vez de uma corrida
                                         # de pontuação de texto sem teto.

# --- Coesão lexical (TextTiling) — pega troca de assunto SEM pausa nem
# palavra-chave (ver _lexical_boundary_scores em clip_selector.py) ---
# Pedido do usuário depois de ver que a v anterior só cobria marcador
# textual/pausa: "o que resolve é ele ler e entender o contexto das frases
# e pegar o começo do assunto até o final". Isso pediria por padrão um LLM
# (ler e ENTENDER de verdade) — descartado aqui não por peso de hardware
# (a API roda no servidor, não na RX580/Xeon local) mas por manter o
# projeto 100% local/offline, sem chave/custo/rede por vídeo, como todo o
# resto já é (whisper.cpp+Vulkan, heurística de texto local etc.). Em vez
# disso: TextTiling (Hearst, 1997), técnica clássica de segmentação de
# texto por VOCABULÁRIO — mede se as palavras de conteúdo mudam de forma
# significativa entre uma janela de frases antes e depois de cada corte
# candidato. Não entende semântica de verdade (não é um LLM), mas pega
# muita virada de assunto que pausa/palavra-chave sozinhas não pegam —
# sem GPU, sem nova dependência (só regex + contagem, Python puro).
TOPIC_LEXICAL_TARGET_WORDS = 40         # quantas palavras de CONTEÚDO (sem
                                         # stopword) juntar de cada lado do
                                         # corte antes de comparar — maior =
                                         # mais estável/menos sensível a
                                         # ruído local, mas reage mais devagar
                                         # a uma virada bem próxima do corte
TOPIC_LEXICAL_MIN_WORDS = 8             # abaixo disso (frase curta demais,
                                         # tipo interjeição isolada, ou perto
                                         # demais da borda do transcript) não
                                         # arrisca marcar como fronteira —
                                         # dado insuficiente pra confiar
TOPIC_LEXICAL_BOUNDARY_MIN = 0.55       # "vale" de similaridade (0-1,
                                         # normalizado pelo vale mais forte
                                         # DESTE vídeo/bloco -- adaptativo,
                                         # igual TOPIC_BOUNDARY_PAUSE_
                                         # RELATIVE_MULT) a partir do qual já
                                         # conta como fim de assunto sozinho,
                                         # mesmo sem pausa/palavra-chave. Suba
                                         # se estiver cortando no meio do
                                         # assunto (falso positivo); desça se
                                         # estiver deixando passar viradas
                                         # óbvias sem pausa nem marcador.
TOPIC_LEXICAL_FIT_BONUS = 1.0           # peso pequeno (pontos) usado só pra
                                         # desempatar entre limites de assunto
                                         # candidatos de duração parecida —
                                         # não decide sozinho quem vence.

# --- Confirmação por lookahead (V9) — pedido explícito do usuário: "ele
# analisa as frases seguintes, e, caso elas continuem o assunto, ele retira
# o ponto de fim de vídeo e joga mais pra frente... até achar o ponto onde
# o assunto acabou". Antes de aceitar um candidato a fronteira (pausa,
# frase de fechamento, troca de assunto, ou pico do TextTiling acima),
# `_boundary_confirmed` em clip_selector.py olha uma janela BEM MAIOR de
# vocabulário dos dois lados — pega o caso de um comentário/pausa curta
# que parece fim de assunto de perto, mas o mesmo papo CONTINUA um pouco
# mais adiante. Quando a confirmação falha, esse ponto deixa de contar como
# fronteira e a busca (que já escaneia em ordem crescente) naturalmente
# considera o próximo candidato mais adiante — sem precisar de nenhum loop
# de busca novo.
TOPIC_LEXICAL_CONFIRM_WORDS = 90        # > TOPIC_LEXICAL_TARGET_WORDS de
                                         # propósito -- precisa enxergar mais
                                         # longe que a detecção padrão pra
                                         # confirmar de verdade, não só
                                         # repetir a mesma checagem
TOPIC_LEXICAL_CONFIRM_MAX_SIMILARITY = 0.22  # similaridade ACIMA disso entre
                                              # as duas janelas largas =
                                              # assunto ainda é o mesmo de
                                              # verdade -- desconfirma esse
                                              # ponto como fronteira. Baixe
                                              # se estiver rejeitando
                                              # fronteiras boas demais; suba
                                              # se estiver deixando passar
                                              # assunto que claramente volta
                                              # logo depois.

# --- Vídeos longos (podcast de horas): transcrever por blocos, não tudo de
# uma vez ---
# Transcrever um podcast de 3h inteiro pra depois usar só 3-4 minutos dele
# em clipes é caro e, na prática, desperdiça a maior parte do processamento.
# Acima de LONG_VIDEO_THRESHOLD_SECONDS, o vídeo (já baixado por inteiro) é
# tratado como uma sequência de blocos lógicos de CHUNK_DURATION_SECONDS
# (só recortes de tempo sobre o arquivo já baixado — não gera arquivos
# separados por bloco, então não há recodificação extra). Um bloco
# aleatório é transcrito e pontuado por vez; se o melhor candidato do bloco
# não alcançar CHUNK_VIRAL_SCORE_MIN, o bloco é descartado e o próximo
# (também aleatório, sem repetição) é transcrito — essa transcrição já
# começa em segundo plano assim que o bloco atual é entregue para montagem,
# então o tempo de montar o clipe (a parte mais lenta depois da
# transcrição) se sobrepõe ao tempo de transcrever o próximo bloco, em vez
# de rodar tudo em sequência estrita. Ver src/long_video.py.
#
# HONESTIDADE sobre CHUNK_VIRAL_SCORE_MIN: a escala de score de
# clip_selector.py (hooks +3, "?" +1.5, energia de áudio variável etc.) não
# tem um teto fixo nem foi calibrada contra vídeos longos reais neste
# ambiente (sem exemplo de podcast de horas disponível para medir a
# distribuição típica de score por bloco) — o valor abaixo é um palpite
# inicial razoável, não uma calibração validada. Se blocos bons estiverem
# sendo descartados com frequência (aviso no console), baixe este valor; se
# blocos fracos estiverem sendo aceitos, suba.
LONG_VIDEO_THRESHOLD_SECONDS = 39 * 60   # 39 minutos
CHUNK_DURATION_SECONDS = 20 * 60         # 20 minutos
CHUNK_VIRAL_SCORE_MIN = 8.0

OUTPUT_DIR = "output"
WORK_DIR = ".autoclip_work"

# --- Mecanismo anti-softlock ---
# Se o programa inteiro (do início ao fim, incluindo o encerramento do
# processo) não terminar sozinho dentro deste tempo, main.py força o
# encerramento (ver _start_watchdog/_force_exit em main.py). Cobre tanto
# um travamento NO MEIO do processamento (ex: um subprocesso ffmpeg/
# whisper.cpp que nunca retorna) quanto um travamento NA SAÍDA do processo
# (ex: driver de GPU com rotina de finalização travada no Windows — o
# motivo original desse mecanismo existir).
#
# 90 minutos é generoso pro uso normal (poucos clipes de um vídeo curto ou
# médio). Se você processa vídeos MUITO longos e/ou pede muitos clipes de
# uma vez e isso legitimamente passa de 90min, suba este valor — não tem
# como o programa "saber" diferenciar processamento genuinamente demorado
# de travamento de verdade, então isso é só um teto de segurança, ajuste
# pro seu uso real.
WATCHDOG_TIMEOUT_SECONDS = 90 * 60

# Watchdog CURTO, específico pro trecho final (depois de já ter impresso o
# resumo de "CONCLUÍDO"/"Arquivos salvos em"). Independente do watchdog
# de 90min acima — esse aqui existe porque foi relatado o processo travar
# bem NESSE ponto mesmo já existindo TerminateProcess logo em seguida (ver
# comentário em main.py, junto do segundo _start_watchdog). Poucos
# segundos é suficiente: se o programa realmente já terminou o trabalho,
# não há motivo legítimo pra essa etapa demorar.
EXIT_WATCHDOG_SECONDS = 6

# --- Modo REACT (ver RELATORIO_PROXIMOS_PASSOS.txt, item 14) ---
# NOVO -- construído em cima do rastreamento de tela do item 3a
# (_ScreenTracker/_compose_screen_frame em reframer.py), que já existia mas
# ficava sempre desligado (VIDEO_IN_VIDEO_ENABLED=False fixo). Agora, em
# vez de um interruptor manual, o próprio vídeo é classificado automatica-
# mente (ver detect_react_video em src/react_detector.py): se for
# detectado como REACT, VIDEO_IN_VIDEO_ENABLED é ligado dinamicamente só
# pra ESSE vídeo (main.py faz isso depois da classificação, antes de
# montar os clipes) -- vídeos comuns de pessoa falando continuam sem pagar
# o custo/risco desse pipeline.
REACT_MODE_AUTO_DETECT = True
# a cada quantos segundos o vídeo inteiro é reamostrado pra decidir se é
# REACT -- pedido explícito do usuário ("a cada 10 segundos"), cobre
# vídeos que alternam entre reagir e ficar em full-cam.
REACT_CHECK_INTERVAL_SECONDS = 10.0
# fração mínima das amostras com uma tela plausível detectada pra
# considerar o vídeo como um todo um REACT. NÃO CALIBRADO contra nenhum
# vídeo real -- é um palpite inicial razoável (1 em cada 4 checagens),
# ajuste se estiver classificando vídeo comum como react ou vice-versa.
REACT_MIN_SCREEN_FRACTION = 0.25

# --- Facecam pequena (layout típico de react: reator numa caixinha
# pequena sobre a tela reagida, às vezes com chat do lado) ---
# quando o rosto detectado ocupa menos que essa fração da ALTURA do
# frame de origem, o crop deixa de usar o zoom "fechado" padrão
# (FACE_CROP_ZOOM) -- que ficaria extremamente ampliado/pixelado numa
# facecam pequena -- e afrouxa suavemente (cresce em direção à altura
# cheia da fonte) na mesma proporção em que o rosto está abaixo desse
# limiar, capturando mais contexto ao redor (na prática, geralmente o
# chat ou parte da tela reagida visível perto da facecam) sem precisar
# detectar essas regiões especificamente.
FACECAM_SMALL_HEIGHT_FRAC = 0.16

# --- Enquadramento pelo TAMANHO do rosto (vídeo comum / podcast) ---
# Achado real (clipes do Flow Podcast): em plano aberto/médio o crop usava a
# altura INTEIRA da fonte, então a pessoa ficava minúscula no meio da tela
# e a cortina escura do estúdio ocupava metade do quadro ("tela preta").
# Agora o crop é dimensionado pra que o rosto ativo ocupe
# SUBJECT_TARGET_FACE_FRAC da altura do quadro final (close da fonte já
# passa disso -> sem zoom extra), limitado por SUBJECT_MAX_UPSCALE (quanto
# a imagem pode ser ampliada antes de ficar borrada: 3.0 = crop de no
# mínimo 640px de altura pra sair em 1920). Plano de GRUPO (2+ rostos em
# quadro e, mesmo com o zoom máximo, rosto abaixo de
# SUBJECT_FIT_GROUP_FACE_FRAC da altura -- ex.: 4 pessoas na mesa) usa o
# layout "fit": quadro inteiro no meio com fundo desfocado, ninguém cortado
# (vale até o próximo corte de câmera). Uma pessoa só fica sempre no
# recorte, a não ser que o rosto não chegue nem a
# SUBJECT_FIT_SINGLE_FACE_FRAC. Não vale no modo REACT (usa a lógica de
# facecam).
SUBJECT_TARGET_FACE_FRAC = 0.22
SUBJECT_MAX_UPSCALE = 3.0
SUBJECT_FIT_GROUP_FACE_FRAC = 0.16
SUBJECT_FIT_SINGLE_FACE_FRAC = 0.07

# quanto tempo (segundos) o enquadramento fica forçado na tela reagida
# depois de detectar uma referência visual no texto ("olha a camisa
# dele") -- ver find_reference_times em src/react_detector.py. Também
# serve de distância mínima entre dois disparos (evita ficar piscando
# entre referências muito próximas uma da outra).
REFERENCE_ZOOM_HOLD_SECONDS = 2.5

# --- Item 17 (pedido do usuário): troca de foco streamer <-> vídeo reagido
# no modo REACT, baseada em quem está "vivo" (falando/acontecendo algo)
# agora, não só em qual dos dois foi detectado por último ---
# Antes desta mudança, a tela reagida tinha prioridade INCONDICIONAL sobre
# o rosto sempre que confirmada com atividade (`_ScreenTracker.active_region`)
# -- não distinguia "streamer calado, vídeo com algo acontecendo" (deve
# focar no vídeo) de "streamer comentando em cima do vídeo que também está
# tocando" (deveria afastar e mostrar os dois, não só cravar na tela e
# cortar o streamer). Ver render_vertical_clip (src/reframer.py) pro estado
# de decisão completo: combina ATIVIDADE DE BOCA do streamer (mesmo placar
# já calculado pra trocar de rosto entre vários falantes, reaproveitado —
# ver active_speaking_activity) com atividade de pixel da tela (já
# existente) numa tabela de 4 casos:
#   streamer calado  + tela ativa  -> foca na tela (pedido do usuário)
#   streamer falando + tela ativa  -> afasta, mostra os dois ("full zoom
#                                      out", pedido do usuário)
#   streamer falando + tela parada -> foca no streamer (comportamento normal)
#   nenhum dos dois claramente ativo -> mesma histerese de confiança de
#                                        rosto de sempre
# "Algo relevante acontecendo no vídeo (ex.: uma batida de carro)" não
# precisou de detecção própria: a atividade de pixel da tela já é genérica
# (qualquer mudança de imagem sustentada conta), então cobre tanto "pessoa
# falando no vídeo" quanto um evento visual chamativo sem lógica extra.
REACT_STREAMER_SPEAKING_ACTIVITY_MIN = 3.0  # mesma ORDEM DE GRANDEZA de
                                             # VIDEO_IN_VIDEO_ACTIVITY_MIN
                                             # (acima) de propósito -- os
                                             # dois medem "diferença média de
                                             # pixel por frame" na mesma
                                             # escala (0-255), só em recortes
                                             # de tamanho diferente (boca
                                             # 24x24 vs tela 32x32). Ainda
                                             # não validado contra um vídeo
                                             # REACT real (ver HONESTIDADE no
                                             # relatório) -- se o streamer
                                             # falando não estiver disparando
                                             # o modo "screen"/"wide" na hora
                                             # certa, esse é o primeiro valor
                                             # a ajustar (baixe pra ficar mais
                                             # sensível a fala baixa/sutil).
