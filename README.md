# Auto Clipper 🎬

Gerador automático de cortes virais para YouTube Shorts / TikTok / Reels,
no estilo Opus Clip — 100% local e gratuito.

Você cola o link de um vídeo do YouTube, diz quantos clipes quer, e o
programa faz **tudo sozinho**:

1. **Baixa** o vídeo (yt-dlp)
2. **Transcreve** a fala com timestamp por palavra (Whisper)
3. **Encontra os melhores trechos** automaticamente, com um sistema de
   pontuação "viral" (ganchos, perguntas, picos de energia na voz, etc.)
4. **Reenquadra** cada corte para vertical 9:16, seguindo o rosto de quem
   fala e dando zoom pelo tamanho do rosto; plano aberto com várias pessoas
   vira layout "fit" (quadro inteiro sobre fundo desfocado), e troca de
   câmera da fonte vira corte seco
5. **Corta as pausas** (jump cuts) de dentro do clipe — só onde o áudio está
   em silêncio de verdade — alternando o zoom a cada corte, como um editor faz
6. **Gera legendas** animadas estilo corte viral: 3 palavras em caixa alta,
   entrada com "pop", palavra falada acesa em amarelo, números/palavras de
   impacto em verde
7. Põe um **título-gancho** no topo nos primeiros segundos, com um *whoosh*
8. Adiciona **música de fundo** (phonk/funk/trap do NCS, liberada pra vídeo
   monetizado) com *ducking* automático (abaixa sozinha quando há fala)
9. Exporta os `.mp4` finais prontos para publicar, uma versão **sem música**
   (pra usar som em alta do app) e um **kit de postagem** (`.post.txt`):
   título, descrição, hashtags para YouTube Shorts / Instagram Reels /
   TikTok e o crédito da música

## ⚠️ Sobre os testes

O pipeline inteiro foi testado de ponta a ponta com vídeo real (entrevista
e trechos de podcast em estúdio, com close, plano médio e plano aberto de
4 pessoas) numa máquina Linux sem GPU. O download do YouTube e a aceleração
pela GPU (AMF / whisper.cpp com Vulkan) só dá pra testar na **sua máquina**:
servidores em nuvem são bloqueados pelo YouTube e não têm a RX 580.

## Pipeline de edição: 1 único passe de encode

Cada clipe passa por: recorte do trecho → reenquadramento vertical seguindo
o rosto → legendas → música com ducking → correção de cor/vinheta → efeitos
de zoom. A versão atual faz **tudo isso numa única codificação de vídeo**
por clipe:

1. uma thread lê o stdout de um processo `ffmpeg` "leitor" que decodifica
   (sem recodificar) só o trecho necessário do vídeo original e empilha os
   frames crus numa fila — assim a decodificação do próximo frame acontece
   em paralelo com o processamento do frame atual, em vez de alternar em
   sequência estrita;
2. o Python recorta cada frame seguindo o rosto (câmera virtual suavizada) e
   aplica os zoom punches. Quando nenhum rosto é encontrado por tempo
   suficiente (plano aberto, corte de câmera, duas pessoas afastadas), o
   programa troca para um modo **"plano aberto"**: mostra o quadro inteiro
   (sem cortar ninguém de fora) sobre um fundo desfocado/escurecido, com uma
   transição suave entre os dois modos — a mesma técnica usada por
   Opus Clip/CapCut, em vez de cravar um crop apertado num ponto qualquer da
   imagem (o que geralmente mostra só mesa/objetos vazios);
3. os frames já recortados são enviados via pipe para um segundo `ffmpeg`,
   que aplica correção de cor + vinheta ("grade cinematográfica"), já
   embute a legenda (fonte própria, sem depender do sistema) **e** mixa o
   áudio final (voz + música com ducking + normalização de volume) — tudo
   na mesma codificação, usando o encoder de GPU detectado automaticamente
   quando disponível.

Isso reduz o trabalho de 3 codificações completas de vídeo por clipe (a
etapa mais cara do pipeline) para apenas 1, sem gerar arquivos de vídeo
intermediários em disco.

## Requisitos

- Python 3.10+
- **ffmpeg** instalado no sistema e disponível no PATH
  - Windows: `winget install Gyan.FFmpeg` (ou `choco install ffmpeg`, ou baixe
    o build "full"/"essentials" em [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)
    e adicione a pasta `bin\` ao PATH — esses builds já incluem suporte a
    `h264_amf`, usado pela RX 580 e outras GPUs AMD)
  - Linux: `sudo apt install ffmpeg`
  - Mac: `brew install ffmpeg`
- (Opcional) GPU acelera o encode de vídeo automaticamente — NVENC
  (NVIDIA, qualquer SO), AMF (AMD no Windows, ex: RX 580) ou VAAPI (AMD/Intel
  no Linux). A transcrição (Whisper) roda na CPU nesses casos, já otimizada
  para usar vários núcleos

## Instalação (Windows)

```powershell
cd opus-clip-clone
py -3 -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

# runtime JavaScript que o yt-dlp usa pra passar pelo desafio do YouTube
# (sem ele o download perde formatos ou falha com 403). Se você já tem o
# Node.js instalado, o programa usa ele e este passo é opcional.
winget install DenoLand.Deno
```

Depois, confirme que `ffmpeg -version` funciona num terminal novo (PATH só
atualiza em janelas abertas depois da instalação).

Se o download começar a falhar do nada (403, "Sign in to confirm you're not
a bot"), quase sempre é o YouTube mudando algo e o yt-dlp já tendo correção:
`pip install -U "yt-dlp[default]"`.

<details>
<summary>Instalação em Linux/Mac</summary>

```bash
cd opus-clip-clone
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
</details>

## Uso

```bash
python main.py
```

O programa vai perguntar:

```
Cole o link do vídeo do YouTube: https://youtube.com/watch?v=...
Quantos clipes você quer gerar? [3]: 5
```

E depois roda tudo sozinho. Os clipes finais aparecem em `output/`, cada
um com estes arquivos ao lado:

- `clip_XX_....mp4` — o clipe pronto, com música de fundo.
- `clip_XX_..._sem_musica.mp4` — a mesma edição só com a voz: poste este e
  escolha um **som em alta pela biblioteca do app** (é o único jeito de usar
  música comercial viral sem levar reivindicação de direitos autorais).

- `clip_XX_....post.txt` — **pronto pra colar na hora de postar**: título,
  descrição, hashtags de cada rede (assunto do clipe + `POST_EXTRA_HASHTAGS`
  do seu canal + `#shorts` / `#reels` / `#fyp`), o crédito da música (se a
  licença exigir) e o link do vídeo original.
- `clip_XX_....contexto.txt` — a transcrição do clipe com 60s de contexto
  antes e depois, pra conferir se o corte começou/terminou no lugar certo.

## Aceleração de hardware (GPU) e processamento paralelo

O programa detecta e testa sozinho, ao iniciar, qual a melhor forma de
codificar vídeo disponível na sua máquina — sem precisar configurar nada:

1. **GPU NVIDIA (NVENC)**, se disponível
2. **GPU AMD via AMF** (Windows) — inclui a **RX 580**, que tem um encoder
   de vídeo dedicado (VCE) plenamente suportado via AMF, mesmo não tendo
   suporte a compute (ROCm). No Linux, o caminho equivalente é **VAAPI**.
3. **CPU (libx264)** — usado automaticamente se não houver GPU compatível,
   ou se o encode por hardware falhar por qualquer motivo

Isso aparece no início da execução:
```
Encode de vídeo: GPU AMD via AMF (Windows — usa o encoder de vídeo dedicado da placa, ex: RX 580)
```

Se aparecer `CPU (libx264)` no lugar disso na sua RX 580, o driver AMD
(Adrenalin) provavelmente está desatualizado, ou o build do ffmpeg no PATH
não tem suporte a AMF — veja "Requisitos" acima.

**Nota sobre transcrição (Whisper):** por padrão roda sempre na CPU. O
motor padrão do programa (`faster-whisper`/CTranslate2) não tem suporte a
AMD ROCm nem a DirectML para GPUs GCN/Polaris como a RX 580 — só CUDA
(NVIDIA) ou CPU. **Só é possível usar a RX 580 pra transcrever trocando de
motor** (veja "Transcrição via GPU/Vulkan" logo abaixo). Sem essa troca, o
programa configura automaticamente o número de threads da CPU usado pela
transcrição com base nos núcleos físicos detectados.

## Transcrição via GPU/Vulkan (usando a RX 580 de verdade)

O motor padrão (`faster-whisper`) só roda a transcrição na CPU nesse
hardware — ele simplesmente não tem um caminho de GPU pra placas AMD, em
nenhum sistema operacional. Pra colocar a RX 580 pra transcrever, é preciso
trocar de motor para o [whisper.cpp](https://github.com/ggerganov/whisper.cpp),
compilado com o backend **Vulkan** (API gráfica multiplataforma que a RX
580 suporta nativamente no Windows, mesmo sem ROCm).

> ATENÇÃO: isso foi implementado e a integração foi conferida linha a linha
> contra o código-fonte do whisper.cpp (flags do CLI, formato do JSON, nome
> do executável), mas não foi possível testar a inferência de verdade numa
> GPU no ambiente onde isso foi escrito (sem GPU física e sem acesso à
> Hugging Face pra baixar os pesos do modelo). Teste na sua máquina — se
> algum nome de flag tiver mudado numa versão mais nova do whisper.cpp, me
> avise que eu ajusto.

### 1. Compilar o whisper.cpp com Vulkan (Windows)

Não existe build Windows com Vulkan pronto pra baixar nas releases
oficiais no momento (só CPU, "BLAS" e CUDA) — precisa compilar:

```powershell
# pre-requisitos: Visual Studio 2022 (com "Desktop development with C++"),
# CMake, e o Vulkan SDK (https://vulkan.lunarg.com/sdk/home#windows)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

O executável fica em `build\bin\Release\whisper-cli.exe`.

### 2. Baixar um modelo

```powershell
cd whisper.cpp
.\models\download-ggml-model.cmd small
```

Isso baixa `models\ggml-small.bin` (multilingue, funciona pra portugues;
modelos com sufixo `.en`, tipo `small.en`, sao so ingles). Use `medium` se
quiser mais precisao e a GPU aguentar; `base` se quiser mais velocidade.

### 3. Configurar em `src/config.py`

```python
TRANSCRIBE_ENGINE = "whispercpp"
WHISPERCPP_BIN = r"C:\caminho\pra\whisper.cpp\build\bin\Release\whisper-cli.exe"
WHISPERCPP_MODEL = r"C:\caminho\pra\whisper.cpp\models\ggml-small.bin"
WHISPERCPP_LANGUAGE = "pt"
```

Pronto — a proxima execucao ja sai usando a GPU. O log no inicio da etapa
`[2/6]` muda de `device=cpu` (faster-whisper) para a chamada do
`whisper-cli` (whisper.cpp); pra confirmar que a GPU esta sendo usada de
verdade, rode o `whisper-cli.exe` manualmente uma vez — ele imprime no
inicio qual backend ggml carregou (deve aparecer mencao a `Vulkan0` com o
nome da sua GPU, algo como "AMD Radeon RX 580").

Pra voltar ao motor padrao, e so `TRANSCRIBE_ENGINE = "faster-whisper"`.

**Clipes em paralelo:** como cada clipe é processado de forma independente
(corte → reenquadramento → legendas → música), o programa processa vários
clipes ao mesmo tempo automaticamente, usando o máximo de núcleos livres da
CPU. Se o encode for por GPU, a concorrência é limitada a 2 clipes
simultâneos (evita disputa pelo único chip de vídeo da placa); se for por
CPU, sobe para até 4 (ajustável em `MAX_PARALLEL_CLIPS_HW` /
`MAX_PARALLEL_CLIPS_CPU` em `config.py`, ou fixe um número exato em
`PARALLEL_CLIPS`).

Se algum encode por GPU falhar no meio do processo (raro — o programa já
testa o driver antes de começar), ele refaz automaticamente aquele passo em
CPU e avisa no terminal, sem travar o restante do pipeline.

**Ajuste automático de threads:** quando vários clipes rodam em paralelo
via CPU, o número de threads de encode de *cada* clipe é dividido pelo
número de clipes simultâneos (em vez de cada um tentar usar todos os
núcleos), evitando que a CPU fique trocando de contexto entre threads
demais competindo pelos mesmos núcleos físicos — importante em CPUs com
muitos núcleos, como o Xeon E5-2680 v4 (14C/28T).

## Personalização (`src/config.py`)

Tudo é configurável em um único arquivo:

| O que ajustar | Variável |
|---|---|
| Duração mínima/máxima/ideal dos clipes | `MIN_CLIP_DURATION`, `MAX_CLIP_DURATION`, `IDEAL_CLIP_DURATION` |
| Tamanho/modelo do Whisper (velocidade x precisão) | `WHISPER_MODEL_SIZE` (`tiny`→`large-v3`) |
| Cor da legenda / cor de destaque | `CAPTION_PRIMARY_BGR`, `CAPTION_HIGHLIGHT_BGR` |
| Quantas palavras aparecem por vez na legenda | `CAPTION_WORDS_PER_GROUP` |
| Intensidade/frequência dos zoom punches | `ZOOM_PUNCH_INTENSITY`, `ZOOM_PUNCH_MIN_GAP` |
| Volume da música de fundo | `MUSIC_VOLUME_DB` |
| Volume alvo do áudio final (padrão streaming) | `LOUDNESS_TARGET_LUFS` (padrão: -14 LUFS) |
| Fonte da legenda | `CAPTION_FONT` (Anton, incluída em `assets/fonts/`) |
| Altura da legenda / margem lateral (zona segura do app) | `CAPTION_MARGIN_V` (padrão 560), `CAPTION_MARGIN_H` |
| Hashtags fixas do seu canal no `.post.txt` | `POST_EXTRA_HASHTAGS` |
| Corte das pausas (jump cuts) — ligar/desligar, sensibilidade | `JUMPCUT_ENABLED`, `JUMPCUT_MIN_GAP_SECONDS`, `JUMPCUT_PUNCH_ZOOM` |
| Título-gancho no topo — ligar/desligar, duração, tamanho | `HOOK_ENABLED`, `HOOK_SECONDS`, `HOOK_FONT_SIZE` |
| Efeito sonoro (whoosh) | `SFX_ENABLED`, `SFX_WHOOSH_DB` |
| Cores da legenda: palavra falada / palavras de impacto | `CAPTION_HIGHLIGHT_BGR`, `CAPTION_EMPHASIS_BGR` |
| Zoom em plano médio / quando usar o layout fit | `SUBJECT_TARGET_FACE_FRAC`, `SUBJECT_MAX_UPSCALE`, `SUBJECT_FIT_GROUP_FACE_FRAC` |
| Aproximação lenta (Ken Burns) no plano aberto | `WIDE_PUSH_IN_PER_SECOND`, `WIDE_PUSH_IN_MAX` |
| Versão sem música de cada clipe | `EXPORT_NO_MUSIC_VERSION` |
| Detector de rosto (`yunet` ou `haar`) | `FACE_DETECTOR` |
| A partir de quanto tempo um vídeo é "longo" (transcrição por blocos) | `LONG_VIDEO_THRESHOLD_SECONDS` (padrão: 39 min) |
| Duração de cada bloco de um vídeo longo | `CHUNK_DURATION_SECONDS` (padrão: 20 min) |
| Viral score mínimo pra aceitar um bloco (senão tenta o próximo) | `CHUNK_VIRAL_SCORE_MIN` (**não calibrado contra vídeo real, ver comentário em `config.py`**) |
| Sensibilidade da detecção de "fim de assunto" (pausa) | `TOPIC_BOUNDARY_PAUSE_SECONDS`, `TOPIC_BOUNDARY_BONUS`, `TOPIC_BOUNDARY_GRACE_SECONDS` |

## Música de fundo

`assets/music/` vem com 23 faixas de edit do **NoCopyrightSounds (NCS)** —
Brazilian Phonk, Phonk, Jersey Club, Future Trap, lançamentos de 2024-2026
(lista completa em `assets/music/CREDITS.txt`). A política do NCS libera o
uso por criadores independentes, **inclusive em vídeo monetizado**, desde
que o crédito vá na descrição — o `.post.txt` de cada clipe já traz a linha
de crédito da faixa sorteada. Cada arquivo é um trecho de ~95s a partir do
drop, normalizado pra -16 LUFS.

Para usar as suas músicas, coloque os `.mp3`/`.wav`/`.m4a` na pasta e
adicione cada uma em:

- `assets/music/track_drops.txt` — `Título - M:SS` (onde começa o
  drop/refrão). **Faixa sem linha aqui é ignorada.**
- `assets/music/track_credits.txt` (opcional) — `Título | linha de crédito`,
  para faixas cuja licença exige atribuição.

> ⚠️ Música "viral" comercial (hits do momento) quase sempre tem direitos
> autorais: no YouTube o clipe leva reivindicação do Content ID (receita vai
> pro dono da música ou o vídeo é bloqueado), e Instagram/TikTok podem
> silenciar o áudio. Para trends com música famosa, poste o
> `*_sem_musica.mp4` e escolha o som pela biblioteca do próprio app.

Se a pasta estiver vazia, o programa gera uma trilha ambiente simples para
não travar o fluxo automático.

## Vídeos longos (podcasts, entrevistas de horas)

Acima de `LONG_VIDEO_THRESHOLD_SECONDS` (padrão: 39 min), o programa não
transcreve o vídeo inteiro de uma vez — ele já foi baixado por completo, mas
é tratado como uma sequência de blocos lógicos de `CHUNK_DURATION_SECONDS`
(padrão: 20 min). Um bloco aleatório é transcrito e pontuado por vez; se o
melhor candidato do bloco não alcançar `CHUNK_VIRAL_SCORE_MIN`, o bloco é
descartado e o próximo (também aleatório, sem repetição) é transcrito — essa
transcrição já roda em segundo plano assim que o bloco anterior é entregue
para montagem, então o tempo de montar o clipe se sobrepõe ao tempo de
transcrever o bloco seguinte, em vez de rodar tudo em sequência estrita. Se
nenhum bloco do vídeo atingir o score mínimo, o programa usa os melhores
candidatos vistos mesmo abaixo do limiar, em vez de terminar sem gerar nada
(avisando no console quando isso acontece).

`CHUNK_VIRAL_SCORE_MIN` é um palpite inicial, não uma calibração validada
contra vídeos longos reais — se blocos bons estiverem sendo descartados com
frequência (muitos avisos de "completando com candidato abaixo do limiar"),
baixe esse valor em `config.py`.

## Como funciona a seleção "viral" dos cortes

O `src/clip_selector.py` não usa nenhuma IA de terceiros — é um sistema de
pontuação transparente que você pode ajustar:

- **Texto**: detecta ganchos comuns ("você sabia", "o maior erro", números,
  perguntas, palavras de forte carga emocional) em português e inglês
- **Áudio**: mede a energia (RMS) da voz — trechos com mais entusiasmo/ênfase
  pontuam mais
- **Estrutura**: nunca corta no meio de uma frase; prioriza a duração ideal
  configurada; garante que os clipes escolhidos não se sobrepõem
- **Fim de assunto**: além de nunca cortar no meio de uma frase, o programa
  prefere terminar o clipe onde o ASSUNTO termina, usando três sinais
  (qualquer um basta): pausa bem acima do normal DESSE vídeo (adaptativo,
  não um número fixo), expressões de fechamento ("e é isso", "resumindo")
  ou a frase seguinte já puxando outro assunto ("mudando de assunto",
  "anyway"), e **coesão lexical** (técnica TextTiling): mede se o
  vocabulário de conteúdo muda de forma significativa ao redor de cada
  corte candidato, pegando viradas de assunto "silenciosas" que não têm
  nem pausa longa nem nenhuma das frases de transição previstas — sem
  precisar de LLM, GPU ou nenhuma dependência nova (só contagem de
  palavra + cosseno). Início do clipe passa pela mesma checagem (começar
  logo após um desses sinais, não no meio de um assunto em andamento).
  Quando nenhum bate perto da duração ideal, a busca pode esticar um
  pouco além de `MAX_CLIP_DURATION` (limitado por
  `TOPIC_BOUNDARY_GRACE_SECONDS`) só para alcançar um desses pontos. É
  heurística (pausa/palavra-chave/vocabulário), não compreensão semântica
  de verdade — ajuste `TOPIC_BOUNDARY_*`/`TOPIC_LEXICAL_*` em `config.py`
  se a busca não estiver parando no lugar certo no seu conteúdo (o script
  `scripts/validate_topic_lexical.py` ajuda a calibrar o sinal lexical
  contra transcrições reais suas, sem precisar rodar o pipeline inteiro).

Quer plugar um modelo de linguagem (Claude, GPT etc.) para pontuar os
trechos com mais inteligência semântica? É só trocar `_text_score()` em
`clip_selector.py` por uma chamada à API do seu provedor preferido — a
arquitetura já foi pensada para isso.

## Reenquadramento (auto-frame)

**Tamanho do rosto decide o enquadramento** (vídeo comum/podcast):

- **close** da fonte: recorte seguindo o rosto;
- **plano médio** de uma pessoa: zoom até o rosto ocupar ~22% da altura
  (`SUBJECT_TARGET_FACE_FRAC`), sem ampliar a imagem mais que 3x
  (`SUBJECT_MAX_UPSCALE`) pra não borrar;
- **plano aberto de grupo** (2+ pessoas, rostos pequenos): layout "fit" —
  quadro inteiro no meio com fundo desfocado, ninguém cortado; vale até o
  próximo corte de câmera;
- **troca de câmera** da fonte: o enquadramento muda junto, em corte seco
  (sem fade).

Usa detecção de rosto via OpenCV (Haar Cascade, incluso na própria lib —
não precisa baixar nada), combinando **dois classificadores**: rosto de
frente e rosto de perfil (testado nos dois lados, via espelhamento). Isso
importa bastante em vídeos de podcast/entrevista com duas pessoas
conversando entre si de lado — só o detector frontal perde o rosto sempre
que alguém vira a cabeça para falar com quem está ao lado, justamente nos
momentos de diálogo mais ativo.

Quando há **mais de um rosto em quadro** (ex.: dois apresentadores lado a
lado), o programa não simplesmente pula para "o maior rosto" a cada
checagem — isso faria a câmera virtual ficar oscilando entre as duas
pessoas. Em vez disso, prioriza o rosto mais próximo de onde já estava
rastreando, e só troca de pessoa se o outro rosto for consideravelmente
maior/mais próximo da câmera.

A detecção roda sobre uma cópia reduzida do frame (480px de largura) para
ser rápida mesmo em vídeo fonte 1080p — só o *centro* do rosto importa, não
precisão de pixel. Para maior precisão ainda (ex.: ângulos muito extremos,
óculos escuros, iluminação difícil), você pode trocar por um detector
baseado em rede neural como **YuNet** ou **MediaPipe Face Detection** em
`src/reframer.py` — a interface da função `render_vertical_clip()` continua
igual; só é preciso baixar o modelo `.onnx`/`.tflite` correspondente numa
máquina com acesso à internet.

## Legendas e título-gancho

Estilo corte viral: grupos de 3 palavras em CAIXA ALTA, cada grupo entra com
um "pop", a palavra falada acende em amarelo e cresce, e números / dinheiro /
palavras de impacto ficam em verde (`CAPTION_*` em `config.py`). Nos
primeiros ~3s aparece o título do clipe num balão branco no topo, com um
*whoosh* sintetizado (`HOOK_*`, `SFX_*`).

A fonte usada nas legendas (**Anton**, Google Fonts, licença OFL livre) vem
embutida em `assets/fonts/` e é carregada diretamente pelo filtro `ass` do
ffmpeg — funciona em qualquer sistema operacional, sem precisar instalar
nada globalmente. Para trocar a fonte, troque `CAPTION_FONT` em
`config.py` e coloque o `.ttf`/`.otf` correspondente em `assets/fonts/`.

A legenda fica em ~70% da altura do vídeo (`CAPTION_MARGIN_V = 560`), acima
da faixa de baixo que o TikTok/Reels/Shorts cobrem com a descrição do post,
o nome da música e os botões — e com margem lateral pra não passar por baixo
dos botões de curtir/comentar da direita.

## Jump cuts (corte das pausas)

`src/jumpcut.py` tira as pausas de dentro do clipe (vão entre palavras maior
que `JUMPCUT_MIN_GAP_SECONDS`), **só se o áudio ali estiver em silêncio** —
risada e reação ficam. Os cortes caem na grade de quadros do vídeo (áudio e
imagem não dessincronizam) e o enquadramento alterna entre normal e 8% mais
fechado a cada corte (`JUMPCUT_PUNCH_ZOOM`). `JUMPCUT_ENABLED = False` desliga.

## Áudio

O volume final de cada clipe é normalizado para **-14 LUFS** (padrão usado
por TikTok, YouTube e Instagram) via `loudnorm`, então os clipes saem com
volume consistente entre si, independente de quão alto/baixo estava o
áudio original ou a trilha de música escolhida.

## Estrutura do projeto

```
opus-clip-clone/
├── main.py                 # CLI principal
├── test_pipeline.py         # teste de integração (sem precisar do YouTube)
├── requirements.txt
├── src/
│   ├── config.py             # todas as configurações
│   ├── downloader.py         # download via yt-dlp
│   ├── transcriber.py        # transcrição via faster-whisper
│   ├── audio.py               # análise de energia do áudio
│   ├── clip_selector.py      # seleção automática dos melhores cortes
│   ├── hwaccel.py             # detecção de GPU (AMF/VAAPI/NVENC) e fallback p/ CPU
│   ├── reframer.py           # recorte+reenquadre+legenda+áudio em 1 passe de encode
│   ├── captioner.py          # legendas estilo karaokê (.ass)
│   ├── effects.py            # detecção de picos p/ zoom punches
│   ├── music.py               # mixagem de música com ducking + normalização
│   ├── video_editor.py       # orquestra a montagem final de cada clipe
│   ├── post_kit.py            # título/descrição/hashtags/crédito (.post.txt)
│   ├── jumpcut.py             # corte das pausas (jump cuts)
│   ├── sfx.py                 # efeitos sonoros sintetizados (whoosh)
│   └── utils.py
├── assets/
│   ├── fonts/                 # fonte Anton (OFL) embutida p/ legendas
│   └── music/                 # trilhas (+ track_drops.txt / track_credits.txt)
```

## Testando sem baixar nada do YouTube

```bash
python test_pipeline.py
```

Isso gera automaticamente um vídeo sintético local (não precisa de
internet) e roda o pipeline completo (seleção de cortes, reenquadramento,
legendas, música, normalização de áudio) para validar que tudo está
instalado e funcionando antes de gastar tempo baixando vídeos reais. O
teste também confere, pixel a pixel, que o corte começa exatamente no
timestamp certo.

## Avisos legais

Baixar e reeditar vídeos do YouTube pode violar os Termos de Serviço da
plataforma e direitos autorais, dependendo do vídeo e do uso que você fizer
do resultado. Use esta ferramenta apenas com vídeos que você tem o direito
de reeditar/republicar (seus próprios vídeos, conteúdo licenciado, ou uso
que se enquadre em exceções legais aplicáveis na sua jurisdição).
