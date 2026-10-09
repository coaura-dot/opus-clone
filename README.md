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

Política de privacidade: [PRIVACY.md](PRIVACY.md) · Termos de uso: [TERMS.md](TERMS.md)

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

## Postagem automática no YouTube (piloto automático)

`autopilot.py` usa o programa acima como "motor" e faz o resto sozinho:
baixa, edita e **posta** no seu canal, com título, descrição, hashtags,
crédito do vídeo original e da música.

- **Duplo clique em `INICIAR_AUTOMATICO.bat`** -- modo automático infinito:
  procura os vídeos que estão ganhando views mais rápido nos canais da sua
  lista (`AUTOPILOT_CHANNELS` em `src/config.py`), corta, posta espaçado e
  repete até você fechar a janela ou desligar o PC. Mantém o PC acordado
  enquanto roda. Se desligar no meio, ao abrir de novo continua de onde parou.
  Se o programa cair por algum erro, o `.bat` religa sozinho em 60 s. Ctrl+C
  na janela para na hora (inclusive no meio de uma edição): **não aperte
  Ctrl+C pra copiar o log** sem ter selecionado o texto antes. O log inteiro
  também fica em `autopilot_data/autopilot.log`.
- **`AUTO_CLIPPER_MENU.bat`** -- menu: colar um link (ou arquivo do PC) e
  postar os cortes na hora, conectar o canal, ver a fila e o que já foi postado.

### Configuração (uma vez só, ~10 min)

1. `pip install -r requirements.txt` (instala as bibliotecas do Google).
2. No [Google Cloud Console](https://console.cloud.google.com/):
   1. crie um projeto (qualquer nome);
   2. em **APIs e serviços → Biblioteca**, ative a **YouTube Data API v3**;
   3. em **Tela de consentimento OAuth**: tipo **Externo**, preencha nome e
      e-mail, adicione o seu e-mail em **Usuários de teste** e depois clique
      em **Publicar app** (em "Teste" o login expira a cada 7 dias e o
      piloto para de postar);
   4. em **Credenciais → Criar credenciais → ID do cliente OAuth**, tipo
      **App para computador**; baixe o JSON e salve como
      `credentials/client_secret.json` (dentro da pasta do programa).
3. Rode `AUTO_CLIPPER_MENU.bat` → opção **3** (ou `python autopilot.py --login`).
   Abre o navegador: escolha o canal e autorize. Como o app é seu e não é
   verificado, o Google mostra um aviso -- clique em **Avançado → Acessar**.
   O login fica salvo em `credentials/youtube_token.json`.

**Importante -- regras do YouTube, não do programa:**

- **Vídeos privados até a auditoria:** vídeos enviados por um projeto da API
  ainda **não auditado** pelo Google ficam **travados como privados**. Para
  postar público, peça a auditoria (é grátis) pelo formulário
  ["YouTube API Services - Audit and Quota Extension"](https://support.google.com/youtube/contact/yt_api_form).
  Até sair, os vídeos sobem privados e você pode torná-los públicos no
  YouTube Studio.
- **Cota:** 10.000 unidades por dia, e cada upload gasta 1.600, então são no
  máximo **6 postagens por dia**. O programa respeita isso sozinho. A
  auditoria também serve para pedir mais cota.
- **Direitos:** use canais que **liberam cortes**. Clipe de canal que não
  libera pode render reivindicação ou strike no seu canal. A descrição de cada
  vídeo já leva o crédito do vídeo original e da música.

### Faxina automática: a pasta de cortes não passa de 1 GB

O piloto limpa `output\autopiloto` sozinho (`src/housekeeping.py`): ao
ligar, antes de cada vídeo novo, depois de cada edição e depois de cada
postagem.
- **Sai na hora:** pastas vazias (downloads que falharam), clipes
  reprovados, sobras de edição e a versão `_sem_musica.mp4`. O piloto posta
  a versão com música; `AUTOPILOT_KEEP_NO_MUSIC = True` guarda as duas.
- **Sai depois de 2 dias:** clipe já postado e público
  (`AUTOPILOT_KEEP_POSTED_DAYS`).
- **Acima de 1 GB** (`AUTOPILOT_OUTPUT_MAX_GB`): apaga primeiro os postados
  mais antigos. Clipe da fila, que ainda vai ser postado, nunca é apagado:
  com a pasta cheia, o piloto só para de produzir até postar e liberar
  espaço.
- **Pasta de trabalho** (`.autoclip_work`): o vídeo baixado, que pode ter
  vários GB, e os temporários saem depois de cada edição.

### Bloqueio do YouTube ("Sign in to confirm you're not a bot" / HTTP 429)

O YouTube bloqueia download anônimo de quem baixa muito. Quando isso
acontece, o piloto pausa os downloads, sem marcar os vídeos como falhos:
15 min, depois 30 min, 1 h, 2 h, até 6 h. A postagem continua normalmente.

A solução de verdade são os cookies de uma conta do Google logada:
1. **Use uma conta secundária, não a do canal.** O YouTube pode restringir a
   conta que baixa demais.
2. Logado nela, no Chrome ou Edge, instale a extensão "Get cookies.txt
   LOCALLY" e abra youtube.com.
3. Clique na extensão → **Export** e salve como
   `credentials\youtube_cookies.txt`, dentro da pasta do programa.
4. Pronto: o download e a busca passam a usar esses cookies. Se o bloqueio
   voltar semanas depois, os cookies venceram, e é só exportar de novo.

(Se você usa o Firefox, dá pra pôr `YTDLP_COOKIES_FROM_BROWSER = "firefox"`
no `config.py` em vez do arquivo.)

O piloto também atualiza o yt-dlp do `.venv` uma vez por dia
(`AUTOPILOT_UPDATE_YTDLP`): o YouTube muda direto, e versão velha falha com
"HTTP Error 403" ou "unable to extract yt initial data".

### Nota de qualidade: o que posta e o que não posta

Cada clipe pronto ganha uma nota de 0 a 100 (`src/quality.py`). A fila
posta sempre a maior nota primeiro, e clipe abaixo de `AUTOPILOT_MIN_QUALITY`
não é postado. O que pesa:

- **descarta**: recado do próprio canal ou patrocínio ("segue a gente no
  Spotify", "se inscreve", "cupom", "link na descrição") e vídeo escuro ou
  congelado;
- **tira nota**: abertura/encerramento do episódio, fala arrastada, clipe
  longo demais (mais de 2,5 min), pouco rosto em quadro;
- **dá nota**: gancho forte no começo (pergunta, "nunca", "ninguém",
  "dinheiro", "polícia"...), ritmo de fala bom, duração de 25-75 s, trecho
  que se destacou no episódio, rosto em quadro.

O log mostra a nota e o motivo de cada clipe; `python autopilot.py --status`
lista a fila com as notas e os últimos descartados.

### Nota de viralidade: no tempo livre, ele escolhe o que postar

**Banco de clipes.** Enquanto não pode postar, o piloto vai gerando clipes
de podcasts diferentes, até `AUTOPILOT_QUEUE_TARGET` (15). Canal que já
tem clipe na fila perde prioridade na busca. Na hora de postar, ele só
posta se a fila tiver pelo menos `AUTOPILOT_POOL_MIN` (6) clipes de
`AUTOPILOT_POOL_MIN_SOURCES` (3) podcasts diferentes, e escolhe o mais
viral entre eles. Se o banco não encher em
`AUTOPILOT_POOL_MAX_WAIT_HOURS` (3 h), por exemplo com o YouTube bloqueando
download, posta o melhor que tiver. Não posta o mesmo vídeo ou podcast em
sequência. Clipe parado na fila há mais de `AUTOPILOT_QUEUE_MAX_AGE_DAYS`
(4) dias sai, pra abrir espaço pra conteúdo novo.

**A nota (`src/virality.py`, ~10 s por clipe, 0 a 100):**
- **gancho:** o que é dito nos primeiros ~4 s (pergunta, afirmação forte,
  número, nome, falar com "você"), se a fala começa logo e se a voz tem
  energia;
- **contexto:** o começo se sustenta sozinho, sem "ele/isso..." de alguém
  que não foi apresentado e sem "Mas..." de uma ideia anterior;
- **conteúdo:** emoção, dinheiro, conflito, história, curiosidade e
  substância (explicação, dado);
- **fechamento:** termina numa frase completa, de preferência com
  conclusão, e não numa pergunta sem resposta;
- **picos de voz, ritmo, imagem** (rosto e movimento), **duração e título**.

Recado do canal, abertura ou **encerramento do episódio** ("quer deixar um
recado final?") fica travado em no máximo 25. A nota final é 60%
viralidade + 40% qualidade. Com o juiz de IA ligado, a nota da IA entra na
média e, antes de cada postagem, a IA compara os 5 melhores da fila entre
si (~US$0,03 por postagem).

**Conferida com as views reais (`src/feedback.py`).** Cada vídeo postado
guarda a nota que teve. Uma vez por dia o piloto busca as views dos vídeos
com 2 a 10 dias de vida (1 unidade de cota da API a cada 50 vídeos). Com 8
ou mais medidos, o log mostra o quanto a nota acerta a ordem das views:
```
[feedback] nota de viralidade x views reais (12 vídeos): correlação +0.41 -- acerta parte
```
Com 15 ou mais, os pesos se ajustam sozinhos: o item que anda junto com as
views ganha peso. O ajuste é pequeno no começo e cresce com a quantidade de
vídeos (até 70%), pra não seguir ruído. Pesos fixados em `VIRAL_WEIGHTS`
continuam mandando.

### Enquanto a auditoria não sai: vídeos privados que se liberam sozinhos

Pode deixar o piloto postando normalmente: os vídeos sobem privados e ficam
anotados. Todo dia (logo que o piloto liga e, depois, a partir das 10h --
`AUTOPILOT_RELEASE_CHECK_HOUR`) ele confere se a auditoria já saiu, tentando
deixar público o privado mais antigo. Saiu: solta os privados **de pouco em
pouco** (`AUTOPILOT_RELEASE_PER_DAY` = 3 por dia, um a cada
`AUTOPILOT_RELEASE_MINUTES_BETWEEN` = 120 min, no horário de postagem). Se um
vídeo antigo continuar travado mesmo depois da aprovação (o YouTube pode
manter travado o que subiu antes dela), ele **reposta o arquivo** que ficou
no PC, já público, e o privado antigo fica pra você apagar no YouTube Studio.
Para conferir na hora: menu, opção **5** (`python autopilot.py --liberar`).

Mudar a visibilidade precisa de uma permissão a mais: depois de atualizar o
programa, faça o login (opção 3) uma vez de novo.

### Ajustes (`src/config.py`, seção "POSTAGEM AUTOMÁTICA")

| Opção | Padrão | O que faz |
|---|---|---|
| `AUTOPILOT_CHANNELS` | Flow, Inteligência Ltda, Podpah, Ticaracaticast, Ciência Sem Fim; humor/TV/assunto do momento: The Noite, Ilha de Barbados, Diva Depressão, Felipe Neto; reacts: orochidois, Maicon Küster, Cortes do Casimito; gringos: DrDonut Clips, Theo Von, Lex Fridman | canais acompanhados (`"link\|en"` = canal em inglês). Programa de TV (The Noite) tem mais chance de reivindicação no Content ID |
| `AUTOPILOT_BLOCK_WORDS` | defante, rango brabo, aqueles caras... | título ou canal com essas palavras é ignorado |
| `AUTOPILOT_SEARCHES` | vazio | buscas extras ("esta semana, mais vistos", só títulos em português) |
| `AUTOPILOT_MIN_VIEWS` | 50000 | só vídeos com pelo menos N views |
| `AUTOPILOT_POPULARITY_WEIGHT` | 0.65 | na escolha, quanto pesa o tamanho do vídeo (views por dia) contra o "bombando pro canal dele"; mais alto = mais conteúdo de gente grande |
| `AUTOPILOT_FOREIGN_FACTOR` | 0.5 | peso dos canais gringos (`\|en`) na escolha, pra não tomarem a fila |
| `AUTOPILOT_DROP_QUEUED_FROM` | PrimoCast, PodPeople, Os Sócios, Market Makers, Roda Viva | clipes já prontos desses canais saem da fila ao ligar |
| `AUTOPILOT_MAX_AGE_DAYS` | 30 | só vídeos de até N dias; entre eles vence o que está indo **melhor que o normal do próprio canal** (views por dia), no máximo 2 por canal na disputa; 0 = qualquer idade |
| `AUTOPILOT_CLIPS_PER_VIDEO` | 3 | cortes por vídeo |
| `AUTOPILOT_PREFER_EPISODE_MINUTES` | 35 | vídeo mais curto que isso (provável corte, não o episódio inteiro) perde até 40% na escolha |
| `AUTOPILOT_EXPERT_BONUS` | 1.3 | título com convidado especialista (cientista, médico, economista, CEO, delegado...) ganha na escolha |
| `AUTOPILOT_WEAK_TOPIC_FACTOR` | 0.7 | título de desafio, zoeira ou pegadinha perde na escolha |
| `AUTOPILOT_QUEUE_TARGET` / `AUTOPILOT_POOL_MIN` / `AUTOPILOT_POOL_MIN_SOURCES` | 15 / 6 / 3 | banco de clipes: quantos gera e o mínimo (de quantos podcasts) pra escolher o mais viral |
| `AUTOPILOT_POSTS_PER_DAY` | 24 | meta; o limite real sai da cota (`YOUTUBE_DAILY_QUOTA`: 10.000 = 6/dia, 40.000 = 24/dia) |
| `AUTOPILOT_MIN_MINUTES_BETWEEN_POSTS` | 60 | intervalo mínimo; o real espalha o limite do dia pela janela (6/dia em 24h = 1 a cada 4h) |
| `AUTOPILOT_POST_HOURS` | (0, 24) | horário em que posta (fora dele só produz) |
| `AUTOPILOT_QUEUE_TARGET` | 30 | clipes prontos esperando na fila |
| `AUTOPILOT_MIN_QUALITY` | 45 | nota mínima (0-100) pra um clipe ser postado; ver "Nota de qualidade" abaixo |
| `YOUTUBE_PRIVACY` | "public" | "public", "unlisted" ou "private" |
| `AUTOPILOT_UPLOAD` | True | False = só gera os clipes (teste) |

Como escolhe o vídeo: entre os últimos 30 vídeos de cada canal, o de **mais
views** (novo ou antigo, com uma leve preferência pelos mais recentes), de 8
min a 4 h, que não seja live nem já usado, variando o canal quando um foi
usado nas últimas 24 h. A fila posta primeiro os cortes com
melhor pontuação.

Robustez: cada vídeo é editado num processo separado com tempo máximo
(`AUTOPILOT_WORKER_TIMEOUT_MINUTES`); se travar, é encerrado e o piloto
segue. Erro de rede no upload tenta de novo sozinho; cota esgotada espera
zerar (meia-noite do horário do Pacífico); disco quase cheio apaga primeiro
clipes já postados. Tudo fica registrado em `autopilot_data/autopilot.log`.

Para iniciar junto com o Windows: `Win+R` → `shell:startup` → cole ali um
atalho do `INICIAR_AUTOMATICO.bat`.

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

**Recomendado (legenda erra bem menos palavra): `large-v3-turbo`.** Baixe o
arquivo pra MESMA pasta do modelo que está em `WHISPERCPP_MODEL`; o programa
usa sozinho o melhor modelo que achar lá (`WHISPERCPP_AUTO_BEST_MODEL`), sem
mexer no config:

```powershell
$models = "C:\Users\igo\Desktop\opus2\opus-clip-clone\whisper.cpp\models"
Invoke-WebRequest "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin" -OutFile "$models\ggml-large-v3-turbo-q5_0.bin"
```

(~550 MB.) Fica uns 2-3x mais lento que o `small` na transcrição, o que
cabe folgado no ritmo do piloto automático. Junto, o programa usa busca em
feixe (`WHISPERCPP_BEAM_SIZE = 5`), o título do vídeo como dica de
vocabulário (nomes próprios) e o alinhamento DTW pro tempo de cada palavra
(`WHISPERCPP_DTW`). Se o seu `whisper-cli` for antigo e não conhecer alguma
dessas opções, ele tenta de novo sem elas.

**Download pela metade.** Se o arquivo do modelo ficou incompleto (o
download caiu), o programa não usa ele: transcreve com o próximo modelo bom
da pasta (no fim, o `WHISPERCPP_MODEL`), e o piloto automático baixa o
arquivo de novo em segundo plano, continuando de onde parou e conferindo o
tamanho e o SHA-256 antes de usar (`src/whisper_models.py`). Um modelo que
o whisper.cpp recusa ao carregar vira `<nome>.incompleto` (ou
`.incompativel`, se o arquivo está inteiro mas a sua versão do whisper.cpp
não lê) e não é tentado de novo. Se nem o modelo configurado carregar, o
piloto pausa a edição por 30 min sem descartar o vídeo escolhido.

**Sincronia da legenda.** Duas correções, valem pra qualquer modelo:
- vídeo com quadros por segundo variável (VFR, aparece como "29.6fps" no
  log): a imagem escorregava ~1 s por minuto em relação ao áudio e à legenda
  (medido: quase 3 s de atraso no fim de um clipe de 28 s). Agora os quadros
  são lidos numa taxa constante;
- o tempo de cada palavra é conferido com o áudio (`CAPTION_SNAP_TO_SPEECH`):
  atraso/adiantamento constante de um trecho é corrigido pelos começos e
  fins de frase (testado: tempos deslocados de -0,4 a +0,3 s voltam com erro
  mediano de 0,05 s), e palavra que começava no silêncio vai pro começo da fala.

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

**Suas músicas:** jogue os arquivos (`.mp3`, `.wav`, `.m4a`, `.ogg`,
`.flac`) direto em `assets/music/`. Não precisa anotar nada: o programa
acha sozinho o "drop" (a parte que explode) de cada uma, sorteia uma por
clipe e evita repetir as últimas. Para conferir o que ele está vendo:
`python diag_music.py`.

- Quer fixar o ponto de início? `assets/music/track_drops.txt` com
  `Título - M:SS`.
- Música que exige crédito? `assets/music/track_credits.txt` com
  `Título | crédito` (vai sozinho na descrição do vídeo).
- Sem nenhuma música na pasta, o clipe sai **só com a voz**.
- A biblioteca **NCS** que vinha com o programa (23 faixas de phonk/funk)
  fica em `assets/music/ncs/`, desligada. Para usar: `MUSIC_USE_NCS = True`.

**Volume acompanhando a voz.** O programa mede a loudness da fala e da
música a cada 0,1 s ao longo do clipe inteiro e ajusta o volume da música
trecho a trecho, pra ela ficar sempre `MUSIC_BELOW_VOICE_DB` (22 dB) abaixo
da fala:
- se o convidado fala mais baixo, a música desce junto;
- no drop, ela é segurada; na parte calma, sobe um pouco.

As mudanças são suaves (média de 1,5 s) e limitadas
(`MUSIC_LEVEL_MAX_CUT_DB` / `MUSIC_LEVEL_MAX_BOOST_DB`), pra não "bombear".
O *ducking* continua por cima, abaixando a música enquanto alguém fala.

Medido em 8 faixas com um trecho real de podcast que tinha 20 s de fala 8
dB mais baixa: com o volume único de antes, a distância entre voz e música
variava de 3 a 9 dB ao longo do clipe, e no trecho baixo a música ficava
só ~17 dB abaixo da fala. Agora a distância fica em 22 dB o tempo todo,
variando em geral menos de 2 dB (4 dB numa faixa com pausa em silêncio).
Para voltar ao volume único: `MUSIC_LEVEL_FOLLOW_VOICE = False`.

> ⚠️ Música comercial (os hits do TikTok/Instagram/rádio) é reconhecida pelo
> Content ID do YouTube: o clipe é reivindicado (a receita vai pra
> gravadora), fica sem som ou é bloqueado, conforme a gravadora. Para trend
> com som famoso, poste o `*_sem_musica.mp4` (gerado quando o clipe tem
> música) e escolha o som em alta pela biblioteca do próprio app, que já
> tem a licença.

## Momentos engraçados

Além de conteúdo "inteligente", o programa reconhece momento engraçado
(`src/humor.py`). O Whisper não escreve risada, então ela aparece como um
buraco na transcrição: som alto, perto do volume da fala, sem palavra
nenhuma, logo depois de alguém falar e no meio da conversa. Pra não
confundir com música, o trecho não pode ter grave: medido em 15 risadas
reais e 23 faixas de música, a energia abaixo de 150 Hz é ~0,1% na
risada e ~50% na música.

- **Seleção:** cada risada ou reação por minuto soma pontos
  (`SELECT_HUMOR_WEIGHT`). Se a risada vem logo depois da última frase, o
  clipe termina nela (é o punchline).
- **Nota de qualidade:** risada não vira palavra, então momento engraçado
  não leva mais a penalidade de "fala arrastada".
- **Nota de viralidade:** entra o item "risadas", e clipe engraçado conta
  como conteúdo, mesmo sem explicar nada.
- **Juiz de IA:** passou a valorizar momento que faz rir sozinho.

**Teste:** risadas reais (ESC-50) coladas no meio de 10 min de fala real
de podcast.
- Achou 20 de 20.
- Confundiu 1 de 5 trechos de música.
- Marcou 3 pontos da fala original. Conferidos de novo com outro modelo
  do Whisper, uma parte era reação e outra era fala que o Whisper não
  tinha transcrito.

Por isso a risada é um bônus na nota, não o que decide sozinho.

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

## Juiz de cortes com IA (opcional, recomendado)

As regras de seleção acham bons *candidatos* (gancho, assunto que começa e
termina, energia da voz), mas não entendem se o assunto é interessante ou se
a fala é inteligente. Com uma chave da API da Anthropic, os até 12 melhores
trechos de cada vídeo vão pro Claude (`src/ai_judge.py`). Ele dá uma nota de
0 a 100 pelo conteúdo e escreve um título melhor pra cada trecho. O que
entra na nota:

- **sobe:** insight, dado ou história que se sustenta sozinho; curiosidade
  nos primeiros segundos; um desfecho.
- **desce:** zoeira sem conteúdo, piada interna, recado do canal, trecho que
  depende do resto do episódio.

Os clipes são escolhidos pela nota da IA. Trecho abaixo de 35
(`AI_REJECT_BELOW`) nem é editado, e vídeo sem nenhum trecho bom é pulado
sem nova tentativa. No vídeo longo, um bloco só vira clipe na hora com nota
≥ 60 (`AI_CHUNK_MIN_SCORE`); senão o programa segue procurando nos outros
blocos.

Como ligar:
1. Crie a chave em https://console.anthropic.com (Settings → API keys) e
   coloque créditos na conta.
2. Salve a chave no arquivo `credentials\claude_api_key.txt`, só a chave,
   sem mais nada. A pasta `credentials` não é tocada nas atualizações. Também
   dá pra usar `CLAUDE_API_KEY` no `config.py` ou a variável de ambiente
   `ANTHROPIC_API_KEY`.
3. `.venv\Scripts\python.exe -m pip install anthropic`.

Custo: ~US$0,05-0,10 por vídeo com o Claude Opus 5.5 (`AI_JUDGE_MODEL`). O
log mostra o custo de cada chamada. Sem chave, tudo funciona do mesmo jeito,
só com as regras.

## Como funciona a seleção "viral" dos cortes

O `src/clip_selector.py` não usa nenhuma IA de terceiros — é um sistema de
pontuação transparente que você pode ajustar:

- **Texto**: detecta ganchos comuns ("você sabia", "o maior erro", números,
  perguntas, palavras de forte carga emocional) em português e inglês
- **Fora**: o trailer de "melhores momentos" dos primeiros 90 s do episódio
  (`SELECT_SKIP_INTRO_SECONDS`) e o encerramento da conversa ("recado
  final", "antes de te liberar", "onde a galera te encontra")
- **Título** (sem IA): a frase do começo do clipe que melhor funciona
  sozinha, com pergunta, número, nome ou assunto concreto. Sai o vocativo
  ("Walter, quer...") e a hesitação, e perde a frase que depende de
  contexto ("ele...", "isso..."). Com o juiz de IA ligado, o título é o dele
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

**Plano por cena** (`src/shot_plan.py`): antes de renderizar, o programa
analisa o trecho do clipe, acha os cortes do vídeo original e decide **um
layout por plano**, do começo ao fim dele. Feito pra vídeos com muito corte,
zoom e câmera na mão (vlog, documentário, cortes já editados):

- **um rosto**: o recorte tem tamanho fixo no plano inteiro, então o zoom que
  a câmera original faz aparece como está, sem "zoom duplo". A câmera do
  programa fica parada se a pessoa mexe pouco. Se a pessoa anda pelo quadro,
  ela segue um caminho suave, sem atraso;
- **conversa** (2+ pessoas que não cabem num recorte): rastreia quem está
  falando, como antes;
- **sem rosto** (b-roll: paisagem, carro, aeroporto): tela cheia, parado, no
  ponto com mais detalhe. Antes saía como uma faixa fina sobre fundo borrado;
- **cartela de texto/título**: a cartela inteira, ampliada até a largura do
  texto, sem cortar nada;
- **close gigante** (rosto mais largo que o recorte vertical, tipo o zoom de
  edição do Podpah): o rosto inteiro, ampliado até onde cabe, sobre fundo
  borrado. Antes, o recorte mostrava só nariz e boca;
- **barras pretas de cinema** (e a legenda do vídeo original dentro delas)
  ficam de fora do recorte;
- **flash e chicote** (planos de menos de 0,4 s) não contam como corte.

Ajustes em `SHOT_*` e `LETTERBOX_*` (`src/config.py`). `SHOT_PLAN_ENABLED =
False` volta pro rastreamento quadro a quadro.

**Vídeo de react** (streamer com facecam sobreposta ao vídeo que está
reagindo): o programa detecta a facecam uma vez por vídeo — um rosto que
fica sempre no mesmo lugar, numa caixa que não muda enquanto o resto da
imagem muda — e os clipes saem em **tela dividida**: o vídeo reagido em cima
(uma janela fixa por clipe, cobrindo quem está falando nele e sem pegar a
facecam) e o streamer embaixo, com a legenda na divisória e o título logo
acima dela. Nos trechos em que a facecam some (troca de cena), volta pro
enquadramento normal — sempre em trechos de pelo menos 3s, sem pisca-pisca.
Ajustes em `REACT_*` (`src/config.py`); `REACT_MODE_AUTO_DETECT = False`
desliga.

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
palavras de impacto ficam em verde (`CAPTION_*` em `config.py`). O clipe
abre com um zoom out suave (~12%, em 1,5 s; `FX_INTRO_*`). O título num
balão branco no topo nos primeiros ~3 s, com *whoosh*, vem desligado
(`HOOK_ENABLED = True` liga de novo). Clipes feitos antes de desligar o
título (balão e whoosh ficam gravados no arquivo) saem da fila sozinhos
quando o piloto liga, e os vídeos privados antigos com o balão não são
liberados (`src/hook_check.py`).

A fonte usada nas legendas (**Anton**, Google Fonts, licença OFL livre) vem
embutida em `assets/fonts/` e é carregada diretamente pelo filtro `ass` do
ffmpeg — funciona em qualquer sistema operacional, sem precisar instalar
nada globalmente. Para trocar a fonte, troque `CAPTION_FONT` em
`config.py` e coloque o `.ttf`/`.otf` correspondente em `assets/fonts/`.

A legenda fica em ~70% da altura do vídeo (`CAPTION_MARGIN_V = 560`), acima
da faixa de baixo que o TikTok/Reels/Shorts cobrem com a descrição do post,
o nome da música e os botões — e com margem lateral pra não passar por baixo
dos botões de curtir/comentar da direita.

## Efeitos de edição (zoom, impacto, emoji)

`src/fx.py` planeja os efeitos **pela fala** de cada clipe — nada é aleatório:

- **Zoom nos momentos-chave**: palavra com peso (número/dinheiro, palavra de
  impacto, palavra com emoji, "!" ou pergunta forte, voz mais alta que o
  normal) ganha um zoom rápido com *overshoot*, segura durante a frase e
  volta suave (zoom out). Espaçados (`FX_ZOOM_MIN_GAP_SECONDS`).
- **Abertura**: o clipe começa com um zoom out de impacto e um flicker curto.
- **Impacto** nos momentos mais fortes (no máx. 1 a cada 8s): flash de luz,
  tremida de câmera, separação RGB e um "boom" grave.
- **Emojis**: quando alguém fala uma palavra-chave (dinheiro 💰, polícia 🚔,
  Brasil 🇧🇷, morreu 💀, loucura 🤯...), o emoji entra com "pop" acima da
  legenda, em estilo adesivo (contorno branco). Ignora negação ("não tem
  vitória" não ganha 🏆) e expressões ("todo mundo" não é 🌎); mapa de
  palavras em `src/emoji_map.py`. Emojis: Twemoji (CC BY 4.0, `assets/emoji/`).
- **Vinheta** mais marcada e grade de cor.

Tudo ligável/desligável e ajustável nos `FX_*` do `config.py`.

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
├── autopilot.py            # piloto automático: baixa, edita e posta no YouTube
├── INICIAR_AUTOMATICO.bat  # 1 clique: modo automático infinito
├── AUTO_CLIPPER_MENU.bat   # menu (um link, login do canal, fila)
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
│   ├── sfx.py                 # efeitos sonoros sintetizados (whoosh, boom)
│   ├── fx.py                  # zoom em momento-chave, impacto, emoji, abertura
│   ├── emoji_map.py           # palavra falada -> emoji
│   ├── opener.py              # a frase de abertura se sustenta sozinha?
│   ├── react_layout.py        # react: acha a facecam e monta a tela dividida
│   ├── react_detector.py      # react: "olha isso" na fala
│   ├── aspect_fix.py          # corrige vídeo salvo esticado/amassado
│   ├── shot_plan.py           # plano por cena: cortes, barras pretas e layout de cada plano
│   ├── quality.py             # nota de qualidade do clipe: o que o piloto posta e o que descarta
│   ├── virality.py            # nota de viralidade (tempo ocioso): o que postar primeiro
│   ├── ai_judge.py            # juiz de cortes com IA (Claude, opcional): nota + título por trecho
│   ├── housekeeping.py        # faxina: pasta de cortes até 1 GB, sem pastas vazias nem sobras
│   ├── autopilot.py           # loop do piloto automático (fila, agenda, cota)
│   ├── discovery.py           # acha o vídeo que mais está bombando
│   ├── youtube_uploader.py    # upload pela API oficial do YouTube (OAuth)
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
