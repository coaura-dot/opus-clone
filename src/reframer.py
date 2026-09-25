"""
Reenquadramento + legendas + áudio em UM ÚNICO passe de encode.

Esta é a etapa mais pesada do pipeline (decodificação + processamento de
frame + codificação de vídeo), então é onde a arquitetura importa mais para
performance. Em vez de gerar um arquivo intermediário por etapa (cortar ->
reenquadrar -> queimar legenda, três encodes completos), este módulo faz
tudo em uma só passagem:

  1. uma thread "leitora" consome o stdout de um processo ffmpeg que
     decodifica (sem recodificar) só o trecho necessário do vídeo original
     (seek preciso via -ss/-t do próprio ffmpeg) e empilha os frames crus
     numa fila — assim o ffmpeg pode decodificar o frame seguinte enquanto
     a thread principal ainda está processando o frame atual, em vez de
     alternar leitura/processamento/escrita em sequência estrita;
  2. a thread principal recorta cada frame seguindo o rosto de quem está
     FALANDO (não simplesmente o maior rosto ou o mais "de frente" pra
     câmera): todos os rostos visíveis são rastreados, e um placar de
     atividade (movimento na região da boca/expressão facial) decide qual
     deles a câmera virtual segue, com uma margem + tempo mínimo antes de
     trocar de rosto — evitando ficar pulando entre os participantes só
     porque a cabeça de quem está calado virou mais pra câmera num
     instante. Quando nenhum rosto é encontrado por tempo suficiente
     (plano aberto, corte de câmera, ninguém de frente), troca para um
     modo "plano aberto" que mostra o quadro inteiro (sem cortar ninguém
     de fora) sobre um fundo desfocado, em vez de cravar o crop apertado
     num ponto qualquer da imagem;
  3. os frames já recortados são enviados via pipe para um segundo processo
     ffmpeg "escritor", que aplica correção de cor + vinheta, já embute a
     legenda (.ass) e mixa o áudio final — usando o encoder de GPU
     detectado automaticamente (VAAPI/NVENC/AMF) quando disponível, com
     fallback silencioso para CPU.

Resultado: 1 encode de vídeo por clipe em vez de 3, sem arquivos
intermediários grandes em disco.
"""
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import config
from . import hwaccel

# tamanho (largura, em px) da cópia reduzida do frame usada só para a
# detecção de rosto — detectar em alta resolução é desperdício, já que só
# precisamos do *centro* do rosto, não de precisão de pixel. Reduz o custo
# do classificador Haar em ~18x (proporcional ao quadrado do fator de escala)
# sem perda perceptível de qualidade de rastreamento.
FACE_DETECT_WIDTH = 480

# quantos frames crus (já decodificados) ficam bufferizados entre a thread
# leitora e a thread principal — pequeno o bastante para não gastar muita
# RAM (cada frame de 1920x1080 BGR24 pesa ~6MB), grande o bastante para
# absorver variações de velocidade entre decode e processamento/encode.
_READ_AHEAD_FRAMES = 8


def _new_cascades() -> dict:
    # instanciado por chamada (não global) para ser seguro em processamento
    # paralelo de vários clipes ao mesmo tempo (threads diferentes).
    #
    # Usamos DOIS classificadores Haar, ambos já embutidos no OpenCV (não
    # precisa baixar nada): "frontalface" (rosto de frente) e "profileface"
    # (rosto de perfil). Isso importa bastante para vídeos de podcast/
    # entrevista com duas pessoas conversando entre si de lado — só o
    # detector frontal costuma perder o rosto sempre que a pessoa vira a
    # cabeça pra falar com quem está ao lado, e a câmera virtual perde o
    # rastreamento justamente nos momentos de diálogo mais ativo.
    frontal_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    return {
        "frontal": cv2.CascadeClassifier(frontal_path),
        "profile": cv2.CascadeClassifier(profile_path),
        "yunet": _new_yunet(),
    }


def _new_yunet():
    """Detector de rosto neural YuNet (OpenCV Zoo, licença MIT; modelo de
    ~230KB em assets/models/, roda em CPU em ~10ms por detecção a 640px).
    Achado real em podcast de estúdio: o Haar via "rostos" na parede de
    espuma, na camisa e no microfone num close de perfil com óculos escuros
    -- 4 detecções pra 1 pessoa, o que fazia o programa achar que era plano
    de grupo. O YuNet dá 1 rosto com ~90% de confiança no mesmo quadro.
    Sem o modelo (ou OpenCV antigo), volta pro Haar."""
    if getattr(config, "FACE_DETECTOR", "yunet") != "yunet" or not hasattr(cv2, "FaceDetectorYN"):
        return None
    model = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"
    if not model.exists():
        return None
    try:
        return cv2.FaceDetectorYN_create(str(model), "", (320, 320),
                                         getattr(config, "YUNET_SCORE_THRESHOLD", 0.6), 0.3, 5000)
    except cv2.error:
        return None


def _detect_yunet(detector, frame_bgr, det_width: int, min_score: float) -> list:
    h, w = frame_bgr.shape[:2]
    scale = det_width / w if w > det_width else 1.0
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return []
    # Centro horizontal: nos OLHOS (não no centro da caixa -- num perfil a
    # caixa pega cabelo/nuca e fica ~100px atrás do rosto, que acabava
    # colado na borda do quadro) + "espaço de olhar": desloca pro lado em
    # que o nariz aponta, deixando o rosto no terço oposto, como um
    # cinegrafista enquadra (FACE_LOOK_ROOM x largura do rosto).
    look_room = getattr(config, "FACE_LOOK_ROOM", 0.3)
    # o recorte padrão tem a altura cheia da fonte em 9:16; o deslocamento
    # de "espaço de olhar" nunca pode tirar a cabeça do quadro. Achado real:
    # num close de perfil bem de perto (rosto ~metade da largura do recorte)
    # o deslocamento empurrava a nuca/cabelo pra fora da borda.
    crop_w = small.shape[0] * 9.0 / 16.0
    margin = 0.08 * crop_w
    out = []
    for f in faces:
        if float(f[-1]) < min_score:
            continue
        x, y, fw, fh = f[0], f[1], f[2], f[3]
        eyes_x = (f[4] + f[6]) / 2.0
        yaw = float(np.clip((f[8] - eyes_x) / max(fw * 0.25, 1e-3), -1.0, 1.0))
        # a caixa do YuNet cobre só a FRENTE do rosto; num perfil a nuca e o
        # cabelo ficam atrás dela (medido num close: caixa em x=566-744, cabeça
        # começando em x~340). Estima a cabeça inteira estendendo a caixa pro
        # lado oposto ao olhar, e só dá "espaço de olhar" se ela couber.
        back = abs(yaw) * 0.9 * fw
        head_l, head_r = (x - back, x + fw) if yaw > 0 else (x, x + fw + back)
        lo = head_r + margin - crop_w / 2.0
        hi = head_l - margin + crop_w / 2.0
        desired = eyes_x + yaw * look_room * fw
        cx = float(np.clip(desired, lo, hi)) if lo <= hi else (head_l + head_r) / 2.0
        out.append((cx / scale, (y + fh / 2.0) / scale, fw / scale, fh / scale))
    return out


def _detect_profile_faces(cascade, gray) -> list:
    """O classificador de perfil do OpenCV só reconhece rostos virados para
    UM lado (a base de treino não é simétrica). Para cobrir os dois lados
    sem precisar de um segundo modelo, rodamos o mesmo classificador também
    na imagem espelhada horizontalmente e "desespelhamos" as coordenadas
    encontradas de volta.

    minNeighbors reduzido de 5→3 e scaleFactor de 1.15→1.10 para capturar
    mais rostos de perfil — o Haar de perfil é inerentemente mais ruidoso
    que o frontal, mas a deduplicação em _dedupe_faces elimina falsos
    positivos duplicados antes de chegarem no rastreador."""
    faces = list(cascade.detectMultiScale(
        gray, scaleFactor=1.10, minNeighbors=4, minSize=(50, 50),
    ))
    flipped = cv2.flip(gray, 1)
    w = gray.shape[1]
    for (x, y, fw, fh) in cascade.detectMultiScale(
            flipped, scaleFactor=1.10, minNeighbors=4, minSize=(50, 50)):
        faces.append((w - x - fw, y, fw, fh))
    return faces


def _dedupe_faces(boxes: list, iou_thresh: float = 0.3) -> list:
    """Mescla detecções sobrepostas (ex.: o mesmo rosto pego pelo
    classificador frontal E pelo de perfil) num único box, mantendo o
    maior. Sem isso, um único rosto físico poderia virar dois "slots"
    diferentes no rastreador."""
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: list = []
    for b in boxes:
        bx, by, bw, bh = b
        dup = False
        for k in kept:
            kx, ky, kw, kh = k
            ix0, iy0 = max(bx, kx), max(by, ky)
            ix1, iy1 = min(bx + bw, kx + kw), min(by + bh, ky + kh)
            iw, ih = max(ix1 - ix0, 0), max(iy1 - iy0, 0)
            inter = iw * ih
            union = bw * bh + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(b)
    return kept


def _detect_all_faces(cascades: dict, frame_bgr, sensitive: bool = False,
                       _max_y_frac: float = None) -> list:
    """Detecta TODOS os rostos no frame (não só o "melhor"), combinando o
    classificador frontal com o de perfil (nos dois sentidos). Rodar os
    dois SEMPRE — em vez de só tentar perfil quando o frontal não acha
    nada, como a versão anterior fazia — é essencial para pegar os dois
    participantes de uma conversa ao mesmo tempo quando um está de frente
    para a câmera e o outro está virado de lado, olhando para quem fala
    (a situação mais comum num podcast de duas pessoas). Sem isso, o
    programa só "via" quem estivesse mais de frente a cada instante, e
    ficava trocando de rosto conforme as cabeças giravam — não porque
    alguém tivesse começado a falar, mas porque o ângulo da cabeça mudou.

    Retorna uma lista de (cx, cy, largura, altura) em coordenadas do frame
    ORIGINAL.

    CORREÇÃO (achado num clipe real — plano aberto/2-shot de um podcast):
    quando a câmera de origem corta pra um ângulo mais aberto (os dois
    participantes visíveis, mas cada rosto ocupando bem menos da tela),
    os rostos ficam pequenos demais pra sobreviver ao downscale de
    detecção padrão (FACE_DETECT_WIDTH=480px) — e isso NÃO é um problema
    de `minSize` (testei de 60 até 15px, nenhum encontrou nada): na
    escala de 480px o rosto já perdeu detalhe de pixel suficiente pro
    classificador Haar reconhecer, não importa o quão permissivo o
    tamanho mínimo seja. Confirmado com um frame real de plano aberto:
    0 rostos em QUALQUER minSize a 480px, mas 2 rostos encontrados de
    cara ao redetectar a 960px. Resultado prático do bug: perder os
    rostos nesse momento faz a confiança do rastreamento cair e o
    programa trocar pro modo "plano aberto" (ninguém em destaque, só o
    quadro inteiro pequeno) — exatamente o oposto do que se quer (seguir
    quem está falando mesmo depois de um corte de câmera).
    Por isso, SÓ QUANDO o passe rápido (480px) não encontra nada, um
    segundo passe é tentado numa escala maior (menos downscale) antes de
    desistir — o custo extra só é pago nos frames em que o passe rápido
    já falhou (não em todo frame), então o caso comum (rosto grande,
    plano fechado) continua com o custo baixo de sempre."""
    h, w = frame_bgr.shape[:2]

    if cascades.get("yunet") is not None:
        det = cascades["yunet"]
        min_score = getattr(config, "YUNET_SCORE_THRESHOLD", 0.6) - (0.1 if sensitive else 0.0)
        faces = _detect_yunet(det, frame_bgr, getattr(config, "YUNET_DETECT_WIDTH", 640), min_score)
        if not faces:
            # rostos pequenos de plano aberto: segunda passada com mais resolução
            faces = _detect_yunet(det, frame_bgr, getattr(config, "FACE_DETECT_WIDTH_FALLBACK", 960),
                                  min_score)
        eff_y_frac = _max_y_frac if _max_y_frac is not None else float(
            getattr(config, "FACE_MAX_Y_FRAC", 0.80))
        return [f for f in faces if f[1] < h * eff_y_frac]

    # modo sensitivo (burst pós-corte): scaleFactor menor + minNeighbors menor
    # + minSize menor → encontra rostos parciais, em ângulo, ou na borda do frame
    sf  = 1.05 if sensitive else 1.15
    mn  = 3    if sensitive else 5
    msz = 30   if sensitive else 60
    msz_fb = 20 if sensitive else 40

    def _run(det_width: int, min_size: int):
        scale = det_width / w if w > det_width else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        raw = list(cascades["frontal"].detectMultiScale(
            gray, scaleFactor=sf, minNeighbors=mn, minSize=(min_size, min_size),
        ))
        raw += _detect_profile_faces(cascades["profile"], gray)
        faces = _dedupe_faces(raw)
        return [
            (x / scale + fw / scale / 2.0, y / scale + fh / scale / 2.0,
             fw / scale, fh / scale)
            for (x, y, fw, fh) in faces
        ]

    faces = _run(FACE_DETECT_WIDTH, msz)
    if not faces:
        fallback_width = getattr(config, "FACE_DETECT_WIDTH_FALLBACK", 960)
        faces = _run(fallback_width, msz_fb)

    h = frame_bgr.shape[0]
    # Limiar vertical: rejeita detecções abaixo desta fração da altura do frame
    # (brinquedos, logos e objetos de mesa geram falsos positivos comuns).
    # Quando chamada com _max_y_frac explícito (ex.: busca escalona da no 1º frame
    # pós-corte), usa esse valor; senão, lê o default do config.
    eff_y_frac = _max_y_frac if _max_y_frac is not None else float(
        getattr(config, "FACE_MAX_Y_FRAC", 0.80))
    faces = [f for f in faces if f[1] < h * eff_y_frac]

    # Aspect-ratio: rejeita boxes muito largos (placa/banner) ou muito estreitos.
    faces = [f for f in faces if f[3] > 0 and 0.30 <= f[2] / f[3] <= 3.5]

    return faces


def _detect_faces_on_cut(cascades: dict, frame_bgr) -> list:
    """Busca escalona da usada exclusivamente no PRIMEIRO frame após um corte
    de câmera — o momento em que o custo de um falso positivo é mais alto
    (snap imediato para a posição errada).

    Estratégia:
    1. Busca sensitive em todo o frame, mas só aceita rostos na metade
       superior (y < CUT_FACE_MAX_Y_FRAC, default 0.55): elimina brinquedos,
       logos e objetos de mesa sem precisar de um limiar geral agressivo.
    2. Se não achou: busca na metade ESQUERDA do frame com resolução dobrada
       (o frame já tem metade da largura → o downscale para FACE_DETECT_WIDTH
       é menor → rostos pequenos ficam maiores relativamente).
    3. Se não achou: idem para a metade DIREITA.
    4. Se ainda não achou: retorna [] — o chamador cai no modo plano aberto.

    ig: "on every detection of a cut… it should ask itself 'there is a face
    in this image' — he asks it in the first frame after the cut. Then, if
    not, he tries to find it in this 1st frame." — é exatamente isso."""
    h, w = frame_bgr.shape[:2]
    y_frac = float(getattr(config, "CUT_FACE_MAX_Y_FRAC", 0.55))

    # Passo 1 — frame inteiro, apenas metade superior.
    faces = _detect_all_faces(cascades, frame_bgr, sensitive=True,
                               _max_y_frac=y_frac)
    if faces:
        return faces

    # Passos 2 e 3 — metades do frame, limiar vertical relaxado para 0.70
    # (dentro da metade já vale uma faixa maior porque eliminamos a parte
    # inferior do frame simplesmente não olhando para lá).
    half_y = float(getattr(config, "CUT_HALF_FACE_MAX_Y_FRAC", 0.70))
    for x_start, x_end in [(0, w // 2), (w // 2, w)]:
        half = frame_bgr[:, x_start:x_end]
        half_faces = _detect_all_faces(cascades, half, sensitive=True,
                                        _max_y_frac=half_y)
        if half_faces:
            # Converte coordenadas de x de volta ao frame completo.
            return [(cx + x_start, cy, fw, fh) for (cx, cy, fw, fh) in half_faces]

    return []


def _detect_screen_region(frame_bgr, min_area_frac: float = 0.08,
                           max_area_frac: float = 0.65):
    """Detecta um retângulo candidato a "tela dentro da tela" (celular,
    monitor, notebook mostrando um vídeo que o youtuber está assistindo/
    reagindo) via bordas (Canny) + contornos poligonais — ver
    RELATORIO_PROXIMOS_PASSOS.txt, item 3a.

    Heurística: um contorno de 4 vértices, convexo, com área dentro de uma
    faixa plausível (nem um ícone pequeno, nem o frame inteiro) e proporção
    largura/altura próxima de formatos comuns de tela (16:9, 4:3, ou 1:1
    para celular). NÃO confirma que há conteúdo REALMENTE tocando ali dentro
    (só a geometria) — isso é responsabilidade de `_ScreenTracker`, que
    exige a mesma região aparecer de forma estável E com atividade de pixel
    sustentada internamente antes de considerar "tela ativa".

    Retorna o maior candidato plausível como (x, y, w, h) em pixels do frame
    original, ou None se nada plausível for encontrado.
    """
    h, w = frame_bgr.shape[:2]
    frame_area = w * h
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * frame_area or area > max_area_frac * frame_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, rw, rh = cv2.boundingRect(approx)
        aspect = rw / max(rh, 1)
        plausible = any(abs(aspect - r) < 0.35 or abs(1.0 / aspect - r) < 0.35
                         for r in (16 / 9, 4 / 3, 1.0))
        if not plausible:
            continue
        if area > best_area:
            best_area, best = area, (x, y, rw, rh)
    return best


class _ScreenTracker:
    """Confirma que um retângulo candidato (de `_detect_screen_region`) é
    de fato uma tela TOCANDO algo, não só um objeto retangular parado
    (janela, quadro na parede, moldura de monitor desligado) — exige duas
    coisas ao longo de várias checagens: (1) posição/tamanho estáveis
    (mesma região aparecendo de novo, não ruído de detecção pulando pelo
    quadro) e (2) atividade de pixel sustentada DENTRO do retângulo (um
    vídeo rodando muda de pixel constantemente; uma foto/quadro parado,
    mesmo que geometricamente pareça uma tela, não muda quase nada de um
    frame pro outro).

    NÃO VALIDADO CONTRA VÍDEO REAL (ver RELATORIO_PROXIMOS_PASSOS.txt, item
    3a — funcionalidade nova, não uma correção de algo existente). Por isso
    fica desligada por padrão (VIDEO_IN_VIDEO_ENABLED=False em config.py);
    ligue e teste com um clipe real de reação/vídeo-dentro-do-vídeo antes de
    usar em produção.
    """

    def __init__(self, fps: float, detect_interval: int):
        self.fps = max(fps, 1e-3)
        check_interval = max(detect_interval / self.fps, 1e-3)
        self.check_interval = check_interval
        stable_s = max(getattr(config, "VIDEO_IN_VIDEO_STABLE_SECONDS", 1.0), check_interval)
        self.stable_checks_needed = max(int(round(stable_s / check_interval)), 1)
        timeout_s = max(getattr(config, "VIDEO_IN_VIDEO_TIMEOUT_SECONDS", 1.0), check_interval)
        self.timeout_checks = max(int(round(timeout_s / check_interval)), 1)

        self.region = None          # (x, y, w, h) confirmado (estável, em uso)
        self.candidate = None       # região vista na última checagem (ainda não confirmada)
        self.stable_streak = 0
        self.missed = 0
        self.activity = 0.0
        self.prev_gray_patch = None
        self.activity_decay = getattr(config, "FACE_ACTIVITY_DECAY", 0.85)
        self.activity_min = float(getattr(config, "VIDEO_IN_VIDEO_ACTIVITY_MIN", 3.0))

    @staticmethod
    def _similar(a, b, tol: float = 0.15) -> bool:
        if a is None or b is None:
            return False
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return (abs(ax - bx) < tol * max(aw, 1) and abs(ay - by) < tol * max(ah, 1)
                and abs(aw - bw) < tol * max(aw, 1) and abs(ah - bh) < tol * max(ah, 1))

    def update_detection(self, region):
        """Chamado a cada checagem (mesmo intervalo da detecção de rosto)."""
        if region is not None and self._similar(region, self.candidate):
            self.stable_streak += 1
        elif region is not None:
            self.candidate, self.stable_streak = region, 1
        else:
            self.missed += 1
            if self.missed > self.timeout_checks:
                self.region, self.candidate, self.stable_streak = None, None, 0
            return

        if region is not None:
            self.missed = 0
            self.candidate = region
            if self.stable_streak >= self.stable_checks_needed:
                self.region = region

    def sample_activity(self, frame_bgr):
        """Roda todo frame (barato: só quando há uma região candidata/
        confirmada) para medir se o conteúdo dentro do retângulo está de
        fato mudando (vídeo tocando) ou parado (imagem estática)."""
        region = self.region or self.candidate
        if region is None:
            self.prev_gray_patch = None
            return
        x, y, w, h = region
        fh, fw = frame_bgr.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, fw), min(y + h, fh)
        if x1 <= x0 or y1 <= y0:
            self.prev_gray_patch = None
            return
        gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        patch = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.prev_gray_patch is not None:
            raw = float(np.mean(np.abs(patch - self.prev_gray_patch)))
            self.activity = self.activity_decay * self.activity + (1 - self.activity_decay) * raw
        self.prev_gray_patch = patch

    def active_region(self):
        """Retorna (x, y, w, h) só quando a região está CONFIRMADA (estável
        por tempo suficiente) E com atividade de pixel acima do piso (algo
        está de fato tocando ali) — senão None (não deve substituir o
        rastreamento de rosto)."""
        if self.region is None:
            return None
        if self.activity < self.activity_min:
            return None
        return self.region

    def last_known_region(self):
        """Região confirmada mais recentemente, IGNORANDO o piso de
        atividade (ver RELATORIO_PROXIMOS_PASSOS.txt, item 14 — modo
        REACT). Usada só pelo zoom de referência visual: mesmo com o
        vídeo reagido pausado/parado no momento, ainda faz sentido
        mostrá-lo quando o comentador aponta pra algo nele ("olha a cor
        da camisa dele"). None se nenhuma região nunca foi confirmada
        neste clipe (vídeo sem tela nenhuma detectável)."""
        return self.region


class _FaceActivityTracker:
    """Rastreia todos os rostos visíveis ao longo do clipe e decide qual
    deles a câmera virtual deve seguir, priorizando quem está FALANDO ou
    reagindo de forma mais expressiva — em vez de simplesmente o maior
    rosto ou o mais recentemente detectado.

    Como funciona: cada rosto detectado vira um "slot" (posição, tamanho,
    e um placar de "atividade"). A cada frame (não só nas checagens de
    detecção, que são mais espaçadas), uma pequena região perto da
    boca/parte inferior do rosto de cada slot é comparada com a mesma
    região no frame anterior — quanto mais essa região muda de um frame
    para o outro, mais "atividade" (fala, riso, expressões) aquele rosto
    está tendo. O rosto com maior atividade acumulada é escolhido como
    ativo, com uma margem + tempo mínimo antes de trocar de um rosto para
    outro (histerese), para não ficar pulando a cada pequena flutuação.

    Essa checagem de movimento só roda de verdade quando há 2+ rostos
    visíveis ao mesmo tempo — com um rosto só, não há ambiguidade a
    resolver, então o custo extra (conversão pra tons de cinza + recorte)
    é evitado por completo, mantendo o caso comum (um rosto) tão rápido
    quanto antes.

    IMPORTANTE (correção deste ciclo): "quanto a região perto da boca
    mudou de um frame pro outro" sozinho NÃO distingue fala de qualquer
    outra coisa que mexa naquela região — em particular, uma mão subindo
    até o queixo/boca (gesto comum de quem está OUVINDO, pensando) gera
    uma variação de pixel tão grande quanto ou maior que a boca abrindo e
    fechando, e o placar antigo contava isso como "atividade de fala" da
    mesma forma. Foi exatamente esse o caso visto num clipe real: por um
    instante a câmera pulou pra quem tinha acabado de levar a mão ao
    queixo, enquanto a outra pessoa continuava com a MESMA frase no meio.
    Duas medidas novas (ver `sample_motion`) atacam isso diretamente:
    contraste interno do recorte (uma boca real tem lábios/dentes/sombra;
    uma mão cobrindo tem pele lisa e uniforme) e sustentação no tempo (fala
    oscila continuamente; uma mão chegando e parando é um pico único
    seguido de silêncio quase total).

    IMPLEMENTADO, VALIDAÇÃO PARCIAL (ver RELATORIO_PROXIMOS_PASSOS.txt, item
    2 e o adendo do ciclo seguinte — NÃO marcar como "resolvido" sem reler
    o adendo): o filtro de contraste em escala de cinza
    (`FACE_OCCLUSION_CONTRAST_MIN`) discriminava mal ("mão no queixo" vs
    "fala real" ficava em ~1.11x — praticamente nada). Trocado pelo
    desvio-padrão de MATIZ/HUE do mesmo recorte (`FACE_OCCLUSION_HUE_STD_MIN`,
    ponderado pela saturação média via `FACE_OCCLUSION_SAT_FLOOR` — hue é
    ruidoso em baixa saturação). Primeira medição: ~2.2x. Uma remedição
    posterior, em outro clipe (mesma pessoa, boca visível vs. mão cobrindo,
    poucos segundos de diferença — mais justa que a primeira, mas ainda não
    é a gravação bruta original com os dois falantes lado a lado que gerou
    o problema), deu só ~1.70x contra ~1.54x do contraste — o hue continua
    ganhando, mas com folga bem mais fraca do que a primeira medição sugeria.
    Ou seja: a direção do resultado (hue > contraste) tem duas medições a
    favor, mas a MAGNITUDE do ganho ainda não está calibrada com confiança.
    Antes de confiar cegamente nisso em produção, rode
    `scripts/validate_occlusion_hue.py` num clipe seu com 2+ pessoas e um
    gesto real de mão-no-rosto. `FACE_OCCLUSION_USE_HUE=False` volta ao modo
    antigo (contraste em cinza) só para comparação/depuração."""

    _PATCH_SIZE = 24
    # tamanho (em nº de amostras por-frame) da janela usada para exigir que
    # a atividade seja SUSTENTADA (oscilando) antes de contar em cheio —
    # ver `_burst_factor`. Calculado a partir de fps na primeira chamada de
    # `sample_motion` (não se sabe fps de antemão no __init__ atual, mas
    # como fps já é passado ao construtor, calculamos direto aqui).
    _BURST_WINDOW_SECONDS = 0.4

    def __init__(self, fps: float, detect_interval: int,
                 energy: Optional[np.ndarray] = None, energy_hop: float = 0.1):
        self.slots: dict = {}
        self._next_id = 0
        self.active_id: Optional[int] = None
        self.candidate_id: Optional[int] = None
        self.candidate_streak = 0
        self.fps = max(fps, 1e-3)

        # curva de energia RMS do áudio do próprio clipe (0..1, uma amostra
        # a cada `energy_hop` segundos) usada para confirmar que o
        # movimento perto da boca de um rosto coincide com áudio de
        # verdade, e não é só reação/agitação de quem está calado.
        self.energy = energy if (energy is not None and len(energy) > 0) else None
        self.energy_hop = max(energy_hop, 1e-3)
        self.energy_gate_on = getattr(config, "FACE_AUDIO_GATE_ENABLED", True)
        self.energy_floor = float(np.clip(
            getattr(config, "FACE_AUDIO_GATE_FLOOR", 0.12), 0.0, 1.0))

        check_interval = max(detect_interval / max(fps, 1e-3), 1e-3)
        self.check_interval = check_interval
        timeout_s = max(getattr(config, "FACE_SLOT_TIMEOUT_SECONDS", 1.2), check_interval)
        self.max_missed_checks = max(int(round(timeout_s / check_interval)), 1)

        # CORREÇÃO: `candidate_streak` (ver `choose_active`) incrementa uma
        # vez por CHECAGEM de rosto (só roda dentro do bloco "a cada
        # FACE_DETECT_EVERY_N_FRAMES frames" do loop principal), não uma vez
        # por frame renderizado. A versão anterior calculava o limiar em
        # unidade de FRAMES (switch_s * fps) — exatamente a mesma unidade
        # que `max_missed_checks`, duas linhas acima, evita de propósito.
        # Com os padrões (FACE_SWITCH_MIN_SECONDS=0.5s,
        # FACE_DETECT_EVERY_N_FRAMES=4 a 30fps => check_interval≈0.133s),
        # isso fazia a troca real de rosto ativo levar ~2s (15 checagens)
        # em vez dos 0.5s configurados — 4x mais devagar que o pedido.
        # Convertendo pra unidade de checagens (switch_s / check_interval,
        # igual a `max_missed_checks` já faz) o valor configurado passa a
        # valer de verdade.
        switch_s = max(getattr(config, "FACE_SWITCH_MIN_SECONDS", 0.5), 0.0)
        self.min_switch_frames = max(int(round(switch_s / check_interval)), 1)

        # Fração mínima de amostras "ativas" (acima de um piso de ruído)
        # dentro da janela de sustentação para um pico contar como fala de
        # verdade, e não como um único evento (mão chegando, cabeça virando
        # rápido). Ver `_burst_factor`.
        self.burst_window = max(int(round(self._BURST_WINDOW_SECONDS * fps)), 3)
        self.burst_min_fraction = float(np.clip(
            getattr(config, "FACE_ACTIVITY_BURST_MIN_FRACTION", 0.35), 0.0, 1.0))
        # IMPLEMENTADO, validação parcial (ver RELATORIO_PROXIMOS_PASSOS.txt,
        # item 2 + adendo): o filtro de oclusão trocou de contraste em
        # escala de cinza (~1.11x de discriminação mão-vs-boca, praticamente
        # inútil) para desvio-padrão de MATIZ/HUE (~2.2x numa 1ª medição;
        # ~1.70x vs 1.54x numa remedição em outro clipe — direção consistente,
        # magnitude do ganho ainda incerta). Mantemos occlusion_contrast_min
        # só como fallback se alguém desligar o modo hue
        # (FACE_OCCLUSION_USE_HUE=False) para comparar/depurar.
        self.occlusion_contrast_min = float(
            getattr(config, "FACE_OCCLUSION_CONTRAST_MIN", 9.0))
        self.occlusion_use_hue = bool(
            getattr(config, "FACE_OCCLUSION_USE_HUE", True))
        self.occlusion_hue_std_min = float(
            getattr(config, "FACE_OCCLUSION_HUE_STD_MIN", 10.0))
        self.occlusion_sat_floor = float(
            getattr(config, "FACE_OCCLUSION_SAT_FLOOR", 40.0))
        self.outlier_cap_mult = float(
            getattr(config, "FACE_ACTIVITY_OUTLIER_CAP_MULT", 3.0))

        # --- Achado num clipe REACT real (facecam pequena reagindo a um
        # vídeo de fundo, com chat) ---: o vídeo de fundo às vezes mostra
        # rostos reais (ex.: uma entrevista sendo reagida), e o Haar
        # detecta esses rostos igual detectaria o do streamer. Como esses
        # rostos "de fundo" costumam aparecer bem maiores em tela que a
        # facecam pequena do canto, o critério antigo de "sem atividade
        # ainda, usa o maior rosto" (ver choose_active) grudava direto no
        # rosto do vídeo reagido em vez do streamer -- o bug relatado
        # como "foca no maior rosto, não em quem tá falando". A facecam
        # do streamer, na prática, está em quadro DESDE O INÍCIO do clipe
        # e continua lá o tempo todo; um rosto de vídeo de fundo aparece e
        # some conforme o conteúdo reagido corta/rola. AGE_RAMP_CHECKS
        # (checagens, não segundos -- consistente com o resto da classe)
        # é quanto tempo um slot RECÉM-CRIADO precisa sobreviver antes de
        # ter "confiança" plena pra competir de igual pra igual: um slot
        # com 0 checagens de vida tem confiança ~0 (não pode vencer nada
        # ainda, nem no critério de atividade nem no de "sem dado ainda");
        # a confiança sobe linearmente até 1.0 depois de AGE_RAMP_CHECKS
        # checagens sobrevivendo. Isso não penaliza pra sempre um rosto
        # novo genuíno (ex.: alguém que entra no enquadramento) -- só
        # atrasa o quanto ele pode "roubar" o foco por um punhado de
        # checagens (frações de segundo), tempo suficiente pra distinguir
        # um rosto que vai ficar de um que é só um corte passageiro do
        # vídeo de fundo.
        self.age_ramp_checks = max(int(getattr(config, "FACE_AGE_RAMP_CHECKS", 6)), 1)

    def _trust(self, s: dict) -> float:
        """0..1: quanto confiar neste slot com base em há quanto tempo ele
        sobrevive (ver comentário de AGE_RAMP_CHECKS acima)."""
        return min(s.get("checks_alive", 0) / self.age_ramp_checks, 1.0)

    def _live_slots(self) -> dict:
        return {si: s for si, s in self.slots.items() if s["missed"] <= self.max_missed_checks}

    def on_scene_cut(self):
        """Descarta todos os slots ao detectar um corte de câmera — as
        posições acumuladas são do enquadramento anterior e não têm relação
        com o novo. A próxima detecção cria slots frescos e choose_active()
        escolhe pelo tamanho/idade sem o viés do histórico antigo."""
        self.slots.clear()
        self.active_id = None
        self.candidate_id = None
        self.candidate_streak = 0

    def update_detections(self, faces: list, frame_idx: int):
        """Casa as novas detecções com os slots existentes (pelo vizinho
        mais próximo, dentro de uma distância proporcional ao tamanho do
        rosto), cria slots para rostos novos, e descarta slots não vistos
        há tempo demais (NO_FACE_FALLBACK aplicado por rosto individual)."""
        pairs = []
        for si, s in self.slots.items():
            for fi, (cx, cy, fw, fh) in enumerate(faces):
                dist = ((cx - s["cx"]) ** 2 + (cy - s["cy"]) ** 2) ** 0.5
                thresh = 0.7 * max(s["fw"], fw, 1.0)
                if dist < thresh:
                    pairs.append((dist, si, fi))
        pairs.sort(key=lambda p: p[0])

        matched_slots, matched_faces = set(), set()
        for dist, si, fi in pairs:
            if si in matched_slots or fi in matched_faces:
                continue
            matched_slots.add(si)
            matched_faces.add(fi)
            cx, cy, fw, fh = faces[fi]
            s = self.slots[si]
            a = 0.5  # suaviza só a posição/tamanho do box (não a câmera)
            s["cx"] = (1 - a) * s["cx"] + a * cx
            s["cy"] = (1 - a) * s["cy"] + a * cy
            s["fw"] = (1 - a) * s["fw"] + a * fw
            s["fh"] = (1 - a) * s["fh"] + a * fh
            s["missed"] = 0
            s["checks_alive"] = s.get("checks_alive", 0) + 1

        for si in self.slots:
            if si not in matched_slots:
                self.slots[si]["missed"] += 1

        for fi, (cx, cy, fw, fh) in enumerate(faces):
            if fi not in matched_faces:
                self._next_id += 1
                self.slots[self._next_id] = {
                    "cx": cx, "cy": cy, "fw": fw, "fh": fh,
                    "activity": 0.0, "missed": 0, "patch": None,
                    "checks_alive": 0,
                    # histórico curto (deque) dos últimos scores BRUTOS
                    # (pré-EMA, pré-gate de áudio) de diferença de patch —
                    # usado por `_burst_factor` pra exigir que a atividade
                    # esteja OSCILANDO (fala real) em vez de ser um pico
                    # isolado (mão chegando, cabeça virando).
                    "history": deque(maxlen=self.burst_window),
                }

        for si in [si for si, s in self.slots.items() if s["missed"] > self.max_missed_checks]:
            del self.slots[si]
            if self.active_id == si:
                self.active_id = None
            if self.candidate_id == si:
                self.candidate_id, self.candidate_streak = None, 0

    def _audio_gate(self, t: float) -> float:
        """Quanto o áudio confirma que HÁ alguém falando no instante `t`
        (0..1). Sem curva de energia disponível, não peneira nada (gate
        neutro = 1.0), mantendo o comportamento antigo como fallback."""
        if not self.energy_gate_on or self.energy is None:
            return 1.0
        idx = int(t / self.energy_hop)
        idx = min(max(idx, 0), len(self.energy) - 1)
        e = float(self.energy[idx])
        # o floor evita que uma pausa curta de respiração no meio de uma
        # fala real derrube a atividade a zero — só o silêncio SUSTENTADO
        # (várias amostras seguidas abaixo do floor) de fato apaga o
        # placar, graças ao EMA em sample_motion.
        return max(e, self.energy_floor)

    def _burst_factor(self, s: dict, raw_score: float) -> float:
        """Quanto a atividade recente deste slot parece SUSTENTADA/
        oscilando (fala real, que abre e fecha a boca várias vezes por
        segundo) em vez de um evento único (uma mão chegando ao queixo e
        parando, uma cabeça virando rápido) — nesse segundo caso, o
        histórico tem UM valor alto seguido de vários valores baixos, e a
        fração de amostras "ativas" na janela fica baixa mesmo que o pico
        isolado tenha sido grande.

        Retorna um fator 0..1 aplicado sobre o score bruto: 1.0 quando o
        padrão já parece sustentado, min(0.25, burst_min_fraction) como
        piso pra não zerar completamente uma fala real que só está
        começando (ainda sem histórico suficiente na janela)."""
        hist = s["history"]
        hist.append(raw_score)
        if len(hist) < 3:
            return 1.0  # sem histórico suficiente ainda: não penaliza
        arr = np.asarray(hist, dtype=np.float32)
        noise_floor = max(float(np.median(arr)) * 0.5, 1.5)
        active_frac = float(np.mean(arr > noise_floor))
        if active_frac >= self.burst_min_fraction:
            return 1.0
        floor = min(self.burst_min_fraction, 0.25)
        return floor + (1.0 - floor) * (active_frac / max(self.burst_min_fraction, 1e-6))

    def sample_motion(self, frame_bgr, t: float = 0.0):
        """Roda TODO frame (não só nas checagens de detecção) para captar
        movimento de fala/reação com boa resolução temporal. Sai
        imediatamente sem custo se houver 0 ou 1 rosto visível — só
        importa comparar atividade quando há mais de um candidato.

        `t` é o instante (em segundos, relativo ao início do clipe) desse
        frame — usado para consultar a curva de energia do áudio e
        confirmar que o movimento captado coincide com fala de verdade,
        em vez de qualquer agitação facial (ver `_audio_gate`).

        O score bruto (diferença média de pixel entre este frame e o
        anterior, na região da boca) passa por TRÊS filtros antes de virar
        "atividade", nesta ordem — cada um ataca um jeito diferente de essa
        região mudar sem que a pessoa esteja de fato falando:

        1. Desvio-padrão de MATIZ/HUE do recorte (inline abaixo, IMPLEMENTADO
           com validação parcial — ver RELATORIO_PROXIMOS_PASSOS.txt item 2
           + adendo): uma boca de verdade (aberta ou fechada) tem cores
           variadas — lábio, dentes, sombra — hue_std alto. Uma mão ou outro
           objeto cobrindo a boca é pele quase uniforme — hue_std baixo,
           mesmo quando a textura de luminância (dedos, sombra entre eles)
           engana um filtro de contraste em cinza. Recortes de hue_std baixo
           (ponderado pela confiança, via saturação média) têm o score bruto
           descontado antes de mais nada. Medido em dois clipes reais
           diferentes: ~2.2x de discriminação mão-vs-boca na 1ª medição,
           ~1.70x (contra 1.54x do contraste) numa remedição mais rigorosa —
           direção do resultado consistente, magnitude ainda não fechada.
           Rode `scripts/validate_occlusion_hue.py` no SEU conteúdo antes de
           confiar cegamente na calibração padrão.
        2. Sustentação no tempo (`_burst_factor`): fala real oscila —
           vários picos por segundo, não um só. Um evento único (mão
           chegando, cabeça virando) aparece como UM pico isolado no
           histórico recente. Só picos que se repetem contam em cheio.
        3. Corte de outlier: mesmo depois dos dois filtros acima, um único
           frame não pode, sozinho, disparar o placar de EMA muito acima
           da própria média recente do slot — evita que um pico
           remanescente (não pego pelos filtros 1-2) grude por vários
           frames por causa do próprio EMA."""
        live = self._live_slots()
        if len(live) < 2:
            return
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        # HSV só é calculado quando o modo hue está ligado (padrão) — é um
        # cvtColor extra por frame, barato (mesma resolução, mesmo custo de
        # ordem de grandeza que o cvtColor pra cinza acima), mas não faz
        # sentido pagar esse custo se occlusion_use_hue=False.
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if self.occlusion_use_hue else None
        decay = getattr(config, "FACE_ACTIVITY_DECAY", 0.85)
        gate = self._audio_gate(t)
        ps = self._PATCH_SIZE
        for s in live.values():
            rw = max(s["fw"] * 0.7, 10)
            rh = max(s["fh"] * 0.5, 10)
            x0 = int(np.clip(s["cx"] - rw / 2, 0, max(w - 1, 0)))
            y0 = int(np.clip(s["cy"] - s["fh"] * 0.05, 0, max(h - 1, 0)))
            x1 = int(np.clip(x0 + rw, 0, w))
            y1 = int(np.clip(y0 + rh, 0, h))
            if x1 <= x0 or y1 <= y0:
                continue
            patch = cv2.resize(gray[y0:y1, x0:x1], (ps, ps),
                                interpolation=cv2.INTER_AREA).astype(np.float32)
            prev = s["patch"]
            if prev is not None:
                raw_score = float(np.mean(np.abs(patch - prev)))

                # 1. desconta oclusão/superfície lisa (mão, pele, cabelo
                # cobrindo a boca) usando o recorte ATUAL. Métrica principal
                # (ver RELATORIO_PROXIMOS_PASSOS.txt, item 2 — resolvido):
                # desvio-padrão de MATIZ (hue), não de luminância. Uma boca
                # real tem cores bem variadas (vermelho do lábio, branco/
                # creme do dente, tons escuros da sombra interna) -> hue_std
                # alto. Uma mão/pele cobrindo é essencialmente um tom só ->
                # hue_std baixo, mesmo quando a TEXTURA de luminância (dedos,
                # sombra entre eles) engana o filtro de contraste antigo.
                # Validado: ~2.2x de razão de discriminação mão-vs-boca,
                # contra ~1.11x do contraste em cinza (praticamente nada).
                if hsv is not None:
                    patch_hsv = cv2.resize(hsv[y0:y1, x0:x1], (ps, ps),
                                            interpolation=cv2.INTER_AREA)
                    hue = patch_hsv[:, :, 0].astype(np.float32)
                    sat = patch_hsv[:, :, 1].astype(np.float32)
                    hue_std = float(np.std(hue))
                    mean_sat = float(np.mean(sat))
                    # hue não é confiável em baixa saturação (tons quase
                    # acinzentados/estourados de brilho — comum em pele muito
                    # clara ou sombra forte) — nesses casos o próprio matiz
                    # vira ruído, então descontamos o quanto confiamos nele
                    # em vez de aplicar hue_std cru. sat_confidence 0 = não
                    # confia nada no hue (usa só o piso 0.4 abaixo);
                    # sat_confidence 1 = confia no hue_std normalmente.
                    sat_confidence = float(np.clip(
                        mean_sat / max(self.occlusion_sat_floor, 1e-3), 0.0, 1.0))
                    hue_factor = float(np.clip(
                        hue_std / max(self.occlusion_hue_std_min, 1e-3), 0.12, 1.0))
                    occlusion_factor = hue_factor * (0.4 + 0.6 * sat_confidence)
                else:
                    # fallback (FACE_OCCLUSION_USE_HUE=False): versão antiga
                    # em contraste de cinza, mantida só para comparação/depuração.
                    contrast = float(np.std(patch))
                    occlusion_factor = float(np.clip(
                        contrast / max(self.occlusion_contrast_min, 1e-3), 0.12, 1.0))

                # 2. exige sustentação/oscilação (fala de verdade) em vez de
                # um pico único (mão chegando, cabeça virando rápido).
                burst_factor = self._burst_factor(s, raw_score)

                score = raw_score * occlusion_factor * burst_factor * gate

                # 3. corte de outlier: um frame isolado não pode empurrar o
                # EMA muito além da própria média recente do slot — evita
                # que um pico remanescente grude no placar por vários
                # frames só por causa do próprio EMA (decay alto = memória
                # longa). Sem piso mínimo aqui (max(..., algo)) porque no
                # início do clipe activity=0 e não queremos travar o
                # primeiro sinal real em zero.
                cap = self.outlier_cap_mult * s["activity"] + 4.0
                score = min(score, cap)

                s["activity"] = decay * s["activity"] + (1 - decay) * score
            s["patch"] = patch

    def choose_active(self) -> Optional[tuple]:
        """Retorna o centro (cx, cy) do rosto que a câmera deve seguir
        agora, ou None se nenhum rosto está visível."""
        live = self._live_slots()
        if not live:
            self.active_id, self.candidate_id, self.candidate_streak = None, None, 0
            return None

        priority_on = getattr(config, "SPEAKING_PRIORITY_ENABLED", True)
        if len(live) == 1 or not priority_on:
            si = max(live, key=lambda k: live[k]["fw"] * live[k]["fh"])
            self.active_id = si
            return (live[si]["cx"], live[si]["cy"])

        if self.active_id not in live:
            # o rosto ativo anterior saiu de quadro: escolhe por atividade
            # (ponderada pela confiança de idade -- ver `_trust`) se já
            # houver algum sinal; sem nenhum histórico ainda, usa o slot
            # mais ANTIGO (mais checagens sobrevivendo), não o maior --
            # ver comentário de AGE_RAMP_CHECKS no __init__: um rosto de
            # vídeo de fundo reagido costuma ser maior em tela que uma
            # facecam pequena, mas o streamer é quem está em quadro desde
            # o início do clipe.
            # Sem histórico de atividade (fresh-after-cut ou burst inicial):
            # prefere o rosto mais ALTO no frame (menor cy). Em conteúdo de
            # podcast/entrevista, rostos reais ficam no terço superior do
            # frame; objetos em mesa, toys ou props ficam abaixo.
            if all(s["activity"] * self._trust(s) < 1e-6 for s in live.values()):
                self.active_id = min(live, key=lambda k: live[k]["cy"])
            else:
                self.active_id = max(live, key=lambda k: live[k]["activity"] * self._trust(live[k]))
            self.candidate_id, self.candidate_streak = None, 0
            return (live[self.active_id]["cx"], live[self.active_id]["cy"])

        current = live[self.active_id]
        challenger_id = max((si for si in live if si != self.active_id),
                             key=lambda k: live[k]["activity"] * self._trust(live[k]),
                             default=None)
        if challenger_id is not None:
            challenger = live[challenger_id]
            margin = getattr(config, "FACE_SWITCH_MARGIN", 1.4)
            challenger_score = challenger["activity"] * self._trust(challenger)
            if challenger_score > current["activity"] * margin + 1e-6:
                if self.candidate_id == challenger_id:
                    self.candidate_streak += 1
                else:
                    self.candidate_id, self.candidate_streak = challenger_id, 1
                if self.candidate_streak >= self.min_switch_frames:
                    self.active_id = challenger_id
                    self.candidate_id, self.candidate_streak = None, 0
            else:
                self.candidate_id, self.candidate_streak = None, 0

        # Two-shot: se há 2+ rostos e o 2º ativo tem atividade relevante
        # (não está claramente calado), retorna o ponto médio entre os dois
        # em vez do centro do vencedor — câmera enquadra os dois em vez de
        # ficar presa num só e cortar o outro.
        if len(live) >= 2 and getattr(config, "FACE_TWO_SHOT_ENABLED", True):
            ratio = getattr(config, "FACE_TWO_SHOT_MAX_RATIO", 2.5)
            winner = live[self.active_id]
            runner_ids = sorted(
                (si for si in live if si != self.active_id),
                key=lambda k: live[k]["activity"] * self._trust(live[k]),
                reverse=True,
            )
            if runner_ids:
                runner = live[runner_ids[0]]
                w_score = max(winner["activity"] * self._trust(winner), 1e-6)
                r_score = runner["activity"] * self._trust(runner)
                if r_score > w_score / ratio:
                    mid_cx = (winner["cx"] + runner["cx"]) / 2.0
                    mid_cy = (winner["cy"] + runner["cy"]) / 2.0
                    return (mid_cx, mid_cy)

        return (live[self.active_id]["cx"], live[self.active_id]["cy"])

    def active_face_size(self) -> Optional[tuple]:
        """Retorna (fw, fh) do rosto ATUALMENTE ativo (o mesmo escolhido por
        `choose_active`), ou None se não há nenhum rosto ativo agora. Usado
        pelo modo REACT (RELATORIO_PROXIMOS_PASSOS.txt, item 14) pra
        detectar facecam pequena e afrouxar o crop -- ver
        FACECAM_SMALL_HEIGHT_FRAC em config.py."""
        live = self._live_slots()
        if self.active_id in live:
            s = live[self.active_id]
            return (s["fw"], s["fh"])
        return None

    def group_face_count(self) -> int:
        """Quantos rostos "de verdade" estão em quadro agora: vistos em pelo
        menos duas checagens (descarta detecção isolada/ruído) e com pelo
        menos metade do tamanho do maior (descarta falso positivo pequeno no
        fundo). Usado pra distinguir plano aberto de grupo (várias pessoas
        na mesa -> layout fit) de plano médio de uma pessoa só (-> recorte
        com zoom)."""
        live = [s for s in self._live_slots().values() if s.get("checks_alive", 0) >= 1]
        if not live:
            return 0
        biggest = max(s["fh"] for s in live)
        return sum(1 for s in live if s["fh"] >= 0.5 * biggest)

    def active_speaking_activity(self) -> float:
        """Placar (EMA) de atividade de boca do rosto ATUALMENTE ativo — o
        mesmo número usado internamente pra decidir troca de falante entre
        VÁRIOS rostos (`choose_active`), reaproveitado aqui pra decidir
        "o streamer está falando agora?" no modo REACT (item 17). 0.0 se
        não há rosto ativo."""
        live = self._live_slots()
        if self.active_id in live:
            return float(live[self.active_id]["activity"])
        return 0.0




def _open_ffmpeg_reader(source_path: str, start: float, duration: float,
                         src_w: int, src_h: int) -> subprocess.Popen:
    """Processo ffmpeg que decodifica (não recodifica) apenas o trecho
    [start, start+duration] do vídeo original e envia os frames crus (BGR)
    via pipe. -ss antes de -i faz o ffmpeg buscar rapidamente a keyframe
    mais próxima e decodificar com precisão até o timestamp exato — mais
    rápido e mais confiável que abrir o arquivo inteiro e buscar via OpenCV."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", str(max(start, 0.0)), "-i", str(source_path),
        "-t", str(max(duration, 0.05)),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{src_w}x{src_h}",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _reader_thread_fn(reader: subprocess.Popen, frame_bytes: int,
                       out_q: "queue.Queue"):
    """Roda numa thread separada: lê frames crus do stdout do ffmpeg leitor
    e empilha na fila, permitindo que a decodificação continue enquanto a
    thread principal processa/escreve o frame anterior. Põe `None` na fila
    ao terminar (fim do stream ou erro) como sentinela."""
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            out_q.put(raw)
    finally:
        out_q.put(None)


def _escape_ffmpeg_filter_path(path: str) -> str:
    """Escapa um caminho de arquivo para uso dentro do valor de uma opção
    do filtro `ass` (ex.: filename, fontsdir) no -vf do ffmpeg.

    O filtro `ass` faz DUAS passagens de parsing sobre o valor de suas
    opções: primeiro o parser do filtergraph (que já trata ':' como
    separador de opção e exige '\\:' para um ':' literal) e depois um
    parser interno do próprio libass. Por isso um ':' literal — como o
    da letra de unidade no Windows (`C:`) — precisa ser escapado DUAS
    vezes (`\\\\:`), não uma (`\\:`). Com escape simples, a primeira
    passagem consome o '\\' e entrega um ':' cru para a segunda
    passagem, que então interpreta como início de outra opção -> erro
    "No option name near ...". Em caminhos POSIX isso nunca aparece
    porque não há ':' no meio do path, então o bug fica escondido até
    rodar em Windows.
    """
    p = str(path).replace("\\", "/")
    p = p.replace(":", "\\\\:")
    return p


def _build_color_vf() -> List[str]:
    """Monta os filtros de correção de cor + vinheta ("grade cinematográfica")
    aplicados no encode final. Ficam ANTES da legenda na cadeia de filtros
    para que a legenda não seja escurecida pela vinheta."""
    parts = []
    if getattr(config, "COLOR_GRADE_ENABLED", True):
        parts.append(
            f"eq=contrast={config.COLOR_EQ_CONTRAST}:"
            f"saturation={config.COLOR_EQ_SATURATION}:"
            f"brightness={config.COLOR_EQ_BRIGHTNESS}:"
            f"gamma={config.COLOR_EQ_GAMMA}"
        )
    if getattr(config, "VIGNETTE_ENABLED", True):
        parts.append(f"vignette={config.VIGNETTE_ANGLE}")
    return parts


def _open_ffmpeg_writer(output_path: str, width: int, height: int, fps: float,
                         audio_path: Optional[str], ass_path: Optional[str],
                         fonts_dir: Optional[str], force_cpu: bool = False) -> subprocess.Popen:
    vf_parts = _build_color_vf()
    if ass_path:
        escaped = _escape_ffmpeg_filter_path(ass_path)
        ass_filter = f"ass={escaped}"
        if fonts_dir:
            fdir = _escape_ffmpeg_filter_path(fonts_dir)
            ass_filter += f":fontsdir={fdir}"
        vf_parts.append(ass_filter)
    extra_vf = ",".join(vf_parts) if vf_parts else None

    pre, vf, codec = hwaccel.encoder_args(extra_vf=extra_vf, force_cpu=force_cpu)
    cmd = [
        "ffmpeg", "-y", "-v", "error", *pre,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(fps), "-i", "-",
    ]
    if audio_path:
        cmd += ["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"]
    if vf:
        cmd += ["-vf", vf]
    cmd += codec
    if audio_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    # +faststart move o índice (moov) pro começo do .mp4: o upload no
    # YouTube/Instagram/TikTok começa a processar sem esperar o arquivo
    # inteiro, e o vídeo abre na hora no celular/navegador.
    cmd += ["-movflags", "+faststart", str(output_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def _compose_tracked_frame(frame, smoothed_x: float, smoothed_y: float,
                            crop_w: int, crop_h: int, src_w: int, src_h: int,
                            out_w: int, out_h: int, zoom_factor: float) -> np.ndarray:
    """Modo "seguindo o rosto": recorta uma janela 9:16 centrada no rosto
    rastreado (com headroom configurável) e reamostra para a resolução de
    saída. `zoom_factor` > 1 aperta ainda mais o crop (usado nos "zoom
    punches" de ênfase)."""
    cur_crop_w = max(min(int(crop_w / zoom_factor), src_w), 2)
    cur_crop_h = max(min(int(crop_h / zoom_factor), src_h), 2)

    x0 = int(np.clip(smoothed_x - cur_crop_w / 2, 0, max(src_w - cur_crop_w, 0)))
    # posiciona verticalmente para que o rosto rastreado (smoothed_y) caia
    # em HEADROOM_RATIO da altura do crop, em vez de sempre exibir a altura
    # inteira da fonte (o que antes tornava HEADROOM_RATIO inofensivo: com
    # crop_h == src_h, y0 era sempre 0 e a config nunca fazia diferença).
    y0 = int(np.clip(smoothed_y - cur_crop_h * config.HEADROOM_RATIO,
                      0, max(src_h - cur_crop_h, 0)))

    cropped = frame[y0:y0 + cur_crop_h, x0:x0 + cur_crop_w]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = frame
    interp = cv2.INTER_AREA if cur_crop_w > out_w else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def _compose_screen_frame(frame, region, src_w: int, src_h: int,
                           out_w: int, out_h: int, padding: float = 0.06) -> np.ndarray:
    """Modo "vídeo dentro do vídeo" (ver RELATORIO_PROXIMOS_PASSOS.txt, item
    3a): recorta e amplia a região da tela detectada (com uma margem/
    padding ao redor pra não cravar a borda exata do retângulo, que costuma
    ser um pouco imprecisa), preservando a proporção 9:16 do quadro de
    saída — igual em espírito a `_compose_tracked_frame`, mas centrado no
    retângulo da tela em vez do rosto."""
    x, y, w, h = region
    pad_x, pad_y = w * padding, h * padding
    cx, cy = x + w / 2.0, y + h / 2.0
    rw, rh = w + 2 * pad_x, h + 2 * pad_y

    # ajusta pra proporção 9:16 do destino, expandindo o menor lado (nunca
    # cortando a região detectada) em vez de espremer/distorcer.
    target_aspect = out_w / out_h
    cur_aspect = rw / max(rh, 1e-3)
    if cur_aspect > target_aspect:
        rh = rw / target_aspect
    else:
        rw = rh * target_aspect

    crop_w = max(min(int(rw), src_w), 2)
    crop_h = max(min(int(rh), src_h), 2)
    x0 = int(np.clip(cx - crop_w / 2, 0, max(src_w - crop_w, 0)))
    y0 = int(np.clip(cy - crop_h / 2, 0, max(src_h - crop_h, 0)))

    cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = frame
    interp = cv2.INTER_AREA if crop_w > out_w else cv2.INTER_LINEAR
    return cv2.resize(cropped, (out_w, out_h), interpolation=interp)


def _compose_wide_frame(frame, src_w: int, src_h: int, out_w: int, out_h: int,
                        push: float = 1.0, focus_x: Optional[float] = None) -> np.ndarray:
    """Modo "plano aberto" (fallback quando nenhum rosto é encontrado por
    tempo suficiente): em vez de cravar um crop apertado num ponto
    qualquer da imagem — o que, num plano largo, quase sempre mostra só
    mesa/objetos, sem ninguém em quadro — mostra o frame ORIGINAL inteiro
    (sem cortar ninguém de fora), encaixado no centro do quadro vertical,
    com um fundo desfocado/escurecido preenchendo o espaço acima/abaixo.
    É a mesma técnica usada por Opus Clip/CapCut para planos largos."""
    # fundo: escala "cover" (preenche o quadro inteiro, pode cortar um pouco
    # das bordas) + desfoque forte + escurecido, pra não competir com o
    # conteúdo principal nem com a legenda.
    bg_scale = max(out_w / src_w, out_h / src_h)
    bg_w, bg_h = max(int(round(src_w * bg_scale)), out_w), max(int(round(src_h * bg_scale)), out_h)
    bg = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_LINEAR)
    bx0 = (bg_w - out_w) // 2
    by0 = (bg_h - out_h) // 2
    bg = bg[by0:by0 + out_h, bx0:bx0 + out_w]
    # o desfoque de fundo é aplicado numa cópia BEM menor e depois
    # redimensionado de volta: em resolução cheia (1080x1920) um
    # GaussianBlur com sigma alto custa ~300ms/frame sozinho (medido) —
    # mais caro que o resto do pipeline inteiro somado, e sem nenhum
    # ganho visual (o resultado, de propósito, é um borrão irreconhecível
    # de qualquer forma). Borrando em baixa resolução o mesmo efeito sai
    # ~50x mais rápido.
    sigma = getattr(config, "FALLBACK_BG_BLUR_SIGMA", 25.0)
    down = 6
    small_w, small_h = max(out_w // down, 8), max(out_h // down, 8)
    small = cv2.resize(bg, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=max(sigma / down, 1.0))
    bg = cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    darken = float(np.clip(getattr(config, "FALLBACK_BG_DARKEN", 0.55), 0.0, 1.0))
    bg = (bg.astype(np.float32) * darken).astype(np.uint8)

    # primeiro plano: escala "contain" (cabe a largura inteira), um pouco
    # ampliada por WIDE_FIT_ZOOM (corta só uma lasquinha das laterais, onde
    # num plano de mesa quase nunca tem ninguém) pra faixa central não
    # ficar pequena demais no celular; centralizado verticalmente.
    # `push` > 1: aproximação lenta (Ken Burns) durante um plano aberto
    # longo, puxando pro lado de `focus_x` (quem está falando).
    fit_zoom = max(float(getattr(config, "WIDE_FIT_ZOOM", 1.0)), 1.0) * max(push, 1.0)
    fg_scale = out_w / src_w * fit_zoom
    fg_w = max(int(round(src_w * fg_scale)), out_w)
    fg_h = max(int(round(src_h * fg_scale)), 1)
    interp = cv2.INTER_AREA if fg_scale < 1 else cv2.INTER_LINEAR
    fg = cv2.resize(frame, (fg_w, fg_h), interpolation=interp)
    cx = fg_w / 2.0 if focus_x is None else focus_x * fg_scale
    fx0 = int(np.clip(cx - out_w / 2.0, 0, fg_w - out_w))
    fg = fg[:, fx0:fx0 + out_w]

    canvas = bg
    fy0 = max((out_h - fg_h) // 2, 0)
    fy1 = min(fy0 + fg_h, out_h)
    src_fy1 = fy1 - fy0
    if src_fy1 > 0:
        canvas[fy0:fy1, :] = fg[0:src_fy1, :]
    return canvas


def render_vertical_clip(source_path: str, start: float, end: float,
                          src_w: int, src_h: int, fps: float,
                          output_path: str,
                          ass_path: Optional[str] = None,
                          audio_path: Optional[str] = None,
                          fonts_dir: Optional[str] = None,
                          zoom_peak_times: Optional[List[float]] = None,
                          force_cpu: bool = False,
                          audio_energy: Optional[np.ndarray] = None,
                          audio_energy_hop: float = 0.1,
                          reference_times: Optional[List[float]] = None,
                          keep_segments: Optional[List[tuple]] = None,
                          fx=None) -> str:
    """Gera o clipe vertical final (9:16) em um único passe: decodifica só o
    trecho necessário do vídeo original, recorta seguindo o rosto do
    orador (com fallback para plano aberto quando não há rosto em quadro),
    aplica zoom punches, corrige cor, aplica vinheta, queima a legenda e
    mixa o áudio final — tudo em uma única codificação de vídeo.

    `audio_energy`/`audio_energy_hop`: curva de energia RMS do áudio deste
    clipe (ver `audio.audio_energy`), usada para confirmar qual rosto está
    REALMENTE falando (ver `_FaceActivityTracker._audio_gate`). Opcional —
    sem ela, o rastreamento volta a depender só do movimento facial.

    `keep_segments`: trechos (segundos relativos a `start`, alinhados na
    grade de quadros) que ficam no vídeo final -- ver src/jumpcut.py. Os
    quadros fora deles (pausas cortadas) são lidos e descartados. Nesse caso
    `audio_energy`, `zoom_peak_times` e `reference_times` já vêm na linha do
    tempo SEM as pausas (a do áudio final), e é nela que tudo que depende de
    tempo é consultado aqui.

    `fx`: plano de efeitos do clipe (src/fx.py — zoom nos momentos-chave,
    impacto, emoji, abertura), consultado pelo tempo da linha final."""
    zoom_peak_times = zoom_peak_times or []
    reference_times = sorted(reference_times or [])
    duration = max(end - start, 0.1)
    cascades = _new_cascades()
    tracker = _FaceActivityTracker(
        fps=fps, detect_interval=config.FACE_DETECT_EVERY_N_FRAMES,
        energy=audio_energy, energy_hop=audio_energy_hop,
    )
    # "Vídeo dentro do vídeo" (RELATORIO_PROXIMOS_PASSOS.txt, item 3a):
    # NÃO validado contra footage real — desligado por padrão em config.py.
    screen_tracker = (
        _ScreenTracker(fps=fps, detect_interval=config.FACE_DETECT_EVERY_N_FRAMES)
        if getattr(config, "VIDEO_IN_VIDEO_ENABLED", False) else None
    )

    out_w, out_h = config.TARGET_WIDTH, config.TARGET_HEIGHT

    # janela de recorte "seguindo o rosto", na resolução original, mantendo
    # a proporção 9:16. FACE_CROP_ZOOM > 1 aperta esse recorte (crop menor
    # que o "maior 9:16 que cabe preservando a altura toda da fonte") para
    # dar um enquadramento mais fechado / "TikTok" no rosto rastreado.
    face_zoom = max(getattr(config, "FACE_CROP_ZOOM", 1.0), 1.0)
    crop_h = max(int(src_h / face_zoom), 2)
    crop_w = int(crop_h * out_w / out_h)
    if crop_w > src_w:
        crop_w = src_w
        crop_h = int(crop_w * out_h / out_w)

    # --- Facecam pequena / modo REACT (RELATORIO item 14) ---
    # crop_w/crop_h acima são o enquadramento "fechado" PADRÃO (derivado só
    # de FACE_CROP_ZOOM, fixo pro clipe inteiro). smoothed_crop_h é a
    # versão DINÂMICA de fato usada pra compor cada frame em modo "face":
    # normalmente igual a crop_h, mas afrouxa (cresce) quando o rosto
    # detectado ocupa uma fatia pequena da altura do frame de origem —
    # típico de facecam pequena sobre uma tela maior — pra não ampliar uma
    # região minúscula até virar um close pixelado; ver
    # FACECAM_SMALL_HEIGHT_FRAC/FACECAM_TARGET_FACE_FRAC em config.py.
    target_crop_h = float(crop_h)
    smoothed_crop_h = float(crop_h)
    facecam_small_frac = getattr(config, "FACECAM_SMALL_HEIGHT_FRAC", 0.16)

    # --- Enquadramento pelo tamanho do rosto (vídeo comum/podcast) ---
    # ver SUBJECT_* em config.py: fora do modo REACT, o crop é dimensionado
    # pelo tamanho do rosto ativo (zoom em plano médio) e, quando nem o zoom
    # máximo deixa o rosto num tamanho decente (plano aberto de verdade),
    # vai pro layout "fit" (quadro inteiro + fundo desfocado).
    subject_framing = screen_tracker is None
    subject_target_frac = getattr(config, "SUBJECT_TARGET_FACE_FRAC", 0.22)
    subject_min_crop_h = min(out_h / max(getattr(config, "SUBJECT_MAX_UPSCALE", 3.0), 1.0),
                             float(crop_h))
    fit_group_frac = getattr(config, "SUBJECT_FIT_GROUP_FACE_FRAC", 0.16)
    fit_single_frac = getattr(config, "SUBJECT_FIT_SINGLE_FACE_FRAC", 0.07)
    subject_face_frac: Optional[float] = None  # EMA do tamanho do rosto no quadro final
    group_shot = False        # plano de grupo confirmado -- vale até o próximo corte
    subject_too_small = False

    center_x = src_w / 2.0
    center_y = src_h * config.HEADROOM_RATIO
    smoothed_x = center_x
    smoothed_y = center_y
    target_x = center_x
    target_y = center_y
    last_detected_xy: Optional[tuple] = None  # última posição CRUA detectada
                                               # (não suavizada) — usada só
                                               # para CONFIRMAR cortes reais
                                               # de câmera (ver abaixo)
    face_confidence = 0.0  # 0 = nenhum rosto rastreado, 1 = rastreando com confiança
    # duração da transição suave entre "seguindo o rosto" e "plano aberto",
    # em frames — derivada de MODE_BLEND_SECONDS (config em segundos) e do
    # fps real deste vídeo, pra a transição durar o mesmo tempo em telas
    # gravadas a 30fps ou a 60fps.
    blend_seconds = max(getattr(config, "MODE_BLEND_SECONDS", 0.6), 0.0)
    blend_frames_total = max(int(round(blend_seconds * fps)), 1)

    # deriva o decaimento de confiança a partir de NO_FACE_FALLBACK_SECONDS
    # (o único knob que o usuário deveria precisar mexer): calcula quantas
    # checagens de rosto (a cada FACE_DETECT_EVERY_N_FRAMES frames) cabem
    # nesse tempo, e o fator de decaimento por checagem sem rosto que faz a
    # confiança cruzar o limiar de fallback (0.35) exatamente depois disso.
    fallback_seconds = max(getattr(config, "NO_FACE_FALLBACK_SECONDS", 0.8), 0.05)
    check_interval = max(config.FACE_DETECT_EVERY_N_FRAMES / fps, 1e-3)
    misses_to_fallback = max(fallback_seconds / check_interval, 1.0)
    confidence_decay = 0.35 ** (1.0 / misses_to_fallback)

    # Detecção de CORTE DE CÂMERA real (troca de plano no vídeo de origem —
    # ex.: podcast de duas câmeras alternando entre quem fala). Quando o
    # rosto detectado numa checagem está MUITO longe do enquadramento atual
    # (mais que CUT_JUMP_RATIO * crop_w), suavizar normalmente faz o crop
    # "arrastar" pela imagem por cima de composições ruins no meio do
    # caminho por 1-2s até convergir — isso é o que aparece como a câmera
    # "travando"/"pulando". Só que uma detecção Haar isolada e ruidosa (uma
    # mão, um objeto, um reflexo) também pode produzir um salto grande só
    # por uma checagem. Por isso o salto só é aceito como corte real quando
    # DUAS checagens seguidas (a ~check_interval de distância uma da outra)
    # concordam sobre a nova posição — nesse caso, em vez de continuar
    # suavizando, o crop PULA direto pra lá (como um corte de verdade),
    # ficando visualmente igual ao que um editor humano faria.
    cut_jump_threshold = getattr(config, "CUT_JUMP_RATIO", 0.35) * crop_w
    cut_confirm_tolerance = getattr(config, "CUT_CONFIRM_RATIO", 0.15) * crop_w

    reader = _open_ffmpeg_reader(source_path, start, duration, src_w, src_h)
    writer = _open_ffmpeg_writer(output_path, out_w, out_h, fps, audio_path,
                                  ass_path, fonts_dir, force_cpu=force_cpu)

    frame_bytes = src_w * src_h * 3
    max_frames = int(round(duration * fps)) + 2  # pequena folga contra arredondamento

    # FACE_SMOOTHING_ALPHA foi calibrado como "quanto a câmera anda em
    # direção ao alvo a cada CHECAGEM de rosto" (uma a cada
    # FACE_DETECT_EVERY_N_FRAMES frames). O bug original aplicava esse
    # alpha só nos frames em que a detecção rodava — nos outros frames
    # smoothed_x/y ficavam parados, e a câmera "pulava" de uma vez a cada
    # N frames em vez de se mover continuamente (era isso que você media
    # como "parada e depois pulando").
    #
    # A correção é mover a câmera virtual um pouquinho a CADA frame, não só
    # nas checagens. Para a curva de convergência até o alvo continuar com
    # a mesma "velocidade" em tempo real de antes (só que distribuída
    # suavemente por todos os frames em vez de saltada a cada N), derivamos
    # um alpha por-frame a partir do alpha por-checagem original:
    #   (1 - alpha_frame) ** N == (1 - alpha_checagem)
    detect_interval = max(config.FACE_DETECT_EVERY_N_FRAMES, 1)
    alpha_per_check = config.FACE_SMOOTHING_ALPHA
    alpha = 1.0 - (1.0 - alpha_per_check) ** (1.0 / detect_interval)
    write_error = None
    frame_idx = 0
    out_idx = 0      # quadros efetivamente escritos (linha do tempo final)
    seg_i = 0        # trecho de keep_segments em que estamos
    smooth_breath = 0.0
    _breath_alpha = getattr(config, "ENERGY_BREATHING_SMOOTHING", 0.97)
    # mínimo de frames que um modo deve ser mantido antes de poder trocar —
    # evita flicker face↔wide quando a detecção de rosto é esporádica.
    mode_min_hold_frames = max(int(round(
        getattr(config, "MODE_MIN_HOLD_SECONDS", 2.5) * fps)), 1)
    mode_hold_remaining = 0

    # --- detecção de corte de câmera ---
    _cut_detect_enabled = getattr(config, "SCENE_CUT_DETECT_ENABLED", True)
    _cut_threshold = float(getattr(config, "SCENE_CUT_THRESHOLD", 28.0))
    _cut_detect_w = int(getattr(config, "SCENE_CUT_DETECT_WIDTH", 160))
    _cut_burst_total = int(getattr(config, "SCENE_CUT_BURST_FRAMES", 20))
    _prev_cut_gray: Optional[np.ndarray] = None
    _burst_remaining = 0  # frames restantes em modo "detecta todo frame"
    frames_since_cut = 10 ** 9  # troca de modo logo depois de um corte = corte seco
    # janela depois de um corte em que a troca de modo ainda conta como parte
    # do corte (seca, sem hold): ~1s, tempo pro detector achar todos os
    # rostos do plano novo
    cut_window_frames = max(_cut_burst_total, int(round(fps)))

    frame_q: "queue.Queue" = queue.Queue(maxsize=_READ_AHEAD_FRAMES)
    reader_thread = threading.Thread(
        target=_reader_thread_fn, args=(reader, frame_bytes, frame_q), daemon=True,
    )
    reader_thread.start()

    wide_frames = 0   # há quantos quadros estamos no plano aberto atual (Ken Burns)
    push_rate = max(getattr(config, "WIDE_PUSH_IN_PER_SECOND", 0.0), 0.0) / max(fps, 1.0)
    push_max = max(getattr(config, "WIDE_PUSH_IN_MAX", 1.0), 1.0)
    mode = "face"  # "face" | "wide" | "screen" — equivalente a in_fallback=False no início
    blend_remaining = 0  # frames restantes de transição suave entre modos
    streamer_speaking_state = False  # histerese do sinal "streamer falando agora" (item 17)

    try:
        while frame_idx < max_frames:
            raw = frame_q.get()
            if raw is None:
                break
            if keep_segments is not None:
                t_src = frame_idx / fps
                while seg_i < len(keep_segments) and t_src >= keep_segments[seg_i][1] - 1e-6:
                    seg_i += 1
                if seg_i >= len(keep_segments) or t_src < keep_segments[seg_i][0] - 1e-6:
                    frame_idx += 1  # quadro dentro de uma pausa cortada
                    continue
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(src_h, src_w, 3)
            t = out_idx / fps  # tempo no vídeo FINAL (= tempo do áudio final)
            frames_since_cut += 1
            tracker.sample_motion(frame, t)
            if screen_tracker is not None:
                screen_tracker.sample_activity(frame)

            # --- detecção de corte de câmera ---
            _cut_this_frame = False
            if _cut_detect_enabled:
                _dw = _cut_detect_w
                _dh = max(int(src_h * _dw / src_w), 1)
                _small = cv2.resize(frame, (_dw, _dh), interpolation=cv2.INTER_AREA)
                _gray = cv2.cvtColor(_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                if _prev_cut_gray is not None:
                    _diff = float(np.mean(np.abs(_gray - _prev_cut_gray)))
                    if _diff > _cut_threshold:
                        _cut_this_frame = True
                        _burst_remaining = max(_burst_remaining, _cut_burst_total)
                        mode_hold_remaining = 0  # permite mudar de modo imediatamente
                        # descarta todos os slots do tracker — as posições
                        # acumuladas são do enquadramento anterior e fariam
                        # choose_active() retornar a posição errada (ex: mão)
                        # no novo enquadramento.
                        tracker.on_scene_cut()
                        face_confidence = 0.0
                        frames_since_cut = 0
                        subject_face_frac = None
                        group_shot = False
                        target_x = float(src_w / 2)
                        target_y = float(src_h / 2)
                _prev_cut_gray = _gray

            _should_detect = (
                frame_idx % config.FACE_DETECT_EVERY_N_FRAMES == 0
                or _cut_this_frame
                or _burst_remaining > 0
            )
            if _burst_remaining > 0:
                _burst_remaining -= 1

            if _should_detect:
                _in_burst = _burst_remaining > 0 or _cut_this_frame
                # No PRIMEIRO frame após um corte usa busca escalona da:
                # tenta o frame inteiro (metade superior apenas), depois
                # metade esquerda e metade direita se não achar nada —
                # é o "pergunta 'tem rosto aqui?' e tenta encontrar neste
                # mesmo frame se não achar de cara" do ig.
                if _cut_this_frame:
                    # 1º frame pós-corte: busca escalona da (frame inteiro
                    # metade superior → metade esq → metade dir).
                    faces = _detect_faces_on_cut(cascades, frame)
                elif _in_burst:
                    # Frames de burst subsequentes: sensitive + filtro y
                    # restritivo para não grudar em falsos positivos de mesa.
                    burst_y = float(getattr(config, "CUT_FACE_MAX_Y_FRAC", 0.55))
                    faces = _detect_all_faces(cascades, frame, sensitive=True,
                                              _max_y_frac=burst_y)
                else:
                    faces = _detect_all_faces(cascades, frame, sensitive=False)
                tracker.update_detections(faces, frame_idx)
                if screen_tracker is not None:
                    screen_tracker.update_detection(_detect_screen_region(frame))
                face_center = tracker.choose_active()
                if face_center is not None:
                    jump_dist = ((face_center[0] - smoothed_x) ** 2
                                 + (face_center[1] - smoothed_y) ** 2) ** 0.5
                    confirmed_cut = False
                    if jump_dist > cut_jump_threshold and last_detected_xy is not None:
                        prev_dist = ((face_center[0] - last_detected_xy[0]) ** 2
                                     + (face_center[1] - last_detected_xy[1]) ** 2) ** 0.5
                        confirmed_cut = prev_dist < cut_confirm_tolerance
                    last_detected_xy = face_center

                    # só atualiza o ALVO aqui (posição crua detectada); a
                    # câmera suavizada (smoothed_x/y) caminha em direção a
                    # esse alvo a CADA frame, fora deste bloco, para que o
                    # movimento seja contínuo mesmo detectando só a cada N
                    # frames.
                    target_x, target_y = face_center
                    # burst pós-corte: snapa imediatamente sem esperar EMA
                    # — o rosto encontrado no burst É o novo enquadramento certo.
                    if _in_burst:
                        smoothed_x, smoothed_y = target_x, target_y
                        face_confidence = 1.0
                    else:
                        # EMA normal fora de burst: sobe gradualmente pra evitar
                        # wide→face flicker por detecções espúrias.
                        face_confidence = min(1.0, face_confidence * 0.65 + 0.35)
                        if confirmed_cut:
                            smoothed_x, smoothed_y = target_x, target_y

                    # facecam pequena (modo REACT, item 14): recalcula o
                    # alvo de altura do crop a partir do tamanho REAL do
                    # rosto detectado nesta checagem.
                    face_size = tracker.active_face_size()
                    if face_size is not None and face_size[1] > 0 and subject_framing:
                        target_crop_h = float(np.clip(face_size[1] / subject_target_frac,
                                                      subject_min_crop_h, crop_h))
                        out_frac = face_size[1] / target_crop_h
                        if subject_face_frac is None or _in_burst:
                            subject_face_frac = out_frac
                        else:
                            subject_face_frac = 0.7 * subject_face_frac + 0.3 * out_frac
                        # plano de GRUPO (várias pessoas, rostos pequenos):
                        # qualquer recorte corta alguém no meio -> fit.
                        # Fica marcado até o próximo corte de câmera, pra
                        # alguém se inclinar pra frente não trocar o layout
                        # no meio do plano.
                        if (tracker.group_face_count() >= 2
                                and subject_face_frac < fit_group_frac):
                            group_shot = True
                        # uma pessoa só: recorte com zoom, a não ser que ela
                        # esteja tão longe que nem o zoom máximo resolva
                        subject_too_small = group_shot or subject_face_frac < fit_single_frac
                    elif face_size is not None and face_size[1] > 0:
                        frac = face_size[1] / src_h
                        if frac < facecam_small_frac:
                            scale = facecam_small_frac / frac
                            target_crop_h = float(np.clip(crop_h * scale, crop_h, src_h))
                        else:
                            target_crop_h = float(crop_h)
                    if _in_burst:
                        # novo plano: o tamanho do crop também muda na hora,
                        # junto com a posição (senão o zoom "respira" por ~1s
                        # depois de cada corte de câmera)
                        smoothed_crop_h = target_crop_h
                else:
                    # sem rosto detectado nesta checagem: NÃO puxa o alvo de
                    # volta pro centro (era o bug original — ao perder o
                    # rosto por um instante, o crop pulava pro meio da
                    # imagem, quase sempre mostrando só mesa/fundo). Mantém
                    # o último alvo conhecido e só perde confiança aos
                    # poucos, até (se durar o bastante) trocar pro modo de
                    # plano aberto.
                    face_confidence *= confidence_decay

            # suaviza em direção ao alvo TODO frame (não só nas checagens de
            # detecção) — é isso que faz a câmera virtual se mover de forma
            # contínua em vez de ficar parada por N frames e pular de uma
            # vez quando a detecção roda de novo.
            smoothed_x = (1 - alpha) * smoothed_x + alpha * target_x
            smoothed_y = (1 - alpha) * smoothed_y + alpha * target_y
            smoothed_crop_h = (1 - alpha) * smoothed_crop_h + alpha * target_crop_h
            # deriva a largura do crop dinâmico mantendo a proporção 9:16,
            # com a mesma folga de "não passar da fonte" já usada no
            # cálculo original de crop_w/crop_h.
            dyn_crop_h = smoothed_crop_h
            dyn_crop_w = dyn_crop_h * out_w / out_h
            if dyn_crop_w > src_w:
                dyn_crop_w = src_w
                dyn_crop_h = dyn_crop_w * out_h / out_w

            # decide o modo deste frame: "screen" (vídeo-dentro-do-vídeo
            # confirmado) tem prioridade sobre o rastreamento de rosto — ver
            # RELATORIO_PROXIMOS_PASSOS.txt item 3a; caindo pra
            # "face"/"wide" pela mesma histerese de confiança de antes
            # quando não há tela ativa.
            screen_region = screen_tracker.active_region() if screen_tracker else None

            # zoom de referência visual (item 14): força o modo "screen" por
            # uma janela breve depois de uma frase tipo "olha a camisa
            # dele", MESMO que o vídeo reagido esteja pausado/parado nesse
            # momento (por isso usa last_known_region, não active_region) —
            # só tem efeito se alguma tela já foi confirmada neste clipe em
            # algum momento; do contrário não há nada pra apontar. Isso é
            # um apontamento EXPLÍCITO do comentador — vence qualquer outra
            # decisão do bloco de troca abaixo.
            screen_confirmed_active = screen_region is not None
            reference_forced = False
            if screen_region is None and screen_tracker is not None and reference_times:
                hold = getattr(config, "REFERENCE_ZOOM_HOLD_SECONDS", 2.5)
                if any(rt <= t <= rt + hold for rt in reference_times):
                    screen_region = screen_tracker.last_known_region()
                    reference_forced = screen_region is not None

            prev_mode = mode
            if reference_forced:
                mode = "screen"
            elif screen_tracker is not None:
                # MODO REACT (item 17, pedido do usuário): decide entre focar
                # na TELA reagida, no STREAMER, ou afastar e mostrar os dois
                # ("full zoom out"), combinando dois sinais já calculados por
                # frame em outro lugar (nenhum sinal novo caro foi
                # adicionado): (a) o streamer está com atividade de boca
                # agora (`active_speaking_activity`, mesmo placar usado pra
                # trocar de rosto entre vários falantes); (b) a tela reagida
                # tem atividade de pixel confirmada agora (`active_region`,
                # já usado antes só como prioridade incondicional). A
                # atividade de pixel da tela, por ser genérica (qualquer
                # mudança de imagem sustentada), cobre tanto "a pessoa no
                # vídeo está falando" quanto "algo relevante aconteceu (ex.:
                # uma batida de carro)" sem precisar de lógica separada pra
                # cada caso — os dois se manifestam do mesmo jeito (pixel
                # mudando bastante e por tempo suficiente pra confirmar).
                #
                # Histerese (limiar de entrada mais alto que o de saída)
                # só no sinal NOVO (streamer falando), mesma técnica já
                # usada em face_confidence (bandas 0.35/0.6) — evita
                # cravar wide/screen a cada oscilação pequena bem em cima
                # do limiar.
                speak_enter = getattr(config, "REACT_STREAMER_SPEAKING_ACTIVITY_MIN", 3.0)
                speak_exit = speak_enter * 0.6
                speak_now = tracker.active_speaking_activity()
                if speak_now >= speak_enter:
                    streamer_speaking_state = True
                elif speak_now < speak_exit:
                    streamer_speaking_state = False
                streamer_speaking = streamer_speaking_state and face_confidence > 0.35

                if screen_confirmed_active and not streamer_speaking:
                    # streamer calado + algo tocando/acontecendo na tela ->
                    # foca no vídeo/pessoa reagida.
                    mode = "screen"
                elif screen_confirmed_active and streamer_speaking:
                    # os dois "vivos" ao mesmo tempo (streamer comentando em
                    # cima do vídeo que também está tocando) -> afasta e
                    # mostra tudo, em vez de brigar entre os dois focos.
                    mode = "wide"
                elif streamer_speaking:
                    mode = "face"
                else:
                    # nem tela nem streamer claramente ativos agora -- cai
                    # na mesma histerese de confiança de rosto de antes,
                    # sem tela envolvida na decisão.
                    if mode == "screen":
                        mode = "face" if face_confidence >= 0.35 else "wide"
                    if face_confidence < 0.35:
                        mode = "wide"
                    elif face_confidence > 0.6:
                        mode = "face"
            else:
                # vídeo comum, sem nenhuma tela detectada neste clipe.
                # MODE_MIN_HOLD: só permite trocar de modo quando o hold
                # expirou — elimina o flicker face↔wide causado por detecção
                # esporádica (rosto perdido por 1-2 frames).
                # Logo depois de um corte de câmera o hold não vale: o plano
                # novo pode pedir outro modo e a troca tem que acontecer
                # junto com o corte, não segundos depois.
                in_cut_window = frames_since_cut <= cut_window_frames
                if mode_hold_remaining > 0 and not in_cut_window:
                    mode_hold_remaining -= 1
                else:
                    if face_confidence < 0.35 or (subject_framing and subject_too_small):
                        mode = "wide"
                    elif face_confidence > 0.6:
                        mode = "face"
                    # banda 0.35-0.6: mantém o modo atual (histerese)
            if prev_mode != mode:
                # troca junto com um corte de câmera da fonte = corte seco
                # (crossfade ali parece erro de edição: dois rostos
                # sobrepostos); fora de corte, transição suave como antes
                at_cut = frames_since_cut <= cut_window_frames
                blend_remaining = 0 if at_cut else blend_frames_total
                mode_hold_remaining = mode_min_hold_frames

            zoom_factor = 1.0
            if keep_segments is not None and mode == "face" and seg_i % 2 == 1:
                # jump cut: trechos alternados ficam um pouco mais fechados
                zoom_factor = 1.0 + (max(getattr(config, "JUMPCUT_PUNCH_ZOOM", 1.0), 1.0) - 1.0) * float(
                    getattr(config, "FX_INTENSITY", 1.0))
            if config.ZOOM_PUNCH_ENABLED and mode == "face":
                ease_s = getattr(config, "ZOOM_PUNCH_EASE_SECONDS", 0.25)
                half_hold = getattr(config, "ZOOM_PUNCH_HOLD", 0.30) / 2.0
                best_weight = 0.0
                for pt in zoom_peak_times:
                    dt = abs(t - pt)
                    if dt <= half_hold:
                        w = 1.0
                    elif dt <= half_hold + ease_s:
                        progress = (dt - half_hold) / ease_s
                        w = 0.5 * (1.0 + np.cos(np.pi * progress))
                    else:
                        w = 0.0
                    if w > best_weight:
                        best_weight = w
                zoom_factor *= 1.0 + config.ZOOM_PUNCH_INTENSITY * best_weight
            fx_zoom = fx.zoom_at(t) if fx is not None else 1.0
            if mode == "face":
                zoom_factor = min(zoom_factor * fx_zoom, getattr(config, "FX_ZOOM_TOTAL_MAX", 1.28))
            if getattr(config, "ENERGY_BREATHING_ENABLED", False) and mode == "face" and audio_energy is not None:
                e_idx = min(int(t / audio_energy_hop), len(audio_energy) - 1)
                smooth_breath = _breath_alpha * smooth_breath + (1.0 - _breath_alpha) * float(audio_energy[e_idx])
                zoom_factor *= 1.0 + smooth_breath * getattr(config, "ENERGY_BREATHING_MAX", 0.025)

            def _compose_mode(m: str) -> np.ndarray:
                if m == "screen" and screen_region is not None:
                    return _compose_screen_frame(frame, screen_region, src_w, src_h, out_w, out_h)
                if m == "face":
                    return _compose_tracked_frame(
                        frame, smoothed_x, smoothed_y, dyn_crop_w, dyn_crop_h, src_w, src_h,
                        out_w, out_h, zoom_factor,
                    )
                # modo "wide": se já detectamos algum rosto neste clipe, mantém
                # o crop NORMAL de rosto ancorado na última posição conhecida
                # (a câmera "congela" onde o rosto foi visto por último, em
                # vez de pular pro letterbox vazio do centro da mesa).
                # Só vai pro letterbox completo se NUNCA houve rosto neste clip.
                if (getattr(config, "WIDE_MODE_MEDIUM_ENABLED", True)
                        and last_detected_xy is not None):
                    return _compose_tracked_frame(
                        frame, smoothed_x, smoothed_y, dyn_crop_w, dyn_crop_h, src_w, src_h,
                        out_w, out_h, 1.0,
                    )
                push = min(1.0 + push_rate * wide_frames, push_max)
                push *= 1.0 + (fx_zoom - 1.0) * 0.6  # momentos-chave também no plano aberto
                focus = smoothed_x if last_detected_xy is not None else None
                return _compose_wide_frame(frame, src_w, src_h, out_w, out_h,
                                           push=push, focus_x=focus)

            # Ken Burns: conta o tempo no plano aberto atual; zera ao sair
            # dele ou num corte de câmera (plano novo começa sem zoom)
            if mode == "wide" and frames_since_cut > 0:
                wide_frames += 1
            else:
                wide_frames = 0

            if blend_remaining > 0:
                # transição suave (crossfade) entre os dois modos envolvidos
                # na troca, pra não dar um "pulo" visual — generaliza o
                # crossfade binário original pra qualquer par de modos
                # (face/wide/screen), fazendo o modo ANTERIOR desaparecer
                # enquanto o NOVO aparece.
                old_frame = _compose_mode(prev_mode)
                new_frame = _compose_mode(mode)
                mix = blend_remaining / blend_frames_total  # 1 -> 0 (peso do modo antigo)
                out_frame = cv2.addWeighted(old_frame, mix, new_frame, 1.0 - mix, 0)
                blend_remaining -= 1
            else:
                out_frame = _compose_mode(mode)

            if fx is not None:
                out_frame = fx.apply_frame_effects(out_frame, t, out_h)

            try:
                writer.stdin.write(out_frame.tobytes())
            except (BrokenPipeError, OSError) as e:
                write_error = e
                break

            frame_idx += 1
            out_idx += 1
    finally:
        # drena a fila pra thread leitora não ficar bloqueada num put() se
        # saímos do loop antes do fim do stream (erro de escrita etc.)
        while True:
            try:
                frame_q.get_nowait()
            except queue.Empty:
                break
        reader_thread.join(timeout=5)
        try:
            reader.stdout.close()
        except Exception:
            pass
        reader_err = reader.stderr.read() if reader.stderr else b""
        reader.wait()
        try:
            writer.stdin.close()
        except Exception:
            pass
        writer_err = writer.stderr.read() if writer.stderr else b""
        writer.wait()

    if reader.returncode != 0 and frame_idx == 0 and write_error is None and writer.returncode == 0:
        # a leitura da origem falhou por conta própria (ex: timestamp de
        # início além da duração do arquivo) — só é a causa raiz quando o
        # escritor terminou bem; se o escritor também falhou, o erro do
        # leitor abaixo é só um efeito colateral (pipe de saída sem ninguém
        # lendo do outro lado) e não a causa real.
        raise RuntimeError(
            f"ffmpeg falhou ao ler o trecho de origem ({reader.returncode}): "
            f"{reader_err.decode(errors='ignore')[-4000:]}"
        )

    hw_failed_mid_stream = write_error is not None or writer.returncode != 0
    if hw_failed_mid_stream and hwaccel.is_hardware_active() and not force_cpu:
        # o encode de hardware falhou no meio do processo (raro, já que a
        # detecção testa antes de começar) — refaz o clipe inteiro em CPU
        print("    [aviso] encode por GPU falhou durante o reenquadramento, refazendo via CPU...")
        hwaccel.mark_hw_failed()
        return render_vertical_clip(
            source_path, start, end, src_w, src_h, fps, output_path,
            ass_path=ass_path, audio_path=audio_path, fonts_dir=fonts_dir,
            zoom_peak_times=zoom_peak_times, force_cpu=True,
            audio_energy=audio_energy, audio_energy_hop=audio_energy_hop,
            reference_times=reference_times, keep_segments=keep_segments, fx=fx,
        )

    if hw_failed_mid_stream:
        raise RuntimeError(
            f"ffmpeg falhou ao gerar o vídeo final ({writer.returncode}): "
            f"{writer_err.decode(errors='ignore')[-4000:]}"
        )

    return output_path
