#!/usr/bin/env python3
"""
Auto Clipper -- PILOTO AUTOMÁTICO
=================================
Baixa, edita e POSTA no YouTube sozinho: título, descrição, hashtags, tudo.

    interface.py (AUTO_CLIPPER.bat)  janela com botões: ligar/desligar, modo da GPU, painel e log
    python autopilot.py              menu
    python autopilot.py --auto       modo automático infinito (1 clique: INICIAR_AUTOMATICO.bat)
    python autopilot.py --url LINK   um vídeo: edita e posta os clipes
    python autopilot.py --login      conecta o seu canal do YouTube (uma vez)
    python autopilot.py --status     mostra fila, postados e cota do dia
    python autopilot.py --liberar    confere agora se a auditoria saiu e solta 1 vídeo privado
    --no-upload                      gera os clipes mas não posta (teste)
    --gpu 25|50|70|100               limite de uso da GPU (fica salvo; 100 = sem limite)

Configuração: src/config.py, seção "POSTAGEM AUTOMÁTICA".
Primeira vez: siga o passo a passo do README ("Postagem automática").
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# aberto sem console (ex.: pela interface no pythonw) o Windows pode entregar
# uma saída "quebrada": o 1º print derrubava o piloto (v30). Testa e, se
# estiver quebrada, segue sem tela -- o log vai pro arquivo de qualquer jeito.
for _name in ("stdout", "stderr"):
    try:
        getattr(sys, _name).write("\n")
        getattr(sys, _name).flush()
    except Exception:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

from src import config
from src import autopilot
from src import throttle


def _login():
    from src import youtube_uploader as yt
    try:
        svc = yt.get_service(interactive=True)
    except yt.UploadError as e:
        print(f"\n[ERRO] {e}")
        return False
    name, problem = yt.channel_check(svc)
    if problem:
        print(f"\n[!] Login feito, mas: {problem}")
        return False
    print(f"\nConectado ao canal: {name}")
    print(f"Login salvo em {yt.token_path()} -- não precisa repetir.")
    return True


def _status():
    st = autopilot.State(autopilot.DATA_DIR / "state.json")
    q, posted = st.data["queue"], st.data["posted"]
    print(f"\nFila pra postar: {len(q)} clipe(s)")
    for c in sorted(q, key=lambda c: (-c.get("quality", 50), -c.get("score", 0)))[:10]:
        try:
            title = json.loads(Path(c["meta"]).read_text(encoding="utf-8"))["title"]
        except (OSError, ValueError, KeyError):
            title = Path(c["video"]).name
        nota = f"nota {c['quality']}" if "quality" in c else "sem nota ainda"
        print(f"  • [{nota}] {title}")
    rej = st.data.get("rejected", [])
    if rej:
        print(f"\nDescartados pela nota de qualidade: {len(rej)} (últimos 5)")
        for r in rej[-5:]:
            print(f"  • [nota {r['quality']}] {r['title']} -- {', '.join(r.get('reasons', []))}")
    print(f"\nPostados: {len(posted)} (hoje: {st.uploads_today()}/{autopilot.daily_limit()})")
    for p in posted[-10:]:
        print(f"  • {datetime.fromtimestamp(p['at']):%d/%m %H:%M}  https://youtube.com/shorts/{p['youtube_id']}"
              f"  {p['title']}")
    priv = [p for p in posted if p.get("privacy") not in (None, "public")
            and not p.get("replaced") and not p.get("gave_up")]
    if priv:
        rel = st.data.get("release", {})
        print(f"\nPrivados esperando a auditoria do YouTube: {len(priv)} "
              f"({'aprovada -- liberando aos poucos' if rel.get('approved') else 'ainda não aprovada'})")
    used = [s for s in st.data["sources"].values() if s.get("status") == "done"]
    print(f"\nVídeos já usados: {len(used)}")


def _release_now():
    from src import release
    from src import youtube_uploader as yt
    st = autopilot.State(autopilot.DATA_DIR / "state.json")
    log = autopilot.Log(autopilot.DATA_DIR / "autopilot.log")
    try:
        svc = yt.get_service(interactive=True)
    except yt.UploadError as e:
        print(f"\n[ERRO] {e}")
        return
    if not release._pending(st) and st.data["posted"]:
        release.refresh_unknown(st, svc)
    if not release._pending(st):
        print("\nNenhum vídeo privado esperando liberação.")
        return
    try:
        release.release_step(st, svc, log, force=True)
    except yt.UploadError as e:
        print(f"\n[ERRO] {e}")


def _parse_gpu(v: str):
    v = (v or "").strip().lower().rstrip("%")
    if v in ("livre", "sem", "sem limite", "irrestrito", "max", "100"):
        return 100
    return int(v) if v.isdigit() and 5 <= int(v) <= 100 else None


def _choose_gpu() -> None:
    cur = throttle.configured_limit()
    print(f"\n  Modo atual: {throttle.describe(cur)}")
    print("  1) 25% da GPU   (PC livre pra usar; edição ~4x mais lenta)")
    print("  2) 50% da GPU")
    print("  3) 70% da GPU")
    print("  4) Sem limite   (o mais rápido)")
    pick = {"1": 25, "2": 50, "3": 70, "4": 100}.get(input("\nEscolha [4]: ").strip() or "4", 100)
    throttle.save_limit(pick)
    print(f"  Salvo: {throttle.describe(pick)} (vale pro piloto automático e pros próximos inícios).")


def _ask_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw.isdigit() and 1 <= int(raw) <= 15 else default


def main():
    ap = argparse.ArgumentParser(description="Auto Clipper -- piloto automático")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--clips", type=int)
    ap.add_argument("--login", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--liberar", action="store_true")
    ap.add_argument("--gpu", default=None, help="25, 50, 70 ou 100 (sem limite)")
    args = ap.parse_args()
    upload = not args.no_upload and getattr(config, "AUTOPILOT_UPLOAD", True)
    if args.gpu is not None:
        n = _parse_gpu(args.gpu)
        if n is None:
            print(f"[!] --gpu {args.gpu}: use 25, 50, 70 ou 100. Seguindo sem limite.")
            n = 100
        throttle.save_limit(n)
    gpu = throttle.install()

    print("=" * 62)
    print("  AUTO CLIPPER — PILOTO AUTOMÁTICO (baixa, edita e posta)")
    print(f"  Modo: {throttle.describe(gpu)}")
    print("=" * 62)

    if args.login:
        _login()
        return
    if args.status:
        _status()
        return
    if args.liberar:
        _release_now()
        return
    if args.auto:
        autopilot.run_forever(upload=upload)
        return
    if args.url:
        autopilot.run_once(args.url, args.clips or getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3), upload)
        return

    print("\n  1) Colar um link: baixa, edita e posta os cortes")
    print("  2) Modo automático: procura vídeos bombando e posta sem parar")
    print("  3) Conectar / trocar o canal do YouTube")
    print("  4) Ver fila e vídeos postados")
    print("  5) Conferir agora se a auditoria saiu e liberar vídeos privados")
    print(f"  6) Modo da GPU: 25% / 50% / 70% / sem limite (agora: {throttle.describe(gpu)})")
    choice = input("\nEscolha [2]: ").strip() or "2"
    if choice == "1":
        url = input("Link do vídeo (ou caminho do arquivo): ").strip()
        if url:
            autopilot.run_once(url, _ask_int("Quantos cortes?", getattr(config, "AUTOPILOT_CLIPS_PER_VIDEO", 3)),
                               upload)
    elif choice == "3":
        _login()
    elif choice == "4":
        _status()
    elif choice == "5":
        _release_now()
    elif choice == "6":
        _choose_gpu()
        print("  Reinicie o programa pra valer agora (ou use os atalhos INICIAR_GPU_*.bat).")
    else:
        autopilot.run_forever(upload=upload)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEncerrado.")
