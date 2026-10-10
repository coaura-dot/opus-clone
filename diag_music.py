"""Mostra quais músicas o programa está enxergando e onde cada uma começa.

    python diag_music.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src import config, music

dirs = music._music_dirs()
print("Pastas de música:", ", ".join(str(d) for d in dirs) or "(nenhuma)")
if not getattr(config, "MUSIC_USE_NCS", False):
    print("(biblioteca NCS desligada -- MUSIC_USE_NCS = False em src/config.py)")
tracks = music.list_usable_tracks()
for t in tracks:
    m, s = divmod(int(t.drop_seconds), 60)
    print(f"  [começa em {m}:{s:02d}]  {t.path.parent.name}/{t.path.name}")
print(f"\nTotal: {len(tracks)} música(s)." if tracks else
      "\nNenhuma música -- os clipes vão sair só com a voz. Coloque arquivos em assets/music/.")
