import sys
sys.path.insert(0, ".")
from pathlib import Path
from src import config, music

manifest = music._load_drop_manifest()
print(f"Linhas reconhecidas no track_drops.txt: {len(manifest)}")

music_dir = Path(config.MUSIC_DIR)
candidates = (list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
              + list(music_dir.glob("*.m4a")))
print(f"Arquivos de áudio encontrados em {music_dir}: {len(candidates)}\n")

usable = 0
for path in sorted(candidates):
    drop = music._match_drop_seconds(path.stem, manifest)
    if drop is None:
        print(f"[SEM MATCH no manifesto]  {path.name}")
        continue
    real = music._is_real_audio(path)
    if not real:
        print(f"[FALHOU ffprobe/corrompido]  {path.name}")
        continue
    usable += 1
    print(f"[OK, drop={drop:.0f}s]  {path.name}")

print(f"\nTotal utilizável de verdade: {usable} de {len(candidates)}")
