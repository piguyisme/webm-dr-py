from pathlib import Path


temp = Path(__file__).resolve().parent.parent / "tmp"
with open(temp / "thingy.txt", "w+") as f:
  for i in range(416):
    f.write(f"file out{i:04d}.ogg\n")