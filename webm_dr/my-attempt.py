
from concurrent.futures import thread
from enum import Enum
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
from sys import stderr
import threading
from time import time_ns

class ModeEnum(Enum):
    RANDOM = 1
    GROWING = 2
    SQUASH = 3
    SPECIAL = 4

class WebmDoctor:
  def __init__(self, mode: int, input_path: os.PathLike, output_path: os.PathLike):
    random.seed(time_ns())
    self.base = Path(__file__).resolve().parent.parent
    self.mode = ModeEnum(mode)
    self.input_path = Path(input_path).resolve()
    self.fps = 0
    self.fps_re = re.compile(r"(\d+\.\d+|\d+) fps")
    self.threads = list[threading.Thread]

    self.output_path = Path(output_path).resolve()
    if self.output_path.is_dir():
        os.makedirs(self.output_path, exist_ok=True)

    self.temp = self.base / "tmp"
    os.makedirs(self.temp, exist_ok=True)
    self.concat_path = self.temp / "concat.txt"
    self.audio_concat_path = self.temp / "audio_concat.txt"

  def __call__(self):
    self.audio_extract_thread = threading.Thread(target=self.extract_audio)
    self.frame_extract_thread = threading.Thread(target=self.extract_frames)

    self.audio_extract_thread.start()
    self.frame_extract_thread.start()

  def extract_frames(self):
    out_path = self.temp / "out%04d.png"
    cmd = subprocess.Popen(
      ["ffmpeg", "-y", "-hide_banner", "-i", str(self.input_path), str(out_path)],
      text=True,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
    )
    for line in cmd.stdout:
      if self.fps == 0:
        if (l_ := re.sub(r"^\s+", "", line).lower()).startswith("stream"):
          match = self.fps_re.search(l_)
          if match is not None:
            self.fps = match.groups()[0]
            print(float(self.fps))

  def extract_audio(self):
    out_path = self.temp / "extracted.ogg"
    subprocess.run(
      ["ffmpeg", "-y", "-hide_banner", "-i", str(self.input_path), "-f", "ogg", "-ab", "192000", "-vn", str(out_path)],
      text=True,
      stdout=subprocess.DEVNULL,
    )
    print(f"Audio extracted at {str(out_path)}")

webm_input = Path("webm_dr/mannn.webm").resolve()
webm_output = Path("webm_dr/mannnnn.webm").resolve()


# Initialize class
webm_dr = WebmDoctor(4, webm_input, webm_output)
try:
  webm_dr()
except Exception as e:
  print(e)
# finally:
#   if webm_dr.temp.exists():
    # shutil.rmtree(webm_dr.temp)
