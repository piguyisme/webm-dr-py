from concurrent.futures import process
from genericpath import exists
from math import pi, sin
import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum
from os import PathLike
from pathlib import Path
from random import SystemRandom
from time import time_ns
import time

from loguru import logger
from PIL import Image

random = SystemRandom()

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    start_time = time.time()
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        estimate = (time.time() - start_time) * total / (iteration+1) - (time.time() - start_time)
        print(f'\r{prefix} [{bar}] {percent}% {suffix} ~{round(estimate, 1)}s', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

class ModeEnum(Enum):
    RANDOM = 1
    GROWING = 2
    SQUASH = 3
    SPECIAL = 4


@logger.catch()
class WebmDynamicResolution:
    def __init__(self, mode: int, input_path: PathLike, output_path: PathLike):
        random.seed(time_ns())
        self.base = Path(__file__).resolve().parent.parent
        self.mode = ModeEnum(mode)
        self.input_path = Path(input_path).resolve()
        self.fps_re = re.compile(r"(\d+\.\d+|\d+) fps")

        self.output_path = Path(output_path).resolve()
        if self.output_path.is_dir():
            os.makedirs(self.output_path, exist_ok=True)

        self.temp = self.base / "tmp"
        os.makedirs(self.temp, exist_ok=True)
        self.concat_path = self.temp / "concat.txt"
        self.audio_concat_path = self.temp / "audio_concat.txt"

    def __call__(self):
        logger.info(f"Doctoring {self.input_path} with mode {self.mode} to {self.output_path}")
        self.extract_audio()
        frame_rate = self.extract_frames()
        frame_bases = self.get_frame_bases()
        self.resize_images(frame_bases, frame_rate)
        self.frames_to_webms(frame_bases, frame_rate)
        self.concat_webms(frame_bases)
        self.add_audio()

        logger.info(f"File created at {self.output_path}")

    def extract_audio(self):
        logger.info("Extracting audio")
        out_path = self.temp / "extracted_audio.ogg"
        cmd = subprocess.run(
          ["ffmpeg", "-i", str(self.input_path), "-f", "ogg", "-ab", "192000", "-vn", str(out_path)],
          text=True,
          stderr=subprocess.PIPE,
        )
        if cmd.returncode != 0:
            logger.error(cmd.stderr)
            sys.exit(cmd.returncode)

    def extract_frame_rate(self, out: str) -> str:
        logger.info("Reading Framerate...")
        lines = out.split("\n")
        for line in lines:
            if (l_ := re.sub(r"^\s+", "", line).lower()).startswith("stream"):
                match = self.fps_re.search(l_)
                if match is not None:
                    return match.groups()[0]
        raise ValueError("No regex match for frame rate.")

    def extract_frames(self) -> str:
        logger.info("Extracting Frames...")
        out_path = self.temp / "out%04d.png"
        cmd = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(self.input_path), str(out_path)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return self.extract_frame_rate(cmd.stdout)

    def get_frame_bases(self) -> list[Path]:
        logger.info("Extracting Frame Bases...")
        return sorted(list(Path(self.temp).glob("*.png")), key=lambda e: e.name)
        # return list(Path(self.temp).glob("*.png"))

    def resize_images(self, frame_bases: list[Path], frame_rate: int):
        logger.info("Resizing images...")
        for i, base in enumerate(progressBar(frame_bases, prefix='Progress:', suffix='Complete', length=50)):
            res_image_path = base.parent / f"{base.stem}_r{base.suffix}"
            with Image.open(base) as f:
                if i == 0:
                    x, y = f.size
                    original_x, original_y = x, y
                    shutil.copy2(base, res_image_path)
                    continue
                if self.mode == ModeEnum.RANDOM:
                    img = f.resize(
                        (random.randint(50, 1000), random.randint(50, 1000)), resample=Image.Resampling.LANCZOS
                    )
                elif self.mode == ModeEnum.GROWING:
                    x += 20
                    y += 20
                    img = f.resize((x, y), resample=Image.Resampling.LANCZOS)
                elif self.mode == ModeEnum.SQUASH:
                    x += 20
                    if y - 20 > 0:
                        y -= 20
                    else:
                        y = 1
                    img = f.resize((x, y), resample=Image.Resampling.LANCZOS)
                elif self.mode == ModeEnum.SPECIAL:
                    bounces_per_second = 1
                    # start_second
                    frames_per_bounce = float(frame_rate)/bounces_per_second
                    y = round(original_y*0.5*sin((i*pi/frames_per_bounce)-(0.5*pi))+original_y*0.5+1)
                    img = f.resize((x, y), resample=Image.Resampling.LANCZOS)

                img.save(res_image_path)

    def frames_to_webms(self, frame_bases: list[Path], frame_rate: str):
        logger.info("Frames --> Webms")
        for i, base in enumerate(progressBar(frame_bases, prefix='Progress:', suffix='Complete', length=50)):
            in_filename = base.parent / f"{base.stem}_r{base.suffix}"
            audio_in = self.temp / f"{i}.ogg"
            out_filename = base.parent / f"{base.stem}.webm"
            cmd = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-framerate",
                    frame_rate,
                    "-f",
                    "image2",
                    "-i",
                    str(in_filename),
                    "-c:v",
                    "libvpx-vp9",
                    "-pix_fmt",
                    "yuva420p",
                    str(out_filename),
                ],
                text=True,
                stderr=subprocess.PIPE,
            )
            if cmd.returncode != 0:
                logger.error(cmd.stderr)
                sys.exit(cmd.returncode)

    def concat_webms(self, frame_bases: list[Path]):
        logger.info("Concatenating webms")
        with open(self.concat_path, "w+") as f:
            for base in frame_bases:
                line = f"file {base.stem}.webm\n"
                f.write(line)
        cmd = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(self.concat_path),
                "-c",
                "copy",
                "-y",
                str(self.temp / "no_audio.webm"),
            ],
            text=True,
            stderr=subprocess.PIPE,
        )
        if cmd.returncode != 0:
            logger.error(cmd.stderr)
            sys.exit(cmd.returncode)
    def add_audio(self):
        logger.info("Adding audio")
        audio_path = self.temp / "extracted_audio.ogg"
        cmd = execute(
          [
            "ffmpeg",
            "-y",
            "-i",
            str(self.temp / "no_audio.webm"),
            "-i",
            str(audio_path),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-shortest",
            str(self.output_path)
          ]
        )
        for output in cmd:
            print(output, end="")


def cli():
    parser = ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, help="1 = random, 2 = growing, 3 = squash, 4 = special. default = 1", default=1)
    parser.add_argument("-o", "--output-path", type=str, help="Path to write output file to.")
    parser.add_argument("input_path", metavar="input_path", type=str, nargs="+", help="Path of input file.")
    args = parser.parse_args()
    if args.mode not in [e.value for e in ModeEnum]:
        raise ValueError(f"Mode must be in {list(ModeEnum)}")
    if args.output_path:
        if not args.output_path.lower().endswith(".webm"):
            raise ValueError('Output file extension must be ".webm"')
        # if exists(args.output_path[0]):

        output_path = args.output_path
    if args.input_path:
        if not exists(args.input_path[0]):
            raise ValueError("Input file doesn't exist")
    else:
        input_path = Path(args.input_path).resolve()
        output_path = input_path.parent / f"{input_path.stem}.webm"
    webm_dr = WebmDynamicResolution(mode=args.mode, input_path=args.input_path[0], output_path=output_path)
    try:
        webm_dr()
    except Exception as e:
        logger.exception(e)
    finally:
        if webm_dr.temp.exists():
            logger.info("Cleaning up tmp...")
            shutil.rmtree(webm_dr.temp)


if __name__ == "__main__":
    cli()
