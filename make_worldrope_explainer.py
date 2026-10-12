"""Render the clean teaser and a separate, captioned WorldRoPE explainer.

Both videos use the original diagram renderer. Only the explainer adds the
patch-to-Gaussians story; neither animation rasterizes token features.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
SOURCE = ROOT / "ICLR2027_SplatRoPE/asset/video/make_worldrope_homepage.py"
FFMPEG = Path("/Applications/ChimeraX-1.6.1.app/Contents/bin/ffmpeg")
FONT = "/System/Library/Fonts/Avenir Next.ttc"
FPS = 24
W, H = 1280, 720
INK = (31, 48, 61)
ORANGE = (222, 104, 48)
MUTED = (105, 119, 128)


def load_renderer():
    spec = importlib.util.spec_from_file_location("worldrope_base", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The source animation uses a blue-gray backdrop and a decorative wash.
    # Remove both so the open canvas is genuinely white.
    module.BG = (255, 255, 255)
    module.WASH = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    return module


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT, size, index=0 if bold else 7)


def draw_explainer(frame: Image.Image, t: float) -> Image.Image:
    im = frame.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 127), fill="white")
    d.text((66, 14), "FROM PATCH TO", font=font(22, True), fill=ORANGE)
    title = "WORLD-CENTRIC 3D GAUSSIANS"
    title_font = font(47, True)
    if d.textbbox((0, 0), title, font=title_font)[2] > W - 132:
        title_font = font(43, True)
    d.text((63, 48), title, font=title_font, fill=INK)
    d.line((64, 125, W - 64, 125), fill=(222, 228, 230), width=2)

    d.rectangle((0, 625, W, H), fill="white")
    d.line((64, 626, W - 64, 626), fill=(222, 228, 230), width=2)
    active = 0 if t < 2.25 else 1 if t < 6.75 else 2
    labels = ("1  PATCH TOKEN", "2  3D GAUSSIAN SUPPORT", "3  CROSS-VIEW ATTENTION")
    centers = (212, 640, 1068)
    for i, (label, x) in enumerate(zip(labels, centers)):
        color = ORANGE if i == active else MUTED
        d.text((x, 645), label, font=font(21, True), fill=color, anchor="mt")
        if i == active:
            d.rounded_rectangle((x - 34, 682, x + 34, 686), radius=2, fill=ORANGE)
    return im


def encoder(path: Path):
    command = [str(FFMPEG), "-loglevel", "error", "-y", "-f", "rawvideo",
               "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def finish(proc):
    proc.stdin.close()
    error = proc.stderr.read().decode("utf-8", "replace")
    code = proc.wait()
    if code:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    base = load_renderer()
    if args.sample:
        frame = base.render(6.5)
        frame.save(ASSETS / "worldrope-clean-sample.png")
        draw_explainer(frame, 6.5).save(ASSETS / "worldrope-explainer-sample.png")
        return

    teaser = encoder(ASSETS / "hero-light.mp4")
    explainer = encoder(ASSETS / "worldrope-explainer.mp4")
    try:
        for n in range(FPS * 12):
            t = n / FPS
            frame = base.render(t)
            teaser.stdin.write(frame.tobytes())
            if n < FPS * 11:
                explained = draw_explainer(frame, t)
                explainer.stdin.write(explained.tobytes())
                if n == FPS * 7:
                    explained.save(ASSETS / "worldrope-explainer-poster.png")
            if n == FPS * 8:
                frame.save(ASSETS / "hero-light-poster.png")
            if n % 48 == 0:
                print(f"Rendered {n}/{FPS * 12} frames", flush=True)
    finally:
        finish(teaser)
        finish(explainer)


if __name__ == "__main__":
    main()
