"""Make the white WorldRoPE teaser, led by patch-to-world geometry.

The underlying camera/Gaussian sequence is the existing paper-page animation.
Projected hypotheses still converge to a smaller expected RoPE operator.
"""

from __future__ import annotations

import argparse
import math

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from make_worldrope_explainer import ASSETS, FPS, H, W, encoder, finish, load_renderer


INK = (31, 48, 61)
PLUM = (91, 76, 120)
MUTED = (99, 117, 125)
COLORS = ((47, 169, 204), (159, 116, 198), (39, 174, 151))
FONT = "/System/Library/Fonts/Avenir Next.ttc"
MATH_FONT = "/System/Library/Fonts/Supplemental/STIXTwoMath.otf"
DURATION = 18


def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT, size, index=0 if bold else 7)


def smooth(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def draw_heading(d: ImageDraw.ImageDraw) -> None:
    d.text((W / 2, 105), "FROM A 2D PATCH TOKEN", font=font(31, True), fill=INK, anchor="mm")
    d.text((W / 2, 180), "TO WORLD-CENTRIC 3D GAUSSIANS", font=font(56, True), fill=PLUM, anchor="mm")
    d.line((174, 231, 1106, 231), fill=(222, 220, 227), width=2)


def draw_support(im: Image.Image, positions) -> Image.Image:
    glow = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    gd = ImageDraw.Draw(glow)
    for x, y, rx, ry, color in positions:
        gd.ellipse((x-rx, y-ry, x+rx, y+ry), fill=(*color, 100))
    glow = glow.filter(ImageFilter.GaussianBlur(17))
    im = Image.alpha_composite(im.convert("RGBA"), glow)
    d = ImageDraw.Draw(im)
    for x, y, rx, ry, color in positions:
        d.ellipse((x-rx, y-ry, x+rx, y+ry), outline=(*color, 210), width=3)
        d.ellipse((x-rx*.62, y-ry*.62, x+rx*.62, y+ry*.62), outline=(*color, 125), width=2)
        d.ellipse((x-5, y-5, x+5, y+5), fill=(*color, 255))
    return im.convert("RGB")


def opening_frame() -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    draw_heading(d)
    # The coordinate triad makes the shared world frame explicit.
    d.line((505, 483, 987, 483), fill=(215, 214, 221), width=2)
    d.line((505, 483, 505, 277), fill=(215, 214, 221), width=2)
    d.line((505, 483, 417, 538), fill=(215, 214, 221), width=2)
    d.text((994, 484), "X", font=font(18, True), fill=MUTED)
    d.text((496, 248), "Y", font=font(18, True), fill=MUTED)
    d.text((394, 535), "Z", font=font(18, True), fill=MUTED)
    for end in ((565, 392), (700, 342), (840, 390)):
        d.line((289, 400, *end), fill=(186, 177, 198), width=2)
    d.rectangle((219, 365, 289, 435), fill=(242, 239, 246), outline=PLUM, width=4)
    d.line((254, 367, 254, 433), fill=(205, 195, 216), width=2)
    d.line((221, 400, 287, 400), fill=(205, 195, 216), width=2)
    d.rectangle((239, 385, 269, 415), fill=PLUM)
    im = draw_support(im, ((565, 392, 59, 35, COLORS[0]),
                           (700, 342, 64, 38, COLORS[1]),
                           (840, 390, 59, 35, COLORS[2])))
    d = ImageDraw.Draw(im)
    d.text((254, 490), "IMAGE PATCH", font=font(21, True), fill=MUTED, anchor="mm")
    d.text((710, 541), "SHARED 3D WORLD", font=font(24, True), fill=PLUM, anchor="mm")
    return im


def result_frame() -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    draw_heading(ImageDraw.Draw(im))
    im = draw_support(im, ((417, 390, 54, 33, COLORS[0]),
                           (522, 347, 60, 36, COLORS[1]),
                           (626, 390, 54, 33, COLORS[2])))
    d = ImageDraw.Draw(im)
    d.line((699, 382, 754, 382), fill=(161, 151, 179), width=4)
    d.polygon(((754, 382), (739, 373), (739, 391)), fill=(161, 151, 179))
    d.text((819, 382), "∫", font=ImageFont.truetype(MATH_FONT, 159), fill=PLUM, anchor="mm")
    d.text((989, 382), "RoPE", font=font(80, True), fill=INK, anchor="mm")
    d.text((911, 487), "EXPECTED ROTARY OPERATOR", font=font(19, True), fill=MUTED, anchor="mm")
    d.text((W / 2, 605), "RoPE integrated over projected 2D Gaussians", font=font(25, True), fill=PLUM, anchor="mm")
    return im


def scene_frame(frame: Image.Image) -> Image.Image:
    im = frame.copy()
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 126), fill="white")
    d.text((64, 57), "2D PATCH TOKEN", font=font(38, True), fill=INK, anchor="lm")
    d.line((414, 57, 454, 57), fill=PLUM, width=4)
    d.polygon(((454, 57), (441, 48), (441, 66)), fill=PLUM)
    d.text((474, 57), "WORLD-CENTRIC 3D GAUSSIANS", font=font(38, True), fill=PLUM, anchor="lm")
    d.line((64, 112, W - 64, 112), fill=(222, 220, 227), width=2)
    return im


def projection_frame(base) -> Image.Image:
    """Pause on the query-view distribution before integrating over it."""
    im = scene_frame(base.render(7.25))
    d = ImageDraw.Draw(im)
    d.text((1014, 143), "PROJECTED 2D GAUSSIANS", font=font(22, True), fill=PLUM, anchor="mm")
    return im


def bezier(start, control, end, u):
    v = 1.0 - u
    return (v * v * start[0] + 2 * v * u * control[0] + u * u * end[0],
            v * v * start[1] + 2 * v * u * control[1] + u * u * end[1])


def convergence_frame(t: float, base) -> Image.Image:
    """Carry the three query-view hypotheses along arrows into ∫ RoPE."""
    tau = t - 11.2
    scene = projection_frame(base)
    white = Image.new("RGB", (W, H), "white")
    # Let the camera drawing recede, while keeping the moving contributions.
    scene = Image.blend(scene, white, smooth((tau - 0.22) / 1.38)).convert("RGBA")
    arrows = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    d = ImageDraw.Draw(arrows, "RGBA")
    starts = ((950, 383), (1015, 347), (1080, 311))
    ends = ((523, 326), (523, 360), (523, 394))
    controls = ((725, 286), (760, 420), (805, 260))
    fade = 1 - smooth((tau - 1.62) / 0.60)
    for i, (start, control, end, color) in enumerate(zip(starts, controls, ends, COLORS)):
        u = smooth((tau - 0.15 - i * 0.12) / 1.18)
        if u <= 0 or fade <= 0:
            continue
        points = [bezier(start, control, end, u * j / 48) for j in range(49)]
        d.line(points, fill=(*color, round(155 * fade)), width=4, joint="curve")
        x, y = points[-1]
        # An arrowhead makes the direction of integration unambiguous.
        prev_x, prev_y = points[-3]
        angle = math.atan2(y - prev_y, x - prev_x)
        wings = []
        for sign in (-1, 1):
            a = angle + sign * 2.55
            wings.append((x + 16 * math.cos(a), y + 16 * math.sin(a)))
        d.polygon([(x, y), *wings], fill=(*color, round(230 * fade)))
        d.ellipse((x - 9, y - 9, x + 9, y + 9), fill=(*color, round(245 * fade)))
    scene = Image.alpha_composite(scene, arrows).convert("RGB")
    # The projected support remains central; expected RoPE appears as its result.
    return Image.blend(scene, result_frame(), smooth((tau - 1.63) / 0.74))


def frame_at(t: float, base) -> Image.Image:
    if t < 1.25:
        return opening_frame()
    if t < 2.05:
        return Image.blend(opening_frame(), scene_frame(base.render(0)), smooth((t - 1.25) / 0.8))
    if t < 9.3:
        return scene_frame(base.render(t - 2.05))
    if t < 11.2:
        return projection_frame(base)
    if t < 13.57:
        return convergence_frame(t, base)
    if t < 17:
        return result_frame()
    # The final encoded frame reaches the opening card exactly. The loop then
    # returns to the same image without a visible jump.
    return Image.blend(result_frame(), opening_frame(), smooth((t - 17) / (1 - 1 / FPS)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(exist_ok=True)
    poster = ASSETS / "hero-integral-light-poster.png"
    opening_frame().save(poster)
    if args.sample:
        return

    base = load_renderer()
    out = ASSETS / "hero-integral-light.mp4"
    proc = encoder(out)
    try:
        for n in range(DURATION * FPS):
            frame = frame_at(n / FPS, base)
            proc.stdin.write(frame.tobytes())
            if n % 48 == 0:
                print(f"Rendered {n}/{DURATION * FPS} frames", flush=True)
    finally:
        finish(proc)
    print(out)


if __name__ == "__main__":
    main()
