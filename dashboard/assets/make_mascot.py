"""Draw the dashboard mascot ("Revi", a little helmeted captain) and render it in 1-bit Atkinson dither.

The character is drawn procedurally with simple 3D shading (a lit sphere for the head, a helmet dome, a visor),
then dithered to black and white like an old Macintosh screen. Frames: idle, blink, talk; plus an animated GIF.
Run:  python dashboard/assets/make_mascot.py   ->  dashboard/assets/mascot_*.png, mascot.gif
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
N, SCALE = 192, 3                      # drawing grid (px) and nearest-neighbour upscale for crisp pixels
INK, PAPER = (15, 61, 53), (253, 243, 236)  # forest-green ink on warm cream (dashboard palette)
LIGHT = np.array([-0.45, -0.6, 0.66]); LIGHT /= np.linalg.norm(LIGHT)

yy, xx = np.mgrid[0:N, 0:N].astype(float)


def sphere(cx, cy, r):
    """Mask and lit shading (0..1) of a sphere seen from the front."""
    dx, dy = (xx - cx) / r, (yy - cy) / r
    d2 = dx ** 2 + dy ** 2
    mask = d2 <= 1
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    lambert = np.clip(dx * LIGHT[0] + dy * LIGHT[1] + nz * LIGHT[2], 0, 1)
    return mask, lambert, nz


def ellipse(cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def draw(eyes="open", mouth="smile") -> np.ndarray:
    """Grayscale image, 1 = paper, 0 = ink."""
    img = np.ones((N, N))

    # shoulders / t-shirt
    body = ellipse(96, 196, 66, 46)
    shade = 0.62 + 0.25 * np.clip(-(xx - 96) / 90, -1, 1) - 0.12 * ((yy - 160) / 40)
    img[body] = np.clip(shade, 0.25, 0.9)[body]
    collar = body & (np.abs(xx - 96) < (yy - 150) * 0.55) & (yy < 176)
    img[collar] = 0.95
    neck = (np.abs(xx - 96) < 13) & (yy > 112) & (yy < 156)
    img[neck] = 0.72 - 0.25 * ((yy[neck] - 112) / 44)

    # ears
    for ex in (52, 140):
        m, lam, _ = sphere(ex, 104, 10)
        img[m] = (0.45 + 0.5 * lam)[m] * 0.92

    # head (skin)
    head, lam, nz = sphere(96, 96, 44)
    img[head] = np.clip(0.58 + 0.45 * lam, 0, 1)[head] * 0.96     # light, gently shaded skin

    # helmet dome (upper part of a slightly bigger sphere): dark and glossy, one small highlight
    helm, hlam, hnz = sphere(96, 92, 50)
    dome = helm & (yy < 82 + 0.10 * np.abs(xx - 96))
    shine = np.exp(-(((xx - 76) / 7) ** 2 + ((yy - 56) / 4.5) ** 2))
    img[dome] = np.clip(0.08 + 0.32 * hlam + 0.85 * shine, 0, 0.92)[dome]
    stripe = dome & (np.abs(xx - 96) < 5)                       # racing stripe
    img[stripe] = np.clip(0.45 + 0.25 * hlam, 0, 1)[stripe]

    # visor band
    visor = helm & (yy >= 76 + 0.10 * np.abs(xx - 96)) & (yy < 86 + 0.10 * np.abs(xx - 96))
    img[visor] = 0.06
    glint = visor & (np.abs(yy - (79 + 0.10 * np.abs(xx - 96))) < 1) & (xx > 70) & (xx < 92)
    img[glint] = 0.9

    # eyes
    for ex in (78, 114):
        if eyes == "open":
            img[ellipse(ex, 102, 5.5, 7.5)] = 0.0
            img[ellipse(ex - 1.8, 99.5, 1.8, 2.2)] = 1.0     # sparkle
        else:
            img[ellipse(ex, 103, 6.5, 1.3)] = 0.0

    # cheeks (soft darker blush, becomes a dotted texture after dithering)
    for cx in (68, 124):
        blush = ellipse(cx, 115, 7, 4) & head
        img[blush] = np.clip(img[blush] - 0.10, 0, 1)

    # nose
    img[ellipse(96, 110, 3, 2.2)] = np.clip(img[ellipse(96, 110, 3, 2.2)] - 0.3, 0, 1)

    # mouth
    if mouth == "smile":
        ring = np.abs(np.hypot(xx - 96, yy - 112) - 13) < 1.6
        img[ring & (yy > 118)] = 0.0
    else:  # talk: open mouth with a lighter tongue
        img[ellipse(96, 124, 9, 7)] = 0.0
        img[ellipse(96, 128, 5, 2.6)] = 0.55
    return img


def atkinson(gray: np.ndarray) -> np.ndarray:
    """1-bit Atkinson dithering (spreads 6/8 of the error to 6 neighbours: crisp, high-contrast)."""
    g = gray.copy()
    h, w = g.shape
    out = np.zeros_like(g)
    for y in range(h):
        for x in range(w):
            new = 1.0 if g[y, x] >= 0.5 else 0.0
            err = (g[y, x] - new) / 8
            out[y, x] = new
            for dx, dy in ((1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)):
                if 0 <= x + dx < w and 0 <= y + dy < h:
                    g[y + dy, x + dx] += err
    return out


def outline(mask: np.ndarray) -> np.ndarray:
    """1-px border of a mask (pixels inside it that touch the outside)."""
    inner = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(np.roll(mask, dy, 0), dx, 1)
    return mask & ~inner


def silhouette_lines() -> np.ndarray:
    """Cartoon outline: whole silhouette, plus the head edge so the chin separates from the neck."""
    head, _, _ = sphere(96, 96, 44)
    helm, _, _ = sphere(96, 92, 50)
    dome = helm & (yy < 86 + 0.10 * np.abs(xx - 96))
    ears = sphere(52, 104, 10)[0] | sphere(140, 104, 10)[0]
    neck = (np.abs(xx - 96) < 13) & (yy > 112) & (yy < 156)
    body = ellipse(96, 196, 66, 46)
    every = head | dome | ears | neck | body
    return outline(every) | (outline(head) & (yy > 96)) | (outline(body) & (yy < 190))


def to_image(bits: np.ndarray) -> Image.Image:
    rgb = np.where(bits[..., None] > 0.5, PAPER, INK).astype(np.uint8)
    return Image.fromarray(rgb).resize((N * SCALE, N * SCALE), Image.NEAREST)


def main() -> None:
    lines = silhouette_lines()
    base = atkinson(draw())
    base[lines] = 0.0
    face = ellipse(96, 112, 34, 22)                  # eyes + mouth region: only these pixels change between frames
    frames = {"idle": base}
    for name, kw in (("blink", {"eyes": "closed"}), ("talk", {"mouth": "talk"})):
        bits = atkinson(draw(**kw))
        bits[lines] = 0.0
        frames[name] = np.where(face, bits, base)     # keep the rest identical so the dots don't shimmer
    for name, bits in frames.items():
        to_image(bits).save(OUT / f"mascot_{name}.png")

    seq = [("idle", 2200), ("blink", 120), ("idle", 1400), ("talk", 160), ("idle", 140), ("talk", 160), ("idle", 900)]
    imgs = [to_image(frames[n]).convert("P", palette=Image.ADAPTIVE, colors=2) for n, _ in seq]
    imgs[0].save(OUT / "mascot.gif", save_all=True, append_images=imgs[1:], duration=[d for _, d in seq], loop=0)
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("mascot*"))))


if __name__ == "__main__":
    main()
