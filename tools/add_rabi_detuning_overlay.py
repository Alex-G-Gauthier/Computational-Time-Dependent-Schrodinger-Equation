"""Add a prominent 'Detuning ... = 0 meV (on resonance)' banner to every
frame of figures/03_rabi_isw.gif.

The banner is rendered as actual image pixels at the top of every frame,
so it survives any matplotlib title quirks and is visible in any GIF
viewer.  Re-run this script after re-executing notebook cell 18, since
``viz.make_gif`` regenerates the file from scratch.

Usage::

    python tools/add_rabi_detuning_overlay.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GIF = ROOT / "figures" / "03_rabi_isw.gif"

# Banner text uses Unicode glyphs so PIL's default font works without
# needing a math-typesetting renderer.
# Use plain ASCII subscripts ("omega_12") because most Windows fonts
# don't include the dedicated Unicode subscript glyphs (\u2081\u2082)
# and they render as tofu (boxes) when missing.
BANNER_TEXT = "Detuning frequency   \u0394 = \u0127(\u03A9 \u2212 \u03C9_12) = 0 meV    (on resonance)"
BANNER_H = 44            # pixels of extra space above each frame
BG_COLOUR = (255, 248, 220)
FG_COLOUR = (170, 0, 0)
FONT_SIZE = 19

# Try a few common Windows / cross-platform fonts that include Greek + math glyphs.
FONT_CANDIDATES = [
    "arial.ttf",
    "Arial.ttf",
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "seguisym.ttf",
    "calibri.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]


def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _strip_existing_banner(frame: Image.Image) -> Image.Image:
    """If this frame already starts with our BG_COLOUR banner, crop it
    off so re-running the overlay does not stack banners.  We probe a
    column close to the left edge (x=2) -- the banner extends full
    width but the centre column hits banner text (FG_COLOUR), so a
    centre probe would terminate prematurely.

    Treat the gray separator line that bottom-caps the banner as part
    of the banner.
    """
    w, h = frame.size
    probe_x = min(2, w - 1)
    pixels = frame.load()
    banner_rows = 0
    for y in range(min(h, 100)):
        r, g, b = pixels[probe_x, y][:3]
        is_bg = (
            abs(r - BG_COLOUR[0]) < 12
            and abs(g - BG_COLOUR[1]) < 12
            and abs(b - BG_COLOUR[2]) < 12
        )
        # Allow exactly one separator-line row: it's gray-ish and lives
        # right at the bottom of the banner.  Anything below it is the
        # original matplotlib figure and must be kept.
        is_separator = (
            banner_rows > 0
            and abs(r - 120) < 25
            and abs(g - 120) < 25
            and abs(b - 120) < 25
        )
        if is_bg or is_separator:
            banner_rows = y + 1
            if is_separator:
                # Stop right after we consume the separator -- no more banner.
                break
        else:
            break
    if banner_rows >= 5:
        return frame.crop((0, banner_rows, w, h))
    return frame


def main() -> None:
    if not GIF.exists():
        raise SystemExit(f"GIF not found: {GIF}")

    src = Image.open(GIF)
    duration = src.info.get("duration", 80)
    loop = src.info.get("loop", 0)
    n_frames = getattr(src, "n_frames", 1)

    font = _load_font()

    new_frames: list[Image.Image] = []
    for k in range(n_frames):
        src.seek(k)
        frame = src.convert("RGB")
        frame = _strip_existing_banner(frame)
        w, h = frame.size

        canvas = Image.new("RGB", (w, h + BANNER_H), BG_COLOUR)
        canvas.paste(frame, (0, BANNER_H))

        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), BANNER_TEXT, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (w - text_w) // 2 - bbox[0]
        y = (BANNER_H - text_h) // 2 - bbox[1]
        draw.text((x, y), BANNER_TEXT, fill=FG_COLOUR, font=font)
        # Thin separating line so it doesn't visually merge with the figure.
        draw.line([(0, BANNER_H - 1), (w, BANNER_H - 1)], fill=(120, 120, 120), width=1)

        new_frames.append(canvas)

    new_frames[0].save(
        GIF,
        save_all=True,
        append_images=new_frames[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
    size_kb = GIF.stat().st_size / 1024
    print(f"wrote {GIF.relative_to(ROOT)}  ({size_kb:.1f} KB, {n_frames} frames)")
    # Windows consoles default to cp1252 which can't encode Greek glyphs;
    # use the ASCII repr and round-trip via .encode/.decode so this never
    # raises UnicodeEncodeError when run from a plain `cmd.exe` shell.
    safe_banner = BANNER_TEXT.encode("ascii", errors="backslashreplace").decode("ascii")
    print(f"banner: {safe_banner!r}")


if __name__ == "__main__":
    main()
