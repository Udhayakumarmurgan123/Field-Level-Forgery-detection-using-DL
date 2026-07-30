"""
Cross-platform TrueType font resolution.

The project originally hardcoded Linux DejaVu font paths (the sandbox this was
built in). Real deployment machines are often Windows, so every font lookup
now goes through `find_font()`, which tries a list of known-good locations per
platform/style and falls back to PIL's built-in default font (always
available, though not scalable/bold) if nothing is found -- the app should
never crash due to a missing font.
"""
from __future__ import annotations

import platform
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# candidate paths per logical style, checked in order, across platforms
_CANDIDATES = {
    "regular": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",          # Linux (Debian/Ubuntu)
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",                    # Linux (RHEL/Fedora)
        "C:\\Windows\\Fonts\\arial.ttf",                             # Windows
        "C:\\Windows\\Fonts\\segoeui.ttf",                           # Windows fallback
        "/System/Library/Fonts/Supplemental/Arial.ttf",              # macOS
        "/System/Library/Fonts/Helvetica.ttc",                       # macOS fallback
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ],
    "oblique": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Oblique.ttf",
        "C:\\Windows\\Fonts\\ariali.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
    ],
    "mono": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ],
}


@lru_cache(maxsize=None)
def find_font_path(style: str = "regular") -> str | None:
    """Returns the first existing font path for `style`, or None if none found."""
    for candidate in _CANDIDATES.get(style, []):
        if Path(candidate).exists():
            return candidate
    return None


def load_font(style: str = "regular", size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Loads a TrueType font for the given logical style/size, falling back to
    PIL's built-in bitmap default font (fixed size, but always available) if no
    system font is found -- guarantees this never raises."""
    path = find_font_path(style)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    # last-resort fallback: PIL's bundled default font (no system dependency)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # older Pillow versions: load_default() doesn't accept a size argument
        return ImageFont.load_default()