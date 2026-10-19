"""
PNG -> collision grid.

1. Read the image as RGBA (Pillow, or Qt's QImage as a fallback).
2. A pixel blocks movement when it is DARK (every RGB channel below `dark_threshold`,
   default 20 - catches #000000 void/walls including compression noise), when it is
   TRANSPARENT (alpha below `alpha_threshold`), or when it matches an extra colour rule
   (default: pure blue water #0000FF within a tolerance of 60 per channel).
3. The image is split into cells of `cell_px` x `cell_px` pixels. cell_px=None picks it
   automatically: an image with few distinct colours is a Tibia minimap (1 px = 1 sqm),
   anything else is treated as a client screenshot (32 px = 1 sqm).
4. A cell is an obstacle (1) when at least `blocked_fraction` of its pixels block;
   everything else is walkable (0). With cell_px=1 every dark pixel is a wall tile.
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_DARK_THRESHOLD = 20
MINIMAP_MAX_COLORS = 256      # Tibia minimaps use a small fixed palette
SCREENSHOT_CELL_PX = 32


@dataclass(frozen=True)
class ColorRule:
    label: str
    rgb: tuple
    tolerance: int = 30


DEFAULT_RULES = (
    ColorRule("Woda (#0000FF)", (0, 0, 255), 60),
)


def load_rgba(path):
    """Return an (H, W, 4) uint8 array for the image at `path`."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    except ImportError:
        pass
    from PyQt5.QtGui import QImage
    image = QImage(path)
    if image.isNull():
        raise OSError(f"Cannot read image: {path}")
    image = image.convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    buffer = image.constBits()
    buffer.setsize(image.byteCount())
    rows = np.frombuffer(buffer, dtype=np.uint8).reshape(height, image.bytesPerLine())
    return rows[:, : width * 4].reshape(height, width, 4).copy()


def count_colors(rgba, limit=MINIMAP_MAX_COLORS + 1, sample=200_000):
    """Number of distinct RGB colours (stops counting past `limit`), on a pixel sample."""
    flat = rgba[..., :3].reshape(-1, 3)
    if len(flat) > sample:
        flat = flat[np.linspace(0, len(flat) - 1, sample).astype(np.int64)]
    packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    return min(len(np.unique(packed)), limit)


class MapParser:
    def __init__(self, cell_px=None, dark_threshold=DEFAULT_DARK_THRESHOLD, rules=DEFAULT_RULES,
                 blocked_fraction=0.5, alpha_threshold=128):
        if cell_px is not None and cell_px < 1:
            raise ValueError("cell_px must be >= 1 (or None for auto)")
        self.cell_px = cell_px
        self.dark_threshold = int(dark_threshold)
        self.rules = tuple(rules)
        self.blocked_fraction = float(blocked_fraction)
        self.alpha_threshold = alpha_threshold
        self.last_cell_px = None
        self.last_was_minimap = None

    @staticmethod
    def looks_like_minimap(rgba):
        return count_colors(rgba) <= MINIMAP_MAX_COLORS

    def resolve_cell_px(self, rgba):
        """Explicit cell_px, or 1 for a minimap / 32 for a screenshot."""
        self.last_was_minimap = self.looks_like_minimap(rgba)
        if self.cell_px is not None:
            return self.cell_px
        return 1 if self.last_was_minimap else SCREENSHOT_CELL_PX

    def pixel_mask(self, rgba):
        """Boolean (H, W) mask of blocking pixels."""
        rgb = rgba[..., :3]
        mask = (rgb < self.dark_threshold).all(axis=2)                     # dark = wall / void
        if rgba.shape[2] == 4:
            mask |= rgba[..., 3] < self.alpha_threshold                    # transparent = void
        if self.rules:
            signed = rgb.astype(np.int16)
            for rule in self.rules:
                diff = np.abs(signed - np.array(rule.rgb, dtype=np.int16)).max(axis=2)
                mask |= diff <= rule.tolerance
        return mask

    def to_grid(self, rgba, cell_px=None):
        """Collision grid (rows, cols) of uint8: 1 = obstacle, 0 = walkable."""
        cell = cell_px or self.resolve_cell_px(rgba)
        self.last_cell_px = cell
        rows, cols = rgba.shape[0] // cell, rgba.shape[1] // cell
        if rows == 0 or cols == 0:
            raise ValueError(f"Image {rgba.shape[1]}x{rgba.shape[0]} px is smaller than one {cell} px cell")
        mask = self.pixel_mask(rgba)[: rows * cell, : cols * cell]
        if cell == 1:
            return mask.astype(np.uint8)
        fraction = mask.reshape(rows, cell, cols, cell).mean(axis=(1, 3))
        return (fraction >= self.blocked_fraction).astype(np.uint8)

    def parse(self, path):
        return self.to_grid(load_rgba(path))
