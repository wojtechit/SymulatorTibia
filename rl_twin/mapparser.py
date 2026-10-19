"""
PNG -> collision grid.

1. Read the image as RGBA (Pillow, or Qt's QImage as a fallback).
2. Mark every pixel whose colour is within `tolerance` (max per-channel
   difference) of a blocking colour rule, or that is transparent, as blocked.
3. Split the image into cells of `cell_px` x `cell_px` pixels (1 for a Tibia
   minimap export where 1 px = 1 sqm, 32 for a client-style screenshot).
4. A cell is an obstacle (1) when at least `blocked_fraction` of its pixels are
   blocked; everything else is walkable (0).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorRule:
    label: str
    rgb: tuple
    tolerance: int = 30


DEFAULT_RULES = (
    ColorRule("Pustka / ściana (#000000)", (0, 0, 0), 30),
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


class MapParser:
    def __init__(self, rules=DEFAULT_RULES, cell_px=32, blocked_fraction=0.5, alpha_threshold=128):
        if cell_px < 1:
            raise ValueError("cell_px must be >= 1")
        self.rules = tuple(rules)
        self.cell_px = int(cell_px)
        self.blocked_fraction = float(blocked_fraction)
        self.alpha_threshold = alpha_threshold

    def pixel_mask(self, rgba):
        """Boolean (H, W) mask of blocking pixels."""
        rgb = rgba[..., :3].astype(np.int16)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        for rule in self.rules:
            diff = np.abs(rgb - np.array(rule.rgb, dtype=np.int16)).max(axis=2)
            mask |= diff <= rule.tolerance
        if rgba.shape[2] == 4:
            mask |= rgba[..., 3] < self.alpha_threshold
        return mask

    def grid_shape(self, width, height):
        return height // self.cell_px, width // self.cell_px

    def to_grid(self, rgba):
        """Collision grid (rows, cols) of uint8: 1 = obstacle, 0 = walkable."""
        rows, cols = self.grid_shape(rgba.shape[1], rgba.shape[0])
        if rows == 0 or cols == 0:
            raise ValueError("Image is smaller than one cell")
        mask = self.pixel_mask(rgba)[: rows * self.cell_px, : cols * self.cell_px]
        fraction = mask.reshape(rows, self.cell_px, cols, self.cell_px).mean(axis=(1, 3))
        return (fraction >= self.blocked_fraction).astype(np.uint8)

    def parse(self, path):
        return self.to_grid(load_rgba(path))
