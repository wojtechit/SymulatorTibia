"""
PNG -> collision grid.

1. Read the image as RGBA (Pillow, or Qt's QImage as a fallback).
2. A pixel blocks movement when it is DARK (every RGB channel below `dark_threshold`,
   default 20), TRANSPARENT, or has one of the `wall_colors` (exact Tibia minimap
   colours, +- `color_tolerance`). Default wall colours: cave/mountain walls, rock,
   water, building walls, lava and trees of the Tibia minimap palette.
3. Cell size (`cell_px`, None = auto):
   - minimap (<= 256 distinct colours): the image is often an upscaled export, so the
     scale is the largest s dividing >= 90% of single-colour run lengths (e.g. 3 when one
     sqm is drawn as 3x3 px) and the grid origin is the most common colour-change phase;
   - anything else is treated as a client screenshot: 32 px per sqm.
4. A cell is an obstacle (1) when at least `blocked_fraction` of its pixels block.
"""

import numpy as np

DEFAULT_DARK_THRESHOLD = 20
DEFAULT_COLOR_TOLERANCE = 10
MINIMAP_MAX_COLORS = 256
SCREENSHOT_CELL_PX = 32
MAX_DETECTED_SCALE = 64

# Tibia minimap palette: rgb -> (Polish label, blocks movement by default)
TIBIA_MINIMAP_COLORS = {
    (0, 0, 0): ("Pustka / nieodkryte", True),
    (0, 102, 0): ("Drzewa / krzaki", True),
    (0, 204, 0): ("Trawa", False),
    (0, 255, 0): ("Bagno", False),
    (51, 102, 153): ("Woda", True),
    (102, 102, 102): ("Skały / góry", True),
    (153, 51, 0): ("Ściana jaskini / góra", True),
    (153, 102, 51): ("Ziemia / podłoga jaskini", False),
    (153, 153, 153): ("Kamień / bruk", False),
    (153, 255, 102): ("Jasna trawa", False),
    (204, 255, 255): ("Lód", False),
    (255, 51, 0): ("Mur / ściana budynku", True),
    (255, 102, 0): ("Lawa", True),
    (255, 204, 153): ("Piasek / podłoga", False),
    (255, 255, 0): ("Schody / drabina / dziura", False),
    (255, 255, 255): ("Śnieg", False),
}
DEFAULT_WALL_COLORS = tuple(rgb for rgb, (_, blocks) in TIBIA_MINIMAP_COLORS.items() if blocks)


def color_label(rgb):
    known = TIBIA_MINIMAP_COLORS.get(tuple(int(v) for v in rgb))
    return known[0] if known else "Nieznany kolor"


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


def _packed(rgb):
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]


def palette(rgba, limit=MINIMAP_MAX_COLORS + 1):
    """[(rgb tuple, fraction)] sorted by frequency (at most `limit` entries)."""
    packed = _packed(rgba[..., :3]).ravel()
    values, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)[:limit]
    total = packed.size
    return [(((int(v) >> 16) & 255, (int(v) >> 8) & 255, int(v) & 255), counts[i] / total)
            for i, v in zip(order, values[order])]


def _runs_and_changes(rgba):
    """Interior single-colour run lengths and colour-change positions along x and y (sampled lines)."""
    packed = _packed(rgba[..., :3])
    runs, changes_x, changes_y = [], [], []
    for lines, sink in ((packed, changes_x), (packed.T, changes_y)):
        step = max(1, lines.shape[0] // 96)
        for line in lines[::step]:
            changes = np.flatnonzero(line[1:] != line[:-1]) + 1
            sink.append(changes)
            if len(changes) > 1:
                runs.append(np.diff(changes))   # interior runs only (both ends are colour changes)
    runs = np.concatenate(runs) if runs else np.zeros(0, dtype=np.int64)
    as_array = lambda parts: np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return runs, as_array(changes_x), as_array(changes_y)


def detect_pixel_scale(rgba, min_share=0.9):
    """
    (scale, offset_x, offset_y) of an upscaled flat-colour image (e.g. a minimap drawn at 3x3 px per sqm).

    scale  = largest s whose multiples cover at least `min_share` of the single-colour runs
             (tolerates a few odd runs from cropping or non-integer resizing);
    offset = most common colour-change phase modulo s, i.e. where the tile grid starts.
    """
    runs, changes_x, changes_y = _runs_and_changes(rgba)
    if len(runs) < 8:
        return 1, 0, 0
    for scale in range(MAX_DETECTED_SCALE, 1, -1):
        if (runs % scale == 0).mean() >= min_share:
            phase = lambda changes: int(np.bincount(changes % scale, minlength=scale).argmax()) if len(changes) else 0
            return scale, phase(changes_x), phase(changes_y)
    return 1, 0, 0


def mask_to_cells(mask, cell, offset=(0, 0), fraction=0.5):
    """Downsample a per-pixel boolean mask to tiles: a tile is set when >= `fraction` of its pixels are."""
    offset_x, offset_y = offset
    mask = mask[offset_y:, offset_x:]
    rows, cols = mask.shape[0] // cell, mask.shape[1] // cell
    mask = mask[: rows * cell, : cols * cell]
    if cell == 1:
        return mask.astype(np.uint8)
    return (mask.reshape(rows, cell, cols, cell).mean(axis=(1, 3)) >= fraction).astype(np.uint8)


class MapParser:
    def __init__(self, cell_px=None, dark_threshold=DEFAULT_DARK_THRESHOLD, wall_colors=DEFAULT_WALL_COLORS,
                 color_tolerance=DEFAULT_COLOR_TOLERANCE, blocked_fraction=0.5, alpha_threshold=128):
        if cell_px is not None and cell_px < 1:
            raise ValueError("cell_px must be >= 1 (or None for auto)")
        self.cell_px = cell_px
        self.dark_threshold = int(dark_threshold)
        self.wall_colors = tuple(tuple(int(v) for v in c) for c in wall_colors)
        self.color_tolerance = int(color_tolerance)
        self.blocked_fraction = float(blocked_fraction)
        self.alpha_threshold = alpha_threshold
        self.last_cell_px = None
        self.last_offset = (0, 0)     # (x, y) pixel where the tile grid starts
        self.last_was_minimap = None

    @staticmethod
    def looks_like_minimap(rgba):
        return len(palette(rgba)) <= MINIMAP_MAX_COLORS

    def resolve_cell_px(self, rgba):
        """Explicit cell_px, else the detected upscale of a minimap, else 32 for a screenshot."""
        self.last_was_minimap = self.looks_like_minimap(rgba)
        self.last_offset = (0, 0)
        if self.cell_px is not None:
            return self.cell_px
        if not self.last_was_minimap:
            return SCREENSHOT_CELL_PX
        scale, offset_x, offset_y = detect_pixel_scale(rgba)
        self.last_offset = (offset_x, offset_y)
        return scale

    def pixel_mask(self, rgba):
        """Boolean (H, W) mask of blocking pixels."""
        rgb = rgba[..., :3]
        mask = (rgb < self.dark_threshold).all(axis=2)                     # dark = wall / void
        if rgba.shape[2] == 4:
            mask |= rgba[..., 3] < self.alpha_threshold                    # transparent = void
        if self.wall_colors:
            signed = rgb.astype(np.int16)
            for color in self.wall_colors:
                diff = np.abs(signed - np.array(color, dtype=np.int16)).max(axis=2)
                mask |= diff <= self.color_tolerance
        return mask

    def to_grid(self, rgba, cell_px=None):
        """Collision grid (rows, cols) of uint8: 1 = obstacle, 0 = walkable."""
        if cell_px:
            self.last_offset = (0, 0)
        cell = cell_px or self.resolve_cell_px(rgba)
        self.last_cell_px = cell
        offset_x, offset_y = self.last_offset
        rgba = rgba[offset_y:, offset_x:]
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
