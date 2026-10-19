"""
Real walkability from the community map data of tibiamaps.io (github.com/tibiamaps/tibia-map-data).

A minimap colour only says what the floor looks like: sand with a stone, a skull or a
stalagmite on it is still drawn as sand, yet nobody can walk there. The pathfinding layer
(floor-ZZ-path.png, same encoding as the client's Minimap_WaypointCost) marks those tiles.

    floor-ZZ-map.png   2560x2048 px, 1 px = 1 sqm, colour minimap of floor ZZ
    floor-ZZ-path.png  yellow = not walkable, magenta = unexplored, grey = walk cost

Pixel (0, 0) of every floor is world (31744, 30976). `locate` finds where a minimap crop
(e.g. a screenshot of tibiamaps.io / the in-game minimap, any integer upscale) lies in the
world, `load_area` builds a TerrainMap for that window with walls, ground speed and stairs.
Downloaded floors are cached in data/cache/tibiamaps/.
"""

import os
import urllib.request

import numpy as np

from .mapparser import detect_pixel_scale, load_rgba
from .terrain import TerrainLayer, TerrainMap, decode_waypoint_cost, link_transitions, stairs_mask

WORLD_X, WORLD_Y = 31744, 30976
FLOORS = tuple(range(16))
SEARCH_ORDER = (7, 8, 9, 10, 11, 12, 6, 13, 14, 15, 5, 4, 3, 2, 1, 0)
URLS = ("https://tibiamaps.github.io/tibia-map-data/floor-{z:02d}-{kind}.png",
        "https://raw.githubusercontent.com/tibiamaps/tibia-map-data/main/data/floor-{z:02d}-{kind}.png")
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "tibiamaps")
VOID = 0                   # packed black = unexplored on the colour map
MIN_MATCH = 0.85           # share of explored tiles that must agree for a location to count


def _packed(rgb):
    return (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]


def floor_path(z, kind, cache_dir=CACHE_DIR, download=True, log=None):
    """Local path of floor z ('map' or 'path'), downloading it on first use."""
    path = os.path.join(cache_dir, f"floor-{z:02d}-{kind}.png")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise FileNotFoundError(path)
    os.makedirs(cache_dir, exist_ok=True)
    errors = []
    for template in URLS:
        url = template.format(z=z, kind=kind)
        try:
            if log:
                log(f"Pobieram {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "rl_twin-simulator"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            with open(path + ".part", "wb") as handle:
                handle.write(data)
            os.replace(path + ".part", path)
            return path
        except OSError as exc:
            errors.append(f"{url}: {exc}")
    raise OSError("Nie udało się pobrać mapy tibiamaps:\n" + "\n".join(errors))


def image_tiles(rgba):
    """One RGB pixel per sqm from a minimap image at any integer upscale (centre pixel of each tile)."""
    scale, offset_x, offset_y = detect_pixel_scale(rgba)
    centre = scale // 2
    rows, cols = (rgba.shape[0] - offset_y) // scale, (rgba.shape[1] - offset_x) // scale   # whole tiles only
    tiles = rgba[offset_y + centre::scale, offset_x + centre::scale, :3][:rows, :cols]
    return tiles, scale, (offset_x, offset_y)


def _anchor(tiles):
    """(y, x) of the rarest non-void colour in the crop - few candidate positions to verify."""
    packed = _packed(tiles)
    values, counts = np.unique(packed[packed != VOID], return_counts=True)
    if values.size == 0:
        raise ValueError("Obraz nie ma żadnych odkrytych pól")
    rare = values[np.argmin(counts)]
    ys, xs = np.nonzero(packed == rare)
    return int(ys[0]), int(xs[0])


def locate(rgba, floors=SEARCH_ORDER, cache_dir=CACHE_DIR, log=None):
    """
    World position of a minimap crop: dict(x, y, z, match, cols, rows, scale, offset), or None.
    x, y = world coordinates of the crop's tile [0, 0].
    """
    tiles, scale, offset = image_tiles(rgba)
    rows, cols = tiles.shape[:2]
    crop = _packed(tiles)
    explored = crop != VOID
    ay, ax = _anchor(tiles)
    best = None
    for z in floors:
        world = _packed(load_rgba(floor_path(z, "map", cache_dir, log=log))[..., :3])
        height, width = world.shape
        for wy, wx in zip(*np.nonzero(world == crop[ay, ax])):
            y0, x0 = wy - ay, wx - ax
            if y0 < 0 or x0 < 0 or y0 + rows > height or x0 + cols > width:
                continue
            window = world[y0:y0 + rows, x0:x0 + cols]
            match = float((window[explored] == crop[explored]).mean())
            if best is None or match > best["match"]:
                best = {"x": WORLD_X + int(x0), "y": WORLD_Y + int(y0), "z": z, "match": match,
                        "cols": cols, "rows": rows, "scale": scale, "offset": offset}
        if best is not None and best["match"] >= MIN_MATCH:
            return best
    return best if best is not None and best["match"] >= MIN_MATCH else None


def load_area(x, y, cols, rows, floors, cache_dir=CACHE_DIR, log=None):
    """TerrainMap of the window [x, x + cols) x [y, y + rows) (world coordinates) on the given floors."""
    terrain = TerrainMap()
    for z in floors:
        try:
            color = load_rgba(floor_path(z, "map", cache_dir, log=log))
            cost = load_rgba(floor_path(z, "path", cache_dir, log=log))
        except OSError:
            if z == floors[0]:
                raise
            continue
        x0, y0 = x - WORLD_X, y - WORLD_Y
        window = np.zeros((rows, cols, 4), dtype=np.uint8)
        cost_window = np.zeros((rows, cols, 4), dtype=np.uint8)
        cost_window[..., 0], cost_window[..., 2], cost_window[..., 3] = 255, 255, 255     # unexplored
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(color.shape[0], y0 + rows), min(color.shape[1], x0 + cols)
        if sy0 >= sy1 or sx0 >= sx1:
            continue
        window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = color[sy0:sy1, sx0:sx1]
        cost_window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = cost[sy0:sy1, sx0:sx1]
        walls, speed = decode_waypoint_cost(cost_window)
        stairs = stairs_mask(window)
        walls = np.where(stairs, 0, walls).astype(np.uint8)    # stairs are walkable entry points
        if not (walls == 0).any():
            continue
        terrain.layers[z] = TerrainLayer(walls, speed, stairs, window, (x, y))
    if not terrain.layers:
        raise ValueError(f"Brak danych tibiamaps dla obszaru ({x}, {y}) na piętrach {list(floors)}")
    terrain.transitions = link_transitions(terrain.layers)
    return terrain


__all__ = ["locate", "load_area", "floor_path", "image_tiles", "WORLD_X", "WORLD_Y", "CACHE_DIR"]
