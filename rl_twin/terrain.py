"""
Client-grade terrain: per-tile ground speed, floor changes, Tibia client minimap import.

Tibia's client stores the explored world in its minimap folder as 256x256 px tiles
(1 px = 1 sqm), named by the world coordinates of their top-left corner:
    Minimap_Color_{X}_{Y}_{Z}.png         - what the in-game minimap shows
    Minimap_WaypointCost_{X}_{Y}_{Z}.png  - walk cost per tile, used by the client's
                                            own pathfinding (map clicks)
Decoding of WaypointCost (assumption, verify on your files): pure yellow (255,255,0)
= not walkable, magenta (255,0,255) = unexplored, any other colour = walkable with a
cost proportional to its red channel. Ground speed is normalised so the most common
walkable cost maps to DEFAULT_GROUND_SPEED (150, grass); slower tiles get a higher value.

Floor changes: yellow tiles of the colour minimap are stairs, ladders, holes and ramps.
`link_transitions` pairs each with the floor above or below (whichever has walkable
ground there) and lands the player on the nearest non-stair walkable tile, so stepping
onto a hole moves (x, y, z) -> (x', y', z+1) like in the game.
"""

import base64
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np

from .mapparser import load_rgba
from .physics import DEFAULT_GROUND_SPEED

FLOOR_CHANGE_RGB = (255, 255, 0)
UNWALKABLE_COST_RGB = (255, 255, 0)
UNEXPLORED_COST_RGB = (255, 0, 255)
CLIENT_TILE = 256
MIN_GROUND_SPEED = 50
MAX_GROUND_SPEED = 1000

_FILE_RE = re.compile(r"Minimap_(Color|WaypointCost)_(\d+)_(\d+)_(\d+)\.png$", re.IGNORECASE)


@dataclass
class TerrainLayer:
    """One floor: walls (1 = blocked), ground speed per tile, floor-change tiles, optional display image."""
    walls: np.ndarray
    speed: np.ndarray = None                     # uint16, ground speed per tile
    stairs: np.ndarray = None                    # bool, floor-change tiles
    image: np.ndarray = None                     # RGBA, 1 px per tile (for the GUI)
    origin: tuple = (0, 0)                       # world (x, y) of tile [0, 0]

    def __post_init__(self):
        shape = self.walls.shape
        if self.speed is None:
            self.speed = np.full(shape, DEFAULT_GROUND_SPEED, dtype=np.uint16)
        if self.stairs is None:
            self.stairs = np.zeros(shape, dtype=bool)


@dataclass
class TerrainMap:
    layers: dict = field(default_factory=dict)   # z -> TerrainLayer
    transitions: list = field(default_factory=list)   # [x, y, z, x2, y2, z2]


# ----- encoding for EnvConfig / scenario JSON ------------------------------ #

def encode_speed(speed):
    return base64.b64encode(np.ascontiguousarray(speed, dtype="<u2").tobytes()).decode("ascii")


def decode_speed(text, rows, cols):
    data = np.frombuffer(base64.b64decode(text), dtype="<u2")
    if data.size != rows * cols:
        raise ValueError(f"Speed layer has {data.size} tiles, expected {rows * cols}")
    return data.reshape(rows, cols).astype(np.uint16)


# ----- decoding ------------------------------------------------------------- #

def stairs_mask(rgba, tolerance=10):
    rgb = rgba[..., :3].astype(np.int16)
    return (np.abs(rgb - np.array(FLOOR_CHANGE_RGB, dtype=np.int16)).max(axis=2) <= tolerance)


def decode_waypoint_cost(rgba):
    """(walls uint8, speed uint16) from a WaypointCost image (see module docstring for the encoding)."""
    rgb = rgba[..., :3].astype(np.int16)
    blocked = ((np.abs(rgb - np.array(UNWALKABLE_COST_RGB)).max(axis=2) <= 8)
               | (np.abs(rgb - np.array(UNEXPLORED_COST_RGB)).max(axis=2) <= 8))
    if rgba.shape[2] == 4:
        blocked |= rgba[..., 3] < 128
    cost = rgb[..., 0].astype(np.float32)
    walkable_costs = cost[~blocked]
    walkable_costs = walkable_costs[walkable_costs > 0]
    speed = np.full(cost.shape, DEFAULT_GROUND_SPEED, dtype=np.uint16)
    if walkable_costs.size:
        typical = float(np.bincount(walkable_costs.astype(np.int64)).argmax())
        scaled = np.where(cost > 0, cost / typical * DEFAULT_GROUND_SPEED, DEFAULT_GROUND_SPEED)
        speed = np.clip(np.rint(scaled), MIN_GROUND_SPEED, MAX_GROUND_SPEED).astype(np.uint16)
    return blocked.astype(np.uint8), speed


def link_transitions(layers, search_radius=2):
    """Pair floor-change tiles with the neighbouring floor; returns [[x, y, z, x2, y2, z2], ...]."""
    transitions = []
    for z, layer in layers.items():
        for y, x in np.argwhere(layer.stairs):
            target = None
            for dz in (1, -1):                       # holes/stairs down are the common case
                other = layers.get(z + dz)
                if other is None:
                    continue
                landing = _landing_tile(other, int(x), int(y), search_radius)
                if landing is not None:
                    # Prefer the floor where the counterpart is also a stair (paired ramp/ladder)
                    paired = other.stairs[max(0, y - 1): y + 2, max(0, x - 1): x + 2].any()
                    if target is None or paired:
                        target = (landing[0], landing[1], z + dz)
                    if paired:
                        break
            if target is not None:
                transitions.append([int(x), int(y), int(z), target[0], target[1], target[2]])
    return transitions


def _landing_tile(layer, x, y, radius):
    """Nearest walkable, non-stair tile around (x, y) - so the player does not bounce straight back."""
    rows, cols = layer.walls.shape
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and not layer.walls[ny, nx] and not layer.stairs[ny, nx]:
                key = (max(abs(dx), abs(dy)), abs(dx) + abs(dy))
                if best is None or key < best[0]:
                    best = (key, (nx, ny))
    return best[1] if best else None


# ----- client minimap folder ----------------------------------------------- #

def scan_client_minimap(folder):
    """{(kind, X, Y, Z): path} for every Minimap_*.png in the folder."""
    files = {}
    for path in glob.glob(os.path.join(folder, "Minimap_*.png")):
        match = _FILE_RE.search(os.path.basename(path))
        if match:
            kind, x, y, z = match.group(1).lower(), int(match.group(2)), int(match.group(3)), int(match.group(4))
            files[(kind, x, y, z)] = path
    return files


def load_client_minimap(folder, center_x, center_y, radius, floors):
    """
    Crop a square of (2 * radius + 1) tiles around world (center_x, center_y) from the client
    minimap for each floor. Returns a TerrainMap with linked floor transitions.
    """
    files = scan_client_minimap(folder)
    if not files:
        raise FileNotFoundError(f"Brak plików Minimap_*.png w {folder}")
    x0, y0 = center_x - radius, center_y - radius
    size = 2 * radius + 1
    terrain = TerrainMap()
    for z in floors:
        color = np.zeros((size, size, 4), dtype=np.uint8)
        cost = np.zeros((size, size, 4), dtype=np.uint8)
        cost[..., 0], cost[..., 2], cost[..., 3] = 255, 255, 255          # unexplored until filled
        found_color = found_cost = False
        for tile_x in range((x0 // CLIENT_TILE) * CLIENT_TILE, x0 + size, CLIENT_TILE):
            for tile_y in range((y0 // CLIENT_TILE) * CLIENT_TILE, y0 + size, CLIENT_TILE):
                for kind, target in (("color", color), ("waypointcost", cost)):
                    path = files.get((kind, tile_x, tile_y, z))
                    if path is None:
                        continue
                    _blit(target, load_rgba(path), tile_x - x0, tile_y - y0)
                    found_color |= kind == "color"
                    found_cost |= kind == "waypointcost"
        if not (found_color or found_cost):
            continue
        if found_cost:
            walls, speed = decode_waypoint_cost(cost)
        else:
            from .mapparser import MapParser
            walls, speed = MapParser(cell_px=1).to_grid(color), None
        stairs = stairs_mask(color) if found_color else np.zeros((size, size), dtype=bool)
        walls = np.where(stairs, 0, walls).astype(np.uint8)    # stairs are walkable entry points
        terrain.layers[z] = TerrainLayer(walls, speed, stairs, color if found_color else None, (x0, y0))
    if not terrain.layers:
        raise ValueError(f"Brak danych minimapy wokół ({center_x}, {center_y}) na piętrach {list(floors)}")
    terrain.transitions = link_transitions(terrain.layers)
    return terrain


def _blit(target, source, offset_x, offset_y):
    """Copy `source` into `target` at (offset_x, offset_y), clipping both."""
    th, tw = target.shape[:2]
    sh, sw = source.shape[:2]
    x1, y1 = max(0, offset_x), max(0, offset_y)
    x2, y2 = min(tw, offset_x + sw), min(th, offset_y + sh)
    if x1 < x2 and y1 < y2:
        target[y1:y2, x1:x2] = source[y1 - offset_y: y2 - offset_y, x1 - offset_x: x2 - offset_x]
