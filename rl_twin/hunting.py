"""
Hunting grounds: named world windows that load terrain + native spawns in one go.

rl_twin/data/hunting_grounds.json lists each ground as a world window (x, y, cols, rows), the
hunting floor z and the floors to load. `build_ground` pulls the walls / ground speed / stairs
from tibiamaps data and the spawns from the spawn database, then places every known monster
on its exact tile (local = world - window origin).
"""

import json
import os
from dataclasses import dataclass, field

from .spawns import default_spawn_file, in_window, load_spawns, place
from .tibiamaps import load_area

GROUNDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hunting_grounds.json")


@dataclass
class HuntingGround:
    name: str
    x: int
    y: int
    z: int
    cols: int
    rows: int
    floors: list
    monsters: list = field(default_factory=list)

    @property
    def label(self):
        return f"{self.name}  (z={self.z})"


@dataclass
class GroundData:
    ground: HuntingGround
    terrain: object                      # terrain.TerrainMap
    spawns: list                         # [[lx, ly, z, name]] placed on walkable tiles
    unknown: dict                        # monster name -> count (not in the monster database)
    blocked: int                         # spawns without a free tile nearby
    warnings: list = field(default_factory=list)


def load_grounds(path=GROUNDS_FILE):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    grounds = []
    for entry in data.get("grounds", []):
        grounds.append(HuntingGround(entry["name"], int(entry["x"]), int(entry["y"]), int(entry["z"]),
                                     int(entry["cols"]), int(entry["rows"]), [int(z) for z in entry["floors"]],
                                     list(entry.get("monsters", []))))
    return grounds


def build_ground(ground, known_monsters, spawn_file=None, log=None):
    """Terrain and spawns for a hunting ground. Spawn problems become warnings, terrain problems raise."""
    terrain = load_area(ground.x, ground.y, ground.cols, ground.rows, ground.floors, log=log)
    warnings = []
    try:
        path = spawn_file or default_spawn_file(log=log)
        spawns = load_spawns(path)
    except (ValueError, OSError) as exc:
        warnings.append(f"Spawny niedostępne: {exc}")
        return GroundData(ground, terrain, [], {}, 0, warnings)
    local = in_window(spawns, ground.x, ground.y, ground.cols, ground.rows, terrain.layers.keys())
    placed, unknown, blocked = place(local, {z: layer.walls for z, layer in terrain.layers.items()}, known_monsters)
    return GroundData(ground, terrain, placed, unknown, blocked, warnings)


__all__ = ["HuntingGround", "GroundData", "load_grounds", "build_ground", "GROUNDS_FILE"]
