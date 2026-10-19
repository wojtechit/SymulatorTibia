"""
Monster spawn databases: OTServ / Canary XML and plain JSON, with exact world [x, y, z] per monster.

Supported formats (auto-detected):
  TFS     <spawns><spawn centerx centery centerz radius><monster name x y z spawntime/></spawn></spawns>
  Canary  <monsters><monster centerx centery centerz radius><monster name x y z spawntime/></monster></monsters>
          (in both: monster x/y are offsets from the centre, z is the absolute floor)
  JSON    [{"name": "Dragon", "x": 33200, "y": 32250, "z": 11}, ...]  or {"spawns": [...]}

The default database is Canary's global map spawns (a community reconstruction of the real
Tibia world, so coordinates line up with tibiamaps.io / the client minimap). It is
downloaded once into data/cache/spawns/.
"""

import json
import os
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache", "spawns")
DEFAULT_SOURCES = {
    "canary": ("https://raw.githubusercontent.com/opentibiabr/canary/main/"
               "data-otservbr-global/world/otservbr-monster.xml"),
}

_cache = {}


class SpawnError(ValueError):
    """Missing, unreadable or malformed spawn database."""


@dataclass(frozen=True)
class Spawn:
    name: str
    x: int
    y: int
    z: int
    spawntime: int = 60


def parse_spawn_xml(path):
    """All monster spawns of a TFS / Canary XML file (streamed, so large files stay cheap)."""
    spawns = []
    centre = None
    try:
        for event, element in ET.iterparse(path, events=("start", "end")):
            if event == "start":
                if element.get("centerx") is not None:
                    centre = (int(element.get("centerx")), int(element.get("centery")), int(element.get("centerz", 7)))
                continue
            if element.get("centerx") is not None:
                centre = None
                element.clear()
                continue
            if element.tag == "monster" and element.get("name") and centre is not None:
                z = element.get("z")
                spawns.append(Spawn(element.get("name"), centre[0] + int(element.get("x", 0)),
                                    centre[1] + int(element.get("y", 0)), int(z) if z is not None else centre[2],
                                    int(element.get("spawntime", 60) or 60)))
    except ET.ParseError as exc:
        raise SpawnError(f"Uszkodzony XML spawnów {os.path.basename(path)}: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise SpawnError(f"Błędne współrzędne w {os.path.basename(path)}: {exc}") from exc
    return spawns


def parse_spawn_json(path):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get("spawns", []) if isinstance(data, dict) else data
    try:
        return [Spawn(str(e["name"]), int(e["x"]), int(e["y"]), int(e["z"]), int(e.get("spawntime", 60)))
                for e in entries]
    except (KeyError, TypeError, ValueError) as exc:
        raise SpawnError(f"{os.path.basename(path)}: wpis bez name/x/y/z ({exc})") from exc


def load_spawns(path):
    """Spawns from an XML or JSON file, cached per (path, mtime)."""
    if not path or not os.path.isfile(path):
        raise SpawnError(f"Brak pliku spawnów: {path}")
    key = os.path.abspath(path)
    mtime = os.path.getmtime(path)
    cached = _cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        spawns = parse_spawn_json(path) if path.lower().endswith(".json") else parse_spawn_xml(path)
    except json.JSONDecodeError as exc:
        raise SpawnError(f"Uszkodzony JSON spawnów {os.path.basename(path)}: {exc}") from exc
    except OSError as exc:
        raise SpawnError(f"Nie można odczytać {os.path.basename(path)}: {exc}") from exc
    if not spawns:
        raise SpawnError(f"{os.path.basename(path)} nie zawiera żadnych potworów")
    _cache[key] = (mtime, spawns)
    return spawns


def default_spawn_file(source="canary", cache_dir=CACHE_DIR, download=True, log=None):
    """Local path of a default spawn database, downloading it on first use."""
    url = DEFAULT_SOURCES[source]
    path = os.path.join(cache_dir, f"{source}-monster.xml")
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    if not download:
        raise SpawnError(f"Brak bazy spawnów {source} w {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    if log:
        log(f"Pobieram bazę spawnów: {url}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rl_twin-simulator"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except OSError as exc:
        raise SpawnError(f"Nie udało się pobrać bazy spawnów ({exc}). Wskaż plik XML ręcznie.") from exc
    with open(path + ".part", "wb") as handle:
        handle.write(data)
    os.replace(path + ".part", path)
    return path


def in_window(spawns, x, y, cols, rows, floors=None):
    """Spawns inside [x, x + cols) x [y, y + rows) on `floors`, as local [lx, ly, z, name]."""
    floors = None if floors is None else set(floors)
    return [[s.x - x, s.y - y, s.z, s.name] for s in spawns
            if x <= s.x < x + cols and y <= s.y < y + rows and (floors is None or s.z in floors)]


def place(local_spawns, walls_by_floor, known_names):
    """
    Split local spawns into (placed, skipped_reasons). A spawn is placed when the monster is known
    and its tile is walkable (a spawn on a blocked tile is moved to the nearest free tile within 2).
    """
    placed, unknown, blocked = [], {}, 0
    taken = set()
    for lx, ly, z, name in local_spawns:
        if name not in known_names:
            unknown[name] = unknown.get(name, 0) + 1
            continue
        walls = walls_by_floor.get(z)
        if walls is None:
            blocked += 1
            continue
        spot = _free_near(walls, lx, ly, z, taken)
        if spot is None:
            blocked += 1
            continue
        taken.add((spot[0], spot[1], z))
        placed.append([spot[0], spot[1], z, name])
    return placed, unknown, blocked


def _free_near(walls, x, y, z, taken, radius=2):
    rows, cols = walls.shape
    for r in range(radius + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and not walls[ny, nx] and (nx, ny, z) not in taken:
                    return nx, ny
    return None


__all__ = ["Spawn", "SpawnError", "load_spawns", "parse_spawn_xml", "parse_spawn_json", "default_spawn_file",
           "in_window", "place", "CACHE_DIR"]
