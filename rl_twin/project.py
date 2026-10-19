"""
Complete project files: scenario (EnvConfig) + editor state + route script in one JSON.

    {"format": "rl_twin.project", "version": 1,
     ...EnvConfig fields (cols, rows, collision, waypoints, spawns, zones, tactics, rewards, ...),
     "floor_maps": {...}, "client_minimap": {...}, "map_wall_colors": [...],   # editor state
     "script": {rl_twin.script export in world coordinates}}

Older scenario files (no "format") load the same way. `load_project` never trusts the file:
every list is validated against the grid, broken parts are dropped with a readable warning
and only an unreadable file raises ProjectError.
"""

import base64
import binascii
import json
import os

from .env import EnvConfig
from .scripts import json_script
from .tactics import ZONE_KINDS

FORMAT = "rl_twin.project"
VERSION = 1
MAX_GRID = 2048


class ProjectError(ValueError):
    """The file cannot be used at all (not JSON, wrong type, no grid size)."""


def script_block(config):
    """rl_twin.script structure (world coordinates) for the config's route, zones and spawns."""
    ox, oy = config.world_origin
    script = json_script([tuple(w) for w in config.waypoints], config.zones, config.transitions, (ox, oy))
    script["spawns"] = [{"name": name, "x": x + ox, "y": y + oy, "z": z} for x, y, z, name in config.spawns]
    return script


def project_dict(config, editor=None):
    data = {"format": FORMAT, "version": VERSION}
    data.update(config.to_dict())
    data.update(editor or {})
    data["script"] = script_block(config)
    return data


def save_project(path, config, editor=None):
    data = project_dict(config, editor)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)            # never leave a half-written project behind
    return data


def load_project(path, monster_names=None):
    """(EnvConfig, raw dict, warnings). Raises ProjectError only when nothing can be recovered."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ProjectError(f"{os.path.basename(path)} nie jest poprawnym JSON-em (linia {exc.lineno}): {exc.msg}")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProjectError(f"Nie można odczytać {os.path.basename(path)}: {exc}")
    if not isinstance(data, dict):
        raise ProjectError("Plik projektu musi zawierać obiekt JSON")
    if data.get("format") not in (None, FORMAT):
        raise ProjectError(f"Nieznany format pliku: {data.get('format')!r} (to może być skrypt trasy - użyj Importu skryptu)")
    if int(data.get("version", VERSION) or VERSION) > VERSION:
        raise ProjectError(f"Projekt z nowszej wersji ({data['version']}) - zaktualizuj symulator")
    warnings = []
    data = sanitize(data, monster_names, warnings)
    try:
        config = EnvConfig.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"Błędne ustawienia scenariusza: {exc}")
    return config, data, warnings


def sanitize(data, monster_names, warnings):
    data = dict(data)
    try:
        cols, rows = int(data.get("cols", 32)), int(data.get("rows", 24))
    except (TypeError, ValueError):
        raise ProjectError("Rozmiar siatki (cols/rows) nie jest liczbą")
    if not (1 <= cols <= MAX_GRID and 1 <= rows <= MAX_GRID):
        raise ProjectError(f"Rozmiar siatki {cols}x{rows} poza zakresem 1..{MAX_GRID}")
    data["cols"], data["rows"] = cols, rows

    def inside(x, y):
        return 0 <= x < cols and 0 <= y < rows

    collision = {}
    for z, lines in (data.get("collision") or {}).items():
        if not isinstance(lines, list) or not all(isinstance(line, str) for line in lines):
            warnings.append(f"Kolizje piętra {z} uszkodzone - piętro pominięte.")
            continue
        bad = [i for i, line in enumerate(lines) if len(line) != cols or set(line) - {"0", "1"}]
        if len(lines) != rows or bad:
            warnings.append(f"Kolizje piętra {z}: {len(bad)} uszkodzonych wierszy / {len(lines)} z {rows} - "
                            f"brakujące pola uznane za przechodnie, sprawdź mapę.")
            lines = ["".join(c if c in "01" else "1" for c in line[:cols]).ljust(cols, "0") for line in lines[:rows]]
        collision[str(z)] = lines
    data["collision"] = collision

    speeds = {}
    for z, text in (data.get("terrain_speed") or {}).items():
        try:
            if len(base64.b64decode(text)) != rows * cols * 2:
                raise ValueError("rozmiar")
            speeds[str(z)] = text
        except (binascii.Error, TypeError, ValueError):
            warnings.append(f"Prędkości podłoża piętra {z} uszkodzone - użyta prędkość domyślna.")
    data["terrain_speed"] = speeds

    def keep(items, check, label):
        good = []
        for item in items or []:
            try:
                if check(item):
                    good.append(item)
                    continue
            except (TypeError, ValueError, IndexError, KeyError):
                pass
        dropped = len(items or []) - len(good)
        if dropped:
            warnings.append(f"Pominięto {dropped} {label} (poza mapą albo uszkodzone).")
        return good

    data["waypoints"] = keep(data.get("waypoints"),
                             lambda w: len(w) in (2, 3) and inside(int(w[0]), int(w[1])), "punktów trasy")
    known = set(monster_names) if monster_names else None
    spawns = keep(data.get("spawns"), lambda s: len(s) in (3, 4) and inside(int(s[0]), int(s[1])), "spawnów")
    if known is not None:
        unknown = sorted({s[-1] for s in spawns if s[-1] not in known})
        if unknown:
            warnings.append(f"Nieznane potwory pominięte (brak w bazie): {', '.join(unknown)}")
            spawns = [s for s in spawns if s[-1] in known]
    data["spawns"] = spawns
    data["transitions"] = keep(data.get("transitions"),
                               lambda t: len(t) == 6 and inside(int(t[0]), int(t[1])) and inside(int(t[3]), int(t[4])),
                               "przejść między piętrami")
    data["zones"] = keep(data.get("zones"),
                         lambda z: z["kind"] in ZONE_KINDS and inside(int(z["x"]), int(z["y"])) and int(z["radius"]) > 0,
                         "stref taktycznych")
    return data


__all__ = ["ProjectError", "FORMAT", "VERSION", "save_project", "load_project", "project_dict", "script_block"]
