"""
Route scripts in world coordinates, for exchanging paths with cavebots.

Formats (picked by file extension):
  .json - rl_twin.script: waypoints with types, tactical zones and the world origin
  .txt  - one waypoint per line, "Type (x, y, z)" e.g.  Node (32369, 32241, 7)
          Types: Node (walk), Stand (exact tile), Ladder (floor up), Hole (floor down).
          Import accepts any line with three integers; a leading word sets the type.

Grid tiles are local (0..cols-1, 0..rows-1); world = local + world_origin. The origin comes
from the client minimap import or is set by hand for PNG maps.
"""

import json
import re

FORMAT = "rl_twin.script"
VERSION = 1
TEXT_TYPES = {"node": "Node", "stand": "Stand", "floor_up": "Ladder", "floor_down": "Hole"}
TEXT_TYPES_REVERSED = {v.lower(): k for k, v in TEXT_TYPES.items()}
_LINE_RE = re.compile(r"^\s*([A-Za-z_]+)?\D*?(-?\d+)\D+(-?\d+)\D+(-?\d+)")


def waypoint_types(waypoints, transitions):
    """Type of each waypoint: floor_up / floor_down when it sits on stairs, else node."""
    lookup = {(t[0], t[1], t[2]): t[5] for t in transitions}
    types = []
    for x, y, z in waypoints:
        target = lookup.get((x, y, z))
        types.append("node" if target is None else ("floor_down" if target > z else "floor_up"))
    return types


def text_script(waypoints, transitions=(), origin=(0, 0), name="trasa"):
    """Cavebot text: one 'Type (x, y, z)' line per waypoint in world coordinates."""
    ox, oy = origin
    types = waypoint_types(waypoints, transitions)
    lines = [f"# {name} - rl_twin, współrzędne świata (origin {ox}, {oy})"]
    lines += [f"{TEXT_TYPES[t]} ({x + ox}, {y + oy}, {z})" for (x, y, z), t in zip(waypoints, types)]
    return "\n".join(lines) + "\n"


def json_script(waypoints, zones=(), transitions=(), origin=(0, 0), name="trasa"):
    """rl_twin.script dict: typed waypoints and tactical zones in world coordinates."""
    ox, oy = origin
    types = waypoint_types(waypoints, transitions)
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": name,
        "origin": [ox, oy],
        "waypoints": [{"type": t, "x": x + ox, "y": y + oy, "z": z} for (x, y, z), t in zip(waypoints, types)],
        "zones": [dict(zone, x=zone["x"] + ox, y=zone["y"] + oy) for zone in zones],
    }


def export_script(path, waypoints, zones=(), transitions=(), origin=(0, 0), name="trasa"):
    if str(path).lower().endswith(".txt"):
        text = text_script(waypoints, transitions, origin, name)
    else:
        text = json.dumps(json_script(waypoints, zones, transitions, origin, name), ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return len(waypoints)


def import_script(path, origin=(0, 0), cols=None, rows=None):
    """Returns (waypoints, zones, skipped) in local tiles; points outside the grid are skipped."""
    ox, oy = origin
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    raw_points, raw_zones = [], []
    if text.lstrip().startswith("{"):
        data = json.loads(text)
        if data.get("format") != FORMAT:
            raise ValueError("To nie jest skrypt rl_twin (pole 'format')")
        raw_points = [(w["x"], w["y"], w["z"]) for w in data.get("waypoints", [])]
        raw_zones = data.get("zones", [])
    else:
        for line in text.splitlines():
            if line.strip().startswith(("#", "//", ";")):
                continue
            match = _LINE_RE.match(line)
            if match:
                raw_points.append((int(match.group(2)), int(match.group(3)), int(match.group(4))))

    def inside(x, y):
        return (cols is None or 0 <= x < cols) and (rows is None or 0 <= y < rows)

    waypoints, skipped = [], 0
    for x, y, z in raw_points:
        lx, ly = int(x) - ox, int(y) - oy
        if inside(lx, ly):
            waypoints.append((lx, ly, int(z)))
        else:
            skipped += 1
    zones = []
    for zone in raw_zones:
        lx, ly = int(zone["x"]) - ox, int(zone["y"]) - oy
        if inside(lx, ly):
            zones.append(dict(zone, x=lx, y=ly))
        else:
            skipped += 1
    return waypoints, zones, skipped
