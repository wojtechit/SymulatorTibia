"""
OpenTibia binary map (.otbm) and item database (items.otb / items.xml) reader.

Both files are OTB node trees:  0xFE <type> <props...> <children...> 0xFF, where any
literal 0xFD/0xFE/0xFF byte inside props is escaped with 0xFD. The reader streams the tree
with a C-level search for the three special bytes, so tile areas outside the requested
window cost only a scan, never a Python object per tile.

OTBM layout used here:
    root (props: version u32, width u16, height u16, items major/minor u32)
      MAP_DATA (2)
        TILE_AREA (4)  props: base x u16, y u16, z u8
          TILE (5) / HOUSETILE (14)  props: dx u8, dy u8, [house id u32], attrs
              attr TILE_FLAGS (3) u32 | attr ITEM (9) u16 inline ground item
            ITEM (6)  props: item id u16, attrs (containers nest more ITEM nodes - ignored)

Walkability needs item flags from items.otb (BLOCK_SOLID, ground speed, minimap colour)
and floor changes from items.xml (attribute floorchange). Without items.otb every tile with
ground counts as walkable and a warning says so.

Note: modern servers (Canary) take item flags from the client's appearances.dat instead of
items.otb; a converted items.otb (e.g. from the server's item editor) works here.
"""

import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .terrain import TerrainLayer, TerrainMap, link_transitions

ESCAPE, NODE_START, NODE_END = 0xFD, 0xFE, 0xFF
_SPECIAL = re.compile(rb"[\xfd\xfe\xff]")

OTBM_MAP_DATA, OTBM_TILE_AREA, OTBM_TILE, OTBM_ITEM, OTBM_HOUSETILE = 2, 4, 5, 6, 14
ATTR_TILE_FLAGS, ATTR_ITEM = 3, 9

FLAG_BLOCK_SOLID = 1 << 0
FLAG_BLOCK_PATHFIND = 1 << 2
ITEM_ATTR_SERVERID, ITEM_ATTR_SPEED, ITEM_ATTR_MINIMAPCOLOR = 0x10, 0x14, 0x21

DEFAULT_SPEED = 150
VOID_RGB = (0, 0, 0)
WALL_RGB = (153, 51, 0)
FLOOR_RGB = (255, 204, 153)
STAIRS_RGB = (255, 255, 0)


class OtbError(ValueError):
    """Corrupted or unsupported OTB/OTBM file."""


def walk_tree(data, enter, leave=None, offset=4):
    """
    Stream the node tree in `data` (bytes). enter(type, props, depth) is called once a node's
    props are complete (props = unescaped bytes); leave(type, depth) at its end. enter may
    return False to skip the node's subtree (it is still scanned, but no callbacks fire).
    """
    stack = []          # [type, props bytearray, props_done, skipped]
    pos = offset
    search = _SPECIAL.search
    size = len(data)
    skip_depth = None
    while True:
        match = search(data, pos)
        if match is None:
            if stack:
                raise OtbError(f"Plik ucięty: {len(stack)} niezamkniętych węzłów")
            return
        i = match.start()
        top = stack[-1] if stack else None
        if top is not None and not top[2]:
            top[1] += data[pos:i]
        byte = data[i]
        if byte == ESCAPE:
            if i + 1 >= size:
                raise OtbError("Plik ucięty po bajcie ucieczki")
            if top is not None and not top[2]:
                top[1].append(data[i + 1])
            pos = i + 2
            continue
        if byte == NODE_START:
            if i + 1 >= size:
                raise OtbError("Plik ucięty po początku węzła")
            if top is not None and not top[2]:
                _finish_props(stack, enter, skip_depth)
                if top[3] and skip_depth is None:
                    skip_depth = len(stack)
            stack.append([data[i + 1], bytearray(), False, False])
            pos = i + 2
            continue
        # NODE_END
        if top is None:
            raise OtbError("Nadmiarowy koniec węzła")
        if not top[2]:
            _finish_props(stack, enter, skip_depth)
        depth = len(stack)
        stack.pop()
        if skip_depth is not None and depth <= skip_depth:
            skip_depth = None
        elif skip_depth is None and leave is not None and not top[3]:
            leave(top[0], depth - 1)
        pos = i + 1
        if not stack:
            return


def _finish_props(stack, enter, skip_depth):
    node = stack[-1]
    node[2] = True
    if skip_depth is not None:
        node[3] = True
        return
    if enter(node[0], bytes(node[1]), len(stack) - 1) is False:
        node[3] = True


def _read_file(path, magic=None):
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise OtbError(f"Nie można odczytać {os.path.basename(path)}: {exc}") from exc
    if len(data) < 6:
        raise OtbError(f"{os.path.basename(path)} jest pusty albo uszkodzony")
    if magic and data[:4] not in (magic, b"\0\0\0\0"):
        raise OtbError(f"{os.path.basename(path)}: nieznany nagłówek {data[:4]!r}")
    return data


# ----- items.otb / items.xml ---------------------------------------------- #

@dataclass
class ItemInfo:
    flags: int = 0
    speed: int = 0
    minimap: int = 0           # 0..215 web-safe palette index, 0 = none
    floor_change: str = None   # "down" or a direction (= up), from items.xml

    @property
    def blocks(self):
        return bool(self.flags & FLAG_BLOCK_SOLID)


@dataclass
class ItemTable:
    items: dict = field(default_factory=dict)     # server id -> ItemInfo
    has_flags: bool = False

    def get(self, item_id):
        info = self.items.get(item_id)
        if info is None:
            info = self.items[item_id] = ItemInfo()
        return info


def load_items_otb(path, table=None):
    """Item flags, ground speed and minimap colour per server id from items.otb."""
    table = table or ItemTable()
    data = _read_file(path)

    def enter(node_type, props, depth):
        if depth != 1 or len(props) < 4:
            return depth < 1
        flags = struct.unpack_from("<I", props, 0)[0]
        pos, server_id, speed, minimap = 4, None, 0, 0
        while pos + 3 <= len(props):
            attr = props[pos]
            length = struct.unpack_from("<H", props, pos + 1)[0]
            value = props[pos + 3: pos + 3 + length]
            pos += 3 + length
            if attr == ITEM_ATTR_SERVERID and length >= 2:
                server_id = struct.unpack_from("<H", value)[0]
            elif attr == ITEM_ATTR_SPEED and length >= 2:
                speed = struct.unpack_from("<H", value)[0]
            elif attr == ITEM_ATTR_MINIMAPCOLOR and length >= 2:
                minimap = struct.unpack_from("<H", value)[0]
        if server_id is not None:
            info = table.get(server_id)
            info.flags, info.speed, info.minimap = flags, speed, minimap
        return False

    walk_tree(data, enter)
    if not table.items:
        raise OtbError(f"{os.path.basename(path)}: brak przedmiotów (to nie jest items.otb?)")
    table.has_flags = True
    return table


def load_items_xml(path, table=None):
    """Floor-change attributes (stairs, holes, ramps) per server id from items.xml."""
    table = table or ItemTable()
    try:
        for _, element in ET.iterparse(path, events=("end",)):
            if element.tag != "item":
                continue
            change = next((a.get("value") for a in element.findall("attribute")
                           if a.get("key", "").lower() == "floorchange"), None)
            if change:
                if element.get("id"):
                    ids = [int(element.get("id"))]
                else:
                    ids = range(int(element.get("fromid", 0)), int(element.get("toid", -1)) + 1)
                for item_id in ids:
                    table.get(item_id).floor_change = change.lower()
            element.clear()
    except (ET.ParseError, ValueError, OSError) as exc:
        raise OtbError(f"{os.path.basename(path)}: {exc}") from exc
    return table


def palette_rgb(index):
    """Tibia minimap colour index (6x6x6 web-safe cube) -> rgb."""
    index = int(index)
    if not 0 < index < 216:
        return None
    return (index // 36) * 51, ((index // 6) % 6) * 51, (index % 6) * 51


# ----- .otbm ------------------------------------------------------------- #

@dataclass
class OtbmInfo:
    version: int
    width: int
    height: int
    description: str = ""
    spawn_file: str = None
    house_file: str = None


def read_otbm_tiles(path, window=None, floors=None):
    """
    (OtbmInfo, {(x, y, z): [item ids, ground first]}) for tiles inside window = (x0, y0, x1, y1)
    (exclusive end) on `floors` (None = everything).
    """
    data = _read_file(path, magic=b"OTBM")
    floors = None if floors is None else set(floors)
    tiles = {}
    state = {"area": None, "tile": None, "info": None}

    def inside(x, y, z):
        if floors is not None and z not in floors:
            return False
        if window is None:
            return True
        x0, y0, x1, y1 = window
        return x0 <= x < x1 and y0 <= y < y1

    def enter(node_type, props, depth):
        if depth == 0:
            if len(props) < 16:
                raise OtbError("Nagłówek OTBM za krótki")
            version, width, height = struct.unpack_from("<IHH", props, 0)
            state["info"] = OtbmInfo(version, width, height)
            return True
        if node_type == OTBM_MAP_DATA:
            _map_attributes(props, state["info"])
            return True
        if node_type == OTBM_TILE_AREA:
            if len(props) < 5:
                raise OtbError("Uszkodzony obszar mapy")
            x, y, z = struct.unpack_from("<HHB", props, 0)
            state["area"] = (x, y, z)
            if floors is not None and z not in floors:
                return False
            if window is not None:
                x0, y0, x1, y1 = window
                if x + 256 <= x0 or x >= x1 or y + 256 <= y0 or y >= y1:
                    return False
            return True
        if node_type in (OTBM_TILE, OTBM_HOUSETILE) and state["area"] is not None:
            if len(props) < 2:
                return False
            ax, ay, z = state["area"]
            x, y = ax + props[0], ay + props[1]
            if not inside(x, y, z):
                state["tile"] = None
                return False
            pos = 2 + (4 if node_type == OTBM_HOUSETILE else 0)
            items = []
            while pos < len(props):
                attr = props[pos]
                if attr == ATTR_TILE_FLAGS and pos + 5 <= len(props):
                    pos += 5
                elif attr == ATTR_ITEM and pos + 3 <= len(props):
                    items.append(struct.unpack_from("<H", props, pos + 1)[0])
                    pos += 3
                else:
                    break
            state["tile"] = (x, y, z, depth)
            tiles[(x, y, z)] = items
            return True
        if node_type == OTBM_ITEM:
            tile = state["tile"]
            if tile is not None and depth == tile[3] + 1 and len(props) >= 2:
                tiles[(tile[0], tile[1], tile[2])].append(struct.unpack_from("<H", props, 0)[0])
            return False                     # container contents are not on the floor
        return node_type not in (12, 15)     # towns / waypoints are not needed

    walk_tree(data, enter)
    if state["info"] is None:
        raise OtbError("Brak węzła głównego OTBM")
    return state["info"], tiles


def _map_attributes(props, info):
    pos = 0
    names = {1: "description", 11: "spawn_file", 13: "house_file"}
    while pos + 3 <= len(props):
        attr = props[pos]
        length = struct.unpack_from("<H", props, pos + 1)[0]
        text = props[pos + 3: pos + 3 + length].decode("latin-1", "replace")
        pos += 3 + length
        if attr in names and info is not None:
            setattr(info, names[attr], text if attr != 1 else (info.description + text))


def load_otbm_area(path, x, y, cols, rows, floors, items=None, log=None):
    """
    TerrainMap for the window [x, x + cols) x [y, y + rows) on `floors` of an .otbm map.
    items: ItemTable (items.otb + items.xml) - without it walls cannot be told from floor.
    Returns (terrain, warnings).
    """
    floors = list(floors)
    warnings = []
    info, tiles = read_otbm_tiles(path, (x, y, x + cols, y + rows), floors)
    if log:
        log(f"OTBM v{info.version} {info.width}x{info.height}: {len(tiles)} pól w oknie.")
    if items is None or not items.has_flags:
        warnings.append("Brak items.otb - ściany rozpoznane tylko po braku podłogi (obiekty nie blokują).")
    terrain = TerrainMap()
    for z in floors:
        walls = np.ones((rows, cols), dtype=np.uint8)
        speed = np.full((rows, cols), DEFAULT_SPEED, dtype=np.uint16)
        stairs = np.zeros((rows, cols), dtype=bool)
        image = np.zeros((rows, cols, 4), dtype=np.uint8)
        image[..., 3] = 255
        count = 0
        for (tx, ty, tz), ids in tiles.items():
            if tz != z or not ids:
                continue
            lx, ly = tx - x, ty - y
            count += 1
            infos = [items.get(i) if items else ItemInfo() for i in ids]
            ground = infos[0]
            blocked = any(i.blocks for i in infos)
            change = next((i.floor_change for i in infos if i.floor_change), None)
            walls[ly, lx] = 1 if blocked and not change else 0
            if ground.speed:
                speed[ly, lx] = ground.speed
            stairs[ly, lx] = change is not None
            colour = next((palette_rgb(i.minimap) for i in reversed(infos) if palette_rgb(i.minimap)), None)
            if colour is None:
                colour = STAIRS_RGB if change else (WALL_RGB if walls[ly, lx] else FLOOR_RGB)
            image[ly, lx, :3] = STAIRS_RGB if change else colour
        if count:
            terrain.layers[z] = TerrainLayer(walls, speed, stairs, image, (x, y))
    if not terrain.layers:
        raise OtbError(f"Brak pól mapy w oknie ({x}, {y}) {cols}x{rows} na piętrach {floors}")
    terrain.transitions = link_transitions(terrain.layers)
    return terrain, warnings


__all__ = ["OtbError", "ItemTable", "ItemInfo", "load_items_otb", "load_items_xml", "read_otbm_tiles",
           "load_otbm_area", "walk_tree", "palette_rgb"]
