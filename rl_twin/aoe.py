"""
Area-of-effect geometry for monster spells (pure Python + numpy, shared with brainstem.py).

Directional spells are defined in the caster's frame as (forward, lateral) tiles:
  beam  - spread 0: one tile wide, `length` tiles in front of the caster
  wave  - cone; half-width at distance d is (d + spread - 2) // spread
          (length 7, spread 3 -> widths 1, 3, 3, 3, 5, 5, 5, as OTServ length/spread areas)
  ball  - targeted: circle of `radius` centred on the target (unavoidable by alignment,
          only by range or line of sight)
Monsters face one of four cardinal directions, so a druid standing exactly diagonal
to a dragon is outside every beam and outside the narrow first rows of a wave.
All shapes are cached; a hit test is a single frozenset lookup.
"""

from functools import lru_cache

import numpy as np

from .physics import has_line_of_sight

NORTH = (0, -1)
EAST = (1, 0)
SOUTH = (0, 1)
WEST = (-1, 0)
CARDINALS = (NORTH, EAST, SOUTH, WEST)

SPELL_BEAM = "beam"
SPELL_WAVE = "wave"
SPELL_BALL = "ball"
DIRECTIONAL_TYPES = (SPELL_BEAM, SPELL_WAVE)


def facing_toward(dx, dy, current=SOUTH):
    """Cardinal direction a creature turns to when looking at a tile (dx, dy) away."""
    if dx == 0 and dy == 0:
        return current
    if abs(dx) > abs(dy):
        return EAST if dx > 0 else WEST
    if abs(dy) > abs(dx):
        return SOUTH if dy > 0 else NORTH
    # Exact diagonal: keep facing if it already points along one of the two axes
    horizontal = EAST if dx > 0 else WEST
    vertical = SOUTH if dy > 0 else NORTH
    return current if current in (horizontal, vertical) else horizontal


def facing_after_step(dx, dy, current=SOUTH):
    """Walking diagonally turns a creature east or west (Tibia/OTServ rule)."""
    if dx != 0:
        return EAST if dx > 0 else WEST
    if dy != 0:
        return SOUTH if dy > 0 else NORTH
    return current


@lru_cache(maxsize=None)
def directional_area(length, spread, facing):
    """Frozenset of (dx, dy) offsets from the caster hit by a beam/wave facing `facing`."""
    fx, fy = facing
    tiles = set()
    for forward in range(1, length + 1):
        half = 0 if spread <= 0 else (forward + spread - 2) // spread
        for lateral in range(-half, half + 1):
            # rotate (forward, lateral) into world space; lateral axis is perpendicular to facing
            tiles.add((fx * forward - fy * lateral, fy * forward + fx * lateral))
    return frozenset(tiles)


@lru_cache(maxsize=None)
def ball_area(radius):
    """Frozenset of (dx, dy) offsets around the target hit by a ball of `radius`."""
    limit = (radius - 0.5) ** 2
    return frozenset((dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
                     if dx * dx + dy * dy <= limit)


def spell_area(spell, facing):
    if spell.type == SPELL_BALL:
        return ball_area(spell.radius)
    return directional_area(spell.length, spell.spread if spell.type == SPELL_WAVE else 0, facing)


def spell_hits(spell, facing, dx, dy):
    """Would `spell`, cast now by a caster facing `facing`, hit a target at offset (dx, dy)?"""
    if spell.type == SPELL_BALL:
        return max(abs(dx), abs(dy)) <= spell.range
    return (dx, dy) in directional_area(spell.length, spell.spread if spell.type == SPELL_WAVE else 0, facing)


@lru_cache(maxsize=None)
def danger_template(spells, facing):
    """
    Offsets (dx, dy arrays) a caster with these directional spells threatens, given that it
    turns toward the player before casting (tie-breaking on its current `facing`).
    """
    offsets = set()
    for direction in CARDINALS:
        for spell in spells:
            for dx, dy in spell_area(spell, direction):
                if facing_toward(dx, dy, facing) == direction:
                    offsets.add((dx, dy))
    ordered = sorted(offsets)
    return (np.array([o[0] for o in ordered], dtype=np.int32),
            np.array([o[1] for o in ordered], dtype=np.int32))


def danger_window(px, py, radius, casters, collision, grid_list=None):
    """
    (2r+1, 2r+1) float32 map around (px, py): 1.0 where standing would put the player inside
    a beam/wave of any caster.

    casters:   iterable of (x, y, facing, directional_spells_tuple) for monsters on the player's floor.
    collision: ndarray [y, x], 1 = wall; walls block areas (line of sight from the caster).
    grid_list: optional collision.tolist() for faster line-of-sight checks.
    """
    size = 2 * radius + 1
    danger = np.zeros((size, size), dtype=np.float32)
    rows, cols = collision.shape
    for mx, my, facing, spells in casters:
        if not spells:
            continue
        dxs, dys = danger_template(spells, facing)
        wx = mx + dxs - (px - radius)
        wy = my + dys - (py - radius)
        inside = (wx >= 0) & (wx < size) & (wy >= 0) & (wy < size)
        if not inside.any():
            continue
        wx, wy = wx[inside], wy[inside]
        # Line of sight only matters when a wall sits between caster and window
        x0, x1 = max(0, min(mx, px - radius)), min(cols, max(mx, px + radius) + 1)
        y0, y1 = max(0, min(my, py - radius)), min(rows, max(my, py + radius) + 1)
        if collision[y0:y1, x0:x1].any():
            grid = grid_list if grid_list is not None else collision.tolist()
            keep = [has_line_of_sight(grid, mx, my, int(x) + px - radius, int(y) + py - radius)
                    for x, y in zip(wx, wy)]
            wx, wy = wx[keep], wy[keep]
        danger[wy, wx] = 1.0
    return danger
