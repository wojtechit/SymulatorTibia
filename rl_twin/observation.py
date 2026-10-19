"""
Observation encoding shared by the training environment and external inference.

Depends only on numpy (plus the pure-Python physics helpers), so brainstem.py can
import it and build the exact feature vector the policy saw during training.
"""

from dataclasses import dataclass, field

import numpy as np

from .physics import VIEW_RANGE_X, VIEW_RANGE_Y, chebyshev

VIEW_RADIUS = 4           # local window: (2 * VIEW_RADIUS + 1) ** 2 cells per channel
POTION_NORMALIZER = 100   # potions_left saturates at this count
STEP_MS_NORMALIZER = 1000.0
MODIFIER_NORMALIZER = 200.0

ACTION_NAMES = [
    "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right",
    "exori_frigo", "exori_tera", "heal", "mana_potion", "wait",
]
ACTION_LABELS_PL = [
    "Góra", "Dół", "Lewo", "Prawo",
    "Ukos ↖", "Ukos ↗", "Ukos ↙", "Ukos ↘",
    "Exori Frigo (lód)", "Exori Tera (ziemia)", "Exura Gran", "Mikstura many", "Czekaj",
]

SCALAR_FEATURES = [
    "player_x", "player_y", "player_z",
    "hp_ratio", "mana_ratio", "utamo_active", "utamo_remaining",
    "cooldown_exori_frigo", "cooldown_exori_tera", "cooldown_heal", "cooldown_potion",
    "potions_left", "player_step_ms",
    "monster_visible", "monster_dx", "monster_dy", "monster_distance", "monster_hp_ratio",
    "monster_attack_cooldown", "monster_in_spell_range", "monster_line_of_sight", "monster_adjacent",
    "monster_ice_modifier", "monster_earth_modifier", "monster_step_ms",
    "aggro_count", "waypoint_dx", "waypoint_dy",
]
_WINDOW = [(dx, dy) for dy in range(-VIEW_RADIUS, VIEW_RADIUS + 1)
           for dx in range(-VIEW_RADIUS, VIEW_RADIUS + 1)]
GRID_FEATURES = ([f"blocked_{dx}_{dy}" for dx, dy in _WINDOW]
                 + [f"monster_{dx}_{dy}" for dx, dy in _WINDOW])
FEATURE_NAMES = SCALAR_FEATURES + GRID_FEATURES
OBS_SIZE = len(FEATURE_NAMES)


@dataclass
class MonsterView:
    x: int
    y: int
    z: int
    hp_ratio: float
    attack_cooldown_ratio: float
    aggro: bool
    ice_modifier: int = 100        # damage taken percent
    earth_modifier: int = 100
    step_ms: int = 300             # orthogonal step duration
    line_of_sight: bool = True


@dataclass
class GameSnapshot:
    """Everything the policy needs to know at one decision point."""
    cols: int
    rows: int
    player_x: int
    player_y: int
    player_z: int
    hp_ratio: float
    mana_ratio: float
    utamo_active: bool
    utamo_remaining_ratio: float
    cooldown_ratios: dict          # keys: exori_frigo, exori_tera, heal, potion -> 0..1
    potions_left: int
    player_step_ms: int
    spell_range: int
    collision: np.ndarray = None   # (rows, cols) uint8 for the player's floor, 1 = blocked
    monsters: list = field(default_factory=list)   # alive MonsterView, any floor
    next_waypoint: tuple = None    # (x, y, z) or None


def _visible(snap, m):
    return (m.z == snap.player_z
            and abs(m.x - snap.player_x) <= VIEW_RANGE_X and abs(m.y - snap.player_y) <= VIEW_RANGE_Y)


def encode_observation(snap):
    """Convert a GameSnapshot into a float32 vector of length OBS_SIZE (values roughly in [-1, 1])."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    px, py = snap.player_x, snap.player_y
    visible = [m for m in snap.monsters if _visible(snap, m)]
    nearest = min(visible, key=lambda m: (chebyshev(px, py, m.x, m.y), m.hp_ratio), default=None)
    cds = snap.cooldown_ratios

    obs[0] = px / max(1, snap.cols - 1)
    obs[1] = py / max(1, snap.rows - 1)
    obs[2] = snap.player_z / 15.0
    obs[3] = snap.hp_ratio
    obs[4] = snap.mana_ratio
    obs[5] = 1.0 if snap.utamo_active else 0.0
    obs[6] = snap.utamo_remaining_ratio
    obs[7] = cds.get("exori_frigo", 0.0)
    obs[8] = cds.get("exori_tera", 0.0)
    obs[9] = cds.get("heal", 0.0)
    obs[10] = cds.get("potion", 0.0)
    obs[11] = min(snap.potions_left, POTION_NORMALIZER) / POTION_NORMALIZER
    obs[12] = snap.player_step_ms / STEP_MS_NORMALIZER
    if nearest is not None:
        dist = chebyshev(px, py, nearest.x, nearest.y)
        obs[13] = 1.0
        obs[14] = (nearest.x - px) / VIEW_RANGE_X
        obs[15] = (nearest.y - py) / VIEW_RANGE_Y
        obs[16] = dist / VIEW_RANGE_X
        obs[17] = nearest.hp_ratio
        obs[18] = nearest.attack_cooldown_ratio
        obs[19] = 1.0 if dist <= snap.spell_range else 0.0
        obs[20] = 1.0 if nearest.line_of_sight else 0.0
        obs[21] = 1.0 if dist <= 1 else 0.0
        obs[22] = nearest.ice_modifier / MODIFIER_NORMALIZER
        obs[23] = nearest.earth_modifier / MODIFIER_NORMALIZER
        obs[24] = nearest.step_ms / STEP_MS_NORMALIZER
    obs[25] = min(sum(1 for m in visible if m.aggro), 5) / 5.0
    if snap.next_waypoint is not None and snap.next_waypoint[2] == snap.player_z:
        obs[26] = np.clip((snap.next_waypoint[0] - px) / VIEW_RANGE_X, -1, 1)
        obs[27] = np.clip((snap.next_waypoint[1] - py) / VIEW_RANGE_Y, -1, 1)

    # Local window, out-of-bounds counts as blocked
    r = VIEW_RADIUS
    size = 2 * r + 1
    blocked = np.ones((size, size), dtype=np.float32)
    x0, x1 = max(0, px - r), min(snap.cols, px + r + 1)
    y0, y1 = max(0, py - r), min(snap.rows, py + r + 1)
    if snap.collision is not None and x0 < x1 and y0 < y1:
        blocked[y0 - (py - r): y1 - (py - r), x0 - (px - r): x1 - (px - r)] = snap.collision[y0:y1, x0:x1]
    elif x0 < x1 and y0 < y1:
        blocked[y0 - (py - r): y1 - (py - r), x0 - (px - r): x1 - (px - r)] = 0.0
    monsters = np.zeros((size, size), dtype=np.float32)
    for m in visible:
        wx, wy = m.x - px + r, m.y - py + r
        if 0 <= wx < size and 0 <= wy < size:
            monsters[wy, wx] = 1.0
    base = len(SCALAR_FEATURES)
    obs[base: base + size * size] = blocked.ravel()
    obs[base + size * size:] = monsters.ravel()
    return obs
