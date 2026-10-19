"""
Observation encoding shared by the training environment and external inference.

This module depends only on numpy, so brainstem.py can import it (or copy it)
and build the exact same feature vector from live game data that the policy
saw during training.
"""

from dataclasses import dataclass, field

import numpy as np

VIEW_RADIUS = 3          # local window: (2 * VIEW_RADIUS + 1) ** 2 cells per channel
SIGHT_RANGE = 8          # cells; monsters further away are invisible to the agent
ATTACK_RANGE = 4         # cells (Chebyshev) for Exori Frigo
POTION_NORMALIZER = 100  # potions_left feature saturates at this count

ACTION_NAMES = ["up", "down", "left", "right", "attack", "heal", "mana_potion"]
ACTION_LABELS_PL = ["Góra", "Dół", "Lewo", "Prawo", "Atak (Exori Frigo)",
                    "Leczenie (Exura Gran)", "Mikstura many"]

SCALAR_FEATURES = [
    "player_x", "player_y",
    "hp_ratio", "mana_ratio", "utamo_active", "utamo_remaining",
    "attack_cooldown", "heal_cooldown", "potion_cooldown", "potions_left",
    "monster_visible", "monster_dx", "monster_dy", "monster_distance",
    "monster_hp_ratio", "monster_attack_cooldown", "monster_in_attack_range", "monster_adjacent",
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
    hp_ratio: float
    attack_cooldown_ratio: float
    aggro: bool


@dataclass
class GameSnapshot:
    """Everything the policy needs to know about one tick of the game."""
    cols: int
    rows: int
    player_x: int
    player_y: int
    hp_ratio: float
    mana_ratio: float
    utamo_active: bool
    utamo_remaining_ratio: float
    attack_cooldown_ratio: float
    heal_cooldown_ratio: float
    potion_cooldown_ratio: float
    potions_left: int
    monsters: list = field(default_factory=list)        # list[MonsterView], alive only
    blocked: set = field(default_factory=set)           # {(x, y)} static obstacles
    next_waypoint: tuple = None                         # (x, y) or None


def chebyshev(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def encode_observation(snap):
    """Convert a GameSnapshot into a float32 vector of length OBS_SIZE in [-1, 1]."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    px, py = snap.player_x, snap.player_y

    visible = [m for m in snap.monsters if chebyshev(px, py, m.x, m.y) <= SIGHT_RANGE]
    nearest = min(visible, key=lambda m: (chebyshev(px, py, m.x, m.y), m.hp_ratio), default=None)

    obs[0] = px / max(1, snap.cols - 1)
    obs[1] = py / max(1, snap.rows - 1)
    obs[2] = snap.hp_ratio
    obs[3] = snap.mana_ratio
    obs[4] = 1.0 if snap.utamo_active else 0.0
    obs[5] = snap.utamo_remaining_ratio
    obs[6] = snap.attack_cooldown_ratio
    obs[7] = snap.heal_cooldown_ratio
    obs[8] = snap.potion_cooldown_ratio
    obs[9] = min(snap.potions_left, POTION_NORMALIZER) / POTION_NORMALIZER
    if nearest is not None:
        dist = chebyshev(px, py, nearest.x, nearest.y)
        obs[10] = 1.0
        obs[11] = np.clip((nearest.x - px) / SIGHT_RANGE, -1, 1)
        obs[12] = np.clip((nearest.y - py) / SIGHT_RANGE, -1, 1)
        obs[13] = dist / SIGHT_RANGE
        obs[14] = nearest.hp_ratio
        obs[15] = nearest.attack_cooldown_ratio
        obs[16] = 1.0 if dist <= ATTACK_RANGE else 0.0
        obs[17] = 1.0 if dist <= 1 else 0.0
    obs[18] = min(sum(1 for m in snap.monsters if m.aggro), 5) / 5.0
    if snap.next_waypoint is not None:
        wx, wy = snap.next_waypoint
        obs[19] = np.clip((wx - px) / SIGHT_RANGE, -1, 1)
        obs[20] = np.clip((wy - py) / SIGHT_RANGE, -1, 1)

    base = len(SCALAR_FEATURES)
    window = len(_WINDOW)
    monster_cells = {(m.x, m.y) for m in snap.monsters}
    for i, (dx, dy) in enumerate(_WINDOW):
        cx, cy = px + dx, py + dy
        out_of_bounds = not (0 <= cx < snap.cols and 0 <= cy < snap.rows)
        if out_of_bounds or (cx, cy) in snap.blocked:
            obs[base + i] = 1.0
        if (cx, cy) in monster_cells:
            obs[base + window + i] = 1.0
    return obs
