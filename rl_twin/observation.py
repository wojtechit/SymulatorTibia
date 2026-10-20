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
RUNE_NORMALIZER = 100     # sd_runes_left saturates at this count
PATH_NORMALIZER = 100.0   # waypoint_path_distance saturates at this travel cost
CLUSTER_NORMALIZER = 5    # avalanche_best_hits saturates at this many monsters
TACTICAL_STATES = ("lure", "pull", "fight", "escape", "recover", "return")
STEP_MS_NORMALIZER = 1000.0
MODIFIER_NORMALIZER = 200.0
MONSTER_SLOTS = 4         # nearest visible monsters described one by one (pack awareness)
TARGET_SLOTS = 4          # spawn groups on the floor, nearest by walking cost (route choice)
TARGET_DISTANCE = 50.0    # target dx/dy saturate at this many tiles
TARGET_PATH = 150.0       # target path cost saturates here
TARGET_AGE_MS = 120_000   # "time since visited" saturates at 2 minutes

ACTION_NAMES = [
    "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right",
    "exori_frigo", "exori_tera", "heal", "mana_potion", "wait",
    "use_sd", "use_avalanche",
    *[f"goto_target_{k}" for k in range(TARGET_SLOTS)], "goto_arena", "escape_stairs",
]
ACTION_LABELS_PL = [
    "Góra", "Dół", "Lewo", "Prawo",
    "Ukos ↖", "Ukos ↗", "Ukos ↙", "Ukos ↘",
    "Exori Frigo (lód)", "Exori Tera (ziemia)", "Exura Gran", "Mikstura many", "Czekaj",
    "Sudden Death (runa)", "Avalanche (runa)",
    *[f"Idź do grupy spawnów {k + 1}" for k in range(TARGET_SLOTS)], "Idź na arenę", "Ucieczka schodami",
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
    "cooldown_sd", "sd_runes_left", "rod_equipped", "rod_cooldown",
    "monster_in_rod_range", "monster_in_rune_range", "monster_death_modifier", "monster_rod_modifier",
    "monster_facing_dx", "monster_facing_dy", "aligned_horizontal", "aligned_vertical", "aligned_diagonal",
    "monster_has_aoe", "monster_spell_cooldown", "danger_here",
    "waypoint_path_dx", "waypoint_path_dy", "waypoint_path_distance",
    "cooldown_avalanche", "avalanche_runes_left", "avalanche_best_hits", "waypoint_floor_delta",
    "state_lure", "state_pull", "state_fight", "state_escape", "state_recover", "state_return",
    "in_arena", "arena_dx", "arena_dy", "orbit_dx", "orbit_dy",
    *[f"m{k}_{f}" for k in range(MONSTER_SLOTS) for f in ("dx", "dy", "hp", "adjacent", "threat", "ranged", "fleeing")],
    *[f"t{k}_{f}" for k in range(TARGET_SLOTS) for f in ("dx", "dy", "path", "alive", "age")],
    "hp_trend", "mana_trend", "damage_rate", "respawn_soon",
    "escape_dx", "escape_dy", "escape_path",
    "condition_damage", "slowed",
]
_WINDOW = [(dx, dy) for dy in range(-VIEW_RADIUS, VIEW_RADIUS + 1)
           for dx in range(-VIEW_RADIUS, VIEW_RADIUS + 1)]
GRID_FEATURES = ([f"blocked_{dx}_{dy}" for dx, dy in _WINDOW]
                 + [f"monster_{dx}_{dy}" for dx, dy in _WINDOW]
                 + [f"danger_{dx}_{dy}" for dx, dy in _WINDOW])
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
    death_modifier: int = 100
    rod_modifier: int = 100        # modifier for the element of the equipped rod
    facing: tuple = (0, 1)         # cardinal direction the monster looks at
    has_aoe: bool = False          # owns beam / wave / ball spells
    spell_cooldown_ratio: float = 1.0
    ranged: bool = False           # keeps distance (Canary targetDistance > 1)
    fleeing: bool = False          # below its run-health: runs and does not melee


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
    cooldown_ratios: dict          # keys: exori_frigo, exori_tera, heal, potion, sd, rod -> 0..1
    potions_left: int
    player_step_ms: int
    spell_range: int
    sd_runes: int = 0
    rune_range: int = 7
    rod_equipped: bool = False
    rod_range: int = 0
    collision: np.ndarray = None   # (rows, cols) uint8 for the player's floor, 1 = blocked
    monsters: list = field(default_factory=list)   # alive MonsterView, any floor
    next_waypoint: tuple = None    # (x, y, z) or None
    danger: np.ndarray = None      # (2r+1, 2r+1) beam/wave threat map from aoe.danger_window, or None
    waypoint_path: tuple = None    # (dx, dy, cost) from physics.path_direction toward the next waypoint
    avalanche_runes: int = 0
    avalanche_hits: int = 0        # monsters the best Avalanche aim would hit right now
    tactical_state: str = None     # tactics.STATES value, None when the hybrid controller is off
    arena: tuple = None            # (dx, dy, inside) toward the nearest arena centre
    orbit: tuple = None            # (dx, dy) unit tangent of the nearest orbit circle
    targets: list = None           # [(dx, dy, path_cost, believed_alive, ms_since_visit)] nearest spawn groups
    memory: tuple = None           # (hp_trend, mana_trend, damage_rate, respawn_soon) short-term history
    escape: tuple = None           # (dx, dy, path_cost) first step toward the nearest stairs / hole
    absolute_position: bool = True # False: player_x / player_y are zeroed (policy does not memorise one map)
    condition_damage: int = 0      # damage still to come from poison / burning / ... on the player
    slowed: bool = False           # player speed lowered by a slow / paralyze
    max_hp: int = 1


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

    if snap.absolute_position:
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
    obs[28] = cds.get("sd", 0.0)
    obs[29] = min(snap.sd_runes, RUNE_NORMALIZER) / RUNE_NORMALIZER
    obs[30] = 1.0 if snap.rod_equipped else 0.0
    obs[31] = cds.get("rod", 0.0)
    if nearest is not None:
        dist = chebyshev(px, py, nearest.x, nearest.y)
        obs[32] = 1.0 if snap.rod_equipped and dist <= snap.rod_range else 0.0
        obs[33] = 1.0 if dist <= snap.rune_range else 0.0
        obs[34] = nearest.death_modifier / MODIFIER_NORMALIZER
        obs[35] = nearest.rod_modifier / MODIFIER_NORMALIZER
        ddx, ddy = nearest.x - px, nearest.y - py
        obs[36] = nearest.facing[0]
        obs[37] = nearest.facing[1]
        obs[38] = 1.0 if ddy == 0 else 0.0
        obs[39] = 1.0 if ddx == 0 else 0.0
        obs[40] = 1.0 if abs(ddx) == abs(ddy) else 0.0
        obs[41] = 1.0 if nearest.has_aoe else 0.0
        obs[42] = nearest.spell_cooldown_ratio
    if snap.danger is not None:
        obs[43] = snap.danger[VIEW_RADIUS, VIEW_RADIUS]
    # Shortest-path step toward the waypoint (walls respected) - tells the agent how to leave a dead end
    if snap.waypoint_path is not None:
        step_x, step_y, cost = snap.waypoint_path
        obs[44] = step_x
        obs[45] = step_y
        obs[46] = 1.0 if cost == float("inf") else min(cost, PATH_NORMALIZER) / PATH_NORMALIZER
    obs[47] = cds.get("avalanche", 0.0)
    obs[48] = min(snap.avalanche_runes, RUNE_NORMALIZER) / RUNE_NORMALIZER
    obs[49] = min(snap.avalanche_hits, CLUSTER_NORMALIZER) / CLUSTER_NORMALIZER
    if snap.next_waypoint is not None:
        obs[50] = float(np.clip(snap.next_waypoint[2] - snap.player_z, -1, 1))
    if snap.tactical_state in TACTICAL_STATES:
        obs[51 + TACTICAL_STATES.index(snap.tactical_state)] = 1.0
    if snap.arena is not None:
        obs[57] = 1.0 if snap.arena[2] else 0.0
        obs[58] = np.clip(snap.arena[0] / VIEW_RANGE_X, -1, 1)
        obs[59] = np.clip(snap.arena[1] / VIEW_RANGE_Y, -1, 1)
    if snap.orbit is not None:
        obs[60], obs[61] = snap.orbit

    i = 62
    # Pack: the nearest visible monsters one by one
    ordered = sorted(visible, key=lambda m: (chebyshev(px, py, m.x, m.y), m.hp_ratio))
    for k in range(MONSTER_SLOTS):
        if k < len(ordered):
            m = ordered[k]
            ddx, ddy = m.x - px, m.y - py
            aligned = ddx == 0 or ddy == 0 or abs(ddx) == abs(ddy)
            obs[i: i + 7] = (ddx / VIEW_RANGE_X, ddy / VIEW_RANGE_Y, m.hp_ratio,
                             1.0 if max(abs(ddx), abs(ddy)) <= 1 else 0.0,
                             1.0 if m.has_aoe and aligned else 0.0,
                             1.0 if m.ranged else 0.0, 1.0 if m.fleeing else 0.0)
        i += 7
    # Spawn groups: where the next pack probably is (memory of what was already killed)
    for k in range(TARGET_SLOTS):
        if snap.targets is not None and k < len(snap.targets):
            tdx, tdy, cost, alive, age = snap.targets[k]
            obs[i: i + 5] = (np.clip(tdx / TARGET_DISTANCE, -1, 1), np.clip(tdy / TARGET_DISTANCE, -1, 1),
                             1.0 if cost == float("inf") else min(cost, TARGET_PATH) / TARGET_PATH,
                             min(alive, CLUSTER_NORMALIZER) / CLUSTER_NORMALIZER,
                             1.0 if age is None else min(age, TARGET_AGE_MS) / TARGET_AGE_MS)
        i += 5
    if snap.memory is not None:
        obs[i: i + 4] = np.clip(snap.memory, -1, 1)
    i += 4
    if snap.escape is not None:
        step_x, step_y, cost = snap.escape
        obs[i: i + 3] = (step_x, step_y, 1.0 if cost == float("inf") else min(cost, PATH_NORMALIZER) / PATH_NORMALIZER)
    i += 3
    obs[i] = min(1.0, snap.condition_damage / max(1, snap.max_hp))
    obs[i + 1] = 1.0 if snap.slowed else 0.0

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
    cells = size * size
    obs[base: base + cells] = blocked.ravel()
    obs[base + cells: base + 2 * cells] = monsters.ravel()
    if snap.danger is not None:
        obs[base + 2 * cells:] = snap.danger.ravel()
    return obs
