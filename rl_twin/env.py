"""
Tibia-accurate Druid hunting environment with a Gymnasium-style API.

    env = DruidHuntEnv(EnvConfig(...))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

Time model (semi-MDP): every step() is one agent decision. A move consumes the
druid's real step duration (diagonal = x1.5), any other action consumes
DECISION_MS. During that time the world advances in 50 ms server ticks, so
monsters move on their own step timers and attack every attack_interval_ms.
"""

import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from .aoe import SPELL_BALL, danger_window, facing_after_step, facing_toward, spell_area, spell_hits
from .aoe import ball_area
from .combat import (CURES, ELEMENT_LABELS_PL, HASTES, PHYSICAL, POTION_EXHAUST_MS, SPELLBOOK,
                     UTAMO_VITA_DURATION_MS)
from .entities import GROUND_FLOOR, Druid, Monster
from .items import ItemDatabase
from .monsters import MonsterDatabase
from .rewards import Economy, RewardConfig
from .tactics import (ESCAPE, FIGHT, PULL, RECOVER, STATE_LABELS_PL, TacticalController, TacticsConfig,
                      orbit_tangent, zone_contains)
from .terrain import decode_speed
from .observation import (ACTION_NAMES, OBS_SIZE, TARGET_SLOTS, VIEW_RADIUS, GameSnapshot, MonsterView,
                          encode_observation)
from .physics import (
    DEFAULT_DIAGONAL_MULTIPLIER,
    DEFAULT_GROUND_SPEED,
    SERVER_TICK_MS,
    a_star,
    chebyshev,
    distance_field,
    has_line_of_sight,
    in_view,
    is_diagonal,
    path_direction,
)

CONDITION_LABELS_PL = {"earth": "Trucizna", "fire": "Podpalenie", "ice": "Zamrożenie", "energy": "Porażenie",
                       "holy": "Oślepienie", "death": "Klątwa", "physical": "Krwawienie"}
DECISION_MS = 100          # game time consumed by a non-movement action
AOE_FLASH_MS = 600         # how long the GUI shows a monster spell area
TIME_UNIT_MS = 500         # the -1 step penalty is charged per 500 ms of game time

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT, ACTION_DOWN_RIGHT = 4, 5, 6, 7
ACTION_EXORI_FRIGO, ACTION_EXORI_TERA, ACTION_HEAL, ACTION_MANA_POTION, ACTION_WAIT = 8, 9, 10, 11, 12
ACTION_USE_SD = 13
ACTION_USE_AVALANCHE = 14
ACTION_GOTO_TARGETS = tuple(range(15, 15 + TARGET_SLOTS))      # walk toward the k-th nearest spawn group
ACTION_GOTO_ARENA = 15 + TARGET_SLOTS
ACTION_ESCAPE = ACTION_GOTO_ARENA + 1                           # walk toward the nearest stairs / hole
ACTION_CURE = ACTION_ESCAPE + 1                                 # Exana Pox / Flam / Vis for the worst condition
ACTION_HASTE = ACTION_CURE + 1                                  # Utani Gran Hur (level 20+) / Utani Hur
FIELD_MS = 30_000          # monster-made fields on the ground last this long
# element -> (damage on entering, condition total, condition tick ms, ticks)   (TFS field items, simplified)
FIELD_EFFECTS = {"fire": (20, 70, 4000, 7), "earth": (0, 80, 4000, 8), "energy": (30, 75, 4000, 3)}
FIELD_LABELS_PL = {"fire": "pole ognia", "earth": "pole trucizny", "energy": "pole energii"}
MAX_SUMMONS_ALIVE = 30     # safety cap for the whole map
CLUSTER_RADIUS = 6         # spawns closer than this (Chebyshev) form one spawn group
HISTORY_MS = 2_000         # hp / mana trend window
DAMAGE_WINDOW_MS = 5_000   # damage_rate window
RESPAWN_SOON_MS = 10_000
HOME_RADIUS = 2            # idle monsters farther than this from their spawn walk back   # respawn_soon counts known kills that come back within this time
MOVE_DELTAS = {
    ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0),
    ACTION_UP_LEFT: (-1, -1), ACTION_UP_RIGHT: (1, -1),
    ACTION_DOWN_LEFT: (-1, 1), ACTION_DOWN_RIGHT: (1, 1),
}
ATTACK_SPELLS = {ACTION_EXORI_FRIGO: "exori_frigo", ACTION_EXORI_TERA: "exori_tera"}
STRIKE_RANGE = SPELLBOOK["exori_frigo"].range
RUNE_RANGE = SPELLBOOK["sudden_death"].range
AVALANCHE = SPELLBOOK["avalanche"]
AVALANCHE_AREA = tuple(ball_area(AVALANCHE.area_radius))
AVALANCHE_AREA_SET = frozenset(AVALANCHE_AREA)


class Discrete:
    """Minimal stand-in for gymnasium.spaces.Discrete."""

    def __init__(self, n):
        self.n = n

    def sample(self, rng=random):
        return rng.randrange(self.n)

    def contains(self, value):
        return isinstance(value, (int, np.integer)) and 0 <= value < self.n


class Box:
    """Minimal stand-in for gymnasium.spaces.Box."""

    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype


def grid_to_rows(grid):
    return ["".join("1" if v else "0" for v in row) for row in np.asarray(grid)]


def rows_to_grid(rows, height, width):
    grid = np.zeros((height, width), dtype=np.uint8)
    for y, line in enumerate(rows[:height]):
        values = np.frombuffer(line[:width].encode("ascii"), dtype=np.uint8) - ord("0")
        grid[y, : len(values)] = values > 0
    return grid


@dataclass
class EnvConfig:
    cols: int = 32
    rows: int = 24
    level: int = 100
    magic_level: int = 60
    armor: int = 30
    use_utamo: bool = True
    mana_potions: int = 100
    collision: dict = field(default_factory=dict)   # {"7": ["0010...", ...]} one string per row
    waypoints: list = field(default_factory=list)   # [[x, y, z], ...]
    spawns: list = field(default_factory=list)      # [[x, y, z, "Giant Spider"], ...]
    floor: int = GROUND_FLOOR                       # used when there are no waypoints
    random_monster_count: int = 3                   # used when spawns is empty
    random_monster_type: str = "Giant Spider"
    respawn: bool = True                            # killed monsters come back at their spawn (Tibia spawntime)
    respawn_ms: int = 60_000                        # default spawntime; a 5th spawn field [.., seconds] overrides
    absolute_position: bool = False                 # True: observe absolute x/y (ties a policy to one map)
    max_steps: int = 600
    ground_speed: int = DEFAULT_GROUND_SPEED
    diagonal_multiplier: float = DEFAULT_DIAGONAL_MULTIPLIER
    monsters_file: str = None                       # None = rl_twin/data/monsters.json
    rod: str = None                                 # rod name from items.json, None = no weapon
    sd_runes: int = 0
    avalanche_runes: int = 0
    items_file: str = None                          # None = rl_twin/data/items.json
    exp_rate: float = 1.0                           # server experience stage / boost multiplier
    terrain_speed: dict = field(default_factory=dict)   # {"7": base64 uint16 ground speed per tile}
    transitions: list = field(default_factory=list)     # [[x, y, z, x2, y2, z2], ...] stairs / holes / ladders
    zones: list = field(default_factory=list)           # tactics.make_zone dicts (lure / arena / orbit)
    tactics: TacticsConfig = field(default_factory=TacticsConfig)
    world_origin: list = field(default_factory=lambda: [0, 0])   # world (x, y) of tile [0, 0] for scripts
    seed: int = None
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        rewards = RewardConfig(**{k: v for k, v in data.pop("rewards", {}).items()
                                  if k in RewardConfig.__dataclass_fields__})
        tactics = TacticsConfig(**{k: v for k, v in data.pop("tactics", {}).items()
                                   if k in TacticsConfig.__dataclass_fields__})
        floor = int(data.get("floor", GROUND_FLOOR))
        # Phase 1 scenarios: 2D waypoints/spawns and an obstacle list
        data["waypoints"] = [list(w) + [floor] if len(w) == 2 else list(w) for w in data.get("waypoints", [])]
        data["spawns"] = [[s[0], s[1], floor, s[2]] if len(s) == 3 else list(s) for s in data.get("spawns", [])]
        legacy = data.pop("obstacles", None)
        if legacy and not data.get("collision"):
            grid = np.zeros((data.get("rows", 24), data.get("cols", 32)), dtype=np.uint8)
            for x, y, *_ in legacy:
                if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                    grid[y, x] = 1
            data["collision"] = {str(floor): grid_to_rows(grid)}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(rewards=rewards, tactics=tactics, **known)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def speed_grid(self, z):
        encoded = self.terrain_speed.get(str(z))
        if encoded is None:
            return np.full((self.rows, self.cols), self.ground_speed, dtype=np.uint16)
        return decode_speed(encoded, self.rows, self.cols)

    def collision_grid(self, z):
        rows = self.collision.get(str(z))
        if rows is None:
            return np.zeros((self.rows, self.cols), dtype=np.uint8)
        return rows_to_grid(rows, self.rows, self.cols)


class DruidHuntEnv:
    """Episode ends on death, when all monsters die (terminated) or at max_steps (truncated)."""

    def __init__(self, config=None, record_events=False, monster_db=None, item_db=None):
        self.config = config or EnvConfig()
        self.record_events = record_events
        self.monster_db = monster_db or MonsterDatabase.load(self.config.monsters_file)
        self.item_db = item_db or ItemDatabase.load(self.config.items_file)
        if self.config.rod and not self.item_db.has_rod(self.config.rod):
            raise ValueError(f"Unknown rod: {self.config.rod}")
        self.action_space = Discrete(len(ACTION_NAMES))
        self.observation_space = Box(-1.0, 1.0, (OBS_SIZE,))
        self.rng = random.Random(self.config.seed)
        self.grids = {}          # z -> np.ndarray, built lazily
        self._grid_lists = {}    # z -> list of lists for fast A* / LOS
        self._waypoint_fields = {}   # (waypoint index, floor) -> distance field (static map, solved once)
        self._speeds = {}            # z -> ground speed grid
        self.transitions = {(int(t[0]), int(t[1]), int(t[2])): (int(t[3]), int(t[4]), int(t[5]))
                            for t in self.config.transitions}
        self.economy = None
        self.tactics = TacticalController(self.config.tactics, [dict(z) for z in self.config.zones])
        self.last_action = None
        self.last_controlled = False
        self._route_fields = {}
        self.waypoints = [tuple(int(v) for v in w[:3]) for w in self.config.waypoints]
        self.waypoints = [w for w in self.waypoints if self._walkable_static(*w)]
        self.druid = None
        self.monsters = []
        self.events = []
        self.aoe_flashes = []    # (tiles, z, element, until_ms) for the visualizer; only with record_events
        self.time_ms = 0
        self.steps = 0
        self.waypoint_index = 0
        self.kills = 0
        self.experience = 0
        self.damage_taken = 0
        self.hp_min_ratio = 1.0
        self.adjacent_ms = 0
        self.heals = 0
        self.episode_reward = 0.0
        self.clusters = []           # spawn groups: {"x", "y", "z", "members", "visited_ms"}
        self._hunt_target = None
        self.floor_changed_ms = None
        self.respawns = 0
        self.escapes_learned = 0
        self._history = deque()      # (time_ms, hp, mana, damage_taken) for the memory features
        self.fields = {}             # (x, y, z) -> (element, until_ms) fire / poison / energy on the ground
        self._monster_names = {name.lower(): name for name in self.monster_db.names()}

    # ----- Gym API --------------------------------------------------------- #

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        cfg = self.config
        x, y, z = self._start_position()
        rod = self.item_db.rod(cfg.rod) if cfg.rod else None
        self.druid = Druid(cfg.level, cfg.magic_level, cfg.armor, cfg.mana_potions, x, y, z,
                           rod=rod, sd_runes=cfg.sd_runes, avalanche_runes=cfg.avalanche_runes)
        self.economy = Economy(self.item_db.supply_prices(), cfg.exp_rate, cfg.rewards.exp_gold_value)
        self.tactics.reset(self.druid.z)
        if self.tactics.config.enabled and not self.escape_routes(self.druid.z):
            self._event("warn", f"Taktyka: brak schodów/dziur na piętrze {self.druid.z} - awaryjna ucieczka niemożliwa "
                                f"(wczytaj sąsiednie piętro, żeby połączyć przejścia).")
        self.monsters = self._create_monsters()
        self.clusters = self._build_clusters()
        self._hunt_target = None
        self.floor_changed_ms = None
        self.fields = {}
        self.respawns = 0
        self.escapes_learned = 0
        self._history = deque()
        self.time_ms = 0
        self.steps = 0
        self.waypoint_index = 1 if len(self.waypoints) > 1 else 0
        self.kills = 0
        self.experience = 0
        self.damage_taken = 0
        self.hp_min_ratio = 1.0
        self.adjacent_ms = 0
        self.heals = 0
        self.episode_reward = 0.0
        self.events = []
        self.aoe_flashes = []
        if rod is not None and self.druid.rod is None:
            self._event("warn", f"{rod.name} wymaga poziomu {rod.level} - walczę bez różdżki.")
        if cfg.sd_runes and self.druid.magic_level < SPELLBOOK["sudden_death"].min_magic_level:
            self._event("warn", "Sudden Death wymaga poziomu magii 15 - runy będą bezużyteczne.")
        if cfg.use_utamo and self.druid.cast_utamo(self.time_ms):
            self._event("shield", "Druid rzuca Utamo Vita.")
        return self._observe(), self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        self.events = []
        aggro_before = len(self.aggressive_monsters())
        z_before = self.druid.z
        tactics = self.config.tactics
        danger_before = self.druid.hp_ratio < tactics.escape_hp or aggro_before > tactics.safe_max
        reward = self._tactical_update()
        action, controlled = self.tactics.choose_action(self, int(action))
        self.last_action, self.last_controlled = action, controlled
        reward += self._aoe_discipline(action)
        step_reward, elapsed = self._apply_action(int(action))
        reward += step_reward
        reward += self._simulate_world(elapsed)
        reward += self.config.rewards.step * elapsed / TIME_UNIT_MS
        reward += self._after_step(z_before, danger_before)

        druid = self.druid
        terminated = False
        if not druid.alive:
            reward += self.config.rewards.death
            terminated = True
            self._event("death", "DRUID ZGINĄŁ! Koniec epizodu.")
        elif not self.config.respawn and not any(m.alive for m in self.monsters):
            terminated = True
            self._event("system", "Wszystkie potwory pokonane - epizod zakończony sukcesem.")
        self.steps += 1
        truncated = not terminated and self.steps >= self.config.max_steps
        if truncated:
            self._event("system", f"Limit {self.config.max_steps} decyzji - epizod przerwany.")
        self.episode_reward += reward
        return self._observe(), reward, terminated, truncated, self._info()

    # ----- agent actions --------------------------------------------------- #

    def _apply_action(self, action):
        """Execute the agent's action. Returns (reward, game milliseconds consumed)."""
        rewards = self.config.rewards
        druid = self.druid
        now = self.time_ms

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            nx, ny = druid.x + dx, druid.y + dy
            blocked = self._movement_blocker(nx, ny, druid.z)
            if blocked is not None:
                # Rejected: the druid stays put, the clock still advances, invalid-action penalty
                self._event("info", f"Ruch zablokowany ({blocked}) na ({nx}, {ny}).")
                return rewards.invalid_action, DECISION_MS
            druid.move_to(nx, ny)
            duration = self._step_duration(druid, is_diagonal(dx, dy))   # ground speed of the entered tile
            reward = self._touch_field()
            self._enter_tile(druid)
            return reward + self._check_waypoint(), duration

        if action == ACTION_CURE:
            element = self._curable_condition()
            spell = SPELLBOOK[CURES[element]] if element else None
            if spell is None or not druid.can_cast(spell, now):
                return rewards.invalid_action, DECISION_MS
            druid.pay_for(spell, now)
            prevented = druid.cure(element)
            self._event("heal", f"{spell.name} ({spell.words}): usunięto efekt, -{prevented} obrażeń w czasie.")
            return 0.0, DECISION_MS

        if action == ACTION_HASTE:
            key = self._haste_spell()
            spell = SPELLBOOK[key] if key else None
            if spell is None or not druid.can_cast(spell, now):
                return rewards.invalid_action, DECISION_MS
            factor, flat, duration, _ = HASTES[key]
            was_slowed = druid.slowed
            druid.pay_for(spell, now)
            druid.haste(factor, flat, duration, now)
            self._event("shield", f"{spell.name} ({spell.words}): prędkość {druid.speed}"
                                  + (" - paraliż zdjęty." if was_slowed else "."))
            return 0.0, DECISION_MS

        if action in ACTION_GOTO_TARGETS or action in (ACTION_GOTO_ARENA, ACTION_ESCAPE):
            step = self._macro_step(action)
            if step is None:
                return rewards.invalid_action, DECISION_MS
            return self._apply_action(self.move_action(step))

        if action in ATTACK_SPELLS:
            spell = SPELLBOOK[ATTACK_SPELLS[action]]
            target = self._select_target(spell.range)
            if target is None or not druid.can_cast(spell, now):
                return rewards.invalid_action, DECISION_MS
            druid.pay_for(spell, now)
            raw = spell.roll(druid.level, druid.magic_level, self.rng)
            return self._hit_monster(target, raw, spell.element, spell.name), DECISION_MS

        if action == ACTION_USE_SD:
            rune = SPELLBOOK["sudden_death"]
            target = self._select_target(rune.range)
            if target is None or not druid.can_use_rune(rune, now):
                return rewards.invalid_action, DECISION_MS
            druid.use_rune(rune, now)
            reward = -self._spend("sudden_death_rune")
            raw = rune.roll(druid.level, druid.magic_level, self.rng)
            hp_before = target.hp
            reward += self._hit_monster(target, raw, rune.element, f"{rune.name} [{druid.sd_runes}]")
            self._book_overkill("sudden_death_rune", [(target, hp_before)])
            return reward, DECISION_MS

        if action == ACTION_USE_AVALANCHE:
            aim = self._avalanche_aim()
            if aim is None or not druid.can_use_rune(AVALANCHE, now):
                return rewards.invalid_action, DECISION_MS
            (cx, cy), targets = aim
            druid.use_rune(AVALANCHE, now)
            reward = -self._spend("avalanche_rune")
            raw = AVALANCHE.roll(druid.level, druid.magic_level, self.rng)   # one roll, per-target modifiers
            if self.record_events:
                self.aoe_flashes.append(([(cx + dx, cy + dy) for dx, dy in AVALANCHE_AREA], druid.z, "ice",
                                         now + 600))
            self._event("attack", f"Avalanche [{druid.avalanche_runes}] na ({cx}, {cy}): trafia {len(targets)} potw.")
            before = [(target, target.hp) for target in targets]
            for target in targets:
                reward += self._hit_monster(target, raw, AVALANCHE.element, "Avalanche")
            self._book_overkill("avalanche_rune", before)
            return reward, DECISION_MS

        if action == ACTION_HEAL:
            spell = SPELLBOOK["exura_gran"]
            if druid.hp >= druid.max_hp or not druid.can_cast(spell, now):
                return rewards.invalid_action, DECISION_MS
            druid.pay_for(spell, now)
            healed = druid.heal(spell.roll(druid.level, druid.magic_level, self.rng))
            self.heals += 1
            self._event("heal", f"Exura Gran: +{healed} HP. HP: {druid.hp}/{druid.max_hp}.")
            return 0.0, DECISION_MS

        if action == ACTION_MANA_POTION:
            if not druid.can_drink_potion(now):
                return rewards.invalid_action, DECISION_MS
            restored, rolled = druid.drink_mana_potion(now, self.rng)
            self.economy.waste("great_mana_potion", 1 - restored / rolled)
            self._event("mana", f"Great Mana Potion: +{restored} many. "
                                f"Mana: {druid.mana}/{druid.max_mana}. Zostało: {druid.potions}.")
            return -self._spend("great_mana_potion"), DECISION_MS

        return 0.0, DECISION_MS  # ACTION_WAIT

    def _hit_monster(self, target, raw, element, source):
        """Apply attack damage (armor/resistance aware). Returns the economic reward of this hit."""
        rewards = self.config.rewards
        dealt, percent = target.take_spell_damage(raw, element, self.rng)
        target.aggro = True
        self._event("attack", f"{source} ({ELEMENT_LABELS_PL[element]}, {percent}%) "
                              f"w {target.name}: {dealt} obr. HP: {target.hp}/{target.max_hp}.")
        reward = rewards.exp_scale * self.economy.exp_for_damage(target, dealt)
        if target.alive:
            return reward
        exp, loot = self.economy.book_kill(target)
        target.died_at = target.known_dead_at = self.time_ms
        for summon in target.summoned:
            if summon.alive:
                summon.hp = 0                                   # summons vanish with their master (no exp)
        target.summoned = []
        self.kills += 1
        self.experience = self.economy.experience
        self._event("kill", f"{target.name} ginie! +{exp:.0f} exp, loot ~{loot} gp.")
        if rewards.loot_in_reward:
            reward += rewards.gold_scale * loot
        return reward + rewards.kill

    def _book_overkill(self, supply, hits):
        """Share of a rune's damage that landed beyond the targets' remaining HP is wasted gold."""
        damage = sum(getattr(m, "last_damage", 0) for m, _ in hits)
        useful = sum(min(getattr(m, "last_damage", 0), hp) for m, hp in hits)
        if damage > 0:
            self.economy.waste(supply, (damage - useful) / damage)

    # ----- hybrid tactics -------------------------------------------------- #

    def _tactical_update(self):
        """Advance the tactical state machine; returns shaping reward for completed transitions."""
        change = self.tactics.update(self)
        if change is None:
            return 0.0
        old, new = change
        rewards = self.config.rewards
        self._event("system", f"Taktyka: {STATE_LABELS_PL[old]} -> {STATE_LABELS_PL[new]}"
                              f" (HP {self.druid.hp_ratio * 100:.0f}%, potwory {len(self.aggressive_monsters())}).")
        if new == FIGHT and old in ("lure", PULL):
            return rewards.lure_bonus * len(self.aggressive_monsters())
        if old == ESCAPE and new == RECOVER:
            return rewards.escape_success
        return 0.0

    def _aoe_discipline(self, action):
        """Penalise single-target attacks while a usable Avalanche would hit a whole pack."""
        if action not in (ACTION_EXORI_FRIGO, ACTION_EXORI_TERA, ACTION_USE_SD):
            return 0.0
        rewards = self.config.rewards
        if not self.druid.can_use_rune(AVALANCHE, self.time_ms):
            return 0.0
        aim = self._avalanche_aim()
        if aim and len(aim[1]) >= rewards.aoe_min_targets:
            return rewards.aoe_missed
        return 0.0

    def _arena_vector(self):
        druid = self.druid
        arena = self.tactics.nearest_zone("arena", druid.x, druid.y, druid.z)
        if arena is None:
            return None
        return arena["x"] - druid.x, arena["y"] - druid.y, zone_contains(arena, druid.x, druid.y, druid.z)

    def _orbit_vector(self):
        druid = self.druid
        orbit = self.tactics.nearest_zone("orbit", druid.x, druid.y, druid.z)
        return None if orbit is None else orbit_tangent(orbit, druid.x, druid.y)

    def aggressive_monsters(self):
        druid = self.druid
        return [m for m in self.monsters if m.alive and m.aggro and m.z == druid.z]

    def escape_routes(self, z, toward=None):
        """(x, y) of floor-change tiles on floor z, optionally only those leading toward floor `toward`."""
        routes = []
        for (x, y, tz), (_, _, target_z) in self.transitions.items():
            if tz != z:
                continue
            if toward is not None and (target_z - z) * (toward - z) <= 0:
                continue
            routes.append((x, y))
        return routes

    def step_toward(self, goals):
        """First (dx, dy) of the shortest path to any goal on the druid's floor (other stairs avoided)."""
        if not goals:
            return None
        return self._free_step(self._route_field(goals))

    def _free_step(self, field):
        """Best neighbouring tile along `field` that is not blocked right now (walls, monsters)."""
        druid = self.druid
        rows, cols = field.shape
        here = float(field[druid.y, druid.x])
        if here == 0.0 or here == float("inf"):
            return None
        options = []
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = druid.x + dx, druid.y + dy
            if 0 <= nx < cols and 0 <= ny < rows and field[ny, nx] < here:
                cost = float(field[ny, nx]) + (self.config.diagonal_multiplier if dx and dy else 1.0)
                options.append((cost, dx, dy))
        options.sort()
        for avoid_fields in (True, False):                  # like the client's autowalk: around fields if possible
            for _, dx, dy in options:
                tile = (druid.x + dx, druid.y + dy, druid.z)
                if avoid_fields and tile in self.fields:
                    continue
                if self._movement_blocker(*tile) is None:
                    return dx, dy
        return None

    def waypoint_step(self):
        if not self.waypoints or self._waypoint_path() is None:
            return None
        field = self._waypoint_fields.get((self.waypoint_index, self.druid.z))
        return None if field is None or field is False else self._free_step(field)

    @staticmethod
    def move_action(step):
        if step is None:
            return ACTION_WAIT
        return next(action for action, delta in MOVE_DELTAS.items() if delta == tuple(step))

    # ----- hunting ground memory, respawn and macro moves --------------------- #

    def _build_clusters(self):
        """Group spawn points (monster homes) that lie close together on one floor."""
        clusters = []
        for monster in self.monsters:
            hx, hy, hz = monster.home
            for cluster in clusters:
                if cluster["z"] == hz and any(chebyshev(hx, hy, *m.home[:2]) <= CLUSTER_RADIUS
                                              for m in cluster["members"]):
                    cluster["members"].append(monster)
                    break
            else:
                clusters.append({"z": hz, "members": [monster], "visited_ms": None})
        for cluster in clusters:
            homes = [m.home for m in cluster["members"]]
            mx = sum(h[0] for h in homes) / len(homes)
            my = sum(h[1] for h in homes) / len(homes)
            cluster["x"], cluster["y"] = min(((h[0], h[1]) for h in homes),
                                             key=lambda p: (p[0] - mx) ** 2 + (p[1] - my) ** 2)
        return clusters

    def believed_alive(self, cluster):
        """
        Monsters the player expects at a spawn group: seen alive now; otherwise not known dead (or due to
        respawn) and not missing when the group was last checked (a monster that ran off is not 'there').
        """
        druid, now = self.druid, self.time_ms
        visited = cluster["visited_ms"]
        count = 0
        for m in cluster["members"]:
            if m.z == druid.z and in_view(druid.x, druid.y, m.x, m.y):
                count += 1 if m.alive else 0
            elif m.known_dead_at is not None and now - m.known_dead_at < m.respawn_ms:
                continue
            elif (visited is not None and now - visited < m.respawn_ms
                  and (m.seen_alive_ms is None or m.seen_alive_ms < visited)):
                continue                                    # checked recently and it was not there
            else:
                count += 1
        return count

    def _remember(self):
        """Update what the player knows: visited spawn groups, monsters seen alive / dead, short history."""
        druid, now = self.druid, self.time_ms
        for cluster in self.clusters:
            if cluster["z"] == druid.z and in_view(druid.x, druid.y, cluster["x"], cluster["y"]):
                cluster["visited_ms"] = now
        for m in self.monsters:
            if m.z == druid.z and in_view(druid.x, druid.y, m.x, m.y):
                if m.alive:
                    m.known_dead_at = None
                    m.seen_alive_ms = now
                elif m.known_dead_at is None:
                    m.known_dead_at = m.died_at
        self._history.append((now, druid.hp, druid.mana, self.damage_taken))
        while len(self._history) > 2 and now - self._history[1][0] >= DAMAGE_WINDOW_MS:
            self._history.popleft()

    def _respawn(self):
        """Dead monsters return to their spawn after respawn_ms unless the druid sees the spawn tile (Tibia rule)."""
        if not self.config.respawn:
            return
        druid, now = self.druid, self.time_ms
        for m in self.monsters:
            if m.alive or m.died_at is None or now - m.died_at < m.respawn_ms or m.master is not None:
                continue
            hx, hy, hz = m.home
            if hz == druid.z and in_view(druid.x, druid.y, hx, hy):
                continue
            if not self._is_free(hx, hy, hz, ignore=m):
                continue
            m.revive()
            self.respawns += 1
            self._event("info", f"{m.name} odradza się na spawnie ({hx}, {hy}, {hz}).")

    def _after_step(self, z_before, danger_before):
        """Respawns, memory and shaping for decisions the RL policy now owns (lure size, escapes)."""
        self._respawn()
        self._remember()
        now = self.time_ms
        if self.fields:
            self.fields = {tile: field for tile, field in self.fields.items() if field[1] > now}
        if any(m.master is not None and not m.alive for m in self.monsters):
            self.monsters = [m for m in self.monsters if m.master is None or m.alive]
        rewards = self.config.rewards
        reward = 0.0
        if not self.tactics.config.enabled:
            # Each spawned monster counts once: losing aggro and pulling it again earns nothing
            fresh = [m for m in self.aggressive_monsters() if not m.lured]
            for m in fresh:
                m.lured = True
            reward += rewards.aggro_gain * len(fresh)
            if danger_before and self.druid.z != z_before and self.druid.alive:
                self.escapes_learned += 1
                reward += rewards.escape_success
                self._event("system", "Ucieczka schodami w niebezpieczeństwie (decyzja polityki).")
        return reward

    def target_slots(self, limit=TARGET_SLOTS):
        """[(cluster, path_cost)] spawn groups on the druid's floor, nearest by walking cost."""
        druid = self.druid
        slots = []
        for cluster in self.clusters:
            if cluster["z"] != druid.z:
                continue
            field = self._route_field([(cluster["x"], cluster["y"])])
            cost = float(field[druid.y, druid.x])
            if cost != float("inf"):
                slots.append((cluster, cost))
        slots.sort(key=lambda item: item[1])
        return slots[:limit]

    def _route_field(self, goals):
        druid = self.druid
        key = (druid.z, tuple(sorted(goals)))
        field = self._route_fields.get(key)
        if field is None:
            walkable = self.collision(druid.z).copy()
            goal_set = set(goals)
            for (x, y, z) in self.transitions:
                if z == druid.z and (x, y) not in goal_set:
                    walkable[y, x] = 1
            field = distance_field(walkable, list(goals), self.config.diagonal_multiplier)
            self._route_fields[key] = field
        return field

    def _macro_step(self, action):
        druid = self.druid
        if action in ACTION_GOTO_TARGETS:
            slots = self.target_slots()
            index = ACTION_GOTO_TARGETS.index(action)
            if index >= len(slots):
                return None
            cluster = slots[index][0]
            return self.step_toward([(cluster["x"], cluster["y"])])
        if action == ACTION_GOTO_ARENA:
            arena = self.tactics.nearest_zone("arena", druid.x, druid.y, druid.z)
            return None if arena is None else self.step_toward([(arena["x"], arena["y"])])
        return self.step_toward(self.escape_routes(druid.z))

    def hunt_step(self):
        """
        Without a route: walk to the spawn group with the most expected monsters per walking cost and keep
        that goal until reached (no flip-flopping). When nothing is expected alive, patrol to the group
        visited longest ago - standing in view of a spawn would block its respawn (Tibia rule).
        """
        druid = self.druid
        slots = self.target_slots(limit=None)
        if not slots:
            self._hunt_target = None
            if any(c["z"] == druid.z for c in self.clusters):
                return None
            # Escaped to a floor without spawns: take the stairs back toward the floor with the most groups
            floors = [c["z"] for c in self.clusters]
            if not floors:
                return None
            home = max(set(floors), key=floors.count)
            return self.step_toward(self.escape_routes(druid.z, toward=home))
        costs = {id(c): cost for c, cost in slots}
        target = self._hunt_target
        if target is not None and (id(target) not in costs or costs[id(target)] < 3):
            target = None                                   # reached or unreachable from this floor
        if target is None:
            alive = [(self.believed_alive(c) / (cost + 10.0), c) for c, cost in slots
                     if cost >= 3 and self.believed_alive(c)]
            if alive:
                target = max(alive, key=lambda item: item[0])[1]
            else:
                far = [(c["visited_ms"] if c["visited_ms"] is not None else -1, c) for c, cost in slots if cost >= 3]
                target = min(far, key=lambda item: item[0])[1] if far else None
        self._hunt_target = target
        if target is None:
            return None
        return self.step_toward([(target["x"], target["y"])])

    def _memory_features(self):
        druid, now = self.druid, self.time_ms
        hp_trend = mana_trend = damage_rate = 0.0
        if self._history:
            past = next((h for h in reversed(self._history) if now - h[0] >= HISTORY_MS), self._history[0])
            hp_trend = (druid.hp - past[1]) / druid.max_hp
            mana_trend = (druid.mana - past[2]) / max(1, druid.max_mana)
            oldest = self._history[0]
            damage_rate = min(1.0, (self.damage_taken - oldest[3]) / druid.max_hp)
        soon = sum(1 for m in self.monsters if not m.alive and m.known_dead_at is not None and m.z == druid.z
                   and 0 <= m.respawn_ms - (now - m.known_dead_at) <= RESPAWN_SOON_MS)
        return hp_trend, mana_trend, damage_rate, min(soon, 5) / 5.0

    def survival_action(self, heal_below, potion_below, cure_above=0.10):
        """Deterministic safety layer: heal, mana, cure a heavy condition, break a paralyze with haste."""
        druid, now = self.druid, self.time_ms
        if druid.hp_ratio < heal_below and druid.can_cast(SPELLBOOK["exura_gran"], now):
            return ACTION_HEAL
        if druid.mana_ratio < potion_below and druid.can_drink_potion(now):
            return ACTION_MANA_POTION
        element = self._curable_condition()
        if (element and druid.conditions[element][0] * druid.conditions[element][1] >= cure_above * druid.max_hp
                and druid.can_cast(SPELLBOOK[CURES[element]], now)):
            return ACTION_CURE
        haste = self._haste_spell()
        if druid.slowed and haste and druid.can_cast(SPELLBOOK[haste], now):
            return ACTION_HASTE
        return None

    def _curable_condition(self):
        """Element of the condition with the most damage left that a druid spell can cure, or None."""
        options = [(state[0] * state[1], element) for element, state in self.druid.conditions.items()
                   if element in CURES]
        return max(options)[1] if options else None

    def _haste_spell(self):
        return next((key for key, (*_, level) in HASTES.items() if self.druid.level >= level), None)

    # ----- fields, summons, invisibility ------------------------------------- #

    def _place_fields(self, monster, spell, now):
        """A field spell covers the area around the player with fire / poison / energy."""
        druid = self.druid
        grid = self._grid_list(monster.z)
        cx, cy = druid.x, druid.y
        radius = max(1, spell.radius)
        tiles = []
        for dx, dy in ball_area(radius):
            x, y = cx + dx, cy + dy
            if not (0 <= x < self.config.cols and 0 <= y < self.config.rows) or grid[y][x]:
                continue
            if (x, y, monster.z) in self.transitions or not has_line_of_sight(grid, cx, cy, x, y):
                continue
            self.fields[(x, y, monster.z)] = (spell.element, now + FIELD_MS)
            tiles.append((x, y))
        if self.record_events and tiles:
            self.aoe_flashes.append((tiles, monster.z, spell.element, now + AOE_FLASH_MS))
        self._event("warn", f"{monster.name}: {FIELD_LABELS_PL.get(spell.element, 'pole')} na {len(tiles)} polach.")
        return self._touch_field()

    def _touch_field(self):
        """Standing on / entering a field: entry damage plus its condition."""
        druid = self.druid
        field = self.fields.get(druid.position)
        if field is None or field[1] <= self.time_ms or not druid.alive:
            return 0.0
        element = field[0]
        hit, total, interval, ticks = FIELD_EFFECTS.get(element, (0, 0, 4000, 1))
        reward = 0.0
        if hit:
            result = druid.take_damage(hit, element, self.rng)
            reward += self._register_player_damage(FIELD_LABELS_PL.get(element, "pole"), result)
        if total:
            druid.add_condition(element, total, interval, self.time_ms, ticks=ticks)
        return reward

    def _summon(self, monster, now):
        """Canary monster.summon: every interval roll the chance while below maxSummons."""
        template = monster.template
        monster.summoned = [s for s in monster.summoned if s.alive]
        if len(monster.summoned) >= template.max_summons:
            return
        if sum(1 for m in self.monsters if m.master is not None and m.alive) >= MAX_SUMMONS_ALIVE:
            return
        for i, (name, chance, interval) in enumerate(template.summons):
            if now < monster.next_summon_ms[i]:
                continue
            monster.next_summon_ms[i] = now + interval
            real = self._monster_names.get(name.lower())
            if real is None or self.rng.random() * 100 >= chance:
                continue
            spot = next(((monster.x + dx, monster.y + dy) for dx, dy in MOVE_DELTAS.values()
                         if self._is_free(monster.x + dx, monster.y + dy, monster.z)), None)
            if spot is None:
                return
            summon = Monster(self.monster_db.get(real), spot[0], spot[1], monster.z, respawn_ms=0)
            summon.master = monster
            summon.experience = 0              # summons give no experience and no loot
            summon.loot_value = 0
            summon.aggro = True
            summon.lured = True
            monster.summoned.append(summon)
            self.monsters.append(summon)
            self._event("warn", f"{monster.name} przyzywa: {real}.")
            if len(monster.summoned) >= template.max_summons:
                return

    def _maybe_vanish(self, monster, now):
        chance, interval, duration = monster.template.invisible
        if now < monster.next_invisible_ms or monster.invisible(now):
            return
        monster.next_invisible_ms = now + interval
        if self.rng.random() * 100 < chance:
            monster.invisible_until_ms = now + duration
            self._event("warn", f"{monster.name} staje się niewidzialny ({duration / 1000:.0f} s).")

    def _spend(self, supply):
        """Book a consumed supply; returns its reward cost (positive number)."""
        return self.config.rewards.gold_scale * self.economy.spend(supply)

    def _avalanche_aim(self):
        """((x, y), targets) for the tile hitting the most monsters, or None when nothing is in reach."""
        druid = self.druid
        grid = self._grid_list(druid.z)
        everyone = [m for m in self.monsters if m.alive and m.z == druid.z and in_view(druid.x, druid.y, m.x, m.y)]
        monsters = [m for m in everyone if not m.invisible(self.time_ms)]   # aim at what the player sees
        if not monsters:
            return None
        rows, cols = len(grid), len(grid[0])
        best = None
        candidates = {(m.x + dx, m.y + dy) for m in monsters for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        for cx, cy in candidates:
            if not (0 <= cx < cols and 0 <= cy < rows) or grid[cy][cx]:
                continue
            if (chebyshev(druid.x, druid.y, cx, cy) > AVALANCHE.range or not in_view(druid.x, druid.y, cx, cy)
                    or not has_line_of_sight(grid, druid.x, druid.y, cx, cy)):
                continue
            hit = [m for m in everyone if (m.x - cx, m.y - cy) in AVALANCHE_AREA_SET
                   and has_line_of_sight(grid, cx, cy, m.x, m.y)]
            score = (len(hit), -sum(m.hp for m in hit))
            if hit and (best is None or score > best[0]):
                best = (score, (cx, cy), hit)
        return None if best is None else (best[1], best[2])

    def _rod_auto_attack(self, now):
        """Weapon swing on its own 2 s timer, independent of agent decisions."""
        druid = self.druid
        if druid.rod is None or now < druid.next_auto_attack_ms:
            return 0.0
        target = self._select_target(druid.rod.range)
        if target is None:
            return 0.0
        rod = druid.try_auto_attack(now)
        if rod is None:
            return 0.0
        return self._hit_monster(target, rod.roll(self.rng), rod.element, rod.name)

    def _select_target(self, attack_range):
        """Lowest-HP monster on screen within range and line of sight (auto-target)."""
        druid = self.druid
        grid = self._grid_list(druid.z)
        candidates = [m for m in self.monsters
                      if m.alive and m.z == druid.z and not m.invisible(self.time_ms)
                      and chebyshev(druid.x, druid.y, m.x, m.y) <= attack_range
                      and in_view(druid.x, druid.y, m.x, m.y)
                      and has_line_of_sight(grid, druid.x, druid.y, m.x, m.y)]
        return min(candidates, key=lambda m: (m.hp, chebyshev(druid.x, druid.y, m.x, m.y)), default=None)

    def _avalanche_hits(self):
        if self.druid.avalanche_runes <= 0:
            return 0
        aim = self._avalanche_aim()
        return len(aim[1]) if aim else 0

    def _enter_tile(self, druid):
        """Floor change when the druid steps onto stairs, a ladder or a hole."""
        target = self.transitions.get(druid.position)
        if target is None:
            return
        tx, ty, tz = target
        if self._movement_blocker(tx, ty, tz) is None:
            old_z = druid.z
            druid.move_to(tx, ty, tz)
            self.floor_changed_ms = self.time_ms
            self._event("move", f"Zmiana piętra: {old_z} -> {tz} ({'w dół' if tz > old_z else 'w górę'}).")

    def _check_waypoint(self):
        if not self.waypoints or self.druid.position != self.waypoints[self.waypoint_index]:
            return 0.0
        self._event("move", f"Osiągnięto punkt trasy #{self.waypoint_index + 1}.")
        self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        return self.config.rewards.waypoint_reached

    # ----- world simulation ------------------------------------------------ #

    def _simulate_world(self, duration_ms):
        """Advance the world in 50 ms server ticks. Returns reward collected meanwhile."""
        reward = 0.0
        druid = self.druid
        end = self.time_ms + duration_ms
        while self.time_ms < end and druid.alive:
            self.time_ms += SERVER_TICK_MS
            now = self.time_ms
            if druid.update_timers(now):
                self._event("shield", "Utamo Vita wygasło.")
            if self.config.use_utamo and not druid.utamo_active and druid.cast_utamo(now):
                self._event("shield", f"Druid odnawia Utamo Vita. Mana: {druid.mana}/{druid.max_mana}.")
            reward += self._rod_auto_attack(now)
            for element, damage in druid.due_condition_ticks(now):
                result = druid.take_damage(damage, element, self.rng)
                reward += self._register_player_damage(f"{CONDITION_LABELS_PL.get(element, element)} (efekt)", result)
                if not druid.alive:
                    return reward
            if any(m.z == druid.z and abs(m.x - druid.x) <= 1 and abs(m.y - druid.y) <= 1 and m.alive
                   for m in self.monsters):
                self.adjacent_ms += SERVER_TICK_MS
            reward += self._monsters_think(now)
        return reward

    def _monsters_think(self, now):
        druid = self.druid
        reward = 0.0
        for monster in list(self.monsters):                  # summons may join during the loop
            if not monster.alive or monster.z != druid.z:
                continue
            visible = in_view(monster.x, monster.y, druid.x, druid.y)
            if not monster.aggro:
                if not visible:
                    self._monster_go_home(monster, now)
                    continue
                monster.aggro = True
                monster.next_attack_ms = now + monster.attack_interval_ms // 2
                for i, spell in enumerate(monster.template.spells):
                    monster.next_spell_ms[i] = now + self.rng.randint(SERVER_TICK_MS, spell.cooldown_ms)
                self._event("warn", f"{monster.name} zauważa druida!")
            elif not visible:
                monster.aggro = False
                self._event("info", f"{monster.name} traci druida z oczu.")
                continue

            if monster.template.heals:
                self._monster_heal(monster, now)
            if monster.template.summons and monster.master is None:
                self._summon(monster, now)
            if monster.template.invisible:
                self._maybe_vanish(monster, now)
            if now >= monster.next_move_ms and self._monster_wants_step(monster):
                self._monster_step(monster, now)

            if monster.template.spells:
                reward += self._monster_cast(monster, now)
                if not druid.alive:
                    break

            if monster.distance_to(druid) <= 1 and now >= monster.next_attack_ms and not monster.fleeing:
                monster.next_attack_ms = now + monster.attack_interval_ms
                reward += self._monster_melee(monster)
                if not druid.alive:
                    break
        return reward

    def _monster_cast(self, monster, now):
        """Each spell rolls its chance every cooldown_ms; it fires only if the druid is inside its area."""
        druid = self.druid
        reward = 0.0
        dx, dy = druid.x - monster.x, druid.y - monster.y
        for i, spell in enumerate(monster.template.spells):
            if now < monster.next_spell_ms[i]:
                continue
            monster.next_spell_ms[i] = now + spell.cooldown_ms
            if self.rng.random() * 100 >= spell.chance:
                continue
            facing = monster.facing if spell.type == SPELL_BALL else facing_toward(dx, dy, monster.facing)
            if not spell_hits(spell, facing, dx, dy):
                continue
            if not has_line_of_sight(self._grid_list(monster.z), monster.x, monster.y, druid.x, druid.y):
                continue
            monster.facing = facing
            if spell.effect == "field":
                reward += self._place_fields(monster, spell, now)
                if not druid.alive:
                    break
                continue
            if spell.effect != "damage":
                self._spell_effect(monster, spell, now)
                continue
            result = druid.take_damage(spell.roll(self.rng), spell.element, self.rng)
            if self.record_events:
                cx, cy = (druid.x, druid.y) if spell.type == SPELL_BALL else (monster.x, monster.y)
                tiles = [(cx + ox, cy + oy) for ox, oy in spell_area(spell, facing)]
                self.aoe_flashes.append((tiles, monster.z, spell.element, now + AOE_FLASH_MS))
            reward += self._register_player_damage(f"{monster.name}: {spell.name} "
                                                   f"({ELEMENT_LABELS_PL[spell.element]})", result)
            if not druid.alive:
                break
        return reward

    def _monster_wants_step(self, monster):
        """Melee monsters close in; ranged ones (target_distance > 1) hold their distance; fleeing ones run."""
        druid = self.druid
        distance = monster.distance_to(druid)
        if monster.fleeing:
            return True
        wanted = monster.template.target_distance
        if wanted <= 1:
            return distance > 1
        if distance != wanted:
            return True
        return not has_line_of_sight(self._grid_list(monster.z), monster.x, monster.y, druid.x, druid.y)

    def _monster_step(self, monster, now):
        druid = self.druid
        occupied = {(m.x, m.y) for m in self.monsters if m.alive and m is not monster and m.z == monster.z}
        distance = monster.distance_to(druid)
        wanted = monster.template.target_distance
        if monster.fleeing or (wanted > 1 and distance < wanted):
            step = self._step_away(monster, occupied, None if monster.fleeing else wanted)
        else:
            path = a_star(self._grid_list(monster.z), (monster.x, monster.y), (druid.x, druid.y),
                          occupied=occupied, diagonal_cost=self.config.diagonal_multiplier)
            step = path[0] if path else None
        if step is None:
            monster.next_move_ms = now + 4 * SERVER_TICK_MS   # blocked: re-plan shortly
            return
        nx, ny = step
        diagonal = is_diagonal(nx - monster.x, ny - monster.y)
        monster.facing = facing_after_step(nx - monster.x, ny - monster.y, monster.facing)
        monster.move_to(nx, ny)
        duration = self._step_duration(monster, diagonal)
        monster.next_move_ms = now + (duration if duration else 10 ** 9)

    def _step_away(self, monster, occupied, keep_at=None):
        """Free neighbour that increases the distance to the druid (up to keep_at, keeping line of sight)."""
        druid = self.druid
        grid = self._grid_list(monster.z)
        rows, cols = len(grid), len(grid[0])
        here = chebyshev(monster.x, monster.y, druid.x, druid.y)
        best = None
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = monster.x + dx, monster.y + dy
            if not (0 <= nx < cols and 0 <= ny < rows) or grid[ny][nx] or (nx, ny) in occupied:
                continue
            if (nx, ny) == (druid.x, druid.y) or (nx, ny, monster.z) in self.transitions:
                continue
            distance = chebyshev(nx, ny, druid.x, druid.y)
            if distance <= here:
                continue
            if keep_at is not None:
                if distance > keep_at or not has_line_of_sight(grid, nx, ny, druid.x, druid.y):
                    continue
                score = (abs(distance - keep_at), dx != 0 and dy != 0)
            else:
                score = (-distance, dx != 0 and dy != 0)
            if best is None or score < best[0]:
                best = (score, (nx, ny))
        return None if best is None else best[1]

    def _monster_go_home(self, monster, now):
        """Without a target a monster drifts back to its spawn (and is found there again)."""
        hx, hy, hz = monster.home
        if now < monster.next_move_ms or hz != monster.z or chebyshev(monster.x, monster.y, hx, hy) <= HOME_RADIUS:
            return
        # Spawn tiles never move: one cached distance field per home instead of an A* every step
        field = distance_field(self.collision(hz), (hx, hy), self.config.diagonal_multiplier)
        dx, dy, cost = path_direction(field, monster.x, monster.y, self.config.diagonal_multiplier)
        nx, ny = monster.x + dx, monster.y + dy
        if (dx, dy) == (0, 0) or cost == float("inf") or not self._is_free(nx, ny, hz, ignore=monster):
            monster.next_move_ms = now + 20 * SERVER_TICK_MS
            return
        monster.facing = facing_after_step(nx - monster.x, ny - monster.y, monster.facing)
        monster.move_to(nx, ny)
        duration = self._step_duration(monster, is_diagonal(dx, dy))
        monster.next_move_ms = now + (duration if duration else 10 ** 9)

    def _monster_heal(self, monster, now):
        for i, (low, high, cooldown, chance) in enumerate(monster.template.heals):
            if now < monster.next_heal_ms[i]:
                continue
            monster.next_heal_ms[i] = now + cooldown
            if monster.hp < monster.max_hp and self.rng.random() * 100 < chance:
                healed = min(monster.max_hp - monster.hp, self.rng.randint(low, high))
                monster.hp += healed
                self._event("info", f"{monster.name} leczy się: +{healed} HP ({monster.hp}/{monster.max_hp}).")

    def _spell_effect(self, monster, spell, now):
        """Non-damage area spells: damage over time (poison cloud, burning) or a slow / paralyze."""
        druid = self.druid
        if spell.effect == "dot":
            total = spell.roll(self.rng)
            if druid.add_condition(spell.element, total, spell.dot_interval_ms, now):
                self._event("damage", f"{monster.name}: {CONDITION_LABELS_PL.get(spell.element, spell.element)} "
                                      f"({total} obr. w czasie).")
        elif spell.effect == "slow":
            druid.slow(spell.speed_change, spell.duration_ms, now)
            self._event("warn", f"{monster.name} spowalnia druida: prędkość {druid.speed}/{druid.base_speed} "
                                f"na {spell.duration_ms / 1000:.0f} s.")

    def _monster_melee(self, monster):
        druid = self.druid
        raw = monster.roll_melee(self.rng)
        if raw <= 0:
            self._event("info", f"{monster.name} chybia.")
            return 0.0
        monster.facing = facing_toward(druid.x - monster.x, druid.y - monster.y, monster.facing)
        result = druid.take_damage(raw, PHYSICAL, self.rng)
        condition = monster.template.melee_condition
        if condition is not None and result.after_mitigation > 0:
            element, total, interval = condition
            if druid.add_condition(element, total, interval, self.time_ms):
                self._event("damage", f"{monster.name}: {CONDITION_LABELS_PL.get(element, element)} "
                                      f"({total} obr. w czasie).")
        if result.after_mitigation <= 0:
            self._event("info", f"{monster.name} uderza za {raw}, pancerz blokuje całość.")
            return 0.0
        return self._register_player_damage(monster.name, result)

    def _register_player_damage(self, source, result):
        """Book-keeping, log line and -5 reward for any hit that reached mana or HP."""
        if result.after_mitigation <= 0:
            return 0.0
        druid = self.druid
        self.damage_taken += result.after_mitigation
        self.hp_min_ratio = min(self.hp_min_ratio, druid.hp_ratio)
        if self.record_events:
            armor_note = (f" (pancerz: -{result.raw - result.after_mitigation})"
                          if result.raw != result.after_mitigation else "")
            if result.absorbed_by_mana and not result.shield_broken:
                self._event("shield", f"{source} zadaje {result.after_mitigation} obr.{armor_note} "
                                      f"Pochłonięte przez Utamo Vita. Pozostała mana: {druid.mana}.")
            elif result.shield_broken:
                self._event("damage", f"{source} zadaje {result.after_mitigation} obr.{armor_note} "
                                      f"TARCZA PĘKŁA! Do HP: {result.hp_damage}. HP: {druid.hp}/{druid.max_hp}.")
            else:
                self._event("damage", f"{source} zadaje {result.after_mitigation} obr.{armor_note} "
                                      f"HP: {druid.hp}/{druid.max_hp}.")
        return self.config.rewards.damage_taken

    # ----- grid helpers ---------------------------------------------------- #

    def collision(self, z):
        if z not in self.grids:
            self.grids[z] = self.config.collision_grid(z)
            self._grid_lists[z] = self.grids[z].tolist()
        return self.grids[z]

    def _grid_list(self, z):
        self.collision(z)
        return self._grid_lists[z]

    def _step_duration(self, entity, diagonal):
        """Step time from the creature's speed and the ground speed of the tile it is standing on."""
        ground = int(self.speed(entity.z)[entity.y, entity.x])
        return entity.step_duration(diagonal, ground, self.config.diagonal_multiplier)

    def speed(self, z):
        if z not in self._speeds:
            self._speeds[z] = self.config.speed_grid(z)
        return self._speeds[z]

    def _walkable_static(self, x, y, z):
        return 0 <= x < self.config.cols and 0 <= y < self.config.rows and not self.collision(z)[y, x]

    def _movement_blocker(self, x, y, z):
        """Why the druid cannot enter (x, y, z), or None when the tile is free."""
        grid = self.collision(z)
        rows, cols = grid.shape
        if not (0 <= x < cols and 0 <= y < rows):
            return "poza mapą"
        if grid[y, x]:
            return "ściana"
        if any(m.alive and m.position == (x, y, z) for m in self.monsters):
            return "potwór"
        return None

    def _is_free(self, x, y, z, ignore=None):
        if not self._walkable_static(x, y, z):
            return False
        if self.druid is not None and ignore is not self.druid and self.druid.position == (x, y, z):
            return False
        return not any(m is not ignore and m.alive and m.position == (x, y, z) for m in self.monsters)

    def _start_position(self):
        if self.waypoints:
            return self.waypoints[0]
        z = self.config.floor
        start = self._start_near_spawns(z)
        if start is not None:
            return start
        free = np.argwhere(self.collision(z) == 0)
        if len(free) == 0:
            raise ValueError("No walkable tile on the selected floor")
        cy, cx = self.config.rows // 2, self.config.cols // 2
        y, x = min(free.tolist(), key=lambda c: chebyshev(c[1], c[0], cx, cy))
        return x, y, z

    def _start_near_spawns(self, z):
        """No route: a random tile 8-16 steps from the spawns of floor z, off the stairs, with a path to them."""
        goals = [(int(sp[0]), int(sp[1])) for sp in self.config.spawns
                 if int(sp[2]) == z and self._walkable_static(int(sp[0]), int(sp[1]), z)]
        if not goals:
            return None
        walkable = self.collision(z).copy()
        for (x, y, tz) in self.transitions:
            if tz == z:
                walkable[y, x] = 1
        field = distance_field(walkable, goals, self.config.diagonal_multiplier)
        for low, high in ((8, 16), (3, 30), (1, float("inf"))):
            ys, xs = np.nonzero((field >= low) & (field <= high) & (walkable == 0))
            if len(xs):
                i = self.rng.randrange(len(xs))
                return int(xs[i]), int(ys[i]), z
        return None

    def _create_monsters(self):
        cfg = self.config
        druid = self.druid
        monsters = []
        taken = {druid.position}
        for spawn in cfg.spawns:
            x, y, z, name = spawn[:4]
            position = (int(x), int(y), int(z))
            if name not in self.monster_db or position in taken or not self._walkable_static(*position):
                continue
            seconds = spawn[4] if len(spawn) > 4 else None
            monsters.append(Monster(self.monster_db.get(name), *position,
                                    respawn_ms=int(seconds * 1000) if seconds else cfg.respawn_ms))
            taken.add(position)
        if monsters or cfg.random_monster_count <= 0:
            return monsters
        template = self.monster_db.get(cfg.random_monster_type)
        free = [(int(x), int(y)) for y, x in np.argwhere(self.collision(druid.z) == 0)]
        candidates = [(x, y) for x, y in free
                      if (x, y, druid.z) not in taken and chebyshev(x, y, druid.x, druid.y) >= 4]
        self.rng.shuffle(candidates)
        for x, y in candidates[: cfg.random_monster_count]:
            monsters.append(Monster(template, x, y, druid.z, respawn_ms=cfg.respawn_ms))
        return monsters

    # ----- observation / info --------------------------------------------- #

    def _event(self, kind, message):
        if self.record_events:
            self.events.append((kind, message))

    def snapshot(self):
        druid = self.druid
        now = self.time_ms
        grid = self._grid_list(druid.z)
        cd = druid.cooldowns
        frigo, tera, heal = SPELLBOOK["exori_frigo"], SPELLBOOK["exori_tera"], SPELLBOOK["exura_gran"]
        sd = SPELLBOOK["sudden_death"]
        rod = druid.rod
        return GameSnapshot(
            cols=self.config.cols,
            rows=self.config.rows,
            player_x=druid.x,
            player_y=druid.y,
            player_z=druid.z,
            hp_ratio=druid.hp_ratio,
            mana_ratio=druid.mana_ratio,
            utamo_active=druid.utamo_active,
            utamo_remaining_ratio=druid.utamo_remaining_ms(now) / UTAMO_VITA_DURATION_MS,
            cooldown_ratios={
                "exori_frigo": cd.spell_remaining(frigo, now) / frigo.cooldown_ms,
                "exori_tera": cd.spell_remaining(tera, now) / tera.cooldown_ms,
                "heal": cd.spell_remaining(heal, now) / heal.cooldown_ms,
                "potion": cd.remaining("potion", now) / POTION_EXHAUST_MS,
                "sd": cd.spell_remaining(sd, now) / sd.cooldown_ms,
                "rod": druid.auto_attack_remaining_ms(now) / rod.attack_interval_ms if rod else 0.0,
                "avalanche": cd.spell_remaining(AVALANCHE, now) / AVALANCHE.cooldown_ms,
                "cure": cd.spell_remaining(SPELLBOOK["exana_pox"], now) / SPELLBOOK["exana_pox"].cooldown_ms,
                "haste": cd.spell_remaining(SPELLBOOK["utani_hur"], now) / SPELLBOOK["utani_hur"].cooldown_ms,
            },
            avalanche_runes=druid.avalanche_runes,
            avalanche_hits=self._avalanche_hits(),
            tactical_state=self.tactics.state if self.tactics.config.enabled else None,
            arena=self._arena_vector(),
            orbit=self._orbit_vector(),
            targets=[(c["x"] - druid.x, c["y"] - druid.y, cost, self.believed_alive(c),
                      None if c["visited_ms"] is None else now - c["visited_ms"]) for c, cost in self.target_slots()],
            memory=self._memory_features(),
            escape=self._escape_vector(),
            absolute_position=self.config.absolute_position,
            condition_damage=druid.condition_damage_left,
            slowed=druid.slowed,
            hasted=druid.hasted,
            field_here=druid.position in self.fields,
            max_hp=druid.max_hp,
            sd_runes=druid.sd_runes,
            rune_range=RUNE_RANGE,
            rod_equipped=rod is not None,
            rod_range=rod.range if rod else 0,
            potions_left=druid.potions,
            player_step_ms=self._step_duration(druid, False) or 0,
            spell_range=STRIKE_RANGE,
            collision=self.collision(druid.z),
            monsters=[MonsterView(
                m.x, m.y, m.z, m.hp_ratio,
                max(0, m.next_attack_ms - now) / m.attack_interval_ms if m.aggro else 0.0,
                m.aggro,
                ice_modifier=m.resistances.get("ice", 100),
                earth_modifier=m.resistances.get("earth", 100),
                step_ms=self._step_duration(m, False) or 0,
                line_of_sight=(m.z == druid.z and in_view(druid.x, druid.y, m.x, m.y)   # off-screen: not observed
                               and has_line_of_sight(grid, druid.x, druid.y, m.x, m.y)),
                death_modifier=m.resistances.get("death", 100),
                rod_modifier=m.resistances.get(rod.element, 100) if rod else 100,
                facing=m.facing,
                has_aoe=bool(m.template.spells),
                spell_cooldown_ratio=self._spell_cooldown_ratio(m, now),
                ranged=m.template.target_distance > 1,
                fleeing=m.fleeing,
            ) for m in self.monsters if m.alive and not m.invisible(now)],
            next_waypoint=self.waypoints[self.waypoint_index] if self.waypoints else None,
            waypoint_path=self._waypoint_path(),
            danger=self._danger_with_fields(danger_window(
                druid.x, druid.y, VIEW_RADIUS,
                [(m.x, m.y, m.facing, m.directional_spells) for m in self.monsters
                 if m.alive and m.z == druid.z and m.directional_spells and not m.invisible(now)
                 and in_view(m.x, m.y, druid.x, druid.y)],
                self.collision(druid.z), grid)),
        )

    def _waypoint_path(self):
        """First step and travel cost along the shortest path to the next waypoint (same floor only)."""
        if not self.waypoints:
            return None
        druid = self.druid
        wx, wy, wz = self.waypoints[self.waypoint_index]
        key = (self.waypoint_index, druid.z)
        field = self._waypoint_fields.get(key)
        if field is None:
            if wz == druid.z:
                goals = [(wx, wy)]
            else:
                # Waypoint on another floor: head for the stairs/holes that move toward its floor
                step = 1 if wz > druid.z else -1
                goals = [(x, y) for (x, y, z), (_, _, tz) in self.transitions.items()
                         if z == druid.z and (tz - z) == step]
                if not goals:
                    self._waypoint_fields[key] = False
                    return None
            # Stairs / holes that are not the goal must be walked around, or the path bounces between floors
            walkable = self.collision(druid.z).copy()
            goal_set = set(goals)
            for (x, y, z) in self.transitions:
                if z == druid.z and (x, y) not in goal_set:
                    walkable[y, x] = 1
            field = distance_field(walkable, goals, self.config.diagonal_multiplier)
            self._waypoint_fields[key] = field
        if field is False:
            return None
        return path_direction(field, druid.x, druid.y, self.config.diagonal_multiplier)

    def _danger_with_fields(self, danger):
        """Fields on the ground are dangerous tiles too (same channel the policy already reads)."""
        if not self.fields:
            return danger
        druid = self.druid
        for (x, y, z) in self.fields:
            wx, wy = x - druid.x + VIEW_RADIUS, y - druid.y + VIEW_RADIUS
            if z == druid.z and 0 <= wx < danger.shape[1] and 0 <= wy < danger.shape[0]:
                danger[wy, wx] = 1.0
        return danger

    def _escape_vector(self):
        routes = self.escape_routes(self.druid.z)
        if not routes:
            return None
        return path_direction(self._route_field(routes), self.druid.x, self.druid.y, self.config.diagonal_multiplier)

    @staticmethod
    def _spell_cooldown_ratio(monster, now):
        """0 = a spell attempt is due now, 1 = a full interval away (aggressive casters only)."""
        spells = monster.template.spells
        if not spells or not monster.aggro:
            return 1.0
        return min(max(0, t - now) / s.cooldown_ms for t, s in zip(monster.next_spell_ms, spells))

    def _observe(self):
        return encode_observation(self.snapshot())

    def _info(self):
        druid = self.druid
        return {
            "steps": self.steps,
            "time_ms": self.time_ms,
            "hp": druid.hp,
            "mana": druid.mana,
            "kills": self.kills,
            "damage_taken": self.damage_taken,
            "monsters_alive": sum(1 for m in self.monsters if m.alive),
            "episode_reward": self.episode_reward,
            **self.economy.summary(self.time_ms),
            "hp_min_pct": round(self.hp_min_ratio * 100, 1),
            "safety": round(self.hp_min_ratio * 100, 1) if druid.alive else 0.0,
            "tactical_state": self.tactics.state if self.tactics.config.enabled else None,
            "escapes": self.tactics.escapes,
            "pulls": self.tactics.pulls,
            "avg_pack": round(sum(self.tactics.pack_sizes) / len(self.tactics.pack_sizes), 2)
            if self.tactics.pack_sizes else 0.0,
            "action": self.last_action,
            "controlled": self.last_controlled,
            "respawns": self.respawns,
            "learned_escapes": self.escapes_learned,
            "adjacent_pct": round(100.0 * self.adjacent_ms / self.time_ms, 1) if self.time_ms else 0.0,
            "heals": self.heals,
            "potions_per_hour": int(self.economy.per_hour(self.economy.used["great_mana_potion"], self.time_ms)),
            "runes_per_hour": int(self.economy.per_hour(self.economy.used["sudden_death_rune"]
                                                        + self.economy.used["avalanche_rune"], self.time_ms)),
        }
