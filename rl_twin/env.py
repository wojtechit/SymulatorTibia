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
from dataclasses import asdict, dataclass, field

import numpy as np

from .aoe import SPELL_BALL, danger_window, facing_after_step, facing_toward, spell_area, spell_hits
from .aoe import ball_area
from .combat import ELEMENT_LABELS_PL, PHYSICAL, SPELLBOOK, UTAMO_VITA_DURATION_MS, POTION_EXHAUST_MS
from .entities import GROUND_FLOOR, Druid, Monster
from .items import ItemDatabase
from .monsters import MonsterDatabase
from .rewards import Economy, RewardConfig
from .tactics import (ESCAPE, FIGHT, PULL, RECOVER, STATE_LABELS_PL, TacticalController, TacticsConfig,
                      orbit_tangent, zone_contains)
from .terrain import decode_speed
from .observation import ACTION_NAMES, OBS_SIZE, VIEW_RADIUS, GameSnapshot, MonsterView, encode_observation
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

DECISION_MS = 100          # game time consumed by a non-movement action
AOE_FLASH_MS = 600         # how long the GUI shows a monster spell area
TIME_UNIT_MS = 500         # the -1 step penalty is charged per 500 ms of game time

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT, ACTION_DOWN_RIGHT = 4, 5, 6, 7
ACTION_EXORI_FRIGO, ACTION_EXORI_TERA, ACTION_HEAL, ACTION_MANA_POTION, ACTION_WAIT = 8, 9, 10, 11, 12
ACTION_USE_SD = 13
ACTION_USE_AVALANCHE = 14
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
        reward = self._tactical_update()
        action, controlled = self.tactics.choose_action(self, int(action))
        self.last_action, self.last_controlled = action, controlled
        reward += self._aoe_discipline(action)
        step_reward, elapsed = self._apply_action(int(action))
        reward += step_reward
        reward += self._simulate_world(elapsed)
        reward += self.config.rewards.step * elapsed / TIME_UNIT_MS

        druid = self.druid
        terminated = False
        if not druid.alive:
            reward += self.config.rewards.death
            terminated = True
            self._event("death", "DRUID ZGINĄŁ! Koniec epizodu.")
        elif not any(m.alive for m in self.monsters):
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
            self._enter_tile(druid)
            return self._check_waypoint(), duration

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
        return self._free_step(field)

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
        for _, dx, dy in sorted(options):
            if self._movement_blocker(druid.x + dx, druid.y + dy, druid.z) is None:
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

    def survival_action(self, heal_below, potion_below):
        """Deterministic safety layer used outside FIGHT: heal first, then mana."""
        druid, now = self.druid, self.time_ms
        if druid.hp_ratio < heal_below and druid.can_cast(SPELLBOOK["exura_gran"], now):
            return ACTION_HEAL
        if druid.mana_ratio < potion_below and druid.can_drink_potion(now):
            return ACTION_MANA_POTION
        return None

    def _spend(self, supply):
        """Book a consumed supply; returns its reward cost (positive number)."""
        return self.config.rewards.gold_scale * self.economy.spend(supply)

    def _avalanche_aim(self):
        """((x, y), targets) for the tile hitting the most monsters, or None when nothing is in reach."""
        druid = self.druid
        grid = self._grid_list(druid.z)
        monsters = [m for m in self.monsters if m.alive and m.z == druid.z and in_view(druid.x, druid.y, m.x, m.y)]
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
            hit = [m for m in monsters if (m.x - cx, m.y - cy) in AVALANCHE_AREA_SET
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
                      if m.alive and m.z == druid.z
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
            if any(m.alive and m.z == druid.z and chebyshev(m.x, m.y, druid.x, druid.y) <= 1 for m in self.monsters):
                self.adjacent_ms += SERVER_TICK_MS
            reward += self._monsters_think(now)
        return reward

    def _monsters_think(self, now):
        druid = self.druid
        reward = 0.0
        for monster in self.monsters:
            if not monster.alive or monster.z != druid.z:
                continue
            visible = in_view(monster.x, monster.y, druid.x, druid.y)
            if not monster.aggro:
                if not visible:
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

            if monster.distance_to(druid) > 1 and now >= monster.next_move_ms:
                self._monster_step(monster, now)

            if monster.template.spells:
                reward += self._monster_cast(monster, now)
                if not druid.alive:
                    break

            if monster.distance_to(druid) <= 1 and now >= monster.next_attack_ms:
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

    def _monster_step(self, monster, now):
        druid = self.druid
        occupied = {(m.x, m.y) for m in self.monsters if m.alive and m is not monster and m.z == monster.z}
        path = a_star(self._grid_list(monster.z), (monster.x, monster.y), (druid.x, druid.y),
                      occupied=occupied, diagonal_cost=self.config.diagonal_multiplier)
        if not path:
            monster.next_move_ms = now + 4 * SERVER_TICK_MS   # blocked: re-plan shortly
            return
        nx, ny = path[0]
        diagonal = is_diagonal(nx - monster.x, ny - monster.y)
        monster.facing = facing_after_step(nx - monster.x, ny - monster.y, monster.facing)
        monster.move_to(nx, ny)
        duration = self._step_duration(monster, diagonal)
        monster.next_move_ms = now + (duration if duration else 10 ** 9)

    def _monster_melee(self, monster):
        druid = self.druid
        raw = monster.roll_melee(self.rng)
        if raw <= 0:
            self._event("info", f"{monster.name} chybia.")
            return 0.0
        monster.facing = facing_toward(druid.x - monster.x, druid.y - monster.y, monster.facing)
        result = druid.take_damage(raw, PHYSICAL, self.rng)
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
        free = np.argwhere(self.collision(z) == 0)
        if len(free) == 0:
            raise ValueError("No walkable tile on the selected floor")
        cy, cx = self.config.rows // 2, self.config.cols // 2
        y, x = min(free.tolist(), key=lambda c: chebyshev(c[1], c[0], cx, cy))
        return x, y, z

    def _create_monsters(self):
        cfg = self.config
        druid = self.druid
        monsters = []
        taken = {druid.position}
        for x, y, z, name in cfg.spawns:
            position = (int(x), int(y), int(z))
            if name not in self.monster_db or position in taken or not self._walkable_static(*position):
                continue
            monsters.append(Monster(self.monster_db.get(name), *position))
            taken.add(position)
        if monsters or cfg.random_monster_count <= 0:
            return monsters
        template = self.monster_db.get(cfg.random_monster_type)
        free = [(int(x), int(y)) for y, x in np.argwhere(self.collision(druid.z) == 0)]
        candidates = [(x, y) for x, y in free
                      if (x, y, druid.z) not in taken and chebyshev(x, y, druid.x, druid.y) >= 4]
        self.rng.shuffle(candidates)
        for x, y in candidates[: cfg.random_monster_count]:
            monsters.append(Monster(template, x, y, druid.z))
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
            },
            avalanche_runes=druid.avalanche_runes,
            avalanche_hits=self._avalanche_hits(),
            tactical_state=self.tactics.state if self.tactics.config.enabled else None,
            arena=self._arena_vector(),
            orbit=self._orbit_vector(),
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
                line_of_sight=m.z == druid.z and has_line_of_sight(grid, druid.x, druid.y, m.x, m.y),
                death_modifier=m.resistances.get("death", 100),
                rod_modifier=m.resistances.get(rod.element, 100) if rod else 100,
                facing=m.facing,
                has_aoe=bool(m.template.spells),
                spell_cooldown_ratio=self._spell_cooldown_ratio(m, now),
            ) for m in self.monsters if m.alive],
            next_waypoint=self.waypoints[self.waypoint_index] if self.waypoints else None,
            waypoint_path=self._waypoint_path(),
            danger=danger_window(
                druid.x, druid.y, VIEW_RADIUS,
                [(m.x, m.y, m.facing, m.directional_spells) for m in self.monsters
                 if m.alive and m.z == druid.z and m.directional_spells
                 and in_view(m.x, m.y, druid.x, druid.y)],
                self.collision(druid.z), grid),
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
            "adjacent_pct": round(100.0 * self.adjacent_ms / self.time_ms, 1) if self.time_ms else 0.0,
            "heals": self.heals,
            "potions_per_hour": int(self.economy.per_hour(self.economy.used["great_mana_potion"], self.time_ms)),
            "runes_per_hour": int(self.economy.per_hour(self.economy.used["sudden_death_rune"]
                                                        + self.economy.used["avalanche_rune"], self.time_ms)),
        }
