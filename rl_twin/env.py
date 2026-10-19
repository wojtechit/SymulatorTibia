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

from .combat import ELEMENT_LABELS_PL, PHYSICAL, SPELLBOOK, UTAMO_VITA_DURATION_MS, POTION_EXHAUST_MS
from .entities import GROUND_FLOOR, Druid, Monster
from .items import ItemDatabase
from .monsters import MonsterDatabase
from .observation import ACTION_NAMES, OBS_SIZE, GameSnapshot, MonsterView, encode_observation
from .physics import (
    DEFAULT_DIAGONAL_MULTIPLIER,
    DEFAULT_GROUND_SPEED,
    SERVER_TICK_MS,
    a_star,
    chebyshev,
    has_line_of_sight,
    in_view,
    is_diagonal,
)

DECISION_MS = 100          # game time consumed by a non-movement action
TIME_UNIT_MS = 500         # the -1 step penalty is charged per 500 ms of game time

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_UP_LEFT, ACTION_UP_RIGHT, ACTION_DOWN_LEFT, ACTION_DOWN_RIGHT = 4, 5, 6, 7
ACTION_EXORI_FRIGO, ACTION_EXORI_TERA, ACTION_HEAL, ACTION_MANA_POTION, ACTION_WAIT = 8, 9, 10, 11, 12
ACTION_USE_SD = 13
MOVE_DELTAS = {
    ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0),
    ACTION_UP_LEFT: (-1, -1), ACTION_UP_RIGHT: (1, -1),
    ACTION_DOWN_LEFT: (-1, 1), ACTION_DOWN_RIGHT: (1, 1),
}
ATTACK_SPELLS = {ACTION_EXORI_FRIGO: "exori_frigo", ACTION_EXORI_TERA: "exori_tera"}
STRIKE_RANGE = SPELLBOOK["exori_frigo"].range
RUNE_RANGE = SPELLBOOK["sudden_death"].range


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


@dataclass
class RewardConfig:
    kill: float = 100.0
    damage_taken: float = -5.0
    death: float = -1000.0
    step: float = -1.0              # per TIME_UNIT_MS of game time
    invalid_action: float = -0.5
    waypoint_reached: float = 2.0


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
    items_file: str = None                          # None = rl_twin/data/items.json
    seed: int = None
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        rewards = RewardConfig(**data.pop("rewards", {}))
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
        return cls(rewards=rewards, **known)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

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
        self.waypoints = [tuple(int(v) for v in w[:3]) for w in self.config.waypoints]
        self.waypoints = [w for w in self.waypoints if self._walkable_static(*w)]
        self.druid = None
        self.monsters = []
        self.events = []
        self.time_ms = 0
        self.steps = 0
        self.waypoint_index = 0
        self.kills = 0
        self.experience = 0
        self.damage_taken = 0
        self.episode_reward = 0.0

    # ----- Gym API --------------------------------------------------------- #

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        cfg = self.config
        x, y, z = self._start_position()
        rod = self.item_db.rod(cfg.rod) if cfg.rod else None
        self.druid = Druid(cfg.level, cfg.magic_level, cfg.armor, cfg.mana_potions, x, y, z,
                           rod=rod, sd_runes=cfg.sd_runes)
        self.monsters = self._create_monsters()
        self.time_ms = 0
        self.steps = 0
        self.waypoint_index = 1 if len(self.waypoints) > 1 else 0
        self.kills = 0
        self.experience = 0
        self.damage_taken = 0
        self.episode_reward = 0.0
        self.events = []
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
        reward, elapsed = self._apply_action(int(action))
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
            if not self._is_free(nx, ny, druid.z):
                return rewards.invalid_action, DECISION_MS
            druid.move_to(nx, ny)
            duration = self._step_duration(druid, is_diagonal(dx, dy))
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
            raw = rune.roll(druid.level, druid.magic_level, self.rng)
            return self._hit_monster(target, raw, rune.element, f"{rune.name} [{druid.sd_runes}]"), DECISION_MS

        if action == ACTION_HEAL:
            spell = SPELLBOOK["exura_gran"]
            if druid.hp >= druid.max_hp or not druid.can_cast(spell, now):
                return rewards.invalid_action, DECISION_MS
            druid.pay_for(spell, now)
            healed = druid.heal(spell.roll(druid.level, druid.magic_level, self.rng))
            self._event("heal", f"Exura Gran: +{healed} HP. HP: {druid.hp}/{druid.max_hp}.")
            return 0.0, DECISION_MS

        if action == ACTION_MANA_POTION:
            if not druid.can_drink_potion(now):
                return rewards.invalid_action, DECISION_MS
            restored = druid.drink_mana_potion(now, self.rng)
            self._event("mana", f"Great Mana Potion: +{restored} many. "
                                f"Mana: {druid.mana}/{druid.max_mana}. Zostało: {druid.potions}.")
            return 0.0, DECISION_MS

        return 0.0, DECISION_MS  # ACTION_WAIT

    def _hit_monster(self, target, raw, element, source):
        """Apply attack damage (armor/resistance aware). Returns the kill reward or 0."""
        dealt, percent = target.take_spell_damage(raw, element, self.rng)
        target.aggro = True
        self._event("attack", f"{source} ({ELEMENT_LABELS_PL[element]}, {percent}%) "
                              f"w {target.name}: {dealt} obr. HP: {target.hp}/{target.max_hp}.")
        if target.alive:
            return 0.0
        self.kills += 1
        self.experience += target.experience
        self._event("kill", f"{target.name} ginie! +{target.experience} exp.")
        return self.config.rewards.kill

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
                self._event("warn", f"{monster.name} zauważa druida!")
            elif not visible:
                monster.aggro = False
                self._event("info", f"{monster.name} traci druida z oczu.")
                continue

            if monster.distance_to(druid) > 1 and now >= monster.next_move_ms:
                self._monster_step(monster, now)

            if monster.distance_to(druid) <= 1 and now >= monster.next_attack_ms:
                monster.next_attack_ms = now + monster.attack_interval_ms
                reward += self._monster_melee(monster)
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
        monster.move_to(nx, ny)
        duration = self._step_duration(monster, diagonal)
        monster.next_move_ms = now + (duration if duration else 10 ** 9)

    def _monster_melee(self, monster):
        druid = self.druid
        raw = monster.roll_melee(self.rng)
        if raw <= 0:
            self._event("info", f"{monster.name} chybia.")
            return 0.0
        result = druid.take_damage(raw, PHYSICAL, self.rng)
        if result.after_mitigation <= 0:
            self._event("info", f"{monster.name} uderza za {raw}, pancerz blokuje całość.")
            return 0.0
        self.damage_taken += result.after_mitigation
        if self.record_events:
            armor_note = f" (pancerz: -{raw - result.after_mitigation})" if raw != result.after_mitigation else ""
            if result.absorbed_by_mana and not result.shield_broken:
                self._event("shield", f"{monster.name} zadaje {result.after_mitigation} obr.{armor_note} "
                                      f"Pochłonięte przez Utamo Vita. Pozostała mana: {druid.mana}.")
            elif result.shield_broken:
                self._event("damage", f"{monster.name} zadaje {result.after_mitigation} obr.{armor_note} "
                                      f"TARCZA PĘKŁA! Do HP: {result.hp_damage}. HP: {druid.hp}/{druid.max_hp}.")
            else:
                self._event("damage", f"{monster.name} zadaje {result.after_mitigation} obr.{armor_note} "
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
        return entity.step_duration(diagonal, self.config.ground_speed, self.config.diagonal_multiplier)

    def _walkable_static(self, x, y, z):
        return 0 <= x < self.config.cols and 0 <= y < self.config.rows and not self.collision(z)[y, x]

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
            },
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
            ) for m in self.monsters if m.alive],
            next_waypoint=self.waypoints[self.waypoint_index] if self.waypoints else None,
        )

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
            "experience": self.experience,
            "damage_taken": self.damage_taken,
            "monsters_alive": sum(1 for m in self.monsters if m.alive),
            "episode_reward": self.episode_reward,
        }
