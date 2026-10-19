"""
Grid-based Druid vs Monster environment with a Gymnasium-style API.

    env = DruidHuntEnv(EnvConfig(...))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)

One step equals one game tick (TICK_SECONDS of game time). The module has no
Qt dependency so it runs headless inside a QThread, a CLI or a test.
"""

import json
import random
from dataclasses import asdict, dataclass, field

import numpy as np

from .observation import (
    ACTION_NAMES,
    ATTACK_RANGE,
    OBS_SIZE,
    GameSnapshot,
    MonsterView,
    chebyshev,
    encode_observation,
)

TICK_SECONDS = 0.5

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
ACTION_ATTACK, ACTION_HEAL, ACTION_MANA_POTION = 4, 5, 6
MOVE_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Spells and supplies (cooldowns in ticks)
UTAMO_VITA_MANA = 50
UTAMO_VITA_TICKS = 400
EXURA_GRAN_MANA = 70
EXURA_GRAN_COOLDOWN = 2
EXORI_FRIGO_MANA = 20
EXORI_FRIGO_COOLDOWN = 4
POTION_COOLDOWN = 2
GREAT_MANA_POTION_RANGE = (120, 160)
REGEN_INTERVAL_TICKS = 4
REGEN_HP = 1
REGEN_MANA = 4

MONSTER_TEMPLATES = {
    "Giant Spider": dict(max_hp=1300, speed=240, damage_range=(0, 300),
                         attack_interval=4, experience=900, aggro_range=7),
    "Dragon": dict(max_hp=1000, speed=172, damage_range=(0, 180),
                   attack_interval=4, experience=700, aggro_range=7),
    "Rotworm": dict(max_hp=65, speed=160, damage_range=(0, 40),
                    attack_interval=4, experience=40, aggro_range=5),
}


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
    step: float = -1.0
    invalid_action: float = -0.5
    waypoint_reached: float = 2.0


@dataclass
class EnvConfig:
    cols: int = 32
    rows: int = 24
    level: int = 100
    magic_level: int = 60
    use_utamo: bool = True
    mana_potions: int = 100
    obstacles: list = field(default_factory=list)   # [[x, y], ...]
    waypoints: list = field(default_factory=list)   # [[x, y], ...]
    spawns: list = field(default_factory=list)      # [[x, y, "Giant Spider"], ...]
    random_monster_count: int = 3                   # used when spawns is empty
    random_monster_type: str = "Giant Spider"
    max_steps: int = 600
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
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(rewards=rewards, **known)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))


@dataclass
class DamageResult:
    total: int
    absorbed_by_mana: int
    hp_damage: int
    shield_broken: bool


class Druid:
    def __init__(self, level, magic_level, potions):
        self.level = max(8, int(level))
        self.magic_level = max(0, int(magic_level))
        self.max_hp = 145 + (self.level - 8) * 5
        self.max_mana = 90 + (self.level - 8) * 30
        self.hp = self.max_hp
        self.mana = self.max_mana
        self.speed = 110 + 2 * (self.level - 1)
        self.potions = int(potions)
        self.utamo_active = False
        self.utamo_ticks = 0
        self.attack_cooldown = 0
        self.heal_cooldown = 0
        self.potion_cooldown = 0
        self.x = 0
        self.y = 0
        self._regen_counter = 0

    @property
    def alive(self):
        return self.hp > 0

    def cast_utamo(self):
        if not self.alive or self.mana < UTAMO_VITA_MANA:
            return False
        self.mana -= UTAMO_VITA_MANA
        self.utamo_active = True
        self.utamo_ticks = UTAMO_VITA_TICKS
        return True

    def take_damage(self, amount):
        """Mana Shield absorbs first; when mana hits 0 it breaks and the rest goes to HP."""
        amount = max(0, int(amount))
        absorbed = 0
        broken = False
        if self.utamo_active and amount > 0:
            absorbed = min(amount, self.mana)
            self.mana -= absorbed
            if self.mana <= 0:
                self.mana = 0
                self.utamo_active = False
                self.utamo_ticks = 0
                broken = True
        hp_damage = min(amount - absorbed, self.hp)
        self.hp -= hp_damage
        return DamageResult(amount, absorbed, hp_damage, broken)

    def heal(self, amount):
        healed = max(0, min(int(amount), self.max_hp - self.hp)) if self.alive else 0
        self.hp += healed
        return healed

    def restore_mana(self, amount):
        restored = max(0, min(int(amount), self.max_mana - self.mana)) if self.alive else 0
        self.mana += restored
        return restored

    def roll_heal(self, rng):
        base = self.level / 5.0
        return int(rng.uniform(base + self.magic_level * 3.184 + 20,
                               base + self.magic_level * 5.59 + 35))

    def roll_attack(self, rng):
        base = self.level / 5.0
        return int(rng.uniform(base + self.magic_level * 1.4 + 8,
                               base + self.magic_level * 2.2 + 14))

    def tick(self):
        """Advance cooldowns, buff timer and regeneration. Returns True if Utamo expired."""
        self.attack_cooldown = max(0, self.attack_cooldown - 1)
        self.heal_cooldown = max(0, self.heal_cooldown - 1)
        self.potion_cooldown = max(0, self.potion_cooldown - 1)
        self._regen_counter += 1
        if self._regen_counter >= REGEN_INTERVAL_TICKS:
            self._regen_counter = 0
            self.heal(REGEN_HP)
            self.restore_mana(REGEN_MANA)
        if self.utamo_active:
            self.utamo_ticks -= 1
            if self.utamo_ticks <= 0:
                self.utamo_active = False
                self.utamo_ticks = 0
                return True
        return False


class Monster:
    def __init__(self, name, x, y, max_hp, speed, damage_range,
                 attack_interval=4, experience=0, aggro_range=7):
        self.name = name
        self.max_hp = int(max_hp)
        self.hp = self.max_hp
        self.speed = speed
        self.damage_range = tuple(damage_range)
        self.attack_interval = attack_interval
        self.experience = experience
        self.aggro_range = aggro_range
        self.x = int(x)
        self.y = int(y)
        self.aggro = False
        self.attack_cooldown = 0
        self.move_progress = 0.0

    @classmethod
    def from_template(cls, name, x, y):
        return cls(name, x, y, **MONSTER_TEMPLATES[name])

    @property
    def alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        dealt = max(0, min(int(amount), self.hp))
        self.hp -= dealt
        return dealt


class DruidHuntEnv:
    """Druid hunting on a grid. Episode ends on death, when all monsters die, or at max_steps."""

    def __init__(self, config=None, record_events=False):
        self.config = config or EnvConfig()
        self.record_events = record_events
        self.action_space = Discrete(len(ACTION_NAMES))
        self.observation_space = Box(-1.0, 1.0, (OBS_SIZE,))
        self.rng = random.Random(self.config.seed)
        self.obstacles = {tuple(c) for c in self.config.obstacles}
        self.waypoints = [tuple(w) for w in self.config.waypoints if tuple(w) not in self.obstacles]
        self.druid = None
        self.monsters = []
        self.events = []
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
        self.druid = Druid(cfg.level, cfg.magic_level, cfg.mana_potions)
        self.druid.x, self.druid.y = self._start_cell()
        self.monsters = self._create_monsters()
        self.steps = 0
        self.waypoint_index = 1 if len(self.waypoints) > 1 else 0
        self.kills = 0
        self.experience = 0
        self.damage_taken = 0
        self.episode_reward = 0.0
        self.events = []
        if cfg.use_utamo and self.druid.cast_utamo():
            self._event("shield", "Druid rzuca Utamo Vita.")
        return self._observe(), self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        return self._advance(int(action))

    def idle_step(self):
        """Advance one tick without a player action (manual mode, no key pressed)."""
        return self._advance(None)

    # ----- core tick ------------------------------------------------------- #

    def _advance(self, action):
        self.events = []
        rewards = self.config.rewards
        druid = self.druid
        reward = rewards.step

        if druid.tick():
            self._event("shield", "Utamo Vita wygasło.")
        if self.config.use_utamo and not druid.utamo_active and druid.cast_utamo():
            self._event("shield", f"Druid odnawia Utamo Vita. Mana: {druid.mana}/{druid.max_mana}.")

        if action is not None:
            reward += self._apply_action(action)
        reward += self._monsters_act()

        terminated = False
        if not druid.alive:
            reward += rewards.death
            terminated = True
            self._event("death", "DRUID ZGINĄŁ! Koniec epizodu.")
        elif not any(m.alive for m in self.monsters):
            terminated = True
            self._event("system", "Wszystkie potwory pokonane - epizod zakończony sukcesem.")

        self.steps += 1
        truncated = not terminated and self.steps >= self.config.max_steps
        if truncated:
            self._event("system", f"Limit {self.config.max_steps} kroków - epizod przerwany.")
        self.episode_reward += reward
        return self._observe(), reward, terminated, truncated, self._info()

    def _apply_action(self, action):
        rewards = self.config.rewards
        druid = self.druid

        if action in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[action]
            nx, ny = druid.x + dx, druid.y + dy
            if not self._is_free(nx, ny):
                return rewards.invalid_action
            druid.x, druid.y = nx, ny
            return self._check_waypoint()

        if action == ACTION_ATTACK:
            targets = [m for m in self.monsters
                       if m.alive and chebyshev(druid.x, druid.y, m.x, m.y) <= ATTACK_RANGE]
            if druid.attack_cooldown > 0 or druid.mana < EXORI_FRIGO_MANA or not targets:
                return rewards.invalid_action
            target = min(targets, key=lambda m: (m.hp, chebyshev(druid.x, druid.y, m.x, m.y)))
            druid.mana -= EXORI_FRIGO_MANA
            druid.attack_cooldown = EXORI_FRIGO_COOLDOWN
            dealt = target.take_damage(druid.roll_attack(self.rng))
            target.aggro = True
            self._event("attack", f"Exori Frigo w {target.name}: {dealt} obr. "
                                  f"HP potwora: {target.hp}/{target.max_hp}.")
            if not target.alive:
                self.kills += 1
                self.experience += target.experience
                self._event("kill", f"{target.name} ginie! +{target.experience} exp.")
                return rewards.kill
            return 0.0

        if action == ACTION_HEAL:
            if druid.heal_cooldown > 0 or druid.mana < EXURA_GRAN_MANA or druid.hp >= druid.max_hp:
                return rewards.invalid_action
            druid.mana -= EXURA_GRAN_MANA
            druid.heal_cooldown = EXURA_GRAN_COOLDOWN
            healed = druid.heal(druid.roll_heal(self.rng))
            self._event("heal", f"Exura Gran: +{healed} HP. HP: {druid.hp}/{druid.max_hp}.")
            return 0.0

        if action == ACTION_MANA_POTION:
            if druid.potion_cooldown > 0 or druid.potions <= 0 or druid.mana >= druid.max_mana:
                return rewards.invalid_action
            druid.potions -= 1
            druid.potion_cooldown = POTION_COOLDOWN
            restored = druid.restore_mana(self.rng.randint(*GREAT_MANA_POTION_RANGE))
            self._event("mana", f"Great Mana Potion: +{restored} many. "
                                f"Mana: {druid.mana}/{druid.max_mana}. Zostało: {druid.potions}.")
            return 0.0

        return 0.0

    def _check_waypoint(self):
        if not self.waypoints:
            return 0.0
        target = self.waypoints[self.waypoint_index]
        if (self.druid.x, self.druid.y) != target:
            return 0.0
        self._event("move", f"Osiągnięto punkt trasy #{self.waypoint_index + 1}.")
        self.waypoint_index = (self.waypoint_index + 1) % len(self.waypoints)
        return self.config.rewards.waypoint_reached

    def _monsters_act(self):
        druid = self.druid
        reward = 0.0
        for monster in self.monsters:
            if not monster.alive:
                continue
            monster.attack_cooldown = max(0, monster.attack_cooldown - 1)
            dist = chebyshev(druid.x, druid.y, monster.x, monster.y)
            if not monster.aggro:
                if dist > monster.aggro_range:
                    continue
                monster.aggro = True
                monster.attack_cooldown = monster.attack_interval // 2
                self._event("warn", f"{monster.name} zauważa druida!")

            # Speed relative to the druid: a slower monster can be kited
            if dist > 1:
                monster.move_progress += monster.speed / druid.speed
                while monster.move_progress >= 1.0 and dist > 1:
                    monster.move_progress -= 1.0
                    if not self._move_monster_towards(monster, druid.x, druid.y):
                        monster.move_progress = 0.0
                        break
                    dist = chebyshev(druid.x, druid.y, monster.x, monster.y)
            else:
                monster.move_progress = 0.0

            if dist <= 1 and monster.attack_cooldown == 0:
                monster.attack_cooldown = monster.attack_interval
                damage = self.rng.randint(*monster.damage_range)
                if damage <= 0:
                    self._event("info", f"{monster.name} chybia.")
                    continue
                result = druid.take_damage(damage)
                self.damage_taken += damage
                reward += self.config.rewards.damage_taken
                self._log_damage(monster, result)
                if not druid.alive:
                    break
        return reward

    def _move_monster_towards(self, monster, tx, ty):
        """Greedy 4-way step that never increases Chebyshev+Manhattan distance."""
        def score(x, y):
            return chebyshev(x, y, tx, ty) * 10 + abs(x - tx) + abs(y - ty)

        current = score(monster.x, monster.y)
        best = None
        for dx, dy in MOVE_DELTAS.values():
            nx, ny = monster.x + dx, monster.y + dy
            if not self._is_free(nx, ny, ignore=monster):
                continue
            s = score(nx, ny)
            if s <= current and (best is None or s < best[0]):
                best = (s, nx, ny)
        if best is None:
            return False
        monster.x, monster.y = best[1], best[2]
        return True

    def _log_damage(self, monster, result):
        if not self.record_events:
            return
        druid = self.druid
        if result.absorbed_by_mana and not result.shield_broken:
            self._event("shield", f"{monster.name} zadaje {result.total} obr. Pochłonięte przez "
                                  f"Utamo Vita. Pozostała mana: {druid.mana}.")
        elif result.shield_broken:
            self._event("damage", f"{monster.name} zadaje {result.total} obr. TARCZA PĘKŁA! "
                                  f"Do HP: {result.hp_damage}. HP: {druid.hp}/{druid.max_hp}.")
        else:
            self._event("damage", f"{monster.name} zadaje {result.total} obr. "
                                  f"HP: {druid.hp}/{druid.max_hp}.")

    # ----- helpers --------------------------------------------------------- #

    def _in_bounds(self, x, y):
        return 0 <= x < self.config.cols and 0 <= y < self.config.rows

    def _is_free(self, x, y, ignore=None):
        if not self._in_bounds(x, y) or (x, y) in self.obstacles:
            return False
        if self.druid is not None and ignore is not None and (x, y) == (self.druid.x, self.druid.y):
            return False
        for m in self.monsters:
            if m is not ignore and m.alive and (m.x, m.y) == (x, y):
                return False
        return True

    def _free_cells(self):
        return [(x, y) for x in range(self.config.cols) for y in range(self.config.rows)
                if (x, y) not in self.obstacles]

    def _start_cell(self):
        if self.waypoints:
            return self.waypoints[0]
        cx, cy = self.config.cols // 2, self.config.rows // 2
        return min(self._free_cells(), key=lambda c: chebyshev(c[0], c[1], cx, cy))

    def _create_monsters(self):
        cfg = self.config
        start = (self.druid.x, self.druid.y)
        monsters = []
        taken = {start}
        for x, y, name in cfg.spawns:
            if (x, y) in self.obstacles or (x, y) in taken or name not in MONSTER_TEMPLATES:
                continue
            monsters.append(Monster.from_template(name, x, y))
            taken.add((x, y))
        if monsters:
            return monsters
        candidates = [c for c in self._free_cells()
                      if c not in taken and chebyshev(c[0], c[1], *start) >= 4]
        self.rng.shuffle(candidates)
        for x, y in candidates[:cfg.random_monster_count]:
            monsters.append(Monster.from_template(cfg.random_monster_type, x, y))
        return monsters

    def _event(self, kind, message):
        if self.record_events:
            self.events.append((kind, message))

    def snapshot(self):
        druid = self.druid
        return GameSnapshot(
            cols=self.config.cols,
            rows=self.config.rows,
            player_x=druid.x,
            player_y=druid.y,
            hp_ratio=druid.hp / druid.max_hp,
            mana_ratio=druid.mana / druid.max_mana,
            utamo_active=druid.utamo_active,
            utamo_remaining_ratio=druid.utamo_ticks / UTAMO_VITA_TICKS,
            attack_cooldown_ratio=druid.attack_cooldown / EXORI_FRIGO_COOLDOWN,
            heal_cooldown_ratio=druid.heal_cooldown / EXURA_GRAN_COOLDOWN,
            potion_cooldown_ratio=druid.potion_cooldown / POTION_COOLDOWN,
            potions_left=druid.potions,
            monsters=[MonsterView(m.x, m.y, m.hp / m.max_hp,
                                  m.attack_cooldown / m.attack_interval, m.aggro)
                      for m in self.monsters if m.alive],
            blocked=self.obstacles,
            next_waypoint=self.waypoints[self.waypoint_index] if self.waypoints else None,
        )

    def _observe(self):
        return encode_observation(self.snapshot())

    def _info(self):
        druid = self.druid
        return {
            "steps": self.steps,
            "hp": druid.hp,
            "mana": druid.mana,
            "kills": self.kills,
            "experience": self.experience,
            "damage_taken": self.damage_taken,
            "monsters_alive": sum(1 for m in self.monsters if m.alive),
            "episode_reward": self.episode_reward,
        }
