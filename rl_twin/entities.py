"""Creatures living on the (x, y, z) grid: base Entity, the Druid and data-driven Monsters."""

from dataclasses import dataclass

from .combat import (
    POTION_EXHAUST_MS,
    SPELLBOOK,
    UTAMO_VITA_DURATION_MS,
    Cooldowns,
    mitigate,
)
from .physics import DEFAULT_DIAGONAL_MULTIPLIER, DEFAULT_GROUND_SPEED, chebyshev, step_duration_ms

GROUND_FLOOR = 7          # Tibia floors: 0 (highest) .. 15 (deepest), 7 = ground level
REGEN_INTERVAL_MS = 2000
REGEN_HP = 1
REGEN_MANA = 4
GREAT_MANA_POTION_RANGE = (120, 160)


@dataclass
class DamageResult:
    raw: int
    after_mitigation: int
    absorbed_by_mana: int
    hp_damage: int
    shield_broken: bool
    percent: int = 100


class Entity:
    def __init__(self, name, x, y, z, max_hp, speed):
        self.name = name
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.max_hp = int(max_hp)
        self.hp = self.max_hp
        self.speed = speed
        self.next_move_ms = 0

    @property
    def position(self):
        return self.x, self.y, self.z

    @property
    def alive(self):
        return self.hp > 0

    @property
    def hp_ratio(self):
        return self.hp / self.max_hp

    def distance_to(self, other):
        """Chebyshev distance on the same floor, infinity across floors."""
        if self.z != other.z:
            return float("inf")
        return chebyshev(self.x, self.y, other.x, other.y)

    def step_duration(self, diagonal=False, ground_speed=DEFAULT_GROUND_SPEED,
                      diagonal_multiplier=DEFAULT_DIAGONAL_MULTIPLIER):
        return step_duration_ms(self.speed, diagonal, ground_speed, diagonal_multiplier)

    def move_to(self, x, y, z=None):
        self.x, self.y = int(x), int(y)
        if z is not None:
            self.z = int(z)


class Druid(Entity):
    def __init__(self, level, magic_level, armor=0, potions=0, x=0, y=0, z=GROUND_FLOOR,
                 rod=None, sd_runes=0, avalanche_runes=0):
        level = max(8, int(level))
        super().__init__("Druid", x, y, z,
                         max_hp=145 + (level - 8) * 5,
                         speed=110 + 2 * (level - 1))
        self.level = level
        self.magic_level = max(0, int(magic_level))
        self.armor = max(0, int(armor))
        self.max_mana = 90 + (level - 8) * 30
        self.mana = self.max_mana
        self.potions = int(potions)
        self.utamo_active = False
        self.utamo_expires_ms = 0
        self.cooldowns = Cooldowns()
        self.next_regen_ms = REGEN_INTERVAL_MS
        self.rod = rod if rod is None or self.level >= rod.level else None
        self.runes = {"sudden_death": max(0, int(sd_runes)), "avalanche": max(0, int(avalanche_runes))}
        self.next_auto_attack_ms = 0

    @property
    def mana_ratio(self):
        return self.mana / self.max_mana

    def utamo_remaining_ms(self, now_ms):
        return max(0, self.utamo_expires_ms - now_ms) if self.utamo_active else 0

    def can_cast(self, spell, now_ms):
        return (self.alive and self.mana >= spell.mana and self.magic_level >= spell.min_magic_level
                and self.cooldowns.ready(spell, now_ms))

    def can_use_rune(self, rune, now_ms):
        return (self.alive and self.runes.get(rune.key, 0) > 0 and self.magic_level >= rune.min_magic_level
                and self.cooldowns.ready(rune, now_ms))

    def use_rune(self, rune, now_ms):
        self.runes[rune.key] -= 1
        self.cooldowns.trigger(rune, now_ms)

    @property
    def sd_runes(self):
        return self.runes["sudden_death"]

    @property
    def avalanche_runes(self):
        return self.runes["avalanche"]

    def auto_attack_remaining_ms(self, now_ms):
        return max(0, self.next_auto_attack_ms - now_ms) if self.rod else 0

    def try_auto_attack(self, now_ms):
        """Pay for one rod attack if the weapon is ready. Returns the Rod or None."""
        rod = self.rod
        if rod is None or not self.alive or now_ms < self.next_auto_attack_ms or self.mana < rod.mana:
            return None
        self.mana -= rod.mana
        self.next_auto_attack_ms = now_ms + rod.attack_interval_ms
        return rod

    def pay_for(self, spell, now_ms):
        self.mana -= spell.mana
        self.cooldowns.trigger(spell, now_ms)

    def cast_utamo(self, now_ms):
        spell = SPELLBOOK["utamo_vita"]
        if not self.can_cast(spell, now_ms):
            return False
        self.pay_for(spell, now_ms)
        self.utamo_active = True
        self.utamo_expires_ms = now_ms + UTAMO_VITA_DURATION_MS
        return True

    def take_damage(self, raw, element, rng):
        """Armor (physical) -> elemental modifier -> Utamo Vita -> HP."""
        damage, percent = mitigate(raw, element, self.armor, None, rng)
        absorbed = 0
        broken = False
        if self.utamo_active and damage > 0:
            absorbed = min(damage, self.mana)
            self.mana -= absorbed
            if self.mana <= 0:
                self.mana = 0
                self.utamo_active = False
                broken = True
        hp_damage = min(damage - absorbed, self.hp)
        self.hp -= hp_damage
        return DamageResult(int(raw), damage, absorbed, hp_damage, broken, percent)

    def heal(self, amount):
        healed = max(0, min(int(amount), self.max_hp - self.hp)) if self.alive else 0
        self.hp += healed
        return healed

    def restore_mana(self, amount):
        restored = max(0, min(int(amount), self.max_mana - self.mana)) if self.alive else 0
        self.mana += restored
        return restored

    def can_drink_potion(self, now_ms):
        return (self.alive and self.potions > 0 and self.mana < self.max_mana
                and self.cooldowns.remaining("potion", now_ms) == 0)

    def drink_mana_potion(self, now_ms, rng):
        """Returns (restored, rolled): mana above the maximum is wasted."""
        self.potions -= 1
        self.cooldowns.set("potion", now_ms + POTION_EXHAUST_MS)
        rolled = rng.randint(*GREAT_MANA_POTION_RANGE)
        return self.restore_mana(rolled), rolled

    def update_timers(self, now_ms):
        """Regeneration and buff expiry. Returns True when Utamo Vita expired."""
        while now_ms >= self.next_regen_ms:
            self.next_regen_ms += REGEN_INTERVAL_MS
            self.heal(REGEN_HP)
            self.restore_mana(REGEN_MANA)
        if self.utamo_active and now_ms >= self.utamo_expires_ms:
            self.utamo_active = False
            return True
        return False


class Monster(Entity):
    def __init__(self, template, x, y, z=GROUND_FLOOR):
        super().__init__(template.name, x, y, z, template.hp, template.speed)
        self.template = template
        self.armor = template.armor
        self.resistances = template.resistances
        self.experience = template.exp
        self.loot_value = template.loot_value
        self.attack_interval_ms = template.attack_interval_ms
        self.aggro = False
        self.next_attack_ms = 0
        self.facing = (0, 1)                       # cardinal (dx, dy); spawns facing south
        self.next_spell_ms = [0] * len(template.spells)
        self.directional_spells = tuple(s for s in template.spells if s.type in ("beam", "wave"))

    def roll_melee(self, rng):
        return rng.randint(self.template.melee_min, self.template.melee_max)

    def take_spell_damage(self, raw, element, rng):
        """Apply armor/resistance and subtract HP. Returns (dealt, percent)."""
        damage, percent = mitigate(raw, element, self.armor, self.resistances, rng)
        self.last_damage = damage                 # before capping at remaining HP (overkill = waste)
        dealt = min(damage, self.hp)
        self.hp -= dealt
        return dealt, percent
