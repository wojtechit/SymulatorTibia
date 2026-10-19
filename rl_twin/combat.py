"""
Combat rules: elements, spellbook, armor and elemental damage modifiers, cooldowns.

Damage pipeline for a hit on a creature (mirrors Tibia / OTServ order):
    1. physical damage only: armor reduction  random(armor // 2, armor - (armor % 2 + 1))
    2. elemental modifier:   damage * resistance_percent / 100
    3. (player only) Utamo Vita: mana absorbs before HP
"""

from dataclasses import dataclass

PHYSICAL = "physical"
ICE = "ice"
EARTH = "earth"
FIRE = "fire"
ENERGY = "energy"
DEATH = "death"
HOLY = "holy"
ELEMENTS = (PHYSICAL, ICE, EARTH, FIRE, ENERGY, DEATH, HOLY)
ELEMENT_LABELS_PL = {
    PHYSICAL: "fizyczne", ICE: "lód", EARTH: "ziemia", FIRE: "ogień",
    ENERGY: "energia", DEATH: "śmierć", HOLY: "święte",
}

GROUP_ATTACK = "attack"
GROUP_HEALING = "healing"
GROUP_SUPPORT = "support"


@dataclass(frozen=True)
class Spell:
    key: str
    name: str
    words: str
    kind: str                 # "attack" | "heal" | "support"
    element: str = None
    mana: int = 0
    cooldown_ms: int = 0
    group: str = GROUP_ATTACK
    group_cooldown_ms: int = 0
    range: int = 0            # Chebyshev distance for targeted spells
    level_factor: float = 0.2
    min_mlvl_factor: float = 0.0
    min_base: float = 0.0
    max_mlvl_factor: float = 0.0
    max_base: float = 0.0

    def roll(self, level, magic_level, rng):
        """Tibia formula: level * 0.2 + magic_level * factor + base, uniform between min and max."""
        base = level * self.level_factor
        low = base + magic_level * self.min_mlvl_factor + self.min_base
        high = base + magic_level * self.max_mlvl_factor + self.max_base
        return int(rng.uniform(low, high))


SPELLBOOK = {
    "exori_frigo": Spell("exori_frigo", "Ice Strike", "exori frigo", "attack", ICE, mana=20,
                         cooldown_ms=2000, group=GROUP_ATTACK, group_cooldown_ms=2000, range=3,
                         min_mlvl_factor=1.403, min_base=8, max_mlvl_factor=2.203, max_base=13),
    "exori_tera": Spell("exori_tera", "Terra Strike", "exori tera", "attack", EARTH, mana=20,
                        cooldown_ms=2000, group=GROUP_ATTACK, group_cooldown_ms=2000, range=3,
                        min_mlvl_factor=1.403, min_base=8, max_mlvl_factor=2.203, max_base=13),
    "exura_gran": Spell("exura_gran", "Intense Healing", "exura gran", "heal", mana=70,
                        cooldown_ms=1000, group=GROUP_HEALING, group_cooldown_ms=1000,
                        min_mlvl_factor=3.184, min_base=20, max_mlvl_factor=5.59, max_base=35),
    "utamo_vita": Spell("utamo_vita", "Magic Shield", "utamo vita", "support", mana=50,
                        cooldown_ms=2000, group=GROUP_SUPPORT, group_cooldown_ms=2000),
}
UTAMO_VITA_DURATION_MS = 200_000
POTION_EXHAUST_MS = 1000


def armor_reduction(armor, rng):
    """Physical damage absorbed by armor (OTServ Creature::blockHit)."""
    if armor > 3:
        return rng.randint(armor // 2, armor - (armor % 2 + 1))
    if armor > 0:
        return 1
    return 0


def mitigate(raw, element, armor, resistances, rng):
    """Damage after armor (physical only) and elemental modifier. Returns (damage, percent)."""
    damage = max(0, int(raw))
    if element == PHYSICAL:
        damage = max(0, damage - armor_reduction(armor, rng))
    percent = resistances.get(element, 100) if resistances else 100
    return max(0, int(damage * percent / 100)), percent


class Cooldowns:
    """Per-spell and per-group readiness, all in simulation milliseconds."""

    def __init__(self):
        self._ready_at = {}

    def remaining(self, key, now_ms):
        return max(0, self._ready_at.get(key, 0) - now_ms)

    def spell_remaining(self, spell, now_ms):
        return max(self.remaining(spell.key, now_ms), self.remaining("group:" + spell.group, now_ms))

    def ready(self, spell, now_ms):
        return self.spell_remaining(spell, now_ms) == 0

    def trigger(self, spell, now_ms):
        self._ready_at[spell.key] = now_ms + spell.cooldown_ms
        group_key = "group:" + spell.group
        self._ready_at[group_key] = max(self._ready_at.get(group_key, 0), now_ms + spell.group_cooldown_ms)

    def set(self, key, ready_at_ms):
        self._ready_at[key] = ready_at_ms
