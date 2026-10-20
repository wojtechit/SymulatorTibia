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
    range: int = 0            # Chebyshev distance for targeted spells and runes
    min_magic_level: int = 0  # required to cast / use
    area_radius: int = 0      # > 0: ball area around the aimed tile (aoe.ball_area convention)
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
    # Runes cost no mana; they consume a charge and share the attack cooldown group with strikes
    "sudden_death": Spell("sudden_death", "Sudden Death", "adori gran mort", "rune", DEATH, mana=0,
                          cooldown_ms=2000, group=GROUP_ATTACK, group_cooldown_ms=2000, range=7,
                          min_magic_level=15,
                          min_mlvl_factor=4.3, min_base=32, max_mlvl_factor=6.8, max_base=44),
    # Avalanche: ice ball rune; area_radius 4 = Tibia's 37-tile circle (rows 3-5-7-7-7-5-3), like GFB
    "avalanche": Spell("avalanche", "Avalanche", "adori mas frigo", "rune", ICE, mana=0,
                       cooldown_ms=2000, group=GROUP_ATTACK, group_cooldown_ms=2000, range=7,
                       min_magic_level=4, area_radius=4,
                       min_mlvl_factor=1.2, min_base=7, max_mlvl_factor=2.85, max_base=16),
    "utamo_vita": Spell("utamo_vita", "Magic Shield", "utamo vita", "support", mana=50,
                        cooldown_ms=2000, group=GROUP_SUPPORT, group_cooldown_ms=2000),
    # Cures: one per condition element; share the healing group with Exura Gran
    "exana_pox": Spell("exana_pox", "Cure Poison", "exana pox", "heal", EARTH, mana=30,
                       cooldown_ms=6000, group=GROUP_HEALING, group_cooldown_ms=1000),
    "exana_flam": Spell("exana_flam", "Cure Burning", "exana flam", "heal", FIRE, mana=30,
                        cooldown_ms=6000, group=GROUP_HEALING, group_cooldown_ms=1000),
    "exana_vis": Spell("exana_vis", "Cure Electrification", "exana vis", "heal", ENERGY, mana=30,
                       cooldown_ms=6000, group=GROUP_HEALING, group_cooldown_ms=1000),
    # Haste removes paralyze in Tibia; Strong Haste from level 20
    "utani_hur": Spell("utani_hur", "Haste", "utani hur", "support", mana=60,
                       cooldown_ms=2000, group=GROUP_SUPPORT, group_cooldown_ms=2000),
    "utani_gran_hur": Spell("utani_gran_hur", "Strong Haste", "utani gran hur", "support", mana=100,
                            cooldown_ms=2000, group=GROUP_SUPPORT, group_cooldown_ms=2000),
}
CURES = {EARTH: "exana_pox", FIRE: "exana_flam", ENERGY: "exana_vis"}   # condition element -> cure spell
# (speed factor, flat, duration ms, required level): new speed = base * factor + flat (TFS haste formulas)
HASTES = {"utani_gran_hur": (1.7, -56, 22_000, 20), "utani_hur": (1.3, -24, 33_000, 14)}
UTAMO_VITA_DURATION_MS = 200_000
MONSTER_SPELL_TYPES = ("beam", "wave", "ball")
SPELL_EFFECTS = ("damage", "dot", "slow", "field")   # hit | damage over time | speed loss | fields on the ground


@dataclass(frozen=True)
class MonsterSpell:
    """Monster area attack. Every cooldown_ms the monster rolls `chance` and casts if the target is in the area."""
    name: str
    type: str                 # "beam" | "wave" | "ball"
    element: str
    damage_min: int
    damage_max: int
    cooldown_ms: int = 2000
    chance: int = 100         # percent per attempt
    length: int = 0           # beam / wave: tiles in front of the caster
    spread: int = 0           # wave: the cone widens by 2 tiles every `spread` tiles
    range: int = 0            # ball: max distance to the target
    radius: int = 0           # ball: area radius around the target
    effect: str = "damage"    # SPELL_EFFECTS; "dot": damage_min..max is the total spread over ticks;
                              # "field": element field (fire / poison / energy) on every tile of the area
    dot_interval_ms: int = 2000
    speed_change: int = 0     # "slow": added to the player's speed (negative) for duration_ms
    duration_ms: int = 0

    def roll(self, rng):
        return rng.randint(self.damage_min, self.damage_max)
AUTO_ATTACK_INTERVAL_MS = 2000


@dataclass(frozen=True)
class Rod:
    """Druid weapon: fixed elemental damage per auto-attack, paid with mana."""
    name: str
    element: str
    damage_min: int
    damage_max: int
    mana: int
    range: int
    level: int = 0
    attack_interval_ms: int = AUTO_ATTACK_INTERVAL_MS

    def roll(self, rng):
        return rng.randint(self.damage_min, self.damage_max)

    def label_pl(self):
        return (f"{self.name} ({ELEMENT_LABELS_PL[self.element]} {self.damage_min}-{self.damage_max}, "
                f"{self.mana} many, zasięg {self.range}, lvl {self.level}+)")
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
