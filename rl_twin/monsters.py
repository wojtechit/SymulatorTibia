"""Data-driven monster definitions loaded from JSON (default: rl_twin/data/monsters.json)."""

import json
import os
from dataclasses import dataclass, field

from .combat import ELEMENTS, MONSTER_SPELL_TYPES, MonsterSpell

DEFAULT_MONSTERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monsters.json")
REQUIRED_FIELDS = ("name", "hp", "exp", "speed", "armor", "elemental_resistances")

_cache = {}


@dataclass(frozen=True)
class MonsterTemplate:
    name: str
    hp: int
    exp: int
    speed: int
    armor: int
    resistances: dict = field(default_factory=dict)   # element -> damage taken percent
    melee_min: int = 0
    melee_max: int = 0
    attack_interval_ms: int = 2000
    sprite: str = None   # TibiaWiki file name override, e.g. "Demon.gif"
    spells: tuple = ()   # MonsterSpell, cast in addition to melee
    loot_value: int = 0  # average gold per kill

    def modifier(self, element):
        return self.resistances.get(element, 100) / 100.0

    def describe_pl(self):
        weak = [f"{e} {p}%" for e, p in self.resistances.items() if p != 100]
        return (f"HP {self.hp}, exp {self.exp}, prędkość {self.speed}, pancerz {self.armor}, "
                f"melee {self.melee_min}-{self.melee_max}"
                + (f", modyfikatory: {', '.join(weak)}" if weak else ""))


class MonsterDatabase:
    def __init__(self, templates, source=None):
        self.source = source
        self._templates = {t.name: t for t in templates}

    @classmethod
    def load(cls, path=None):
        """Load (and cache) a monsters JSON file."""
        path = os.path.abspath(path or DEFAULT_MONSTERS_FILE)
        mtime = os.path.getmtime(path)
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        database = cls.from_dict(data, source=path)
        _cache[path] = (mtime, database)
        return database

    @classmethod
    def from_dict(cls, data, source=None):
        entries = data["monsters"] if isinstance(data, dict) else data
        templates = [cls._parse_entry(entry, i) for i, entry in enumerate(entries)]
        names = [t.name for t in templates]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"Duplicate monster names: {sorted(duplicates)}")
        return cls(templates, source)

    @staticmethod
    def _parse_entry(entry, index):
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"Monster #{index} ({entry.get('name', '?')}) is missing fields: {missing}")
        raw_res = {str(k).lower(): int(v) for k, v in entry["elemental_resistances"].items()}
        unknown = set(raw_res) - set(ELEMENTS)
        if unknown:
            raise ValueError(f"Monster {entry['name']}: unknown elements {sorted(unknown)}")
        resistances = {element: raw_res.get(element, 100) for element in ELEMENTS}
        melee = entry.get("melee", [0, 0])
        if int(entry["hp"]) <= 0 or int(melee[0]) > int(melee[1]):
            raise ValueError(f"Monster {entry['name']}: invalid hp or melee range")
        return MonsterTemplate(
            name=str(entry["name"]),
            hp=int(entry["hp"]),
            exp=int(entry["exp"]),
            speed=int(entry["speed"]),
            armor=int(entry["armor"]),
            resistances=resistances,
            melee_min=int(melee[0]),
            melee_max=int(melee[1]),
            attack_interval_ms=int(entry.get("attack_interval_ms", 2000)),
            sprite=entry.get("sprite"),
            spells=tuple(MonsterDatabase._parse_spell(entry["name"], s) for s in entry.get("spells", [])),
            loot_value=int(entry.get("loot_value", 0)),
        )

    @staticmethod
    def _parse_spell(monster, entry):
        kind = str(entry.get("type", "")).lower()
        element = str(entry.get("element", "")).lower()
        if kind not in MONSTER_SPELL_TYPES or element not in ELEMENTS:
            raise ValueError(f"Monster {monster}: invalid spell type/element in {entry}")
        low, high = int(entry["damage"][0]), int(entry["damage"][1])
        spell = MonsterSpell(
            name=str(entry.get("name", kind)),
            type=kind,
            element=element,
            damage_min=low,
            damage_max=high,
            cooldown_ms=int(entry.get("cooldown_ms", 2000)),
            chance=int(entry.get("chance", 100)),
            length=int(entry.get("length", 0)),
            spread=int(entry.get("spread", 0)),
            range=int(entry.get("range", 0)),
            radius=int(entry.get("radius", 0)),
        )
        invalid = (low > high or not 0 <= spell.chance <= 100
                   or (kind in ("beam", "wave") and spell.length < 1)
                   or (kind == "wave" and spell.spread < 1)
                   or (kind == "ball" and (spell.range < 1 or spell.radius < 1)))
        if invalid:
            raise ValueError(f"Monster {monster}: invalid parameters for spell {spell.name}")
        return spell

    def names(self):
        return list(self._templates)

    def get(self, name):
        return self._templates[name]

    def __contains__(self, name):
        return name in self._templates

    def __len__(self):
        return len(self._templates)
