"""Data-driven monster definitions loaded from JSON (default: rl_twin/data/monsters.json)."""

import json
import os
from dataclasses import dataclass, field

from .combat import ELEMENTS

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
        )

    def names(self):
        return list(self._templates)

    def get(self, name):
        return self._templates[name]

    def __contains__(self, name):
        return name in self._templates

    def __len__(self):
        return len(self._templates)
