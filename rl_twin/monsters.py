"""Data-driven monster definitions loaded from JSON (default: rl_twin/data/monsters.json)."""

import json
import os
from dataclasses import dataclass, field

from .combat import ELEMENTS, MONSTER_SPELL_TYPES, SPELL_EFFECTS, MonsterSpell

DEFAULT_MONSTERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monsters.json")
# Canary import (rl_twin.canary_monsters), merged under the hand-tuned file when the default database is loaded
EXTENSION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monsters_canary.json")
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
    target_distance: int = 1     # >1: ranged monster, keeps this distance to the player
    run_health: int = 0          # flees (no melee) while its HP is at or below this
    heals: tuple = ()            # (min, max, cooldown_ms, chance %) self-heals
    melee_condition: tuple = None  # (element, total damage, tick interval ms) applied by a landed melee hit
    summons: tuple = ()          # (name, chance %, interval ms) creatures it calls while fighting
    max_summons: int = 0
    invisible: tuple = None      # (chance %, interval ms, duration ms) turns invisible (cannot be targeted)

    def modifier(self, element):
        return self.resistances.get(element, 100) / 100.0

    def describe_pl(self):
        weak = [f"{e} {p}%" for e, p in self.resistances.items() if p != 100]
        behaviour = []
        if self.target_distance > 1:
            behaviour.append(f"dystans {self.target_distance}")
        if self.run_health:
            behaviour.append(f"ucieka przy HP ≤ {self.run_health}")
        if self.heals:
            behaviour.append("leczy się")
        if self.melee_condition:
            behaviour.append(f"melee: {self.melee_condition[0]} DoT")
        if self.summons:
            behaviour.append(f"przyzywa {', '.join(s[0] for s in self.summons)} (max {self.max_summons})")
        if self.invisible:
            behaviour.append("znika (niewidzialność)")
        return (f"HP {self.hp}, exp {self.exp}, prędkość {self.speed}, pancerz {self.armor}, "
                f"melee {self.melee_min}-{self.melee_max}"
                + (f", modyfikatory: {', '.join(weak)}" if weak else "")
                + (f", {', '.join(behaviour)}" if behaviour else ""))


class MonsterDatabase:
    def __init__(self, templates, source=None, featured=None):
        self.source = source
        self._templates = {t.name: t for t in templates}
        self.featured = list(featured) if featured is not None else list(self._templates)   # hand-tuned, listed first

    @classmethod
    def load(cls, path=None):
        """
        Load (and cache) a monsters JSON file. Without a path: the hand-tuned default file plus the
        Canary extension (if present) for every other monster; hand-tuned entries win.
        """
        if path is None and os.path.isfile(EXTENSION_FILE):
            key = (DEFAULT_MONSTERS_FILE, EXTENSION_FILE)
            stamp = (os.path.getmtime(DEFAULT_MONSTERS_FILE), os.path.getmtime(EXTENSION_FILE))
            cached = _cache.get(key)
            if cached and cached[0] == stamp:
                return cached[1]
            base = cls.load(DEFAULT_MONSTERS_FILE)
            extension = cls.load(EXTENSION_FILE)
            templates = [base.get(n) for n in base.names()]
            templates += [extension.get(n) for n in extension.names() if n not in base]
            database = cls(templates, source=DEFAULT_MONSTERS_FILE, featured=base.names())
            _cache[key] = (stamp, database)
            return database
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
        seen, duplicates = set(), set()
        for template in templates:
            (duplicates if template.name in seen else seen).add(template.name)
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
            target_distance=max(1, int(entry.get("target_distance", 1))),
            run_health=max(0, int(entry.get("run_health", 0))),
            heals=tuple((int(h["heal"][0]), int(h["heal"][1]), int(h.get("cooldown_ms", 2000)), int(h.get("chance", 100)))
                        for h in entry.get("heals", [])),
            melee_condition=MonsterDatabase._parse_condition(entry["name"], entry.get("melee_condition")),
            summons=tuple((str(x["name"]), int(x.get("chance", 10)), int(x.get("interval_ms", 2000)))
                          for x in (entry.get("summons") or {}).get("list", [])),
            max_summons=int((entry.get("summons") or {}).get("max", 0)),
            invisible=(lambda v: None if not v else (int(v.get("chance", 10)), int(v.get("cooldown_ms", 2000)),
                                                     int(v.get("duration_ms", 10_000))))(entry.get("invisible")),
        )

    @staticmethod
    def _parse_condition(monster, entry):
        if not entry:
            return None
        element = str(entry.get("element", "")).lower()
        if element not in ELEMENTS or int(entry.get("damage", 0)) <= 0:
            raise ValueError(f"Monster {monster}: invalid melee_condition {entry}")
        return element, int(entry["damage"]), int(entry.get("interval_ms", 4000))

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
            effect=str(entry.get("effect", "damage")),
            dot_interval_ms=int(entry.get("dot_interval_ms", 2000)),
            speed_change=int(entry.get("speed_change", 0)),
            duration_ms=int(entry.get("duration_ms", 0)),
        )
        invalid = (low > high or not 0 <= spell.chance <= 100 or spell.effect not in SPELL_EFFECTS
                   or (spell.effect == "slow" and (spell.speed_change >= 0 or spell.duration_ms <= 0))
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
