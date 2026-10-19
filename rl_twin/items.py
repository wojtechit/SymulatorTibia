"""Data-driven equipment loaded from JSON (default: rl_twin/data/items.json)."""

import json
import os

from .combat import ELEMENTS, Rod

DEFAULT_ITEMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "items.json")
REQUIRED_ROD_FIELDS = ("name", "element", "damage", "mana", "range")

_cache = {}


class ItemDatabase:
    def __init__(self, rods, source=None, supplies=None):
        self.source = source
        self._rods = {rod.name: rod for rod in rods}
        self.supplies = dict(supplies or {})     # key -> {"name", "price"}

    @classmethod
    def load(cls, path=None):
        """Load (and cache) an items JSON file."""
        path = os.path.abspath(path or DEFAULT_ITEMS_FILE)
        mtime = os.path.getmtime(path)
        cached = _cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        with open(path, encoding="utf-8") as handle:
            database = cls.from_dict(json.load(handle), source=path)
        _cache[path] = (mtime, database)
        return database

    @classmethod
    def from_dict(cls, data, source=None):
        rods = [cls._parse_rod(entry, i) for i, entry in enumerate(data.get("rods", []))]
        names = [r.name for r in rods]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate rod names")
        supplies = {str(k): {"name": str(v.get("name", k)), "price": int(v.get("price", 0))}
                    for k, v in data.get("supplies", {}).items()}
        return cls(rods, source, supplies)

    @staticmethod
    def _parse_rod(entry, index):
        missing = [f for f in REQUIRED_ROD_FIELDS if f not in entry]
        if missing:
            raise ValueError(f"Rod #{index} ({entry.get('name', '?')}) is missing fields: {missing}")
        element = str(entry["element"]).lower()
        if element not in ELEMENTS:
            raise ValueError(f"Rod {entry['name']}: unknown element {element}")
        low, high = int(entry["damage"][0]), int(entry["damage"][1])
        if low > high or int(entry["range"]) < 1:
            raise ValueError(f"Rod {entry['name']}: invalid damage or range")
        return Rod(
            name=str(entry["name"]),
            element=element,
            damage_min=low,
            damage_max=high,
            mana=int(entry["mana"]),
            range=int(entry["range"]),
            level=int(entry.get("level", 0)),
            attack_interval_ms=int(entry.get("attack_interval_ms", 2000)),
        )

    def supply_prices(self):
        return {key: entry["price"] for key, entry in self.supplies.items()}

    def rod_names(self):
        return list(self._rods)

    def rod(self, name):
        return self._rods[name]

    def has_rod(self, name):
        return name in self._rods
