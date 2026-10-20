"""
Monster database import from Canary's Lua monster files (github.com/opentibiabr/canary).

    git clone --depth 1 --filter=blob:none --sparse https://github.com/opentibiabr/canary.git
    git -C canary sparse-checkout set data-otservbr-global/monster
    python -m rl_twin.canary_monsters canary/data-otservbr-global/monster

Writes rl_twin/data/monsters_canary.json, which MonsterDatabase.load() merges under the
hand-tuned rl_twin/data/monsters.json (hand-tuned entries win).

Conversion:
  speed          Canary uses the post-12.x speed scale (about half of the old one) -> x2
  elements       Canary 'percent' is resistance (100 = immune, -10 = 10 % weakness) -> 100 - percent
  melee          |minDamage|..|maxDamage| of the 'melee' attack, its interval
  area attacks   length + spread>0 -> wave, length -> beam, radius + target -> ball on the player,
                 radius without target -> area around the monster (ball, range = radius),
                 target without area -> single-target ranged hit (ball, radius 1)
  loot_value     expected coins only (gold, platinum, crystal); item values are not in the files,
                 so profit for imported monsters is a lower bound
"""

import argparse
import json
import os
import re

OUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "monsters_canary.json")
SKIP_DIRS = {"trainers", "traps", "familiars", "event_creatures"}
ELEMENT_TYPES = {
    "COMBAT_PHYSICALDAMAGE": "physical", "COMBAT_ENERGYDAMAGE": "energy", "COMBAT_EARTHDAMAGE": "earth",
    "COMBAT_FIREDAMAGE": "fire", "COMBAT_ICEDAMAGE": "ice", "COMBAT_HOLYDAMAGE": "holy",
    "COMBAT_DEATHDAMAGE": "death",
}
SPELL_ELEMENTS = dict(ELEMENT_TYPES, COMBAT_LIFEDRAIN="death", COMBAT_DROWNDAMAGE="ice")   # physical = arrows, spears
COINS = {"gold coin": 1, "platinum coin": 100, "crystal coin": 10_000}

_NAME = re.compile(r'createMonsterType\(\s*"([^"]+)"')
_FIELD = r"monster\.{}\s*=\s*(-?\d+)"
_BLOCK = r"monster\.{}\s*=\s*\{{(.*?)\n\}}"
_PAIR = re.compile(r"(\w+)\s*=\s*(\"[^\"]*\"|[^,]+)")


def _number(text, name, default=None):
    match = re.search(_FIELD.format(name), text)
    return int(match.group(1)) if match else default


def _block(text, name):
    match = re.search(_BLOCK.format(name), text, re.S)
    return match.group(1) if match else ""


def _entries(block):
    """Every top-level {key = value, ...} table of a Lua block as a dict of strings (nested tables dropped)."""
    result, depth, buffer = [], 0, []
    for char in block:
        if char == "{":
            depth += 1
            if depth == 1:
                buffer = []
        elif char == "}":
            if depth == 1:
                pairs = {k: v.strip().strip('"') for k, v in _PAIR.findall("".join(buffer))}
                if pairs:
                    result.append(pairs)
            depth = max(0, depth - 1)
        elif depth == 1:
            buffer.append(char)
    return result


def _int(pairs, key, default=0):
    try:
        return int(float(pairs.get(key, default)))
    except ValueError:
        return default


def parse_monster(text):
    """One monster dict in the monsters.json schema, or None when the file is not a usable monster."""
    name_match = _NAME.search(text)
    hp = _number(text, "health")
    if not name_match or not hp or hp <= 0:
        return None
    name = name_match.group(1)
    defenses = _block(text, "defenses")
    armor = re.search(r"\barmor\s*=\s*(\d+)", defenses)
    resistances = {element: 100 for element in ELEMENT_TYPES.values()}
    for entry in _entries(_block(text, "elements")):
        element = ELEMENT_TYPES.get(entry.get("type", ""))
        if element:
            resistances[element] = max(0, min(300, 100 - _int(entry, "percent")))
    melee, interval, spells = [0, 0], 2000, []
    for entry in _entries(_block(text, "attacks")):
        kind = entry.get("name", "")
        low, high = sorted((abs(_int(entry, "minDamage")), abs(_int(entry, "maxDamage"))))
        if kind == "melee":
            melee, interval = [low, high], _int(entry, "interval", 2000) or 2000
            continue
        element = SPELL_ELEMENTS.get(entry.get("type", ""))
        if kind != "combat" or element is None or high <= 0:
            continue
        spell = {"element": element, "damage": [low, high], "cooldown_ms": _int(entry, "interval", 2000) or 2000,
                 "chance": max(0, min(100, _int(entry, "chance", 100)))}
        length, spread, radius = _int(entry, "length"), _int(entry, "spread"), _int(entry, "radius")
        # A missile (shootEffect) with a range flies at the player even when the file says target = false
        target = entry.get("target", "false").strip() == "true" or ("shootEffect" in entry and "range" in entry)
        reach = _int(entry, "range", 7) or 7
        if length > 0 and spread > 0:
            spell.update(type="wave", length=length, spread=spread)
        elif length > 0:
            spell.update(type="beam", length=length)
        elif radius > 0 and target:
            spell.update(type="ball", range=reach, radius=radius)
        elif radius > 0:
            spell.update(type="ball", range=radius, radius=1)
        elif target:
            spell.update(type="ball", range=reach, radius=1)
        else:
            continue
        spell["name"] = f"{element.capitalize()} {spell['type']}"
        spells.append(spell)
    loot = 0.0
    for entry in _entries(_block(text, "loot")):
        value = COINS.get(entry.get("name", "").lower())
        if value:
            count = _int(entry, "maxCount", 1) or 1
            loot += _int(entry, "chance") / 100_000 * (count + 1) / 2 * value
    monster = {
        "name": name,
        "hp": hp,
        "exp": max(0, _number(text, "experience", 0)),
        "speed": max(1, 2 * _number(text, "speed", 100)),
        "armor": int(armor.group(1)) if armor else 0,
        "melee": melee,
        "attack_interval_ms": interval,
        "elemental_resistances": resistances,
        "loot_value": int(round(loot)),
    }
    if spells:
        monster["spells"] = spells
    return monster


def import_folder(folder):
    monsters, skipped = {}, 0
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for file_name in sorted(files):
            if not file_name.endswith(".lua"):
                continue
            with open(os.path.join(root, file_name), encoding="utf-8", errors="replace") as handle:
                monster = parse_monster(handle.read())
            if monster is None:
                skipped += 1
            else:
                monsters.setdefault(monster["name"], monster)
    return sorted(monsters.values(), key=lambda m: m["name"]), skipped


def main():
    parser = argparse.ArgumentParser(description="Import potworów z plików Lua Canary")
    parser.add_argument("folder", help="canary/data-otservbr-global/monster")
    parser.add_argument("--out", default=OUT_FILE)
    args = parser.parse_args()
    monsters, skipped = import_folder(args.folder)
    data = {
        "version": 1,
        "_source": "Imported from opentibiabr/canary data-otservbr-global/monster (Lua) by rl_twin.canary_monsters; "
                   "loot_value = expected coins only (lower bound).",
        "monsters": monsters,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=0, separators=(",", ":"))
    print(f"Zaimportowano {len(monsters)} potworów (pominięto {skipped} plików) -> {args.out}")


if __name__ == "__main__":
    main()
