"""
Scripted fight policy: a hand-written baseline so routes and tactics can be tested without a trained model.

Priorities each decision (like a careful druid on a cavebot):
  1. heal below heal_below HP, mana potion below potion_below mana
  2. leave a beam / wave tile (danger map, same one the RL agent observes)
  3. Avalanche when it hits >= min_avalanche_hits monsters (or every monster on screen)
  4. Exori Frigo / Tera on the target, element picked by its resistances
  5. Sudden Death on a target that is not almost dead (SD overkill wastes gold)
  6. step off melee range while spells are on cooldown (kiting), else close in to spell range
  7. follow the waypoint route when nothing is aggressive
"""

from dataclasses import dataclass

from .aoe import danger_window
from .combat import SPELLBOOK
from .env import (ACTION_EXORI_FRIGO, ACTION_EXORI_TERA, ACTION_USE_AVALANCHE,
                  ACTION_USE_SD, ACTION_WAIT, AVALANCHE, MOVE_DELTAS, STRIKE_RANGE, RUNE_RANGE)
from .physics import chebyshev, in_view

DANGER_RADIUS = 2


@dataclass
class RuleConfig:
    heal_below: float = 0.60
    potion_below: float = 0.40
    min_avalanche_hits: int = 3
    sd_min_target_hp: float = 0.25     # SD only on targets above this HP ratio
    kite: bool = True


def _danger(env, druid):
    """(2r+1)^2 beam/wave threat around the druid, indexed [dy + r][dx + r]."""
    casters = [(m.x, m.y, m.facing, m.directional_spells) for m in env.monsters
               if m.alive and m.z == druid.z and m.directional_spells and in_view(m.x, m.y, druid.x, druid.y)]
    if not casters:
        return None
    grid = env.collision(druid.z)
    return danger_window(druid.x, druid.y, DANGER_RADIUS, casters, grid, env._grid_list(druid.z))


def _best_tile(env, druid, danger, monsters, want_distance):
    """Free neighbour without danger, closest to `want_distance` from the nearest monster."""
    best = None
    for action, (dx, dy) in MOVE_DELTAS.items():
        nx, ny = druid.x + dx, druid.y + dy
        if env._movement_blocker(nx, ny, druid.z) is not None:
            continue
        threat = float(danger[dy + DANGER_RADIUS, dx + DANGER_RADIUS]) if danger is not None else 0.0
        nearest = min((chebyshev(nx, ny, m.x, m.y) for m in monsters), default=want_distance)
        adjacent = sum(1 for m in monsters if chebyshev(nx, ny, m.x, m.y) <= 1)
        score = (threat > 0, adjacent, abs(nearest - want_distance), bool(dx and dy))
        if best is None or score < best[0]:
            best = (score, action)
    return None if best is None else best[1]


def rule_action(env, config=None):
    """Action index for the current state of a DruidHuntEnv."""
    cfg = config or RuleConfig()
    druid, now = env.druid, env.time_ms
    survival = env.survival_action(cfg.heal_below, cfg.potion_below)
    if survival is not None:
        return survival

    aggro = env.aggressive_monsters()
    if not aggro:
        return env.move_action(env.waypoint_step())

    danger = _danger(env, druid)
    if danger is not None and danger[DANGER_RADIUS, DANGER_RADIUS] > 0:
        escape = _best_tile(env, druid, danger, aggro, 2)
        if escape is not None:
            return escape

    if druid.avalanche_runes > 0 and druid.can_use_rune(AVALANCHE, now):
        aim = env._avalanche_aim()
        if aim and len(aim[1]) >= min(cfg.min_avalanche_hits, len(aggro)):
            return ACTION_USE_AVALANCHE

    target = env._select_target(STRIKE_RANGE)
    if target is not None:
        frigo, tera = SPELLBOOK["exori_frigo"], SPELLBOOK["exori_tera"]
        options = sorted(((target.resistances.get(s.element, 100), action, s)
                          for action, s in ((ACTION_EXORI_FRIGO, frigo), (ACTION_EXORI_TERA, tera))),
                         key=lambda o: -o[0])
        for modifier, action, spell in options:
            if modifier > 0 and druid.can_cast(spell, now):
                return action

    sd = SPELLBOOK["sudden_death"]
    rune_target = env._select_target(RUNE_RANGE)
    if (rune_target is not None and druid.sd_runes > 0 and druid.can_use_rune(sd, now)
            and rune_target.hp_ratio > cfg.sd_min_target_hp):
        return ACTION_USE_SD

    nearest = min(chebyshev(druid.x, druid.y, m.x, m.y) for m in aggro)
    if cfg.kite and nearest <= 1:
        step = _best_tile(env, druid, danger, aggro, 2)
        if step is not None:
            return step
    if target is None:
        goal = min(aggro, key=lambda m: chebyshev(druid.x, druid.y, m.x, m.y))
        step = env.step_toward([(goal.x, goal.y)])
        if step is not None:
            return env.move_action(step)
    return ACTION_WAIT


__all__ = ["RuleConfig", "rule_action"]
