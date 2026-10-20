"""
Hybrid tactics: spatial zones + a deterministic state machine around the RL policy.

Zones (drawn in the GUI, stored in EnvConfig.zones as dicts):
  lure   - inside its radius the druid keeps walking the route until `min`..`max` monsters follow
  arena  - where a gathered pack is dragged to and fought (kiting + Avalanche)
  orbit  - a loop around an obstacle; its tangent direction is given to the policy for kiting

States (TacticalController.state):
  LURE    - walk the waypoint route, gather monsters            (controller moves)
  PULL    - drag the pack to the nearest arena                  (controller moves)
  FIGHT   - micro-combat inside the arena / where the pack is   (RL policy acts)
  ESCAPE  - HP below escape_hp or more than safe_max monsters:  (controller, overrides RL)
            run to the nearest stairs / hole / ladder and change floor
  RECOVER - wait on the safe floor until HP >= recover_hp and recover_ms passed
  RETURN  - take the stairs back to the hunting floor, then LURE again

Outside FIGHT a survival layer heals (Exura Gran) and drinks potions before moving.
"""

import math
from dataclasses import dataclass, field

from .physics import DIRECTIONS_8, chebyshev, distance_field, path_direction

LURE, PULL, FIGHT, ESCAPE, RECOVER, RETURN = "lure", "pull", "fight", "escape", "recover", "return"
STATES = (LURE, PULL, FIGHT, ESCAPE, RECOVER, RETURN)
STATE_LABELS_PL = {
    LURE: "LURE (zbieranie)", PULL: "PULL (na arenę)", FIGHT: "WALKA (RL)",
    ESCAPE: "UCIECZKA", RECOVER: "REGENERACJA", RETURN: "POWRÓT",
}
ZONE_KINDS = ("lure", "arena", "orbit")
ZONE_LABELS_PL = {"lure": "Lure", "arena": "Arena", "orbit": "Orbita"}
DEFAULT_ZONE_RADIUS = {"lure": 6, "arena": 4, "orbit": 3}


@dataclass
class TacticsConfig:
    enabled: bool = False
    lure_min: int = 1              # outside lure zones: fight as soon as this many follow
    lure_max: int = 4
    safe_max: int = 6              # more aggressive monsters than this -> ESCAPE
    escape_hp: float = 0.40
    recover_hp: float = 0.90
    recover_ms: int = 30_000
    heal_below: float = 0.70       # survival layer outside FIGHT
    potion_below: float = 0.60
    pull_leash: int = 3            # PULL waits while a pack member is further behind than this (slow monsters)
    lure_timeout_ms: int = 12_000  # fight anyway when the pack has not reached `min` this long after first aggro


def make_zone(kind, x, y, z, radius=None, min_monsters=3, max_monsters=4):
    if kind not in ZONE_KINDS:
        raise ValueError(f"Unknown zone kind: {kind}")
    return {"kind": kind, "x": int(x), "y": int(y), "z": int(z),
            "radius": int(radius or DEFAULT_ZONE_RADIUS[kind]),
            "min": int(min_monsters), "max": int(max_monsters)}


def zone_contains(zone, x, y, z):
    return zone["z"] == z and (x - zone["x"]) ** 2 + (y - zone["y"]) ** 2 <= (zone["radius"] + 0.5) ** 2


def orbit_tangent(zone, x, y, clockwise=True):
    """Unit (dx, dy) along the orbit circle at the druid's angle (toward the ring when off it)."""
    angle = math.atan2(y - zone["y"], x - zone["x"])
    step = 0.6 if clockwise else -0.6
    tx = zone["x"] + zone["radius"] * math.cos(angle + step)
    ty = zone["y"] + zone["radius"] * math.sin(angle + step)
    dx, dy = tx - x, ty - y
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


@dataclass
class TacticalController:
    config: TacticsConfig
    zones: list = field(default_factory=list)
    state: str = LURE
    home_floor: int = 7
    recover_since: int = 0
    lure_aggro_since: int = None
    escapes: int = 0
    pulls: int = 0
    pack_sizes: list = field(default_factory=list)

    def reset(self, home_floor):
        self.state = LURE
        self.home_floor = home_floor
        self.recover_since = 0
        self.lure_aggro_since = None
        self.escapes = 0
        self.pulls = 0
        self.pack_sizes = []

    # ----- queries -------------------------------------------------------- #

    def zones_on(self, kind, z):
        return [zone for zone in self.zones if zone["kind"] == kind and zone["z"] == z]

    def nearest_zone(self, kind, x, y, z):
        candidates = self.zones_on(kind, z)
        return min(candidates, key=lambda zone: chebyshev(x, y, zone["x"], zone["y"]), default=None)

    def lure_limits(self, druid):
        zone = next((zone for zone in self.zones_on("lure", druid.z)
                     if zone_contains(zone, druid.x, druid.y, druid.z)), None)
        if zone is None:
            return self.config.lure_min, self.config.lure_max
        return zone["min"], zone["max"]

    @property
    def rl_in_control(self):
        return not self.config.enabled or self.state == FIGHT

    # ----- state machine -------------------------------------------------- #

    def update(self, env):
        """Advance the state machine before the decision. Returns (old_state, new_state) or None."""
        if not self.config.enabled:
            return None
        cfg = self.config
        druid = env.druid
        aggro = env.aggressive_monsters()
        old = self.state
        danger = druid.hp_ratio < cfg.escape_hp or len(aggro) > cfg.safe_max
        if self.state in (LURE, PULL, FIGHT) and danger and env.escape_routes(druid.z):
            self.state = ESCAPE
            self.escapes += 1
        elif self.state == ESCAPE and druid.z != self.home_floor:
            self.state = RECOVER
            self.recover_since = env.time_ms
        elif self.state == RECOVER:
            healthy = druid.hp_ratio >= cfg.recover_hp and env.time_ms - self.recover_since >= cfg.recover_ms
            if healthy:
                self.state = RETURN
        elif self.state == RETURN and druid.z == self.home_floor:
            self.state = LURE
        elif self.state == LURE:
            low, high = self.lure_limits(druid)
            if not aggro:
                self.lure_aggro_since = None
            elif self.lure_aggro_since is None:
                self.lure_aggro_since = env.time_ms
            waited = self.lure_aggro_since is not None and env.time_ms - self.lure_aggro_since >= cfg.lure_timeout_ms
            if aggro and (len(aggro) >= min(low, high) or waited):
                self.lure_aggro_since = None
                arena = self.nearest_zone("arena", druid.x, druid.y, druid.z)
                inside = arena is not None and zone_contains(arena, druid.x, druid.y, druid.z)
                self.state = FIGHT if arena is None or inside else PULL
                if self.state == FIGHT:
                    self.pack_sizes.append(len(aggro))
        elif self.state == PULL:
            arena = self.nearest_zone("arena", druid.x, druid.y, druid.z)
            if not aggro:
                self.state = LURE
            elif arena is None or chebyshev(druid.x, druid.y, arena["x"], arena["y"]) <= max(1, arena["radius"] - 1):
                self.state = FIGHT
                self.pulls += 1
                self.pack_sizes.append(len(aggro))
        elif self.state == FIGHT and not aggro:
            self.state = LURE
        return (old, self.state) if old != self.state else None

    def choose_action(self, env, rl_action):
        """The action to execute: the RL action in FIGHT, otherwise survival layer + navigation."""
        if self.rl_in_control:
            return rl_action, False
        heal_below = max(self.config.heal_below, self.config.recover_hp) if self.state == RECOVER else self.config.heal_below
        survival = env.survival_action(heal_below, self.config.potion_below)
        if survival is not None:
            return survival, True
        druid = env.druid
        if self.state == LURE:
            step = env.waypoint_step() if env.waypoints else env.hunt_step()
        elif self.state == PULL:
            arena = self.nearest_zone("arena", druid.x, druid.y, druid.z)
            straggler = max((chebyshev(m.x, m.y, druid.x, druid.y) for m in env.aggressive_monsters()), default=0)
            if straggler > self.config.pull_leash:
                step = None                                     # let the slow pack catch up (keeps aggro)
            else:
                step = env.step_toward([(arena["x"], arena["y"])]) if arena else None
        elif self.state == ESCAPE:
            step = env.step_toward(env.escape_routes(druid.z))
        elif self.state == RETURN:
            step = env.step_toward(env.escape_routes(druid.z, toward=self.home_floor))
        else:                                                   # RECOVER
            step = None
        return env.move_action(step), True


def step_from_field(collision, goals, x, y, diagonal_cost):
    """First (dx, dy) of the shortest path toward any goal, or None."""
    if not goals:
        return None
    field = distance_field(collision, goals, diagonal_cost)
    dx, dy, cost = path_direction(field, x, y, diagonal_cost)
    if (dx, dy) == (0, 0) or math.isinf(cost):
        return None
    return dx, dy


__all__ = ["TacticsConfig", "TacticalController", "make_zone", "zone_contains", "orbit_tangent",
           "step_from_field", "STATES", "STATE_LABELS_PL", "ZONE_KINDS", "ZONE_LABELS_PL",
           "DEFAULT_ZONE_RADIUS", "DIRECTIONS_8", "LURE", "PULL", "FIGHT", "ESCAPE", "RECOVER", "RETURN"]
