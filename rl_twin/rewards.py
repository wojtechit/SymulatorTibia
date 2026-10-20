"""
Economic reward shaping: experience and loot in, supplies out.

    reward = exp_scale  * experience earned by damage (exp * exp_rate * dealt / max_hp)
           + gold_scale * loot value of kills          (if loot_in_reward)
           - gold_scale * price of consumed supplies   (potions, runes)
           + safety terms (damage taken, death, time, invalid actions, waypoints)

Experience is credited proportionally to damage so the signal is dense; the sum over a
kill equals the monster's full experience. Real (displayed) experience and loot are only
booked on the kill, like in Tibia.
"""

from collections import Counter
from dataclasses import dataclass

MS_PER_HOUR = 3_600_000


@dataclass
class RewardConfig:
    exp_scale: float = 0.1          # reward per experience point (Dragon 700 exp -> +70)
    gold_scale: float = 0.05        # reward per gold coin (Great Mana Potion 144 gp -> -7.2)
    loot_in_reward: bool = True
    kill: float = 0.0               # legacy flat bonus per kill (phases 1-4 used +100)
    damage_taken: float = -5.0
    death: float = -1000.0
    step: float = -1.0              # per TIME_UNIT_MS of game time
    invalid_action: float = -0.5
    waypoint_reached: float = 2.0
    lure_bonus: float = 5.0         # per monster brought into the arena (LURE/PULL -> FIGHT)
    aggro_gain: float = 1.0         # without the tactics automat: per monster newly following (learned luring)
    aoe_missed: float = -2.0        # single-target attack while Avalanche would hit >= aoe_min_targets
    aoe_min_targets: int = 3
    escape_success: float = 2.0     # emergency floor change completed alive
    exp_gold_value: float = 0.0     # gp per experience point for "net value/h" (0 = profit only)


class Economy:
    """Per-episode ledger of experience, loot and supplies."""

    def __init__(self, prices, exp_rate=1.0, exp_gold_value=0.0):
        self.prices = dict(prices)          # supply key -> gold per unit
        self.exp_rate = float(exp_rate)
        self.exp_gold_value = float(exp_gold_value)
        self.wasted = 0.0                   # gold of supplies that did nothing (overflow, overkill)
        self.experience = 0.0               # booked on kills (what the client would show)
        self.loot = 0
        self.used = Counter()               # supply key -> units
        self.supply_cost = 0

    def exp_for_damage(self, monster, dealt):
        return monster.experience * self.exp_rate * dealt / monster.max_hp

    def book_kill(self, monster):
        exp = monster.experience * self.exp_rate
        self.experience += exp
        self.loot += monster.loot_value
        return exp, monster.loot_value

    def spend(self, key, units=1):
        cost = self.prices.get(key, 0) * units
        self.used[key] += units
        self.supply_cost += cost
        return cost

    def waste(self, key, fraction):
        """Book `fraction` (0..1) of one unit of `key` as wasted."""
        self.wasted += self.prices.get(key, 0) * max(0.0, min(1.0, fraction))

    @property
    def profit(self):
        return self.loot - self.supply_cost

    @staticmethod
    def per_hour(value, time_ms):
        return value * MS_PER_HOUR / time_ms if time_ms > 0 else 0.0

    def summary(self, time_ms):
        return {
            "experience": int(self.experience),
            "loot": int(self.loot),
            "supply_cost": int(self.supply_cost),
            "profit": int(self.profit),
            "exp_per_hour": int(self.per_hour(self.experience, time_ms)),
            "profit_per_hour": int(self.per_hour(self.profit, time_ms)),
            "supplies_used": dict(self.used),
            "waste_pct": round(100.0 * self.wasted / self.supply_cost, 1) if self.supply_cost else 0.0,
            "net_value_per_hour": int(self.per_hour(self.experience * self.exp_gold_value + self.profit, time_ms)),
        }
