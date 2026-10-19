#!/usr/bin/env python3
"""
Symulator Tibia - symulator polowania Druida (PyQt5).

Left pane:  interactive map (QGraphicsView) for drawing the hunting route.
Right pane: character setup, simulation controls, live status and event log.

The RPG engine (Druid, Monster, HuntSimulation) is pure Python and has no Qt
dependency, so it can be tested or reused headless. The GUI only reads its
state and renders the events it produces every tick.
"""

import html
import math
import os
import random
import sys
from dataclasses import dataclass

from PyQt5.QtCore import QPoint, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TICK_MS = 500
TICK_SECONDS = TICK_MS / 1000.0
TILE_PX = 32  # one Tibia square meter (sqm) in map pixels

MAP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map.png")
DEFAULT_SCENE_SIZE = (3200, 2400)

# Spells and supplies (mana costs and durations follow Tibia values)
UTAMO_VITA_MANA = 50
UTAMO_VITA_DURATION = 200.0
EXURA_GRAN_MANA = 70
EXORI_FRIGO_MANA = 20
GREAT_MANA_POTION_RANGE = (120, 160)

HEAL_COOLDOWN = 1.0
ATTACK_COOLDOWN = 2.0
POTION_COOLDOWN = 1.0
HEAL_THRESHOLD = 0.60

# Passive regeneration: every REGEN_INTERVAL seconds
REGEN_INTERVAL = 2.0
REGEN_HP = 1
REGEN_MANA = 4

MELEE_RANGE = TILE_PX * 1.5
SPELL_RANGE = TILE_PX * 5
MONSTER_RESPAWN_SECONDS = 60.0


def format_sim_time(seconds):
    """Format simulation seconds as MM:SS.s, e.g. 15.5 -> '00:15.5'."""
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:04.1f}"


def distance(ax, ay, bx, by):
    return math.hypot(bx - ax, by - ay)


# --------------------------------------------------------------------------- #
# RPG engine
# --------------------------------------------------------------------------- #

@dataclass
class DamageResult:
    total: int
    absorbed_by_mana: int
    hp_damage: int
    shield_broken: bool


class Druid:
    """Player character. HP/Mana scale with level, spells scale with magic level."""

    def __init__(self, level, magic_level, x=0.0, y=0.0):
        self.level = max(8, int(level))
        self.magic_level = max(0, int(magic_level))
        self.max_hp = 145 + (self.level - 8) * 5
        self.max_mana = 90 + (self.level - 8) * 30
        self.hp = self.max_hp
        self.mana = self.max_mana
        self.utamo_active = False
        self.utamo_remaining = 0.0
        self.alive = True
        self.x = float(x)
        self.y = float(y)
        # Tibia base speed formula
        self.speed = 110 + 2 * (self.level - 1)
        self._regen_timer = 0.0

    @property
    def pixels_per_second(self):
        # Approximation: 100 speed points ~ 1 sqm per second
        return self.speed / 100.0 * TILE_PX

    @property
    def hp_ratio(self):
        return self.hp / self.max_hp

    @property
    def mana_ratio(self):
        return self.mana / self.max_mana

    def cast_utamo(self):
        """Cast Utamo Vita (mana shield). Recasting refreshes the duration."""
        if not self.alive or self.mana < UTAMO_VITA_MANA:
            return False
        self.mana -= UTAMO_VITA_MANA
        self.utamo_active = True
        self.utamo_remaining = UTAMO_VITA_DURATION
        return True

    def take_damage(self, amount):
        """Apply damage. With Utamo Vita, mana absorbs damage before HP."""
        amount = max(0, int(amount))
        absorbed = 0
        shield_broken = False
        if self.utamo_active and amount > 0:
            absorbed = min(amount, self.mana)
            self.mana -= absorbed
            if self.mana <= 0:
                self.mana = 0
                self.utamo_active = False
                self.utamo_remaining = 0.0
                shield_broken = True
        hp_damage = min(amount - absorbed, self.hp)
        self.hp -= hp_damage
        if self.hp <= 0:
            self.hp = 0
            self.alive = False
        return DamageResult(amount, absorbed, hp_damage, shield_broken)

    def heal(self, amount):
        """Restore HP up to max_hp. Returns the amount actually healed."""
        if not self.alive:
            return 0
        healed = max(0, min(int(amount), self.max_hp - self.hp))
        self.hp += healed
        return healed

    def restore_mana(self, amount):
        """Restore mana up to max_mana. Returns the amount actually restored."""
        if not self.alive:
            return 0
        restored = max(0, min(int(amount), self.max_mana - self.mana))
        self.mana += restored
        return restored

    def spend_mana(self, amount):
        if self.mana < amount:
            return False
        self.mana -= amount
        return True

    def update_buffs(self, dt):
        """Advance buff timers. Returns True if Utamo Vita expired this step."""
        if not self.utamo_active:
            return False
        self.utamo_remaining -= dt
        if self.utamo_remaining <= 0:
            self.utamo_active = False
            self.utamo_remaining = 0.0
            return True
        return False

    def regenerate(self, dt):
        self._regen_timer += dt
        while self._regen_timer >= REGEN_INTERVAL:
            self._regen_timer -= REGEN_INTERVAL
            self.heal(REGEN_HP)
            self.restore_mana(REGEN_MANA)

    def roll_exura_gran(self):
        """Intense Healing formula (Tibia): level/5 + mlvl * [3.184 .. 5.59] + [20 .. 35]."""
        base = self.level / 5.0
        low = base + self.magic_level * 3.184 + 20
        high = base + self.magic_level * 5.59 + 35
        return int(random.uniform(low, high))

    def roll_exori_frigo(self):
        """Ice Strike formula (Tibia): level/5 + mlvl * [1.4 .. 2.2] + [8 .. 14]."""
        base = self.level / 5.0
        low = base + self.magic_level * 1.4 + 8
        high = base + self.magic_level * 2.2 + 14
        return int(random.uniform(low, high))


class Monster:
    """Hostile creature that chases and attacks the Druid in melee."""

    def __init__(self, name, max_hp, speed, damage_range, x, y,
                 experience=0, attack_interval=2.0, aggro_range_sqm=7):
        self.name = name
        self.max_hp = int(max_hp)
        self.hp = self.max_hp
        self.speed = speed
        self.damage_range = (int(damage_range[0]), int(damage_range[1]))
        self.experience = experience
        self.attack_interval = attack_interval
        self.aggro_range = aggro_range_sqm * TILE_PX
        self.spawn_x = float(x)
        self.spawn_y = float(y)
        self.x = float(x)
        self.y = float(y)
        self.attack_cooldown = 0.0
        self.respawn_timer = 0.0
        self.engaged = False

    @classmethod
    def giant_spider(cls, x, y):
        return cls("Giant Spider", max_hp=1300, speed=240, damage_range=(0, 300),
                   x=x, y=y, experience=900)

    @property
    def alive(self):
        return self.hp > 0

    @property
    def hp_ratio(self):
        return self.hp / self.max_hp

    @property
    def pixels_per_second(self):
        return self.speed / 100.0 * TILE_PX

    def roll_damage(self):
        return random.randint(*self.damage_range)

    def take_damage(self, amount):
        dealt = max(0, min(int(amount), self.hp))
        self.hp -= dealt
        return dealt

    def move_towards(self, tx, ty, dt, stop_distance):
        dist = distance(self.x, self.y, tx, ty)
        if dist <= stop_distance:
            return
        step = min(self.pixels_per_second * dt, dist - stop_distance)
        self.x += (tx - self.x) / dist * step
        self.y += (ty - self.y) / dist * step

    def respawn(self):
        self.hp = self.max_hp
        self.x = self.spawn_x
        self.y = self.spawn_y
        self.attack_cooldown = 0.0
        self.respawn_timer = 0.0
        self.engaged = False


@dataclass
class SimEvent:
    time: float
    kind: str
    message: str


class HuntSimulation:
    """Tick-based hunt: Druid walks the waypoint loop and fights monsters on the way."""

    def __init__(self, level, magic_level, use_utamo, waypoints,
                 monster_count, mana_potions, seed=None):
        if len(waypoints) < 2:
            raise ValueError("At least two waypoints are required")
        self.rng = random.Random(seed)
        self.waypoints = [(float(x), float(y)) for x, y in waypoints]
        start_x, start_y = self.waypoints[0]
        self.druid = Druid(level, magic_level, start_x, start_y)
        self.use_utamo = use_utamo
        self.mana_potions = int(mana_potions)
        self.target_index = 1
        self.time = 0.0
        self.finished = False
        self.monsters = self._spawn_monsters(monster_count)

        self.heal_cooldown = 0.0
        self.attack_cooldown = 0.0
        self.potion_cooldown = 0.0
        self._warned_no_heal_mana = False
        self._warned_no_utamo_mana = False
        self._warned_no_potions = False

        self.laps = 0
        self.kills = 0
        self.experience = 0
        self.damage_received = 0
        self.damage_absorbed = 0
        self.damage_to_hp = 0
        self.heals_cast = 0
        self.hp_healed = 0
        self.potions_used = 0
        self._events = []

    # ----- setup ----------------------------------------------------------- #

    def _spawn_monsters(self, count):
        monsters = []
        segments = len(self.waypoints) - 1
        start_x, start_y = self.waypoints[0]
        for i in range(count):
            ax, ay = self.waypoints[i % segments]
            bx, by = self.waypoints[i % segments + 1]
            seg_len = distance(ax, ay, bx, by) or 1.0
            # Several attempts to keep spawns out of the starting area
            for _ in range(10):
                t = self.rng.uniform(0.25, 0.9)
                offset = self.rng.uniform(1.0, 3.0) * TILE_PX * self.rng.choice((-1, 1))
                nx, ny = -(by - ay) / seg_len, (bx - ax) / seg_len
                x = ax + (bx - ax) * t + nx * offset
                y = ay + (by - ay) * t + ny * offset
                if distance(x, y, start_x, start_y) > 8 * TILE_PX:
                    break
            monsters.append(Monster.giant_spider(x, y))
        return monsters

    # ----- event helpers --------------------------------------------------- #

    def _log(self, kind, message):
        self._events.append(SimEvent(self.time, kind, message))

    @property
    def experience_per_hour(self):
        if self.time <= 0:
            return 0
        return int(self.experience / self.time * 3600)

    # ----- main loop ------------------------------------------------------- #

    def tick(self):
        """Advance the simulation by one tick and return the produced events."""
        self._events = []
        if self.finished:
            return self._events

        dt = TICK_SECONDS
        self.time += dt
        druid = self.druid
        self.heal_cooldown = max(0.0, self.heal_cooldown - dt)
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.potion_cooldown = max(0.0, self.potion_cooldown - dt)

        if druid.update_buffs(dt):
            self._log("shield", "Utamo Vita wygasło.")
        self._process_respawns(dt)
        self._maintain_utamo()

        engaged = self._update_aggro()
        if engaged:
            for monster in engaged:
                monster.move_towards(druid.x, druid.y, dt, TILE_PX)
        else:
            self._move_druid(dt)

        self._monster_attacks(engaged, dt)
        if not druid.alive:
            self._finish_with_death()
            return self._events

        self._druid_heal()
        self._druid_potion()
        self._druid_attack(engaged)
        druid.regenerate(dt)
        return self._events

    def _process_respawns(self, dt):
        for monster in self.monsters:
            if monster.alive:
                continue
            monster.respawn_timer -= dt
            if monster.respawn_timer <= 0:
                monster.respawn()
                self._log("info", f"{monster.name} odrodził się na swoim spawnie.")

    def _maintain_utamo(self):
        druid = self.druid
        if not self.use_utamo or druid.utamo_active:
            return
        if druid.cast_utamo():
            self._warned_no_utamo_mana = False
            self._log("shield", f"Druid rzuca Utamo Vita. Tarcza many aktywna "
                                f"({int(UTAMO_VITA_DURATION)} s). Mana: {druid.mana}/{druid.max_mana}.")
        elif not self._warned_no_utamo_mana:
            self._warned_no_utamo_mana = True
            self._log("warn", "Za mało many, aby rzucić Utamo Vita!")

    def _update_aggro(self):
        druid = self.druid
        engaged = []
        for monster in self.monsters:
            if not monster.alive:
                continue
            if not monster.engaged and distance(monster.x, monster.y, druid.x, druid.y) <= monster.aggro_range:
                monster.engaged = True
                monster.attack_cooldown = monster.attack_interval / 2
                self._log("warn", f"{monster.name} zauważa druida i rusza do ataku!")
            if monster.engaged:
                engaged.append(monster)
        return engaged

    def _move_druid(self, dt):
        druid = self.druid
        remaining = druid.pixels_per_second * dt
        # Bounded loop guards against waypoints stacked on the same spot
        for _ in range(len(self.waypoints) + 1):
            if remaining <= 0:
                break
            tx, ty = self.waypoints[self.target_index]
            dist = distance(druid.x, druid.y, tx, ty)
            if dist <= remaining:
                druid.x, druid.y = tx, ty
                remaining -= dist
                self._arrive_at_waypoint()
            else:
                druid.x += (tx - druid.x) / dist * remaining
                druid.y += (ty - druid.y) / dist * remaining
                remaining = 0

    def _arrive_at_waypoint(self):
        reached = self.target_index
        if reached == 0:
            self.laps += 1
            self._log("move", f"Ukończono okrążenie trasy #{self.laps}.")
        else:
            self._log("move", f"Druid dotarł do punktu #{reached + 1}.")
        self.target_index = (reached + 1) % len(self.waypoints)

    def _monster_attacks(self, engaged, dt):
        druid = self.druid
        for monster in engaged:
            if distance(monster.x, monster.y, druid.x, druid.y) > MELEE_RANGE:
                continue
            monster.attack_cooldown -= dt
            if monster.attack_cooldown > 0:
                continue
            monster.attack_cooldown = monster.attack_interval
            damage = monster.roll_damage()
            if damage == 0:
                self._log("info", f"{monster.name} atakuje, ale chybia.")
                continue
            result = druid.take_damage(damage)
            self.damage_received += result.total
            self.damage_absorbed += result.absorbed_by_mana
            self.damage_to_hp += result.hp_damage
            self._log_damage(monster, result)
            if not druid.alive:
                return

    def _log_damage(self, monster, result):
        druid = self.druid
        name = monster.name
        if result.absorbed_by_mana and not result.shield_broken:
            self._log("shield", f"{name} zadaje {result.total} obr. Pochłonięte przez Utamo Vita. "
                                f"Pozostała mana: {druid.mana}.")
        elif result.shield_broken:
            self._log("damage", f"{name} zadaje {result.total} obr. Utamo Vita pochłonęło "
                                f"{result.absorbed_by_mana} - TARCZA PĘKŁA! Do HP: {result.hp_damage}. "
                                f"HP: {druid.hp}/{druid.max_hp}.")
        else:
            self._log("damage", f"{name} zadaje {result.total} obr. "
                                f"Pozostałe HP: {druid.hp}/{druid.max_hp}.")

    def _druid_heal(self):
        druid = self.druid
        if druid.hp_ratio >= HEAL_THRESHOLD or self.heal_cooldown > 0:
            return
        if not druid.spend_mana(EXURA_GRAN_MANA):
            if not self._warned_no_heal_mana:
                self._warned_no_heal_mana = True
                self._log("warn", f"HP poniżej {int(HEAL_THRESHOLD * 100)}%, "
                                  f"ale brak many na Exura Gran!")
            return
        self._warned_no_heal_mana = False
        healed = druid.heal(druid.roll_exura_gran())
        self.heal_cooldown = HEAL_COOLDOWN
        self.heals_cast += 1
        self.hp_healed += healed
        self._log("heal", f"Druid rzuca Exura Gran i leczy {healed} HP. "
                          f"HP: {druid.hp}/{druid.max_hp}.")

    def _druid_potion(self):
        druid = self.druid
        threshold = 0.85 if (self.use_utamo and druid.utamo_active) else 0.5
        if druid.mana_ratio >= threshold or self.potion_cooldown > 0:
            return
        if self.mana_potions <= 0:
            if not self._warned_no_potions:
                self._warned_no_potions = True
                self._log("warn", "Skończyły się mikstury many!")
            return
        restored = druid.restore_mana(self.rng.randint(*GREAT_MANA_POTION_RANGE))
        self.mana_potions -= 1
        self.potions_used += 1
        self.potion_cooldown = POTION_COOLDOWN
        self._log("mana", f"Druid pije Great Mana Potion (+{restored} many). "
                          f"Mana: {druid.mana}/{druid.max_mana}. Pozostało mikstur: {self.mana_potions}.")

    def _druid_attack(self, engaged):
        druid = self.druid
        if self.attack_cooldown > 0:
            return
        in_range = [m for m in engaged
                    if distance(m.x, m.y, druid.x, druid.y) <= SPELL_RANGE]
        if not in_range:
            return
        # Keep mana for the shield or for healing before spending it on offense
        if druid.utamo_active:
            if druid.mana_ratio < 0.35:
                return
        elif druid.mana < EXORI_FRIGO_MANA + EXURA_GRAN_MANA:
            return
        target = min(in_range, key=lambda m: (m.hp, distance(m.x, m.y, druid.x, druid.y)))
        druid.spend_mana(EXORI_FRIGO_MANA)
        self.attack_cooldown = ATTACK_COOLDOWN
        dealt = target.take_damage(druid.roll_exori_frigo())
        self._log("attack", f"Druid rzuca Exori Frigo w {target.name} za {dealt} obr. "
                            f"HP potwora: {target.hp}/{target.max_hp}.")
        if not target.alive:
            target.engaged = False
            target.respawn_timer = MONSTER_RESPAWN_SECONDS
            self.kills += 1
            self.experience += target.experience
            self._log("kill", f"{target.name} ginie! +{target.experience} doświadczenia "
                              f"(zabitych: {self.kills}).")

    def _finish_with_death(self):
        self.finished = True
        self._log("death", "DRUID ZGINĄŁ! Koniec symulacji.")
        self._log("system", f"Podsumowanie: czas {format_sim_time(self.time)}, zabitych {self.kills}, "
                            f"exp {self.experience}, okrążeń {self.laps}, "
                            f"zużytych mikstur {self.potions_used}.")


# --------------------------------------------------------------------------- #
# Map canvas
# --------------------------------------------------------------------------- #

class EntityGraphic:
    """Circle with a name label and HP (optionally mana) bar drawn on the map."""

    BAR_WIDTH = 34

    def __init__(self, scene, label, color, radius=11, with_mana_bar=False, z=10):
        self.body = QGraphicsEllipseItem(-radius, -radius, radius * 2, radius * 2)
        self.body.setBrush(QBrush(color))
        self.body.setPen(QPen(color.darker(170), 2))
        self.body.setZValue(z)
        scene.addItem(self.body)

        bar_y = -radius - 9
        half = self.BAR_WIDTH / 2
        self.hp_bg = QGraphicsRectItem(-half, bar_y, self.BAR_WIDTH, 4, self.body)
        self.hp_bg.setBrush(QBrush(QColor("#111")))
        self.hp_bg.setPen(QPen(Qt.NoPen))
        self.hp_fg = QGraphicsRectItem(-half, bar_y, self.BAR_WIDTH, 4, self.body)
        self.hp_fg.setPen(QPen(Qt.NoPen))

        self.mana_fg = None
        if with_mana_bar:
            mana_bg = QGraphicsRectItem(-half, bar_y + 5, self.BAR_WIDTH, 3, self.body)
            mana_bg.setBrush(QBrush(QColor("#111")))
            mana_bg.setPen(QPen(Qt.NoPen))
            self.mana_fg = QGraphicsRectItem(-half, bar_y + 5, self.BAR_WIDTH, 3, self.body)
            self.mana_fg.setBrush(QBrush(QColor("#2979ff")))
            self.mana_fg.setPen(QPen(Qt.NoPen))

        self.shield = QGraphicsEllipseItem(-radius - 5, -radius - 5, (radius + 5) * 2, (radius + 5) * 2, self.body)
        shield_pen = QPen(QColor("#40c4ff"), 2, Qt.DashLine)
        self.shield.setPen(shield_pen)
        self.shield.setVisible(False)

        self.label = QGraphicsSimpleTextItem(label, self.body)
        font = QFont("Sans", 7, QFont.Bold)
        self.label.setFont(font)
        self.label.setBrush(QBrush(QColor("#ffffff")))
        rect = self.label.boundingRect()
        self.label.setPos(-rect.width() / 2, bar_y - rect.height() - 1)
        self._bar_y = bar_y

    def update(self, x, y, hp_ratio, mana_ratio=None, shield=False, visible=True):
        self.body.setVisible(visible)
        if not visible:
            return
        self.body.setPos(x, y)
        hp_ratio = max(0.0, min(1.0, hp_ratio))
        half = self.BAR_WIDTH / 2
        self.hp_fg.setRect(-half, self._bar_y, self.BAR_WIDTH * hp_ratio, 4)
        if hp_ratio > 0.6:
            color = "#00e676"
        elif hp_ratio > 0.3:
            color = "#ffea00"
        else:
            color = "#ff1744"
        self.hp_fg.setBrush(QBrush(QColor(color)))
        if self.mana_fg is not None and mana_ratio is not None:
            mana_ratio = max(0.0, min(1.0, mana_ratio))
            self.mana_fg.setRect(-half, self._bar_y + 5, self.BAR_WIDTH * mana_ratio, 3)
        self.shield.setVisible(shield)

    def remove(self, scene):
        scene.removeItem(self.body)


class MapCanvas(QGraphicsView):
    """Map view: LMB adds waypoint, RMB removes last, MMB pans, wheel zooms."""

    waypoints_changed = pyqtSignal(int)
    edit_blocked = pyqtSignal()
    cursor_moved = pyqtSignal(float, float)
    zoom_changed = pyqtSignal(float)

    MIN_ZOOM = 0.1
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.15
    WAYPOINT_RADIUS = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.NoContextMenu)

        self.editing_enabled = True
        self._zoom = 1.0
        self._panning = False
        self._pan_last = QPoint()
        self._map_item = None
        self._waypoints = []
        self._waypoint_items = []
        self._segment_items = []
        self._loop_item = None
        self._druid_graphic = None
        self._monster_graphics = []

        self.load_map(MAP_FILE)

    # ----- map ------------------------------------------------------------- #

    def load_map(self, path):
        """Load a map image. Returns True on success; falls back to a dark grid."""
        if self._map_item is not None:
            self._scene.removeItem(self._map_item)
            self._map_item = None
        pixmap = QPixmap(path) if path and os.path.isfile(path) else QPixmap()
        if pixmap.isNull():
            width, height = DEFAULT_SCENE_SIZE
            self._scene.setSceneRect(0, 0, width, height)
            loaded = False
        else:
            self._map_item = QGraphicsPixmapItem(pixmap)
            self._map_item.setZValue(-10)
            self._map_item.setTransformationMode(Qt.SmoothTransformation)
            self._scene.addItem(self._map_item)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            loaded = True
        self.centerOn(self._scene.sceneRect().center())
        return loaded

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#2b2b2b"))
        if self._map_item is not None:
            return
        scene_rect = self._scene.sceneRect()
        area = rect.intersected(scene_rect)
        painter.fillRect(area, QColor("#333333"))
        if self._zoom < 0.35:
            return
        painter.setPen(QPen(QColor("#3c3c3c"), 0))
        left = int(area.left()) - int(area.left()) % TILE_PX
        top = int(area.top()) - int(area.top()) % TILE_PX
        x = left
        while x <= area.right():
            painter.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))
            x += TILE_PX
        y = top
        while y <= area.bottom():
            painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            y += TILE_PX

    # ----- waypoints ------------------------------------------------------- #

    def waypoints(self):
        return list(self._waypoints)

    def add_waypoint(self, x, y):
        index = len(self._waypoints)
        self._waypoints.append((x, y))

        r = self.WAYPOINT_RADIUS
        color = QColor("#00e676") if index == 0 else QColor("#00b0ff")
        circle = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setBrush(QBrush(color))
        circle.setPen(QPen(QColor("#ffffff"), 1.5))
        circle.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
        circle.setPos(x, y)
        circle.setZValue(5)
        label = QGraphicsSimpleTextItem(str(index + 1), circle)
        label.setFont(QFont("Sans", 8, QFont.Bold))
        label.setBrush(QBrush(QColor("#ffffff")))
        label.setPos(r + 2, -r - 6)
        self._scene.addItem(circle)
        self._waypoint_items.append(circle)

        if index > 0:
            px, py = self._waypoints[index - 1]
            line = QGraphicsLineItem(px, py, x, y)
            pen = QPen(QColor("#ffc400"), 2)
            pen.setCosmetic(True)
            line.setPen(pen)
            line.setZValue(4)
            self._scene.addItem(line)
            self._segment_items.append(line)

        self._update_loop_line()
        self.waypoints_changed.emit(len(self._waypoints))

    def remove_last_waypoint(self):
        if not self._waypoints:
            return
        self._waypoints.pop()
        self._scene.removeItem(self._waypoint_items.pop())
        if self._segment_items:
            self._scene.removeItem(self._segment_items.pop())
        self._update_loop_line()
        self.waypoints_changed.emit(len(self._waypoints))

    def clear_waypoints(self):
        while self._waypoints:
            self.remove_last_waypoint()

    def _update_loop_line(self):
        """Dashed line from the last waypoint back to the first (route is a loop)."""
        if self._loop_item is not None:
            self._scene.removeItem(self._loop_item)
            self._loop_item = None
        if len(self._waypoints) < 3:
            return
        (fx, fy), (lx, ly) = self._waypoints[0], self._waypoints[-1]
        self._loop_item = QGraphicsLineItem(lx, ly, fx, fy)
        pen = QPen(QColor("#ffc400"), 1.5, Qt.DashLine)
        pen.setCosmetic(True)
        self._loop_item.setPen(pen)
        self._loop_item.setZValue(4)
        self._scene.addItem(self._loop_item)

    # ----- entities -------------------------------------------------------- #

    def sync_entities(self, simulation):
        if self._druid_graphic is None:
            self._druid_graphic = EntityGraphic(self._scene, "Druid", QColor("#00c853"),
                                                radius=12, with_mana_bar=True, z=20)
        if len(self._monster_graphics) != len(simulation.monsters):
            self._clear_monster_graphics()
            self._monster_graphics = [
                EntityGraphic(self._scene, m.name, QColor("#d50000"), radius=11, z=15)
                for m in simulation.monsters
            ]
        druid = simulation.druid
        self._druid_graphic.update(druid.x, druid.y, druid.hp_ratio, druid.mana_ratio,
                                   shield=druid.utamo_active, visible=True)
        for monster, graphic in zip(simulation.monsters, self._monster_graphics):
            graphic.update(monster.x, monster.y, monster.hp_ratio, visible=monster.alive)

    def _clear_monster_graphics(self):
        for graphic in self._monster_graphics:
            graphic.remove(self._scene)
        self._monster_graphics = []

    def clear_entities(self):
        self._clear_monster_graphics()
        if self._druid_graphic is not None:
            self._druid_graphic.remove(self._scene)
            self._druid_graphic = None

    def follow(self, x, y):
        """Keep the given point in view, recentering only when it nears the edge."""
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        margin_x = view_rect.width() * 0.15
        margin_y = view_rect.height() * 0.15
        inner = view_rect.adjusted(margin_x, margin_y, -margin_x, -margin_y)
        if not inner.contains(x, y):
            self.centerOn(x, y)

    # ----- mouse & wheel --------------------------------------------------- #

    def mousePressEvent(self, event):
        button = event.button()
        if button == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if button in (Qt.LeftButton, Qt.RightButton):
            event.accept()
            if not self.editing_enabled:
                self.edit_blocked.emit()
                return
            if button == Qt.LeftButton:
                pos = self.mapToScene(event.pos())
                if self._scene.sceneRect().contains(pos):
                    self.add_waypoint(pos.x(), pos.y())
            else:
                self.remove_last_waypoint()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        pos = self.mapToScene(event.pos())
        self.cursor_moved.emit(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = self.ZOOM_STEP if delta > 0 else 1 / self.ZOOM_STEP
        new_zoom = self._zoom * factor
        if new_zoom < self.MIN_ZOOM or new_zoom > self.MAX_ZOOM:
            return
        self._zoom = new_zoom
        self.scale(factor, factor)
        self.zoom_changed.emit(self._zoom)
        event.accept()


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

LOG_COLORS = {
    "info": "#b0bec5",
    "move": "#78909c",
    "damage": "#ff5252",
    "shield": "#40c4ff",
    "heal": "#69f0ae",
    "mana": "#448aff",
    "attack": "#ffd740",
    "kill": "#e040fb",
    "warn": "#ffab40",
    "death": "#ff1744",
    "system": "#ffffff",
}

STYLESHEET = """
QMainWindow, QWidget { background-color: #1e1f22; color: #dcdcdc; font-size: 10pt; }
QGroupBox {
    border: 1px solid #3a3d42; border-radius: 6px; margin-top: 14px; padding: 8px;
    font-weight: bold; color: #e0e0e0;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QSpinBox {
    background-color: #2b2d31; border: 1px solid #44474d; border-radius: 4px;
    padding: 3px 6px; color: #ffffff;
}
QSpinBox:disabled { color: #777; }
QPushButton {
    background-color: #2f3237; border: 1px solid #4a4e55; border-radius: 5px;
    padding: 7px 12px; color: #ffffff; font-weight: bold;
}
QPushButton:hover { background-color: #3a3e44; }
QPushButton:disabled { color: #666; border-color: #33363a; background-color: #26282c; }
QPushButton#startButton { background-color: #1b5e20; border-color: #2e7d32; }
QPushButton#startButton:hover { background-color: #2e7d32; }
QPushButton#startButton:disabled { background-color: #26282c; border-color: #33363a; }
QPushButton#resetButton { background-color: #7f1d1d; border-color: #b71c1c; }
QPushButton#resetButton:hover { background-color: #b71c1c; }
QProgressBar {
    border: 1px solid #44474d; border-radius: 4px; background-color: #111;
    text-align: center; color: #ffffff; height: 18px; font-weight: bold;
}
QProgressBar#hpBar::chunk { background-color: #c62828; border-radius: 3px; }
QProgressBar#manaBar::chunk { background-color: #1565c0; border-radius: 3px; }
QTextEdit#eventLog {
    background-color: #000000; color: #cfcfcf; border: 1px solid #333;
    font-family: "DejaVu Sans Mono", "Consolas", monospace; font-size: 9pt;
}
QLabel#hint { color: #8a8f98; font-size: 9pt; }
QStatusBar { background-color: #17181a; color: #9aa0a6; }
QSplitter::handle { background-color: #2b2d31; }
"""


class MainWindow(QMainWindow):
    STATE_IDLE = "idle"
    STATE_RUNNING = "running"
    STATE_PAUSED = "paused"
    STATE_FINISHED = "finished"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Symulator Tibia - Polowanie Druida")
        self.resize(1520, 900)

        self.simulation = None
        self.state = self.STATE_IDLE

        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.on_tick)

        self._build_ui()
        self._update_character_preview()
        self._refresh_status()
        self._apply_state(self.STATE_IDLE)

        if os.path.isfile(MAP_FILE):
            self.log_system(f"Wczytano mapę: {os.path.basename(MAP_FILE)}")
        else:
            self.log_system("Nie znaleziono map.png - używam ciemnego tła z siatką.")
        self.log_system("Narysuj trasę (min. 2 punkty) i kliknij \"Start symulacji\".")

    # ----- UI construction ------------------------------------------------- #

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        # Left pane: map + mouse hints
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 3, 6)
        self.canvas = MapCanvas()
        self.canvas.waypoints_changed.connect(self._on_waypoints_changed)
        self.canvas.edit_blocked.connect(self._on_edit_blocked)
        self.canvas.cursor_moved.connect(self._on_cursor_moved)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        left_layout.addWidget(self.canvas, 1)
        hint = QLabel("LPM: dodaj punkt trasy   |   PPM: usuń ostatni punkt   |   "
                      "ŚPM (przeciągnij): przesuwanie mapy   |   Kółko: przybliżenie")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(hint)
        splitter.addWidget(left)

        # Right pane: controls + status + log
        right = QWidget()
        right.setMinimumWidth(500)
        right.setMaximumWidth(640)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(3, 6, 6, 6)
        right_layout.setSpacing(6)
        right_layout.addWidget(self._build_character_group())
        right_layout.addWidget(self._build_controls_group())
        right_layout.addWidget(self._build_status_group())
        right_layout.addWidget(self._build_log_group(), 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1000, 520])
        self.setCentralWidget(splitter)

        self.cursor_label = QLabel("X: -  Y: -")
        self.zoom_label = QLabel("Zoom: 100%")
        self.waypoint_label = QLabel("Punkty trasy: 0")
        self.statusBar().addWidget(self.cursor_label)
        self.statusBar().addWidget(self.waypoint_label)
        self.statusBar().addPermanentWidget(self.zoom_label)

    def _build_character_group(self):
        group = QGroupBox("Postać - Druid")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.level_input = QSpinBox()
        self.level_input.setRange(8, 2000)
        self.level_input.setValue(100)
        self.level_input.valueChanged.connect(self._update_character_preview)

        self.magic_level_input = QSpinBox()
        self.magic_level_input.setRange(0, 150)
        self.magic_level_input.setValue(60)
        self.magic_level_input.valueChanged.connect(self._update_character_preview)

        self.monster_count_input = QSpinBox()
        self.monster_count_input.setRange(1, 40)
        self.monster_count_input.setValue(6)

        self.potions_input = QSpinBox()
        self.potions_input.setRange(0, 10000)
        self.potions_input.setValue(300)

        self.utamo_checkbox = QCheckBox("Używaj Utamo Vita (Mana Shield)")
        self.utamo_checkbox.setChecked(True)

        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #9ccc65;")
        self.preview_label.setWordWrap(True)

        form.addRow("Poziom:", self.level_input)
        form.addRow("Poziom magii:", self.magic_level_input)
        form.addRow("Giant Spidery:", self.monster_count_input)
        form.addRow("Great Mana Potion:", self.potions_input)
        form.addRow("", self.utamo_checkbox)
        form.addRow("", self.preview_label)
        self._setup_inputs = [self.level_input, self.magic_level_input,
                              self.monster_count_input, self.potions_input, self.utamo_checkbox]
        return group

    def _build_controls_group(self):
        group = QGroupBox("Sterowanie")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.start_button = QPushButton("▶ Start symulacji")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_simulation)
        self.pause_button = QPushButton("⏸ Pauza")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button = QPushButton("⟲ Reset")
        self.reset_button.setObjectName("resetButton")
        self.reset_button.clicked.connect(self.reset_simulation)
        row.addWidget(self.start_button, 2)
        row.addWidget(self.pause_button, 1)
        row.addWidget(self.reset_button, 1)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self.clear_route_button = QPushButton("Wyczyść trasę")
        self.clear_route_button.clicked.connect(self.clear_route)
        self.load_map_button = QPushButton("Wczytaj mapę…")
        self.load_map_button.clicked.connect(self.load_map_dialog)
        row2.addWidget(self.clear_route_button)
        row2.addWidget(self.load_map_button)
        clear_log_button = QPushButton("Wyczyść dziennik")
        clear_log_button.clicked.connect(lambda: self.event_log.clear())
        row2.addWidget(clear_log_button)
        layout.addLayout(row2)
        return group

    def _build_status_group(self):
        group = QGroupBox("Status")
        layout = QVBoxLayout(group)

        self.hp_bar = QProgressBar()
        self.hp_bar.setObjectName("hpBar")
        self.mana_bar = QProgressBar()
        self.mana_bar.setObjectName("manaBar")
        bars = QFormLayout()
        bars.addRow("HP:", self.hp_bar)
        bars.addRow("Mana:", self.mana_bar)
        layout.addLayout(bars)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        self.status_values = {}
        fields = [
            ("time", "Czas symulacji:"), ("utamo", "Utamo Vita:"),
            ("kills", "Zabite potwory:"), ("exp", "Doświadczenie:"),
            ("exp_h", "Exp/h:"), ("laps", "Okrążenia:"),
            ("absorbed", "Pochłonięte:"), ("hp_dmg", "Obrażenia HP:"),
            ("potions", "Mikstury:"), ("heals", "Exura Gran:"),
        ]
        for i, (key, text) in enumerate(fields):
            name = QLabel(text)
            name.setStyleSheet("color: #9aa0a6;")
            value = QLabel("-")
            value.setStyleSheet("color: #ffffff; font-weight: bold;")
            row, col = divmod(i, 2)
            grid.addWidget(name, row, col * 2)
            grid.addWidget(value, row, col * 2 + 1)
            self.status_values[key] = value
        layout.addLayout(grid)
        return group

    def _build_log_group(self):
        group = QGroupBox("Dziennik zdarzeń")
        layout = QVBoxLayout(group)
        self.event_log = QTextEdit()
        self.event_log.setObjectName("eventLog")
        self.event_log.setReadOnly(True)
        self.event_log.document().setMaximumBlockCount(3000)
        layout.addWidget(self.event_log)
        return group

    # ----- state handling -------------------------------------------------- #

    def _apply_state(self, state):
        self.state = state
        idle = state == self.STATE_IDLE
        self.start_button.setEnabled(idle)
        self.pause_button.setEnabled(state in (self.STATE_RUNNING, self.STATE_PAUSED))
        self.pause_button.setText("▶ Wznów" if state == self.STATE_PAUSED else "⏸ Pauza")
        self.reset_button.setEnabled(not idle or self.simulation is not None)
        self.clear_route_button.setEnabled(idle)
        self.load_map_button.setEnabled(idle)
        for widget in self._setup_inputs:
            widget.setEnabled(idle)
        self.canvas.editing_enabled = idle

    def start_simulation(self):
        waypoints = self.canvas.waypoints()
        if len(waypoints) < 2:
            self.log_event(0.0, "warn", "Nie można wystartować: wyznacz co najmniej 2 punkty trasy.")
            QMessageBox.warning(self, "Brak trasy",
                                "Wyznacz co najmniej 2 punkty trasy lewym przyciskiem myszy.")
            return
        self.simulation = HuntSimulation(
            level=self.level_input.value(),
            magic_level=self.magic_level_input.value(),
            use_utamo=self.utamo_checkbox.isChecked(),
            waypoints=waypoints,
            monster_count=self.monster_count_input.value(),
            mana_potions=self.potions_input.value(),
        )
        druid = self.simulation.druid
        self.log_system("=" * 46)
        self.log_system(f"Start symulacji: Druid lvl {druid.level}, ML {druid.magic_level}, "
                        f"HP {druid.max_hp}, Mana {druid.max_mana}, "
                        f"Utamo Vita: {'TAK' if self.simulation.use_utamo else 'NIE'}.")
        self.log_system(f"Trasa: {len(waypoints)} punktów, potwory: {len(self.simulation.monsters)}, "
                        f"tick: {TICK_MS} ms.")
        self.canvas.sync_entities(self.simulation)
        self._refresh_status()
        self._apply_state(self.STATE_RUNNING)
        self.timer.start()

    def toggle_pause(self):
        if self.state == self.STATE_RUNNING:
            self.timer.stop()
            self._apply_state(self.STATE_PAUSED)
            self.log_event(self.simulation.time, "system", "Symulacja wstrzymana.")
        elif self.state == self.STATE_PAUSED:
            self._apply_state(self.STATE_RUNNING)
            self.log_event(self.simulation.time, "system", "Symulacja wznowiona.")
            self.timer.start()

    def reset_simulation(self):
        self.timer.stop()
        if self.simulation is not None:
            self.log_event(self.simulation.time, "system", "Symulacja zresetowana.")
        self.simulation = None
        self.canvas.clear_entities()
        self._refresh_status()
        self._apply_state(self.STATE_IDLE)

    def clear_route(self):
        self.canvas.clear_waypoints()
        self.log_system("Trasa wyczyszczona.")

    def load_map_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj mapę", os.path.dirname(MAP_FILE),
                                              "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        if self.canvas.load_map(path):
            self.log_system(f"Wczytano mapę: {os.path.basename(path)}")
        else:
            self.log_system(f"Nie udało się wczytać mapy: {os.path.basename(path)}")

    # ----- simulation tick ------------------------------------------------- #

    def on_tick(self):
        sim = self.simulation
        if sim is None:
            self.timer.stop()
            return
        for event in sim.tick():
            self.log_event(event.time, event.kind, event.message)
        self.canvas.sync_entities(sim)
        self.canvas.follow(sim.druid.x, sim.druid.y)
        self._refresh_status()
        if sim.finished:
            self.timer.stop()
            self._apply_state(self.STATE_FINISHED)

    # ----- status & logging ------------------------------------------------ #

    def _update_character_preview(self):
        preview = Druid(self.level_input.value(), self.magic_level_input.value())
        heal_low = int(preview.level / 5 + preview.magic_level * 3.184 + 20)
        heal_high = int(preview.level / 5 + preview.magic_level * 5.59 + 35)
        self.preview_label.setText(
            f"Maks. HP: {preview.max_hp}   |   Maks. mana: {preview.max_mana}\n"
            f"Exura Gran: {heal_low}-{heal_high} HP   |   Prędkość: {preview.speed}")
        if self.simulation is None:
            self._set_bar(self.hp_bar, preview.hp, preview.max_hp)
            self._set_bar(self.mana_bar, preview.mana, preview.max_mana)

    @staticmethod
    def _set_bar(bar, value, maximum):
        bar.setMaximum(max(1, maximum))
        bar.setValue(int(value))
        bar.setFormat(f"{int(value)} / {maximum}")

    def _refresh_status(self):
        sim = self.simulation
        values = self.status_values
        if sim is None:
            for value in values.values():
                value.setText("-")
            values["time"].setText(format_sim_time(0))
            self._update_character_preview()
            return
        druid = sim.druid
        self._set_bar(self.hp_bar, druid.hp, druid.max_hp)
        self._set_bar(self.mana_bar, druid.mana, druid.max_mana)
        values["time"].setText(format_sim_time(sim.time))
        if druid.utamo_active:
            values["utamo"].setText(f"Aktywne ({int(druid.utamo_remaining)} s)")
            values["utamo"].setStyleSheet("color: #40c4ff; font-weight: bold;")
        else:
            values["utamo"].setText("Nieaktywne")
            values["utamo"].setStyleSheet("color: #ff8a80; font-weight: bold;")
        values["kills"].setText(str(sim.kills))
        values["exp"].setText(f"{sim.experience:,}".replace(",", " "))
        values["exp_h"].setText(f"{sim.experience_per_hour:,}".replace(",", " "))
        values["laps"].setText(str(sim.laps))
        values["absorbed"].setText(str(sim.damage_absorbed))
        values["hp_dmg"].setText(str(sim.damage_to_hp))
        values["potions"].setText(f"{sim.potions_used} / {sim.mana_potions}")
        values["heals"].setText(f"{sim.heals_cast}x (+{sim.hp_healed} HP)")

    def log_event(self, sim_time, kind, message):
        color = LOG_COLORS.get(kind, LOG_COLORS["info"])
        weight = "bold" if kind in ("death", "kill") else "normal"
        line = (f'<span style="color:#6d6d6d;">[{format_sim_time(sim_time)}]</span> '
                f'<span style="color:{color}; font-weight:{weight};">{html.escape(message)}</span>')
        self.event_log.append(line)
        scrollbar = self.event_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def log_system(self, message):
        sim_time = self.simulation.time if self.simulation is not None else 0.0
        self.log_event(sim_time, "system", message)

    # ----- canvas callbacks ------------------------------------------------ #

    def _on_waypoints_changed(self, count):
        self.waypoint_label.setText(f"Punkty trasy: {count}")

    def _on_edit_blocked(self):
        self.statusBar().showMessage("Edycja trasy jest zablokowana - zresetuj symulację.", 3000)

    def _on_cursor_moved(self, x, y):
        self.cursor_label.setText(f"X: {int(x)}  Y: {int(y)}  (sqm {int(x // TILE_PX)}, {int(y // TILE_PX)})")

    def _on_zoom_changed(self, zoom):
        self.zoom_label.setText(f"Zoom: {int(zoom * 100)}%")


def apply_dark_palette(app):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1e1f22"))
    palette.setColor(QPalette.WindowText, QColor("#dcdcdc"))
    palette.setColor(QPalette.Base, QColor("#2b2d31"))
    palette.setColor(QPalette.AlternateBase, QColor("#26282c"))
    palette.setColor(QPalette.Text, QColor("#dcdcdc"))
    palette.setColor(QPalette.Button, QColor("#2f3237"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#2979ff"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


def main():
    app = QApplication(sys.argv)
    apply_dark_palette(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
