"""
PyQt5 digital twin: map editor (collision grid per floor, PNG parser, spawns,
waypoints), real-time manual / agent visualizer and background DQN training.
"""

import html
import json
import os
import sys

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .agent import DQNAgent
from .assets import AssetManager, sprite_entries
from .entities import GROUND_FLOOR, Druid
from .env import (
    ACTION_DOWN,
    ACTION_DOWN_LEFT,
    ACTION_DOWN_RIGHT,
    ACTION_EXORI_FRIGO,
    ACTION_EXORI_TERA,
    ACTION_HEAL,
    ACTION_LEFT,
    ACTION_MANA_POTION,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_UP_LEFT,
    ACTION_UP_RIGHT,
    ACTION_WAIT,
    STRIKE_RANGE,
    DruidHuntEnv,
    EnvConfig,
    grid_to_rows,
)
from .mapparser import ColorRule, MapParser, load_rgba
from .monsters import MonsterDatabase
from .observation import ACTION_LABELS_PL
from .physics import DEFAULT_DIAGONAL_MULTIPLIER, DEFAULT_GROUND_SPEED, step_duration_ms
from .trainer import run_training

CELL = 32                       # scene units per tile
VISUAL_TICK_MS = 100
MAX_DECISIONS_PER_FRAME = 50
DEFAULT_GRID = (32, 24)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_FILE = os.path.join(PROJECT_DIR, "map.png")
SPRITES_DIR = os.path.join(PROJECT_DIR, "sprites")
SPRITE_SIZE = CELL

TOOL_WAYPOINT = "waypoint"
TOOL_MONSTER = "monster"
TOOL_OBSTACLE = "obstacle"


def format_game_time(ms):
    seconds = ms / 1000.0
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"


# --------------------------------------------------------------------------- #
# Sprites
# --------------------------------------------------------------------------- #

class SpriteStore:
    """Loads sprites from sprites/ (GIF first frame or PNG), trims transparent margins, fits them to a tile."""

    def __init__(self, manager):
        self.manager = manager
        self._cache = {}

    def pixmap(self, name, size=SPRITE_SIZE):
        key = (name, size)
        if key not in self._cache:
            path = self.manager.local_path(name)
            self._cache[key] = self._load(path, size) if path else None
        return self._cache[key]

    def invalidate(self, name=None):
        for key in [k for k in self._cache if name is None or k[0] == name]:
            del self._cache[key]

    @staticmethod
    def _load(path, size):
        image = QImage(path)
        if image.isNull():
            return None
        image = image.convertToFormat(QImage.Format_ARGB32)
        width, height = image.width(), image.height()
        buffer = image.constBits()
        buffer.setsize(image.byteCount())
        pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, image.bytesPerLine())
        alpha = pixels[:, : width * 4].reshape(height, width, 4)[..., 3]   # ARGB32 in memory: B, G, R, A
        ys, xs = np.nonzero(alpha)
        if len(xs):
            image = image.copy(int(xs.min()), int(ys.min()),
                               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        # Pixel art: smooth only when shrinking, nearest-neighbour when enlarging
        mode = Qt.SmoothTransformation if max(image.width(), image.height()) > size else Qt.FastTransformation
        return QPixmap.fromImage(image.scaled(size, size, Qt.KeepAspectRatio, mode))


class AssetWorker(QThread):
    """Runs AssetManager.ensure() off the GUI thread and reports each sprite as it lands."""

    message = pyqtSignal(str)
    sprite_ready = pyqtSignal(str)
    summary = pyqtSignal(dict)

    def __init__(self, entries, parent=None):
        super().__init__(parent)
        self.entries = entries
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _log(self, text):
        print(text, flush=True)
        self.message.emit(text)

    def run(self):
        manager = AssetManager(SPRITES_DIR, log=self._log)
        counts = {}
        for entry in self.entries:
            if self._stop_requested:
                break
            for result in manager.ensure([entry]):
                counts[result.status] = counts.get(result.status, 0) + 1
                if result.status == "downloaded":
                    self.sprite_ready.emit(result.name)
        self.summary.emit(counts)


# --------------------------------------------------------------------------- #
# Map canvas
# --------------------------------------------------------------------------- #

class GridCanvas(QGraphicsView):
    """LMB: use tool, RMB: erase tile, MMB drag: pan, wheel: zoom. Shows one floor (Z) at a time."""

    scenario_changed = pyqtSignal()
    edit_blocked = pyqtSignal()
    cell_hovered = pyqtSignal(int, int)

    MIN_ZOOM = 0.05
    MAX_ZOOM = 6.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)

        self.cols, self.rows = DEFAULT_GRID
        self.floor = GROUND_FLOOR
        self.tool = TOOL_WAYPOINT
        self.monster_type = None
        self.editing_enabled = True
        self.map_path = None
        self.map_cell_px = CELL
        self.sprites = None          # SpriteStore, set by the main window

        self.collision = {}          # z -> np.ndarray (rows, cols) uint8
        self.waypoints = []          # [(x, y, z)]
        self.spawns = {}             # (x, y, z) -> monster name
        self._map_item = None
        self._overlay_item = None
        self._overlay_buffer = None
        self._waypoint_items = []
        self._spawn_items = {}
        self._zoom = 1.0
        self._panning = False
        self._pan_last = QPoint()
        self._paint_button = None
        self._druid_item = None
        self._range_item = None
        self._monster_items = []

        self._scene.setSceneRect(0, 0, self.cols * CELL, self.rows * CELL)
        self._render_overlay()

    # ----- grid & floors --------------------------------------------------- #

    def grid(self, z=None):
        z = self.floor if z is None else z
        if z not in self.collision:
            self.collision[z] = np.zeros((self.rows, self.cols), dtype=np.uint8)
        return self.collision[z]

    def resize_grid(self, cols, rows):
        for z, grid in list(self.collision.items()):
            resized = np.zeros((rows, cols), dtype=np.uint8)
            h, w = min(rows, grid.shape[0]), min(cols, grid.shape[1])
            resized[:h, :w] = grid[:h, :w]
            self.collision[z] = resized
        self.cols, self.rows = cols, rows
        self._scene.setSceneRect(0, 0, cols * CELL, rows * CELL)
        self.waypoints = [w for w in self.waypoints if self._in_bounds(w[0], w[1])]
        for key in [k for k in self.spawns if not self._in_bounds(k[0], k[1])]:
            del self.spawns[key]
        self.refresh()

    def set_floor(self, z):
        self.floor = int(z)
        self.refresh()

    def refresh(self):
        self._render_overlay()
        self._redraw_spawns()
        self._redraw_waypoints()
        self.scenario_changed.emit()

    def load_map(self, path, cell_px, parser=None):
        """Show the image as background; with a parser, replace this floor's collision with the parsed grid."""
        if self._map_item is not None:
            self._scene.removeItem(self._map_item)
            self._map_item = None
        self.map_path = None
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            return None
        self.map_path = path
        self.map_cell_px = cell_px
        self._map_item = QGraphicsPixmapItem(pixmap)
        self._map_item.setZValue(-10)
        self._map_item.setScale(CELL / cell_px)
        self._scene.addItem(self._map_item)
        self.resize_grid(max(1, pixmap.width() // cell_px), max(1, pixmap.height() // cell_px))
        parsed = np.zeros((self.rows, self.cols), dtype=np.uint8)
        if parser is not None:
            parsed = parser.to_grid(load_rgba(path))
            self.collision[self.floor] = parsed
            self.refresh()
        self.centerOn(self._scene.sceneRect().center())
        return parsed

    def _render_overlay(self):
        grid = self.grid()
        rgba = np.zeros((self.rows, self.cols, 4), dtype=np.uint8)
        rgba[grid > 0] = (150, 40, 40, 150)
        self._overlay_buffer = np.ascontiguousarray(rgba)
        image = QImage(self._overlay_buffer.data, self.cols, self.rows, self.cols * 4, QImage.Format_RGBA8888)
        if self._overlay_item is None:
            self._overlay_item = QGraphicsPixmapItem()
            self._overlay_item.setZValue(1)
            self._overlay_item.setScale(CELL)
            self._overlay_item.setTransformationMode(Qt.FastTransformation)
            self._scene.addItem(self._overlay_item)
        self._overlay_item.setPixmap(QPixmap.fromImage(image))

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor("#1b1c1f"))
        painter.fillRect(rect.intersected(self._scene.sceneRect()), QColor("#3a3a3a"))

    def drawForeground(self, painter, rect):
        if self._zoom < 0.4:
            return
        area = rect.intersected(self._scene.sceneRect())
        pen = QPen(QColor(255, 255, 255, 28))
        pen.setCosmetic(True)
        painter.setPen(pen)
        x = int(area.left()) // CELL * CELL
        while x <= area.right():
            painter.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))
            x += CELL
        y = int(area.top()) // CELL * CELL
        while y <= area.bottom():
            painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            y += CELL

    def _in_bounds(self, x, y):
        return 0 <= x < self.cols and 0 <= y < self.rows

    def cell_at(self, view_pos):
        pos = self.mapToScene(view_pos)
        return int(pos.x() // CELL), int(pos.y() // CELL)

    # ----- editing --------------------------------------------------------- #

    def _redraw_spawns(self):
        for item in self._spawn_items.values():
            self._scene.removeItem(item)
        self._spawn_items = {}
        for (x, y, z), name in self.spawns.items():
            if z != self.floor:
                continue
            item = QGraphicsRectItem(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
            item.setBrush(QBrush(QColor(213, 0, 0, 90)))
            item.setPen(QPen(QColor("#ff5252"), 2, Qt.DashLine))
            item.setZValue(2)
            pixmap = self.sprites.pixmap(name) if self.sprites else None
            if pixmap is not None:
                icon = QGraphicsPixmapItem(pixmap, item)
                icon.setOpacity(0.6)
                icon.setPos(x * CELL + (CELL - pixmap.width()) / 2, y * CELL + (CELL - pixmap.height()) / 2)
            else:
                label = QGraphicsSimpleTextItem(name[:2], item)
                label.setFont(QFont("Sans", 8, QFont.Bold))
                label.setBrush(QBrush(QColor("#ffffff")))
                label.setPos(x * CELL + 8, y * CELL + 8)
            item.setToolTip(f"Spawn: {name} (z={z})")
            self._scene.addItem(item)
            self._spawn_items[(x, y, z)] = item

    def _redraw_waypoints(self):
        for item in self._waypoint_items:
            self._scene.removeItem(item)
        self._waypoint_items = []
        on_floor = [(i, x * CELL + CELL / 2, y * CELL + CELL / 2)
                    for i, (x, y, z) in enumerate(self.waypoints) if z == self.floor]
        pen = QPen(QColor("#ffc400"), 2)
        pen.setCosmetic(True)
        for (ia, ax, ay), (ib, bx, by) in zip(on_floor, on_floor[1:]):
            line = QGraphicsLineItem(ax, ay, bx, by)
            line.setPen(pen if ib == ia + 1 else QPen(QColor("#ffc400"), 1, Qt.DotLine))
            line.setZValue(3)
            self._scene.addItem(line)
            self._waypoint_items.append(line)
        if len(on_floor) >= 3 and len(on_floor) == len(self.waypoints):
            loop = QGraphicsLineItem(on_floor[-1][1], on_floor[-1][2], on_floor[0][1], on_floor[0][2])
            loop_pen = QPen(QColor("#ffc400"), 1.5, Qt.DashLine)
            loop_pen.setCosmetic(True)
            loop.setPen(loop_pen)
            loop.setZValue(3)
            self._scene.addItem(loop)
            self._waypoint_items.append(loop)
        for i, cx, cy in on_floor:
            circle = QGraphicsEllipseItem(cx - 7, cy - 7, 14, 14)
            circle.setBrush(QBrush(QColor("#00e676" if i == 0 else "#00b0ff")))
            circle.setPen(QPen(QColor("#ffffff"), 1.5))
            circle.setZValue(4)
            label = QGraphicsSimpleTextItem(str(i + 1), circle)
            label.setFont(QFont("Sans", 7, QFont.Bold))
            label.setBrush(QBrush(QColor("#ffffff")))
            label.setPos(cx + 7, cy - 16)
            self._scene.addItem(circle)
            self._waypoint_items.append(circle)

    def _apply_tool(self, cell, dragging):
        x, y = cell
        if not self._in_bounds(x, y):
            return
        key = (x, y, self.floor)
        grid = self.grid()
        if self.tool == TOOL_OBSTACLE:
            if grid[y, x]:
                return
            grid[y, x] = 1
            self.spawns.pop(key, None)
            self.waypoints = [w for w in self.waypoints if w != key]
            self.refresh()
            return
        if dragging or grid[y, x]:
            return
        if self.tool == TOOL_WAYPOINT:
            if self.waypoints and self.waypoints[-1] == key:
                return
            self.waypoints.append(key)
            self._redraw_waypoints()
        elif self.tool == TOOL_MONSTER and self.monster_type:
            self.spawns[key] = self.monster_type
            self._redraw_spawns()
        self.scenario_changed.emit()

    def _erase(self, cell, dragging):
        x, y = cell
        if not self._in_bounds(x, y):
            return
        key = (x, y, self.floor)
        grid = self.grid()
        had_content = bool(grid[y, x]) or key in self.spawns or key in self.waypoints
        grid[y, x] = 0
        self.spawns.pop(key, None)
        if key in self.waypoints:
            self.waypoints = [w for w in self.waypoints if w != key]
        elif not had_content and not dragging and self.tool == TOOL_WAYPOINT and self.waypoints:
            self.waypoints.pop()
        if had_content or not dragging:
            self.refresh()

    def clear_floor(self):
        self.collision[self.floor] = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.spawns = {k: v for k, v in self.spawns.items() if k[2] != self.floor}
        self.waypoints = [w for w in self.waypoints if w[2] != self.floor]
        self.refresh()

    def collision_rows(self):
        return {str(z): grid_to_rows(g) for z, g in sorted(self.collision.items()) if g.any()}

    def apply_config(self, config):
        self.cols, self.rows = config.cols, config.rows
        self._scene.setSceneRect(0, 0, self.cols * CELL, self.rows * CELL)
        self.collision = {int(z): config.collision_grid(int(z)) for z in config.collision}
        self.waypoints = [tuple(int(v) for v in w[:3]) for w in config.waypoints]
        self.spawns = {(int(x), int(y), int(z)): name for x, y, z, name in config.spawns}
        self.refresh()

    # ----- live entities --------------------------------------------------- #

    def sync_entities(self, env):
        if self._druid_item is None:
            self._range_item = QGraphicsRectItem()
            self._range_item.setBrush(QBrush(QColor(0, 230, 118, 22)))
            self._range_item.setPen(QPen(QColor(0, 230, 118, 90), 1, Qt.DotLine))
            self._range_item.setZValue(5)
            self._scene.addItem(self._range_item)
            self._druid_item = self._make_entity(QColor("#00c853"), "Druid", 22, mana_bar=True)
        if len(self._monster_items) != len(env.monsters):
            self._clear_monsters()
            self._monster_items = [self._make_entity(QColor("#d50000"), m.name, 15) for m in env.monsters]

        druid = env.druid
        span = STRIKE_RANGE * 2 + 1
        self._range_item.setRect((druid.x - STRIKE_RANGE) * CELL, (druid.y - STRIKE_RANGE) * CELL,
                                 span * CELL, span * CELL)
        self._range_item.setVisible(druid.z == self.floor)
        self._update_entity(self._druid_item, druid.x, druid.y, druid.hp_ratio, druid.mana_ratio,
                            druid.utamo_active, druid.alive and druid.z == self.floor)
        for monster, item in zip(env.monsters, self._monster_items):
            self._update_entity(item, monster.x, monster.y, monster.hp_ratio, None, False,
                                monster.alive and monster.z == self.floor)

    def _make_entity(self, color, label, z, mana_bar=False):
        r = CELL / 2 - 4
        body = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        body.setZValue(z)
        item = {"body": body, "name": label, "color": color, "sprite": None, "r": r}
        self._style_entity(item)
        bg = QGraphicsRectItem(-15, -r - 8, 30, 4, body)
        bg.setBrush(QBrush(QColor("#111")))
        bg.setPen(QPen(Qt.NoPen))
        hp = QGraphicsRectItem(-15, -r - 8, 30, 4, body)
        hp.setPen(QPen(Qt.NoPen))
        mana = None
        if mana_bar:
            mana = QGraphicsRectItem(-15, -r - 4, 30, 3, body)
            mana.setBrush(QBrush(QColor("#2979ff")))
            mana.setPen(QPen(Qt.NoPen))
        shield = QGraphicsEllipseItem(-r - 4, -r - 4, 2 * r + 8, 2 * r + 8, body)
        shield.setPen(QPen(QColor("#40c4ff"), 2, Qt.DashLine))
        shield.setVisible(False)
        text = QGraphicsSimpleTextItem(label, body)
        text.setFont(QFont("Sans", 6, QFont.Bold))
        text.setBrush(QBrush(QColor("#ffffff")))
        text.setPos(-text.boundingRect().width() / 2, -r - 19)
        self._scene.addItem(body)
        item.update(hp=hp, mana=mana, shield=shield)
        self._attach_sprite(item)
        return item

    def _style_entity(self, item):
        """Solid disc without a sprite; with a sprite only a faint team-coloured base ring."""
        color = item["color"]
        if item["sprite"] is None:
            item["body"].setBrush(QBrush(color))
            item["body"].setPen(QPen(color.darker(170), 2))
        else:
            base = QColor(color)
            base.setAlpha(60)
            item["body"].setBrush(QBrush(base))
            ring = QColor(color)
            ring.setAlpha(170)
            item["body"].setPen(QPen(ring, 1.5))

    def _attach_sprite(self, item):
        if item["sprite"] is not None or self.sprites is None:
            return
        pixmap = self.sprites.pixmap(item["name"])
        if pixmap is None:
            return
        sprite = QGraphicsPixmapItem(pixmap, item["body"])
        sprite.setOffset(-pixmap.width() / 2, -pixmap.height() / 2)
        sprite.setTransformationMode(Qt.SmoothTransformation)
        sprite.setZValue(-1)   # above the base disc, below the HP bars and name
        item["sprite"] = sprite
        self._style_entity(item)

    def refresh_sprites(self):
        """Re-apply sprites after new files were downloaded."""
        for item in ([self._druid_item] if self._druid_item else []) + self._monster_items:
            self._attach_sprite(item)
        self._redraw_spawns()

    @staticmethod
    def _update_entity(item, x, y, hp_ratio, mana_ratio, shield, visible):
        body = item["body"]
        body.setVisible(visible)
        if not visible:
            return
        body.setPos(x * CELL + CELL / 2, y * CELL + CELL / 2)
        r = item["r"]
        hp_ratio = max(0.0, min(1.0, hp_ratio))
        item["hp"].setRect(-15, -r - 8, 30 * hp_ratio, 4)
        color = "#00e676" if hp_ratio > 0.6 else "#ffea00" if hp_ratio > 0.3 else "#ff1744"
        item["hp"].setBrush(QBrush(QColor(color)))
        if item["mana"] is not None and mana_ratio is not None:
            item["mana"].setRect(-15, -r - 4, 30 * max(0.0, min(1.0, mana_ratio)), 3)
        item["shield"].setVisible(shield)

    def _clear_monsters(self):
        for item in self._monster_items:
            self._scene.removeItem(item["body"])
        self._monster_items = []

    def clear_entities(self):
        self._clear_monsters()
        if self._druid_item is not None:
            self._scene.removeItem(self._druid_item["body"])
            self._scene.removeItem(self._range_item)
            self._druid_item = None
            self._range_item = None

    # ----- input ----------------------------------------------------------- #

    def mousePressEvent(self, event):
        button = event.button()
        if button == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            return
        if button in (Qt.LeftButton, Qt.RightButton):
            if not self.editing_enabled:
                self.edit_blocked.emit()
                return
            self._paint_button = button
            cell = self.cell_at(event.pos())
            if button == Qt.LeftButton:
                self._apply_tool(cell, dragging=False)
            else:
                self._erase(cell, dragging=False)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return
        cell = self.cell_at(event.pos())
        self.cell_hovered.emit(*cell)
        if self._paint_button is not None and self.editing_enabled:
            if self._paint_button == Qt.LeftButton:
                self._apply_tool(cell, dragging=True)
            else:
                self._erase(cell, dragging=True)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.viewport().unsetCursor()
        elif event.button() == self._paint_button:
            self._paint_button = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        if not self.MIN_ZOOM <= self._zoom * factor <= self.MAX_ZOOM:
            return
        self._zoom *= factor
        self.scale(factor, factor)


# --------------------------------------------------------------------------- #
# Training thread & reward plot
# --------------------------------------------------------------------------- #

class TrainingWorker(QThread):
    progress = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config, agent, episodes, parent=None):
        super().__init__(parent)
        self.config = config
        self.agent = agent
        self.episodes = episodes
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            env = DruidHuntEnv(self.config)
            run_training(env, self.agent, self.episodes,
                         on_progress=self.progress.emit,
                         should_stop=lambda: self._stop_requested,
                         progress_every=0.25)
        except Exception as exc:  # surfaced in the GUI log instead of dying silently
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class RewardPlot(QWidget):
    """Episode reward history (faint) with a 50-episode moving average (bright)."""

    MAX_POINTS = 5000
    WINDOW = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rewards = []
        self.setMinimumHeight(130)

    def add(self, values):
        self.rewards.extend(values)
        self.rewards = self.rewards[-self.MAX_POINTS:]
        self.update()

    def clear(self):
        self.rewards = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0d0d0f"))
        painter.setPen(QPen(QColor("#333"), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if len(self.rewards) < 2:
            painter.setPen(QColor("#777"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Brak danych - rozpocznij trening")
            return
        averages = []
        running = 0.0
        for i, value in enumerate(self.rewards):
            running += value
            if i >= self.WINDOW:
                running -= self.rewards[i - self.WINDOW]
            averages.append(running / min(i + 1, self.WINDOW))
        low, high = min(self.rewards), max(self.rewards)
        if high - low < 1e-6:
            high = low + 1
        margin = 18
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin
        n = len(self.rewards)

        def point(i, value):
            return margin + width * i / (n - 1), margin + height * (1 - (value - low) / (high - low))

        for series, color, line_width in ((self.rewards, QColor(120, 144, 156, 90), 1),
                                          (averages, QColor("#ffd740"), 2)):
            path = QPainterPath()
            path.moveTo(*point(0, series[0]))
            for i in range(1, n):
                path.lineTo(*point(i, series[i]))
            painter.setPen(QPen(color, line_width))
            painter.drawPath(path)
        if low < 0 < high:
            _, zero_y = point(0, 0)
            painter.setPen(QPen(QColor("#555"), 1, Qt.DashLine))
            painter.drawLine(margin, int(zero_y), margin + width, int(zero_y))
        painter.setPen(QColor("#9aa0a6"))
        painter.drawText(4, 13, f"max {high:.0f}")
        painter.drawText(4, self.height() - 4, f"min {low:.0f}   |   śr. {self.WINDOW}: {averages[-1]:.1f}")


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

LOG_COLORS = {
    "info": "#b0bec5", "move": "#78909c", "damage": "#ff5252", "shield": "#40c4ff",
    "heal": "#69f0ae", "mana": "#448aff", "attack": "#ffd740", "kill": "#e040fb",
    "warn": "#ffab40", "death": "#ff1744", "system": "#ffffff", "train": "#80cbc4",
}

STYLESHEET = """
QWidget { background-color: #1e1f22; color: #dcdcdc; font-size: 10pt; }
QGroupBox { border: 1px solid #3a3d42; border-radius: 6px; margin-top: 14px; padding: 8px;
            font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QSpinBox, QDoubleSpinBox, QComboBox { background-color: #2b2d31; border: 1px solid #44474d;
                                      border-radius: 4px; padding: 3px 6px; color: #ffffff; }
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #777; }
QPushButton { background-color: #2f3237; border: 1px solid #4a4e55; border-radius: 5px;
              padding: 6px 10px; color: #ffffff; font-weight: bold; }
QPushButton:hover { background-color: #3a3e44; }
QPushButton:disabled { color: #666; border-color: #33363a; background-color: #26282c; }
QPushButton#primary { background-color: #1b5e20; border-color: #2e7d32; }
QPushButton#primary:hover { background-color: #2e7d32; }
QPushButton#danger { background-color: #7f1d1d; border-color: #b71c1c; }
QPushButton#danger:hover { background-color: #b71c1c; }
QPushButton#export { background-color: #0d47a1; border-color: #1565c0; }
QPushButton#export:hover { background-color: #1565c0; }
QPushButton:disabled#primary, QPushButton:disabled#danger, QPushButton:disabled#export {
    background-color: #26282c; border-color: #33363a; }
QProgressBar { border: 1px solid #44474d; border-radius: 4px; background-color: #111;
               text-align: center; color: #ffffff; font-weight: bold; }
QProgressBar#hpBar::chunk { background-color: #c62828; }
QProgressBar#manaBar::chunk { background-color: #1565c0; }
QProgressBar#trainBar::chunk { background-color: #00897b; }
QTextEdit#eventLog { background-color: #000000; color: #cfcfcf; border: 1px solid #333;
                     font-family: "DejaVu Sans Mono", "Consolas", monospace; font-size: 9pt; }
QLabel#hint { color: #8a8f98; font-size: 9pt; }
QScrollArea { border: none; }
QStatusBar { background-color: #17181a; color: #9aa0a6; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cyfrowy Bliźniak - Trening agenta RL (Druid)")
        self.resize(1580, 960)

        self.monster_db = MonsterDatabase.load()
        self.monsters_file = None
        self.asset_manager = AssetManager(SPRITES_DIR, log=print)
        self.sprite_store = SpriteStore(self.asset_manager)
        self.asset_worker = None
        self._sprite_resync = False
        self.agent = DQNAgent()
        self.env = None
        self.obs = None
        self.worker = None
        self.pending_action = None
        self.manual_running = False
        self.visual_clock_ms = 0.0

        self.visual_timer = QTimer(self)
        self.visual_timer.setTimerType(Qt.PreciseTimer)
        self.visual_timer.setInterval(VISUAL_TICK_MS)
        self.visual_timer.timeout.connect(self.on_visual_tick)

        self._build_ui()
        self.canvas.sprites = self.sprite_store
        self._bind_keys()
        self._populate_monsters()
        self._update_preview()
        self._update_model_label()
        self._refresh_controls()

        self.log("system", f"Baza potworów: {len(self.monster_db)} pozycji "
                           f"({os.path.basename(self.monster_db.source)}).")
        if os.path.isfile(MAP_FILE):
            self.load_map_file(MAP_FILE)
        else:
            self.log("system", f"Brak map.png - pusta siatka {self.canvas.cols}x{self.canvas.rows} pól.")
        self.log("system", "Narysuj trasę, przeszkody i spawny potworów, potem wybierz tryb pracy.")
        self.sync_sprites()

    # ----- UI construction ------------------------------------------------- #

    def _build_ui(self):
        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 3, 6)
        self.canvas = GridCanvas()
        self.canvas.scenario_changed.connect(self._on_scenario_changed)
        self.canvas.edit_blocked.connect(
            lambda *_: self.statusBar().showMessage("Edycja mapy zablokowana podczas symulacji/treningu.", 3000))
        self.canvas.cell_hovered.connect(
            lambda x, y: self.cell_label.setText(f"Pole: ({x}, {y}, {self.canvas.floor})"))
        left_layout.addWidget(self.canvas, 1)
        hint = QLabel("LPM: użyj narzędzia   |   PPM: wymaż pole (z narzędziem Trasa: usuń ostatni punkt)   |   "
                      "ŚPM: przesuwanie   |   Kółko: zoom")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(hint)
        splitter.addWidget(left)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(3, 0, 6, 0)
        controls_layout.addWidget(self._build_character_group())
        controls_layout.addWidget(self._build_editor_group())
        controls_layout.addWidget(self._build_mode_group())
        controls_layout.addWidget(self._build_model_group())
        controls_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(scroll)
        right_split.addWidget(self._build_log_group())
        right_split.setSizes([700, 240])
        right_split.setMinimumWidth(580)
        right_split.setMaximumWidth(740)
        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([980, 600])
        self.setCentralWidget(splitter)

        self.cell_label = QLabel("Pole: -")
        self.scenario_label = QLabel()
        self.statusBar().addWidget(self.cell_label)
        self.statusBar().addWidget(self.scenario_label)
        self._on_scenario_changed()

    def _build_character_group(self):
        group = QGroupBox("Postać - Druid")
        grid = QGridLayout(group)
        self.level_input = self._spin(8, 2000, 100)
        self.magic_input = self._spin(0, 150, 60)
        self.armor_input = self._spin(0, 200, 30)
        self.potions_input = self._spin(0, 5000, 100)
        self.max_steps_input = self._spin(50, 100_000, 600)
        for spin in (self.level_input, self.magic_input):
            spin.valueChanged.connect(self._update_preview)
        self.utamo_checkbox = QCheckBox("Utamo Vita (auto)")
        self.utamo_checkbox.setChecked(True)
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #9ccc65;")
        self.preview_label.setWordWrap(True)
        fields = [("Poziom:", self.level_input), ("Poziom magii:", self.magic_input),
                  ("Pancerz (eq):", self.armor_input), ("Great Mana Potion:", self.potions_input),
                  ("Limit decyzji:", self.max_steps_input), ("", self.utamo_checkbox)]
        for i, (text, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            grid.addWidget(QLabel(text), row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
        grid.addWidget(self.preview_label, 3, 0, 1, 4)
        return group

    def _build_editor_group(self):
        group = QGroupBox("Mapa, kolizje i potwory")
        layout = QVBoxLayout(group)

        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        for i, (tool, text) in enumerate(((TOOL_WAYPOINT, "Punkty trasy"),
                                          (TOOL_MONSTER, "Spawn potwora"),
                                          (TOOL_OBSTACLE, "Przeszkody"))):
            radio = QRadioButton(text)
            radio.setProperty("tool", tool)
            radio.setChecked(i == 0)
            self.tool_group.addButton(radio)
            tools.addWidget(radio)
        self.tool_group.buttonClicked.connect(lambda b, *_: setattr(self.canvas, "tool", b.property("tool")))
        layout.addLayout(tools)

        grid = QGridLayout()
        self.monster_combo = QComboBox()
        self.monster_combo.currentIndexChanged.connect(
            lambda *_: setattr(self.canvas, "monster_type", self.monster_combo.currentData()))
        self.random_monsters_input = self._spin(0, 50, 3)
        self.random_monsters_input.setToolTip("Używane tylko, gdy na mapie nie ma spawnów")
        self.floor_input = self._spin(0, 15, GROUND_FLOOR)
        self.floor_input.setToolTip("Piętro Z (Tibia: 7 = poziom gruntu)")
        self.floor_input.valueChanged.connect(lambda z: self.canvas.set_floor(z))
        self.cell_px_input = self._spin(1, 256, CELL)
        self.cell_px_input.setToolTip("Ile pikseli PNG przypada na jedno pole (1 dla minimapy Tibii)")
        self.tolerance_input = self._spin(0, 255, 40)
        self.tolerance_input.setToolTip("Maks. różnica na kanał RGB od #000000 / #0000FF")
        self.fraction_input = self._spin(1, 100, 50)
        self.fraction_input.setSuffix(" %")
        self.diagonal_input = QDoubleSpinBox()
        self.diagonal_input.setRange(1.0, 3.0)
        self.diagonal_input.setSingleStep(0.1)
        self.diagonal_input.setValue(DEFAULT_DIAGONAL_MULTIPLIER)
        self.diagonal_input.valueChanged.connect(self._update_preview)
        self.ground_speed_input = self._spin(50, 500, DEFAULT_GROUND_SPEED)
        self.ground_speed_input.setToolTip("Prędkość podłoża (trawa ≈ 150)")
        self.ground_speed_input.valueChanged.connect(self._update_preview)
        fields = [("Potwór:", self.monster_combo), ("Losowe potwory:", self.random_monsters_input),
                  ("Piętro (Z):", self.floor_input), ("Piksele na pole:", self.cell_px_input),
                  ("Tolerancja koloru:", self.tolerance_input), ("Próg zajętości:", self.fraction_input),
                  ("Mnożnik ukosu:", self.diagonal_input), ("Prędkość podłoża:", self.ground_speed_input)]
        for i, (text, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            grid.addWidget(QLabel(text), row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
        layout.addLayout(grid)

        buttons = QGridLayout()
        self.load_map_button = QPushButton("Wczytaj mapę PNG…")
        self.load_map_button.clicked.connect(self.load_map_dialog)
        self.reparse_button = QPushButton("Parsuj kolizje ponownie")
        self.reparse_button.clicked.connect(self.reparse_map)
        self.clear_map_button = QPushButton("Wyczyść piętro")
        self.clear_map_button.clicked.connect(lambda *_: self.canvas.clear_floor())
        self.load_db_button = QPushButton("Baza potworów (JSON)…")
        self.load_db_button.clicked.connect(self.load_monster_db)
        self.save_scenario_button = QPushButton("Zapisz scenariusz")
        self.save_scenario_button.clicked.connect(self.save_scenario)
        self.load_scenario_button = QPushButton("Wczytaj scenariusz")
        self.load_scenario_button.clicked.connect(self.load_scenario)
        for i, button in enumerate((self.load_map_button, self.reparse_button, self.clear_map_button,
                                    self.load_db_button, self.save_scenario_button, self.load_scenario_button)):
            buttons.addWidget(button, i // 2, i % 2)
        layout.addLayout(buttons)
        return group

    def _build_mode_group(self):
        group = QGroupBox("Tryb pracy")
        layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self.manual_radio = QRadioButton("Tryb testu ręcznego")
        self.training_radio = QRadioButton("Tryb treningu (headless)")
        self.manual_radio.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.manual_radio)
        self.mode_group.addButton(self.training_radio)
        self.mode_group.buttonClicked.connect(lambda *_: self._on_mode_changed())
        row.addWidget(self.manual_radio)
        row.addWidget(self.training_radio)
        layout.addLayout(row)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_manual_page())
        self.mode_stack.addWidget(self._build_training_page())
        layout.addWidget(self.mode_stack)
        return group

    def _build_manual_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.agent_control_checkbox = QCheckBox("Steruje agent (polityka)")
        self.speed_combo = QComboBox()
        for label, factor in (("x0.5", 0.5), ("x1 (czas rzeczywisty)", 1.0), ("x2", 2.0), ("x4", 4.0), ("x10", 10.0)):
            self.speed_combo.addItem(label, factor)
        self.speed_combo.setCurrentIndex(1)
        row.addWidget(self.agent_control_checkbox)
        row.addWidget(QLabel("Prędkość:"))
        row.addWidget(self.speed_combo)
        layout.addLayout(row)
        keys = QLabel("Klawisze: W/A/S/D lub strzałki - ruch, Q/E/Z/C - ukosy, 1 - Exori Frigo, "
                      "2 - Exori Tera, 3 - Exura Gran, 4 - mikstura many")
        keys.setObjectName("hint")
        keys.setWordWrap(True)
        layout.addWidget(keys)

        buttons = QHBoxLayout()
        self.manual_start_button = QPushButton("▶ Start")
        self.manual_start_button.setObjectName("primary")
        self.manual_start_button.clicked.connect(self.toggle_manual)
        self.manual_reset_button = QPushButton("⟲ Reset epizodu")
        self.manual_reset_button.setObjectName("danger")
        self.manual_reset_button.clicked.connect(lambda *_: self.reset_manual())
        buttons.addWidget(self.manual_start_button)
        buttons.addWidget(self.manual_reset_button)
        layout.addLayout(buttons)

        bars = QFormLayout()
        self.hp_bar = QProgressBar()
        self.hp_bar.setObjectName("hpBar")
        self.mana_bar = QProgressBar()
        self.mana_bar.setObjectName("manaBar")
        bars.addRow("HP:", self.hp_bar)
        bars.addRow("Mana:", self.mana_bar)
        layout.addLayout(bars)

        self.manual_values = self._value_grid(layout, [
            ("time", "Czas gry:"), ("reward", "Nagroda epizodu:"),
            ("kills", "Zabite:"), ("last_reward", "Ostatnia nagroda:"),
            ("utamo", "Utamo Vita:"), ("action", "Ostatnia akcja:"),
        ])
        return page

    def _build_training_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        form = QFormLayout()
        self.episodes_input = self._spin(1, 1_000_000, 2000)
        form.addRow("Liczba epizodów:", self.episodes_input)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.train_start_button = QPushButton("▶ Rozpocznij trening")
        self.train_start_button.setObjectName("primary")
        self.train_start_button.clicked.connect(self.start_training)
        self.train_stop_button = QPushButton("■ Zatrzymaj")
        self.train_stop_button.setObjectName("danger")
        self.train_stop_button.clicked.connect(self.stop_training)
        row.addWidget(self.train_start_button)
        row.addWidget(self.train_stop_button)
        layout.addLayout(row)

        self.train_progress = QProgressBar()
        self.train_progress.setObjectName("trainBar")
        layout.addWidget(self.train_progress)
        self.train_values = self._value_grid(layout, [
            ("episode", "Epizod:"), ("avg_reward", "Śr. nagroda (100):"),
            ("avg_kills", "Śr. zabicia (100):"), ("death_rate", "Zgony (100):"),
            ("epsilon", "Epsilon:"), ("speed", "Decyzje/s:"),
            ("loss", "Strata (Huber):"), ("steps", "Decyzje łącznie:"),
        ])
        self.reward_plot = RewardPlot()
        layout.addWidget(self.reward_plot)
        return page

    def _build_model_group(self):
        group = QGroupBox("Model / polityka")
        layout = QVBoxLayout(group)
        self.model_label = QLabel()
        self.model_label.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(self.model_label)
        row = QHBoxLayout()
        self.export_button = QPushButton("⇩ Eksportuj politykę / model")
        self.export_button.setObjectName("export")
        self.export_button.clicked.connect(self.export_model)
        self.load_model_button = QPushButton("Wczytaj model")
        self.load_model_button.clicked.connect(self.load_model)
        self.new_model_button = QPushButton("Nowy model")
        self.new_model_button.clicked.connect(self.new_model)
        row.addWidget(self.export_button, 2)
        row.addWidget(self.load_model_button, 1)
        row.addWidget(self.new_model_button, 1)
        layout.addLayout(row)
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

    @staticmethod
    def _spin(low, high, value):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(value)
        return spin

    @staticmethod
    def _value_grid(parent_layout, fields):
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        values = {}
        for i, (key, text) in enumerate(fields):
            name = QLabel(text)
            name.setStyleSheet("color: #9aa0a6;")
            value = QLabel("-")
            value.setStyleSheet("color: #ffffff; font-weight: bold;")
            row, col = divmod(i, 2)
            grid.addWidget(name, row, col * 2)
            grid.addWidget(value, row, col * 2 + 1)
            values[key] = value
        parent_layout.addLayout(grid)
        return values

    def _bind_keys(self):
        """Keyboard control via an app-wide event filter, active only while a manual episode runs."""
        self.key_actions = {
            Qt.Key_W: ACTION_UP, Qt.Key_Up: ACTION_UP,
            Qt.Key_S: ACTION_DOWN, Qt.Key_Down: ACTION_DOWN,
            Qt.Key_A: ACTION_LEFT, Qt.Key_Left: ACTION_LEFT,
            Qt.Key_D: ACTION_RIGHT, Qt.Key_Right: ACTION_RIGHT,
            Qt.Key_Q: ACTION_UP_LEFT, Qt.Key_E: ACTION_UP_RIGHT,
            Qt.Key_Z: ACTION_DOWN_LEFT, Qt.Key_C: ACTION_DOWN_RIGHT,
            Qt.Key_1: ACTION_EXORI_FRIGO, Qt.Key_2: ACTION_EXORI_TERA,
            Qt.Key_3: ACTION_HEAL, Qt.Key_4: ACTION_MANA_POTION,
        }
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress and self.manual_running
                and not self.agent_control_checkbox.isChecked()
                and event.key() in self.key_actions):
            self.pending_action = self.key_actions[event.key()]
            return True
        return super().eventFilter(obj, event)

    def _populate_monsters(self):
        current = self.monster_combo.currentData()
        self.monster_combo.blockSignals(True)
        self.monster_combo.clear()
        for name in self.monster_db.names():
            icon = self.sprite_store.pixmap(name, 24)
            self.monster_combo.addItem(QIcon(icon) if icon else QIcon(), name, name)
            self.monster_combo.setItemData(self.monster_combo.count() - 1,
                                           self.monster_db.get(name).describe_pl(), Qt.ToolTipRole)
        index = self.monster_combo.findData(current)
        if index < 0:
            index = max(0, self.monster_combo.findData("Giant Spider"))
        self.monster_combo.setCurrentIndex(index)
        self.monster_combo.blockSignals(False)
        self.canvas.monster_type = self.monster_combo.currentData()

    # ----- state ----------------------------------------------------------- #

    @property
    def training(self):
        return self.worker is not None and self.worker.isRunning()

    def _refresh_controls(self):
        training = self.training
        busy = training or self.manual_running
        manual_mode = self.manual_radio.isChecked()
        editable = not busy and self.env is None

        self.canvas.editing_enabled = editable
        for widget in (self.level_input, self.magic_input, self.armor_input, self.potions_input,
                       self.max_steps_input, self.utamo_checkbox, self.random_monsters_input,
                       self.cell_px_input, self.tolerance_input, self.fraction_input,
                       self.diagonal_input, self.ground_speed_input,
                       self.load_map_button, self.reparse_button, self.clear_map_button,
                       self.load_db_button, self.load_scenario_button):
            widget.setEnabled(editable)
        self.floor_input.setEnabled(not training)
        self.manual_radio.setEnabled(not busy)
        self.training_radio.setEnabled(not busy)

        self.manual_start_button.setEnabled(manual_mode and not training)
        self.manual_start_button.setText("⏸ Pauza" if self.manual_running else "▶ Start")
        self.manual_reset_button.setEnabled(manual_mode and self.env is not None)
        self.agent_control_checkbox.setEnabled(not training)

        self.train_start_button.setEnabled(not busy and not manual_mode and self.env is None)
        self.train_stop_button.setEnabled(training)
        self.export_button.setEnabled(not training)
        self.load_model_button.setEnabled(not busy)
        self.new_model_button.setEnabled(not busy)

    def _on_mode_changed(self):
        self.mode_stack.setCurrentIndex(0 if self.manual_radio.isChecked() else 1)
        if self.training_radio.isChecked() and self.env is not None:
            self.reset_manual(log=False)
        self._refresh_controls()

    def build_config(self):
        canvas = self.canvas
        return EnvConfig(
            cols=canvas.cols,
            rows=canvas.rows,
            level=self.level_input.value(),
            magic_level=self.magic_input.value(),
            armor=self.armor_input.value(),
            use_utamo=self.utamo_checkbox.isChecked(),
            mana_potions=self.potions_input.value(),
            collision=canvas.collision_rows(),
            waypoints=[list(w) for w in canvas.waypoints],
            spawns=[[x, y, z, name] for (x, y, z), name in sorted(canvas.spawns.items())],
            floor=canvas.floor,
            random_monster_count=self.random_monsters_input.value(),
            random_monster_type=self.monster_combo.currentData(),
            max_steps=self.max_steps_input.value(),
            ground_speed=self.ground_speed_input.value(),
            diagonal_multiplier=self.diagonal_input.value(),
            monsters_file=self.monsters_file,
        )

    # ----- manual / visual mode -------------------------------------------- #

    def toggle_manual(self):
        if self.manual_running:
            self.visual_timer.stop()
            self.manual_running = False
            self.log("system", "Wstrzymano.")
        else:
            if self.env is None and not self._new_manual_episode():
                return
            self.manual_running = True
            self.visual_timer.start()
            who = "agent" if self.agent_control_checkbox.isChecked() else "gracz (klawiatura)"
            self.log("system", f"Symulacja w czasie rzeczywistym ({self.speed_combo.currentText()}), "
                               f"steruje: {who}.")
        self._refresh_controls()

    def _new_manual_episode(self):
        config = self.build_config()
        if not self.agent_control_checkbox.isChecked():
            # Keyboard play: idle 100 ms waits are decisions too, so the training limit would end it early
            config.max_steps = 10 ** 9
        try:
            self.env = DruidHuntEnv(config, record_events=True, monster_db=self.monster_db)
            self.obs, info = self.env.reset()
        except ValueError as exc:
            self.env = None
            QMessageBox.warning(self, "Błąd scenariusza", str(exc))
            return False
        druid = self.env.druid
        self.visual_clock_ms = 0.0
        if druid.z != self.canvas.floor:
            self.floor_input.setValue(druid.z)
        self.log("system", "=" * 44)
        self.log("system", f"Nowy epizod: Druid lvl {druid.level}, ML {druid.magic_level}, pancerz {druid.armor}, "
                           f"HP {druid.max_hp}, Mana {druid.max_mana}, potwory: {len(self.env.monsters)}.")
        self._log_env_events()
        self.canvas.sync_entities(self.env)
        self._update_manual_status(info, 0.0, None)
        return True

    def reset_manual(self, log=True):
        self.visual_timer.stop()
        self.manual_running = False
        self.env = None
        self.obs = None
        self.pending_action = None
        self.canvas.clear_entities()
        if log:
            self.log("system", "Epizod zresetowany.")
        self._update_preview()
        self._refresh_controls()

    def on_visual_tick(self):
        """Advance real time; run decisions until the simulation clock catches up."""
        if self.env is None:
            return
        self.visual_clock_ms += VISUAL_TICK_MS * self.speed_combo.currentData()
        for _ in range(MAX_DECISIONS_PER_FRAME):
            env = self.env
            if env is None or env.time_ms >= self.visual_clock_ms:
                break
            if self.agent_control_checkbox.isChecked():
                action = self.agent.act(self.obs, greedy=True)
            else:
                action, self.pending_action = self.pending_action, None
                action = ACTION_WAIT if action is None else action
            self.obs, reward, terminated, truncated, info = env.step(action)
            self._log_env_events()
            self._update_manual_status(info, reward, action)
            if terminated or truncated:
                self._end_manual_episode(info)
                break
        if self.env is not None:
            self.canvas.sync_entities(self.env)

    def _end_manual_episode(self, info):
        self.canvas.sync_entities(self.env)
        self.log("system", f"Koniec epizodu: nagroda {info['episode_reward']:.1f}, zabite {info['kills']}, "
                           f"decyzje {info['steps']}, czas {format_game_time(info['time_ms'])}.")
        if self.agent_control_checkbox.isChecked():
            self._new_manual_episode()
        else:
            self.visual_timer.stop()
            self.manual_running = False
            self.env = None
            self._refresh_controls()

    def _update_manual_status(self, info, reward, action):
        env = self.env
        druid = env.druid
        self._set_bar(self.hp_bar, druid.hp, druid.max_hp)
        self._set_bar(self.mana_bar, druid.mana, druid.max_mana)
        v = self.manual_values
        v["time"].setText(format_game_time(info["time_ms"]))
        v["reward"].setText(f"{info['episode_reward']:.1f}")
        v["kills"].setText(f"{info['kills']} / {len(env.monsters)}")
        v["last_reward"].setText(f"{reward:+.1f}")
        v["utamo"].setText(f"Aktywne ({druid.utamo_remaining_ms(env.time_ms) / 1000:.0f} s)"
                           if druid.utamo_active else "Nieaktywne")
        if action is not None and action != ACTION_WAIT:
            v["action"].setText(ACTION_LABELS_PL[action])

    def _log_env_events(self):
        if self.env is None:
            return
        for kind, message in self.env.events:
            self.log(kind, message, time_ms=self.env.time_ms)

    # ----- training mode --------------------------------------------------- #

    def start_training(self):
        config = self.build_config()
        try:
            DruidHuntEnv(config, monster_db=self.monster_db).reset()
        except ValueError as exc:
            QMessageBox.warning(self, "Błąd scenariusza", str(exc))
            return
        episodes = self.episodes_input.value()
        self.reward_plot.clear()
        self.train_progress.setRange(0, episodes)
        self.train_progress.setValue(0)
        self.worker = TrainingWorker(config, self.agent, episodes, self)
        self.worker.progress.connect(self._on_training_progress)
        self.worker.failed.connect(lambda msg: self.log("death", f"Błąd treningu: {msg}"))
        self.worker.finished.connect(self._on_training_finished)
        blocked = sum(row.count("1") for rows in config.collision.values() for row in rows)
        self.log("train", f"Start treningu: {episodes} epizodów, siatka {config.cols}x{config.rows}, "
                          f"zablokowane pola {blocked}, spawny {len(config.spawns) or 'losowe'}, "
                          f"ukos x{config.diagonal_multiplier:.2f}.")
        self.worker.start()
        self._refresh_controls()

    def stop_training(self):
        if self.training:
            self.worker.stop()
            self.log("train", "Zatrzymywanie treningu po bieżącym epizodzie…")

    def _on_training_progress(self, stats):
        self.train_progress.setValue(stats["episode"])
        self.train_progress.setFormat(f"{stats['episode']} / {stats['episodes_total']}")
        v = self.train_values
        v["episode"].setText(str(stats["episode"]))
        v["avg_reward"].setText(f"{stats['avg_reward']:.1f}")
        v["avg_kills"].setText(f"{stats['avg_kills']:.2f}")
        v["death_rate"].setText(f"{stats['death_rate'] * 100:.0f}%")
        v["epsilon"].setText(f"{stats['epsilon']:.3f}")
        v["speed"].setText(f"{stats['steps_per_second']:.0f}")
        v["loss"].setText(f"{stats['loss']:.4f}")
        v["steps"].setText(f"{stats['total_steps']:,}".replace(",", " "))
        self.reward_plot.add(stats["new_rewards"])
        self._update_model_label()

    def _on_training_finished(self):
        self.log("train", f"Trening zakończony. Epizody łącznie: {self.agent.episodes_trained}, "
                          f"decyzje: {self.agent.total_steps}.")
        self.worker = None
        self._update_model_label()
        self._refresh_controls()

    # ----- model ----------------------------------------------------------- #

    def export_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Eksportuj politykę",
                                              os.path.join(PROJECT_DIR, "druid_policy.npz"),
                                              "Polityka numpy (*.npz)")
        if not path:
            return
        written = self.agent.export_policy(path, {"env_config": self.build_config().to_dict()})
        self.log("system", "Wyeksportowano model: " + ", ".join(os.path.basename(p) for p in written))
        if not any(p.endswith(".pt") for p in written):
            self.log("info", "PyTorch nie jest zainstalowany - zapisano tylko format .npz.")

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj model", PROJECT_DIR, "Polityka numpy (*.npz)")
        if not path:
            return
        try:
            metadata = self.agent.load_policy(path)
        except (ValueError, KeyError, OSError) as exc:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać modelu:\n{exc}")
            return
        self.log("system", f"Wczytano model {os.path.basename(path)} (utworzony {metadata.get('created')}).")
        self._update_model_label()

    def new_model(self):
        self.agent = DQNAgent()
        self.reward_plot.clear()
        self.log("system", "Utworzono nowy, niewytrenowany model.")
        self._update_model_label()

    def _update_model_label(self):
        sizes = " → ".join(str(s) for s in self.agent.online.sizes)
        self.model_label.setText(
            f"Double DQN (numpy), warstwy: {sizes}\n"
            f"Epizody: {self.agent.episodes_trained}   |   decyzje: {self.agent.total_steps}   |   "
            f"epsilon: {self.agent.epsilon:.3f}")

    # ----- map, database, scenario files ----------------------------------- #

    def _parser(self):
        tolerance = self.tolerance_input.value()
        rules = (ColorRule("Pustka / ściana (#000000)", (0, 0, 0), tolerance),
                 ColorRule("Woda (#0000FF)", (0, 0, 255), tolerance))
        return MapParser(rules, cell_px=self.cell_px_input.value(),
                         blocked_fraction=self.fraction_input.value() / 100.0)

    def load_map_file(self, path):
        try:
            parsed = self.canvas.load_map(path, self.cell_px_input.value(), self._parser())
        except (OSError, ValueError) as exc:
            self.log("warn", f"Nie udało się sparsować mapy: {exc}")
            return
        if parsed is None:
            self.log("warn", f"Nie udało się wczytać mapy: {os.path.basename(path)}")
            return
        self.log("system", f"Wczytano mapę {os.path.basename(path)}: siatka {self.canvas.cols}x{self.canvas.rows}, "
                           f"zablokowane pola: {int(parsed.sum())} ({parsed.mean() * 100:.1f}%), "
                           f"piętro {self.canvas.floor}.")

    def load_map_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj mapę", PROJECT_DIR,
                                              "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_map_file(path)

    def reparse_map(self):
        if not self.canvas.map_path:
            self.log("warn", "Brak wczytanej mapy do sparsowania.")
            return
        self.load_map_file(self.canvas.map_path)

    def load_monster_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "Baza potworów", os.path.dirname(self.monster_db.source),
                                              "JSON (*.json)")
        if not path:
            return
        try:
            self.monster_db = MonsterDatabase.load(path)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Błąd", f"Niepoprawny plik potworów:\n{exc}")
            return
        self.monsters_file = path
        self._populate_monsters()
        self.sync_sprites()
        self.log("system", f"Wczytano bazę potworów: {os.path.basename(path)} ({len(self.monster_db)} pozycji).")

    def save_scenario(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz scenariusz",
                                              os.path.join(PROJECT_DIR, "scenario.json"), "JSON (*.json)")
        if not path:
            return
        data = self.build_config().to_dict()
        data["map_image"] = self.canvas.map_path
        data["map_cell_px"] = self.canvas.map_cell_px
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
        self.log("system", f"Zapisano scenariusz: {os.path.basename(path)}")

    def load_scenario(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj scenariusz", PROJECT_DIR, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            config = EnvConfig.from_dict(data)
            db = MonsterDatabase.load(config.monsters_file) if config.monsters_file else self.monster_db
        except (OSError, ValueError, TypeError, KeyError) as exc:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać scenariusza:\n{exc}")
            return
        self.monster_db, self.monsters_file = db, config.monsters_file
        self._populate_monsters()
        self.sync_sprites()
        map_image = data.get("map_image")
        if map_image and os.path.isfile(map_image):
            self.cell_px_input.setValue(int(data.get("map_cell_px", CELL)))
            self.canvas.load_map(map_image, self.cell_px_input.value())
        self.canvas.apply_config(config)
        self.floor_input.setValue(config.floor)
        self.level_input.setValue(config.level)
        self.magic_input.setValue(config.magic_level)
        self.armor_input.setValue(config.armor)
        self.potions_input.setValue(config.mana_potions)
        self.max_steps_input.setValue(config.max_steps)
        self.utamo_checkbox.setChecked(config.use_utamo)
        self.random_monsters_input.setValue(config.random_monster_count)
        self.diagonal_input.setValue(config.diagonal_multiplier)
        self.ground_speed_input.setValue(config.ground_speed)
        index = self.monster_combo.findData(config.random_monster_type)
        if index >= 0:
            self.monster_combo.setCurrentIndex(index)
        self.log("system", f"Wczytano scenariusz: {os.path.basename(path)}")

    # ----- sprites --------------------------------------------------------- #

    def sync_sprites(self):
        """Download missing sprites for the druid and every monster in the database (background thread)."""
        if self.asset_worker is not None and self.asset_worker.isRunning():
            self._sprite_resync = True
            return
        entries = sprite_entries(self.monster_db)
        if all(self.asset_manager.local_path(name) for name, _ in entries):
            self.log("system", f"Sprite'y: komplet {len(entries)} na dysku ({os.path.basename(SPRITES_DIR)}/).")
            return
        self.log("system", "Sprite'y: pobieram brakujące obrazki z TibiaWiki w tle…")
        self.asset_worker = AssetWorker(entries, self)
        self.asset_worker.message.connect(self._on_asset_message)
        self.asset_worker.sprite_ready.connect(self._on_sprite_ready)
        self.asset_worker.summary.connect(self._on_assets_done)
        self.asset_worker.start()

    def _on_asset_message(self, text):
        kind = "warn" if text.startswith("[WARN]") else "info"
        if kind == "warn" or "OK!" in text:
            self.log(kind, text.split("] ", 1)[-1])

    def _on_sprite_ready(self, name):
        self.sprite_store.invalidate(name)
        self.canvas.refresh_sprites()
        index = self.monster_combo.findData(name)
        if index >= 0:
            icon = self.sprite_store.pixmap(name, 24)
            self.monster_combo.setItemIcon(index, QIcon(icon) if icon else QIcon())

    def _on_assets_done(self, counts):
        labels = {"downloaded": "pobrane", "cached": "z dysku", "failed": "nieudane", "skipped": "pominięte"}
        summary = ", ".join(f"{labels.get(k, k)}: {v}" for k, v in counts.items())
        self.log("system", f"Sprite'y gotowe ({summary}). Brakujące postacie rysuję jako kółka.")
        if self._sprite_resync:
            self._sprite_resync = False
            QTimer.singleShot(0, self.sync_sprites)

    # ----- misc ------------------------------------------------------------ #

    def _on_scenario_changed(self):
        c = self.canvas
        self.scenario_label.setText(f"Siatka {c.cols}x{c.rows}   |   piętro {c.floor}   |   "
                                    f"trasa: {len(c.waypoints)}   |   spawny: {len(c.spawns)}   |   "
                                    f"kolizje (piętro): {int(c.grid().sum())}")

    def _update_preview(self):
        druid = Druid(self.level_input.value(), self.magic_input.value())
        ground = self.ground_speed_input.value()
        multiplier = self.diagonal_input.value()
        straight = step_duration_ms(druid.speed, False, ground, multiplier)
        diagonal = step_duration_ms(druid.speed, True, ground, multiplier)
        self.preview_label.setText(f"Maks. HP: {druid.max_hp}  |  Maks. mana: {druid.max_mana}  |  "
                                   f"Prędkość: {druid.speed}  |  Krok: {straight} ms, ukos: {diagonal} ms")
        if self.env is None:
            self._set_bar(self.hp_bar, druid.max_hp, druid.max_hp)
            self._set_bar(self.mana_bar, druid.max_mana, druid.max_mana)

    @staticmethod
    def _set_bar(bar, value, maximum):
        bar.setMaximum(max(1, maximum))
        bar.setValue(int(value))
        bar.setFormat(f"{int(value)} / {maximum}")

    def log(self, kind, message, time_ms=None):
        color = LOG_COLORS.get(kind, LOG_COLORS["info"])
        weight = "bold" if kind in ("death", "kill") else "normal"
        if time_ms is None:
            time_ms = self.env.time_ms if self.env is not None else 0
        self.event_log.append(f'<span style="color:#6d6d6d;">[{format_game_time(time_ms)}]</span> '
                              f'<span style="color:{color}; font-weight:{weight};">{html.escape(message)}</span>')
        bar = self.event_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def closeEvent(self, event):
        if self.training:
            self.worker.stop()
            self.worker.wait(5000)
        if self.asset_worker is not None and self.asset_worker.isRunning():
            self.asset_worker.stop()
            self.asset_worker.wait(15000)
        super().closeEvent(event)


def apply_dark_palette(app):
    app.setStyle("Fusion")
    palette = QPalette()
    for role, color in ((QPalette.Window, "#1e1f22"), (QPalette.WindowText, "#dcdcdc"),
                        (QPalette.Base, "#2b2d31"), (QPalette.Text, "#dcdcdc"),
                        (QPalette.Button, "#2f3237"), (QPalette.ButtonText, "#ffffff"),
                        (QPalette.Highlight, "#2979ff"), (QPalette.HighlightedText, "#ffffff")):
        palette.setColor(role, QColor(color))
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
