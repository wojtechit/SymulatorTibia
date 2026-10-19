"""
PyQt5 digital twin: map editor (collision grid per floor, PNG parser, spawns,
waypoints), real-time manual / agent visualizer and background DQN training.
"""

import html
import json
import os
import sys

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                         QPolygonF)
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
from .combat import ELEMENT_LABELS_PL, SPELLBOOK
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
    ACTION_USE_AVALANCHE,
    ACTION_USE_SD,
    ACTION_WAIT,
    STRIKE_RANGE,
    DruidHuntEnv,
    EnvConfig,
    grid_to_rows,
)
from .mapparser import DEFAULT_WALL_COLORS, MapParser, color_label, load_rgba, mask_to_cells, palette
from .terrain import encode_speed, load_client_minimap, stairs_mask
from .items import ItemDatabase
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

ELEMENT_COLORS = {
    "fire": (255, 120, 0), "ice": (80, 190, 255), "earth": (90, 200, 60), "energy": (190, 90, 255),
    "death": (120, 120, 140), "holy": (255, 230, 120), "physical": (220, 220, 220),
}

TOOL_WAYPOINT = "waypoint"
TOOL_MONSTER = "monster"
TOOL_OBSTACLE = "obstacle"
TOOL_MOVE = "move"
MARKER_PICK_PX = 14   # click radius (screen pixels) for picking spawn / waypoint markers


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

    def __init__(self, entries, parent=None, force=False):
        super().__init__(parent)
        self.entries = entries
        self.force = force
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
            for result in manager.ensure([entry], force=self.force):
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
    move_rejected = pyqtSignal()
    cell_hovered = pyqtSignal(int, int)

    MIN_ZOOM = 0.02
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
        self.floor_maps = {}         # z -> {"path", "cell_px", "offset", "client"} background per floor
        self.speeds = {}             # z -> uint16 ground speed per tile (client minimap import)
        self.stairs = {}             # z -> bool floor-change tiles (yellow on the minimap)
        self.transitions = []        # [[x, y, z, x2, y2, z2], ...] linked stairs / holes / ladders
        self.replaced_floors = []    # floors dropped by the last load_map (different map size)
        self.sprites = None          # SpriteStore, set by the main window

        self.collision = {}          # z -> np.ndarray (rows, cols) uint8
        self.waypoints = []          # [(x, y, z)]
        self.spawns = {}             # (x, y, z) -> monster name
        self._floor_items = {}       # z -> QGraphicsPixmapItem
        self._transition_items = []
        self._overlay_item = None
        self._undo, self._redo = [], []
        self.floor_label = QLabel(self)
        self.floor_label.setStyleSheet("background: rgba(0,0,0,170); color: #ffffff; padding: 3px 8px; "
                                       "border-radius: 4px; font-weight: bold;")
        self._overlay_buffer = None
        self._waypoint_items = []
        self._spawn_items = {}
        self._zoom = 1.0
        self._panning = False
        self._pan_last = QPoint()
        self._paint_button = None
        self._drag_marker = None
        self._druid_item = None
        self._range_item = None
        self._monster_items = []
        self._overlay_items = []     # AoE flashes and danger map, rebuilt every sync
        self.show_danger = False

        self._scene.setSceneRect(0, 0, self.cols * CELL, self.rows * CELL)
        self._render_overlay()
        self._update_floor_label()

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
        for layers, fill, dtype in ((self.speeds, DEFAULT_GROUND_SPEED, np.uint16), (self.stairs, False, bool)):
            for z, layer in list(layers.items()):
                resized = np.full((rows, cols), fill, dtype=dtype)
                h, w = min(rows, layer.shape[0]), min(cols, layer.shape[1])
                resized[:h, :w] = layer[:h, :w]
                layers[z] = resized
        self.cols, self.rows = cols, rows
        self._scene.setSceneRect(0, 0, cols * CELL, rows * CELL)
        self.transitions = [t for t in self.transitions if self._in_bounds(t[0], t[1]) and self._in_bounds(t[3], t[4])]
        self.waypoints = [w for w in self.waypoints if self._in_bounds(w[0], w[1])]
        for key in [k for k in self.spawns if not self._in_bounds(k[0], k[1])]:
            del self.spawns[key]
        self.refresh()

    # Current floor's background (None when this floor has no map)
    @property
    def map_path(self):
        return self.floor_maps.get(self.floor, {}).get("path")

    @property
    def map_floor(self):
        return self.floor if self.floor in self.floor_maps else None

    @property
    def map_cell_px(self):
        return self.floor_maps.get(self.floor, {}).get("cell_px", CELL)

    @property
    def map_offset(self):
        return tuple(self.floor_maps.get(self.floor, {}).get("offset", (0, 0)))

    def set_floor(self, z):
        self.floor = int(z)
        for floor, item in self._floor_items.items():
            item.setVisible(floor == self.floor)
        self._update_floor_label()
        self.refresh()

    def _update_floor_label(self):
        """Tibia floors: 7 = ground, 0-6 above, 8-15 = -1..-8 underground."""
        level = GROUND_FLOOR - self.floor
        where = "parter" if level == 0 else ("nad ziemią" if level > 0 else "pod ziemią")
        loaded = " • mapa" if self.floor in self.floor_maps else ""
        self.floor_label.setText(f"piętro {level:+d} (z={self.floor}, {where}){loaded}   PgUp/PgDn")
        self.floor_label.adjustSize()
        self.floor_label.move(8, self.height() - self.floor_label.height() - 8)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_floor_label()

    # ----- undo / redo (Ctrl+Z / Ctrl+Y) ----------------------------------- #

    def _snapshot(self):
        return ({z: g.copy() for z, g in self.collision.items()}, list(self.waypoints), dict(self.spawns),
                {z: m.copy() for z, m in self.stairs.items()}, [list(t) for t in self.transitions])

    def _restore(self, state):
        self.collision, self.waypoints, self.spawns, self.stairs, self.transitions = state
        self.refresh()

    def push_undo(self):
        self._undo.append(self._snapshot())
        del self._undo[:-50]
        self._redo.clear()

    def undo(self):
        if self._undo:
            self._redo.append(self._snapshot())
            self._restore(self._undo.pop())

    def redo(self):
        if self._redo:
            self._undo.append(self._snapshot())
            self._restore(self._redo.pop())

    def refresh(self):
        self._render_overlay()
        self._redraw_spawns()
        self._redraw_waypoints()
        self._redraw_transitions()
        self.scenario_changed.emit()

    def _set_floor_item(self, z, pixmap, scale, pos):
        old = self._floor_items.pop(z, None)
        if old is not None:
            self._scene.removeItem(old)
        item = QGraphicsPixmapItem(pixmap)
        item.setZValue(-10)
        item.setTransformationMode(Qt.FastTransformation)   # crisp pixel-art minimaps
        item.setScale(scale)
        item.setPos(*pos)
        item.setVisible(z == self.floor)
        self._scene.addItem(item)
        self._floor_items[z] = item

    def clear_floor_maps(self):
        for item in self._floor_items.values():
            self._scene.removeItem(item)
        self._floor_items = {}
        self.floor_maps = {}

    def set_stairs(self, z, mask):
        """Floor-change tiles of a floor; stairs are walkable. Re-links transitions across floors."""
        self.stairs[int(z)] = mask.astype(bool)
        self.relink_transitions()

    def relink_transitions(self):
        from .terrain import TerrainLayer, link_transitions
        layers = {z: TerrainLayer(self.grid(z), stairs=mask) for z, mask in self.stairs.items()
                  if mask.shape == (self.rows, self.cols)}
        self.transitions = link_transitions(layers)
        self.refresh()

    def _redraw_transitions(self):
        for item in self._transition_items:
            self._scene.removeItem(item)
        self._transition_items = []
        for x, y, z, _x2, _y2, z2 in self.transitions:
            if z != self.floor:
                continue
            down = z2 > z
            points = [QPoint(-7, -5), QPoint(7, -5), QPoint(0, 7)] if down else [QPoint(-7, 5), QPoint(7, 5), QPoint(0, -7)]
            item = QGraphicsPolygonItem(QPolygonF([QPointF(p) for p in points]))
            item.setFlag(QGraphicsPolygonItem.ItemIgnoresTransformations)
            item.setPos(x * CELL + CELL / 2, y * CELL + CELL / 2)
            item.setBrush(QBrush(QColor("#ffeb3b")))
            item.setPen(QPen(QColor("#000000"), 1.5))
            item.setZValue(9)
            item.setToolTip(f"{'Zejście' if down else 'Wejście'} na piętro {z2} ({_x2}, {_y2})")
            self._scene.addItem(item)
            self._transition_items.append(item)

    def load_map(self, path, cell_px, grid=None, offset=(0, 0), fit=True):
        """
        Show the image as background of the current floor; `grid` (parsed walls) replaces that floor's
        collision. `offset` is the pixel where the tile grid starts (detected for upscaled minimaps).
        """
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            return None
        scale = CELL / cell_px
        offset_x, offset_y = offset
        cols = max(1, (pixmap.width() - offset_x) // cell_px)
        rows = max(1, (pixmap.height() - offset_y) // cell_px)
        others = [z for z in self.floor_maps if z != self.floor]
        self.replaced_floors = []
        if others and (cols, rows) != (self.cols, self.rows):
            # A map of a different area starts a new project: other floors cannot line up with it
            self.replaced_floors = sorted(others)
            self.clear_floor_maps()
            self.collision, self.speeds, self.stairs, self.transitions = {}, {}, {}, []
            self.waypoints, self.spawns = [], {}
        self._set_floor_item(self.floor, pixmap, scale, (-offset_x * scale, -offset_y * scale))
        self.floor_maps[self.floor] = {"path": path, "cell_px": cell_px, "offset": [offset_x, offset_y],
                                       "client": False}
        self.resize_grid(cols, rows)
        if grid is not None:
            if grid.shape != (self.rows, self.cols):
                raise ValueError(f"Parsed grid {grid.shape} does not match the map {self.rows}x{self.cols}")
            self.collision[self.floor] = grid.astype(np.uint8)
            self.refresh()
        if fit:
            self.fit_map()
        return self.collision.get(self.floor)

    def set_collision(self, z, grid):
        self.collision[int(z)] = grid.astype(np.uint8)
        self.refresh()

    def fit_map(self):
        """Zoom so the whole map fills the view."""
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._zoom = self.transform().m11()
        self._apply_marker_scale()

    def _marker_scale(self):
        """Scale for live entities so they never shrink below ~20 px on screen when zoomed out."""
        return max(1.0, 0.8 / max(self._zoom, 1e-3))

    def _apply_marker_scale(self):
        scale = self._marker_scale()
        for item in ([self._druid_item] if self._druid_item else []) + self._monster_items:
            item["body"].setScale(scale)

    def _render_overlay(self):
        grid = self.grid()
        rgba = np.zeros((self.rows, self.cols, 4), dtype=np.uint8)
        rgba[grid > 0] = (150, 40, 40, 150)
        speed = self.speeds.get(self.floor)
        if speed is not None and speed.shape == grid.shape:
            rgba[(speed > DEFAULT_GROUND_SPEED) & (grid == 0)] = (60, 120, 255, 70)   # slow ground (mud, swamp)
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
            # Fixed on-screen size (ItemIgnoresTransformations) so spawns stay visible at any zoom
            item = QGraphicsRectItem(-11, -11, 22, 22)
            item.setFlag(QGraphicsRectItem.ItemIgnoresTransformations)
            item.setPos(x * CELL + CELL / 2, y * CELL + CELL / 2)
            item.setBrush(QBrush(QColor(213, 0, 0, 150)))
            item.setPen(QPen(QColor("#ffffff"), 1.5))
            item.setZValue(6)
            pixmap = self.sprites.pixmap(name, 20) if self.sprites else None
            if pixmap is not None:
                icon = QGraphicsPixmapItem(pixmap, item)
                icon.setPos(-pixmap.width() / 2, -pixmap.height() / 2)
            else:
                label = QGraphicsSimpleTextItem(name[:2], item)
                label.setFont(QFont("Sans", 8, QFont.Bold))
                label.setBrush(QBrush(QColor("#ffffff")))
                label.setPos(-label.boundingRect().width() / 2, -label.boundingRect().height() / 2)
            item.setToolTip(f"Spawn: {name} (z={z})")
            self._scene.addItem(item)
            self._spawn_items[(x, y, z)] = item

    def _redraw_waypoints(self):
        for item in self._waypoint_items:
            self._scene.removeItem(item)
        self._waypoint_items = []
        on_floor = [(i, x * CELL + CELL / 2, y * CELL + CELL / 2)
                    for i, (x, y, z) in enumerate(self.waypoints) if z == self.floor]
        pen = QPen(QColor("#ffc400"), 3)
        pen.setCosmetic(True)
        gap_pen = QPen(QColor("#ffc400"), 2, Qt.DotLine)
        gap_pen.setCosmetic(True)
        for (ia, ax, ay), (ib, bx, by) in zip(on_floor, on_floor[1:]):
            line = QGraphicsLineItem(ax, ay, bx, by)
            line.setPen(pen if ib == ia + 1 else gap_pen)
            line.setZValue(7)
            self._scene.addItem(line)
            self._waypoint_items.append(line)
        if len(on_floor) >= 3 and len(on_floor) == len(self.waypoints):
            loop = QGraphicsLineItem(on_floor[-1][1], on_floor[-1][2], on_floor[0][1], on_floor[0][2])
            loop_pen = QPen(QColor("#ffc400"), 2, Qt.DashLine)
            loop_pen.setCosmetic(True)
            loop.setPen(loop_pen)
            loop.setZValue(7)
            self._scene.addItem(loop)
            self._waypoint_items.append(loop)
        for i, cx, cy in on_floor:
            circle = QGraphicsEllipseItem(-8, -8, 16, 16)
            circle.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations)
            circle.setPos(cx, cy)
            circle.setBrush(QBrush(QColor("#00e676" if i == 0 else "#00b0ff")))
            circle.setPen(QPen(QColor("#000000"), 2))
            circle.setZValue(8)
            label = QGraphicsSimpleTextItem(str(i + 1), circle)
            label.setFont(QFont("Sans", 9, QFont.Bold))
            label.setBrush(QBrush(QColor("#ffffff")))
            label.setPen(QPen(QColor("#000000"), 0.6))
            label.setPos(9, -20)
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

    def marker_at(self, view_pos):
        """Spawn or waypoint marker under the cursor (screen-distance hit test, works at any zoom)."""
        best, best_dist = None, MARKER_PICK_PX
        for (x, y, z) in self.spawns:
            if z == self.floor:
                center = self.mapFromScene(x * CELL + CELL / 2, y * CELL + CELL / 2)
                dist = (center - view_pos).manhattanLength()
                if dist <= best_dist:
                    best, best_dist = ("spawn", (x, y, z)), dist
        for index, (x, y, z) in enumerate(self.waypoints):
            if z == self.floor:
                center = self.mapFromScene(x * CELL + CELL / 2, y * CELL + CELL / 2)
                dist = (center - view_pos).manhattanLength()
                if dist < best_dist:
                    best, best_dist = ("waypoint", index), dist
        return best

    def delete_marker(self, marker):
        kind, key = marker
        if kind == "spawn":
            self.spawns.pop(key, None)
        else:
            del self.waypoints[key]
        self.refresh()

    def move_marker(self, marker, cell):
        """Move a spawn / waypoint to `cell` on this floor; refuses walls, map edges and occupied spawn tiles."""
        x, y = cell
        if not self._in_bounds(x, y) or self.grid()[y, x]:
            return False
        kind, key = marker
        target = (x, y, self.floor)
        if kind == "spawn":
            if target in self.spawns and target != key:
                return False
            self.spawns[target] = self.spawns.pop(key)
        else:
            self.waypoints[key] = target
        self.refresh()
        return True

    def clear_spawns(self, floor_only=True):
        self.spawns = {k: v for k, v in self.spawns.items() if floor_only and k[2] != self.floor}
        self.refresh()

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
        self.clear_entities()
        self.collision[self.floor] = np.zeros((self.rows, self.cols), dtype=np.uint8)
        self.spawns = {k: v for k, v in self.spawns.items() if k[2] != self.floor}
        self.waypoints = [w for w in self.waypoints if w[2] != self.floor]
        self.refresh()

    def collision_rows(self):
        return {str(z): grid_to_rows(g) for z, g in sorted(self.collision.items()) if g.any()}

    def apply_terrain(self, terrain):
        """Install a client-minimap TerrainMap: walls, ground speed, stairs, floor images and transitions."""
        self.clear_floor_maps()
        first = next(iter(terrain.layers.values()))
        rows, cols = first.walls.shape
        self.collision, self.speeds, self.stairs = {}, {}, {}
        self.resize_grid(cols, rows)
        for z, layer in terrain.layers.items():
            self.collision[z] = layer.walls.astype(np.uint8)
            self.speeds[z] = layer.speed
            self.stairs[z] = layer.stairs
            image = layer.image if layer.image is not None else np.dstack(
                [np.where(layer.walls, 40, 200).astype(np.uint8)] * 3 + [np.full(layer.walls.shape, 255, np.uint8)])
            buffer = np.ascontiguousarray(image)
            qimage = QImage(buffer.data, cols, rows, cols * 4, QImage.Format_RGBA8888).copy()
            self._set_floor_item(z, QPixmap.fromImage(qimage), CELL, (0, 0))
            self.floor_maps[z] = {"path": None, "cell_px": 1, "offset": [0, 0], "client": True}
        self.transitions = [list(t) for t in terrain.transitions]
        self.refresh()
        self.fit_map()

    def apply_config(self, config):
        self.cols, self.rows = config.cols, config.rows
        self._scene.setSceneRect(0, 0, self.cols * CELL, self.rows * CELL)
        self.collision = {int(z): config.collision_grid(int(z)) for z in config.collision}
        self.speeds = {int(z): config.speed_grid(int(z)) for z in config.terrain_speed}
        self.transitions = [list(t) for t in config.transitions]
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
            self._apply_marker_scale()
        if len(self._monster_items) != len(env.monsters):
            self._clear_monsters()
            self._monster_items = [self._make_entity(QColor("#d50000"), m.name, 15) for m in env.monsters]
            self._apply_marker_scale()

        druid = env.druid
        span = STRIKE_RANGE * 2 + 1
        self._range_item.setRect((druid.x - STRIKE_RANGE) * CELL, (druid.y - STRIKE_RANGE) * CELL,
                                 span * CELL, span * CELL)
        self._range_item.setVisible(druid.z == self.floor)
        self._update_entity(self._druid_item, druid.x, druid.y, druid.hp_ratio, druid.mana_ratio,
                            druid.utamo_active, druid.alive and druid.z == self.floor)
        for monster, item in zip(env.monsters, self._monster_items):
            self._update_entity(item, monster.x, monster.y, monster.hp_ratio, None, False,
                                monster.alive and monster.z == self.floor, facing=monster.facing)
        self._draw_combat_overlays(env)

    def _draw_combat_overlays(self, env):
        """Flash monster spell areas for AOE_FLASH_MS and optionally show the agent's danger map."""
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        env.aoe_flashes = [f for f in env.aoe_flashes if f[3] > env.time_ms]
        for tiles, z, element, _until in env.aoe_flashes:
            if z != self.floor:
                continue
            color = QColor(*ELEMENT_COLORS.get(element, (255, 255, 255)), 120)
            for tx, ty in tiles:
                if self._in_bounds(tx, ty):
                    self._add_overlay_rect(tx, ty, QBrush(color), QPen(Qt.NoPen), 12)
        if self.show_danger and env.druid.z == self.floor:
            danger = env.snapshot().danger
            radius = danger.shape[0] // 2
            pen = QPen(QColor(255, 82, 82, 220), 2, Qt.DashLine)
            for wy, wx in zip(*np.nonzero(danger)):
                tx, ty = env.druid.x - radius + int(wx), env.druid.y - radius + int(wy)
                self._add_overlay_rect(tx, ty, QBrush(QColor(255, 82, 82, 35)), pen, 11)

    def _add_overlay_rect(self, tx, ty, brush, pen, z):
        rect = QGraphicsRectItem(tx * CELL + 1, ty * CELL + 1, CELL - 2, CELL - 2)
        rect.setBrush(brush)
        rect.setPen(pen)
        rect.setZValue(z)
        self._scene.addItem(rect)
        self._overlay_items.append(rect)

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
        facing = None
        if not mana_bar:
            facing = QGraphicsLineItem(0, 0, 0, r + 6, body)
            facing.setPen(QPen(QColor("#ffd740"), 3, Qt.SolidLine, Qt.RoundCap))
        text = QGraphicsSimpleTextItem(label, body)
        text.setFont(QFont("Sans", 6, QFont.Bold))
        text.setBrush(QBrush(QColor("#ffffff")))
        text.setPos(-text.boundingRect().width() / 2, -r - 19)
        self._scene.addItem(body)
        item.update(hp=hp, mana=mana, shield=shield, facing=facing)
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
    def _update_entity(item, x, y, hp_ratio, mana_ratio, shield, visible, facing=None):
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
        if item.get("facing") is not None and facing is not None:
            reach = r + 6
            item["facing"].setLine(0, 0, facing[0] * reach, facing[1] * reach)

    def _clear_monsters(self):
        for item in self._monster_items:
            self._scene.removeItem(item["body"])
        self._monster_items = []

    def clear_entities(self):
        self._clear_monsters()
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
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
            self.push_undo()                          # one undo step per click / drag
            cell = self.cell_at(event.pos())
            marker = self.marker_at(event.pos())
            if button == Qt.RightButton and marker is not None:
                self.delete_marker(marker)          # RMB on a marker deletes exactly that marker
                self._paint_button = None
            elif button == Qt.LeftButton and self.tool == TOOL_MOVE:
                self._drag_marker = marker
                if marker is not None:
                    self.viewport().setCursor(Qt.ClosedHandCursor)
            elif button == Qt.LeftButton:
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
            if self.tool == TOOL_MOVE and self._paint_button == Qt.LeftButton:
                return                               # the marker is dropped on release
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
            if self._drag_marker is not None:
                if not self.move_marker(self._drag_marker, self.cell_at(event.pos())):
                    self.move_rejected.emit()
                self._drag_marker = None
                self.viewport().unsetCursor()
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
        self._apply_marker_scale()


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
        self.item_db = ItemDatabase.load()
        self.wall_colors = set(DEFAULT_WALL_COLORS)
        self._map_rgba = None
        self._client_minimap = None
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
            lambda *_: self.statusBar().showMessage(
                "Edycja mapy zablokowana - kliknij 'Reset epizodu' (albo zatrzymaj trening), aby edytować.", 5000))
        self.canvas.move_rejected.connect(
            lambda *_: self.statusBar().showMessage("Nie można tu przenieść: ściana, poza mapą albo zajęte pole.", 4000))
        self.canvas.cell_hovered.connect(
            lambda x, y: self.cell_label.setText(f"Pole: ({x}, {y}, {self.canvas.floor})"))
        left_layout.addWidget(self.canvas, 1)
        hint = QLabel("LPM: narzędzie (Przesuń: chwyć i upuść)  |  PPM na znaczniku: usuń  |  ŚPM: widok  |  "
                      "Kółko: zoom  |  PgUp/PgDn: piętro  |  Ctrl+Z / Ctrl+Y: cofnij / ponów  |  Delete: ostatni punkt")
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
        self.rod_combo = QComboBox()
        self.rod_combo.addItem("Brak różdżki", None)
        for name in self.item_db.rod_names():
            rod = self.item_db.rod(name)
            self.rod_combo.addItem(f"{rod.name} ({ELEMENT_LABELS_PL[rod.element]})", rod.name)
            self.rod_combo.setItemData(self.rod_combo.count() - 1, rod.label_pl(), Qt.ToolTipRole)
        default_rod = self.rod_combo.findData("Hailstorm Rod")
        self.rod_combo.setCurrentIndex(max(0, default_rod))
        self.rod_combo.currentIndexChanged.connect(lambda *_: self._update_preview())
        for combo in (self.rod_combo,):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
        self.sd_input = self._spin(0, 10_000, 100)
        self.sd_input.setToolTip("Liczba run Sudden Death w plecaku (0 = brak amunicji)")
        self.sd_input.valueChanged.connect(lambda *_: self._update_preview())
        self.avalanche_input = self._spin(0, 10_000, 100)
        self.avalanche_input.setToolTip("Liczba run Avalanche (lodowa kula 37 pól)")
        self.avalanche_input.valueChanged.connect(lambda *_: self._update_preview())
        self.exp_rate_input = QDoubleSpinBox()
        self.exp_rate_input.setRange(0.1, 20.0)
        self.exp_rate_input.setSingleStep(0.1)
        self.exp_rate_input.setValue(1.0)
        self.exp_rate_input.setToolTip("Mnożnik expa serwera (stage, boost, double exp)")
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #9ccc65;")
        self.preview_label.setWordWrap(True)
        fields = [("Poziom:", self.level_input), ("Poziom magii:", self.magic_input),
                  ("Pancerz (eq):", self.armor_input), ("Great Mana Potion:", self.potions_input),
                  ("Limit decyzji:", self.max_steps_input), ("", self.utamo_checkbox),
                  ("Różdżka:", self.rod_combo), ("Runy SD:", self.sd_input),
                  ("Runy Avalanche:", self.avalanche_input), ("Mnożnik expa:", self.exp_rate_input)]
        for i, (text, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            grid.addWidget(QLabel(text), row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
        grid.addWidget(self.preview_label, 5, 0, 1, 4)
        return group

    def _build_editor_group(self):
        group = QGroupBox("Mapa, kolizje i potwory")
        layout = QVBoxLayout(group)

        tools = QHBoxLayout()
        self.tool_group = QButtonGroup(self)
        for i, (tool, text) in enumerate(((TOOL_WAYPOINT, "Punkty trasy"),
                                          (TOOL_MONSTER, "Spawn potwora"),
                                          (TOOL_OBSTACLE, "Przeszkody"),
                                          (TOOL_MOVE, "Przesuń"))):
            radio = QRadioButton(text)
            radio.setProperty("tool", tool)
            radio.setChecked(i == 0)
            self.tool_group.addButton(radio)
            tools.addWidget(radio)
        self.tool_group.buttonClicked.connect(lambda b, *_: setattr(self.canvas, "tool", b.property("tool")))
        layout.addLayout(tools)

        grid = QGridLayout()
        self.monster_combo = QComboBox()
        self.monster_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.monster_combo.setMinimumContentsLength(12)
        self.monster_combo.currentIndexChanged.connect(
            lambda *_: setattr(self.canvas, "monster_type", self.monster_combo.currentData()))
        self.random_monsters_input = self._spin(0, 50, 3)
        self.random_monsters_input.setToolTip("Używane tylko, gdy na mapie nie ma spawnów")
        self.floor_input = self._spin(0, 15, GROUND_FLOOR)
        self.floor_input.setToolTip("Piętro Z (Tibia: 7 = poziom gruntu)")
        self.floor_input.valueChanged.connect(self._on_floor_changed)
        self.cell_px_input = self._spin(0, 256, 0)
        self.cell_px_input.setSpecialValueText("auto")
        self.cell_px_input.setToolTip("Ile pikseli PNG przypada na jedno pole. auto: minimapa Tibii = 1, "
                                      "zrzut ekranu klienta = 32")
        self.tolerance_input = self._spin(1, 255, 20)
        self.tolerance_input.setToolTip("Piksel jest ścianą, gdy KAŻDY kanał RGB jest poniżej tej wartości "
                                        "(czerń #000000 i ciemny szum). Przezroczystość też = ściana.")
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
                  ("Próg czerni:", self.tolerance_input), ("Próg zajętości:", self.fraction_input),
                  ("Mnożnik ukosu:", self.diagonal_input), ("Prędkość podłoża:", self.ground_speed_input)]
        for i, (text, widget) in enumerate(fields):
            row, col = divmod(i, 2)
            grid.addWidget(QLabel(text), row, col * 2)
            grid.addWidget(widget, row, col * 2 + 1)
        layout.addLayout(grid)

        palette_label = QLabel("Kolory mapy - zaznaczone = ściana (nieprzechodnie):")
        palette_label.setObjectName("hint")
        layout.addWidget(palette_label)
        self.palette_list = QListWidget()
        self.palette_list.setMaximumHeight(130)
        self.palette_list.setToolTip("Odznacz/zaznacz kolor, aby zmienić, czy jest ścianą. Mapa parsuje się od razu.")
        self.palette_list.itemChanged.connect(self._on_palette_changed)
        layout.addWidget(self.palette_list)

        buttons = QGridLayout()
        self.load_map_button = QPushButton("Wczytaj mapę PNG…")
        self.load_map_button.clicked.connect(self.load_map_dialog)
        self.reparse_button = QPushButton("Parsuj kolizje ponownie")
        self.reparse_button.clicked.connect(self.reparse_map)
        self.clear_map_button = QPushButton("Wyczyść piętro")
        self.clear_map_button.clicked.connect(lambda *_: self.canvas.clear_floor())
        self.load_db_button = QPushButton("Baza potworów (JSON)…")
        self.load_db_button.clicked.connect(self.load_monster_db)
        self.clear_spawns_button = QPushButton("Usuń potwory (piętro)")
        self.clear_spawns_button.clicked.connect(self._clear_spawns)
        self.client_map_button = QPushButton("Minimapa klienta…")
        self.client_map_button.setToolTip("Import folderu minimapy klienta Tibii (Minimap_Color / WaypointCost): "
                                          "ściany, koszt pól i schody między piętrami")
        self.client_map_button.clicked.connect(self.import_client_minimap)
        self.sprites_button = QPushButton("Pobierz grafiki ponownie")
        self.sprites_button.clicked.connect(lambda *_: self.sync_sprites(force=True))
        self.save_scenario_button = QPushButton("Zapisz scenariusz")
        self.save_scenario_button.clicked.connect(self.save_scenario)
        self.load_scenario_button = QPushButton("Wczytaj scenariusz")
        self.load_scenario_button.clicked.connect(self.load_scenario)
        for i, button in enumerate((self.load_map_button, self.reparse_button, self.clear_map_button,
                                    self.clear_spawns_button, self.client_map_button, self.sprites_button,
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
        self.danger_checkbox = QCheckBox("Strefy zagrożenia")
        self.danger_checkbox.setToolTip("Pokaż pola w zasięgu fal/belek potworów - to samo widzi agent w obserwacji")
        self.danger_checkbox.toggled.connect(lambda on: setattr(self.canvas, "show_danger", on))
        row.addWidget(self.danger_checkbox)
        row.addWidget(QLabel("Prędkość:"))
        row.addWidget(self.speed_combo)
        layout.addLayout(row)
        keys = QLabel("Klawisze: W/A/S/D lub strzałki - ruch, Q/E/Z/C - ukosy, 1 - Exori Frigo, "
                      "2 - Exori Tera, 3 - Exura Gran, 4 - mikstura many, 5 - runa SD, 6 - Avalanche")
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
            ("rod", "Różdżka:"), ("sd", "Runy SD / Avalanche:"),
            ("exp_h", "Exp/h:"), ("profit_h", "Zysk/h:"),
            ("loot", "Loot / zapasy:"), ("profit", "Bilans (gp):"),
            ("hp_min", "HP min:"), ("adjacent", "Obok potwora:"),
            ("pots_h", "Mikstury/h:"), ("runes_h", "Runy/h:"),
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
            ("exp_h", "Śr. exp/h (100):"), ("profit_h", "Śr. zysk/h (100):"),
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
            Qt.Key_3: ACTION_HEAL, Qt.Key_4: ACTION_MANA_POTION, Qt.Key_5: ACTION_USE_SD, Qt.Key_6: ACTION_USE_AVALANCHE,
        }
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return super().eventFilter(obj, event)
        if (self.manual_running and not self.agent_control_checkbox.isChecked()
                and event.key() in self.key_actions):
            self.pending_action = self.key_actions[event.key()]
            return True
        # Map editor shortcuts (like a cavebot): only when no text field has focus
        focus = QApplication.focusWidget()
        typing = isinstance(focus, (QLineEdit, QSpinBox, QDoubleSpinBox))
        if event.key() in (Qt.Key_PageUp, Qt.Key_PageDown) and not self.training:
            self.floor_input.setValue(self.floor_input.value() + (-1 if event.key() == Qt.Key_PageUp else 1))
            return True
        if typing or not self.canvas.editing_enabled:
            return super().eventFilter(obj, event)
        ctrl = event.modifiers() & Qt.ControlModifier
        if ctrl and event.key() == Qt.Key_Z:
            self.canvas.undo()
            return True
        if ctrl and event.key() == Qt.Key_Y:
            self.canvas.redo()
            return True
        if event.key() == Qt.Key_Delete and self.canvas.waypoints:
            self.canvas.push_undo()
            self.canvas.waypoints.pop()
            self.canvas.refresh()
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
                       self.rod_combo, self.sd_input, self.avalanche_input, self.exp_rate_input,
                       self.client_map_button,
                       self.cell_px_input, self.tolerance_input, self.fraction_input,
                       self.diagonal_input, self.ground_speed_input,
                       self.load_map_button, self.reparse_button, self.clear_map_button,
                       self.load_db_button, self.load_scenario_button, self.clear_spawns_button):
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
            rod=self.rod_combo.currentData(),
            sd_runes=self.sd_input.value(),
            avalanche_runes=self.avalanche_input.value(),
            exp_rate=self.exp_rate_input.value(),
            terrain_speed={str(z): encode_speed(sp) for z, sp in canvas.speeds.items()
                           if sp.shape == (canvas.rows, canvas.cols) and (sp != DEFAULT_GROUND_SPEED).any()},
            transitions=[list(t) for t in canvas.transitions],
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
        self._warn_if_no_walls(config)
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
            self.canvas.clear_entities()   # live figures vanish; editable spawn markers stay
            self.log("system", "Mapa wróciła do trybu edycji (spawny można przesuwać i usuwać).")
            self._refresh_controls()

    def _clear_spawns(self):
        count = sum(1 for k in self.canvas.spawns if k[2] == self.canvas.floor)
        self.canvas.clear_spawns()
        self.log("system", f"Usunięto {count} spawnów z piętra {self.canvas.floor}.")

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
        if druid.rod is None:
            v["rod"].setText("brak")
        else:
            ready = druid.auto_attack_remaining_ms(env.time_ms)
            v["rod"].setText(f"{druid.rod.name} ({'gotowa' if ready == 0 else f'{ready / 1000:.1f} s'})")
        v["sd"].setText(f"{druid.sd_runes} / {druid.avalanche_runes}")
        v["exp_h"].setText(f"{info['exp_per_hour']:,}".replace(",", " "))
        v["profit_h"].setText(f"{info['profit_per_hour']:,}".replace(",", " "))
        v["loot"].setText(f"{info['loot']:,} / {info['supply_cost']:,}".replace(",", " "))
        v["profit"].setText(f"{info['profit']:,}".replace(",", " "))
        v["hp_min"].setText(f"{info['hp_min_pct']:.0f}%")
        v["adjacent"].setText(f"{info['adjacent_pct']:.1f}%")
        v["pots_h"].setText(str(info["potions_per_hour"]))
        v["runes_h"].setText(str(info["runes_per_hour"]))

    def _log_env_events(self):
        if self.env is None:
            return
        for kind, message in self.env.events:
            self.log(kind, message, time_ms=self.env.time_ms)

    # ----- training mode --------------------------------------------------- #

    def start_training(self):
        config = self.build_config()
        self._warn_if_no_walls(config)
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
        v["exp_h"].setText(f"{stats['avg_exp_per_hour']:,.0f}".replace(",", " "))
        v["profit_h"].setText(f"{stats['avg_profit_per_hour']:,.0f}".replace(",", " "))
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
        cell_px = self.cell_px_input.value() or None   # 0 = auto
        return MapParser(cell_px=cell_px, dark_threshold=self.tolerance_input.value(),
                         wall_colors=sorted(self.wall_colors),
                         blocked_fraction=self.fraction_input.value() / 100.0)

    def load_map_file(self, path, fit=True):
        """Parse walls first, then show the image; a failed parse leaves the previous map untouched."""
        name = os.path.basename(path)
        parser = self._parser()
        try:
            rgba = load_rgba(path)
            grid = parser.to_grid(rgba)
            loaded = self.canvas.load_map(path, parser.last_cell_px, grid, parser.last_offset, fit=fit)
        except Exception as exc:  # any decoder/shape problem must be visible, never a silent 0-wall map
            self.log("warn", f"Nie udało się sparsować mapy {name}: {exc}")
            QMessageBox.warning(self, "Błąd mapy", f"Nie udało się sparsować mapy {name}:\n{exc}")
            return False
        if loaded is None:
            self.log("warn", f"Qt nie potrafi wyświetlić obrazu {name}.")
            return False
        if self.canvas.replaced_floors:
            self.log("warn", f"Mapa {name} ma inny rozmiar niż piętra {self.canvas.replaced_floors} - "
                             f"usunięto je razem z trasą i spawnami (nowy projekt mapy).")
            self._client_minimap = None
        self._map_rgba = rgba
        self._populate_palette(rgba)
        stairs = mask_to_cells(stairs_mask(rgba), parser.last_cell_px, parser.last_offset, 0.3)
        if stairs.shape == (self.canvas.rows, self.canvas.cols):
            self.canvas.set_stairs(self.canvas.floor, stairs)
            if stairs.any():
                self.log("system", f"Schody/dziury na piętrze {self.canvas.floor}: {int(stairs.sum())} pól, "
                                   f"połączone przejścia: {len(self.canvas.transitions)} (wczytaj sąsiednie piętro, "
                                   f"aby je połączyć).")
        if parser.last_was_minimap:
            kind = f"minimapa ({parser.last_cell_px}x{parser.last_cell_px} px = 1 pole, start siatki {parser.last_offset})"
        else:
            kind = f"zrzut ekranu ({parser.last_cell_px} px = 1 pole)"
        walls = int(grid.sum())
        self.log("system", f"Wczytano mapę {name}: {kind}, siatka {self.canvas.cols}x{self.canvas.rows} pól, "
                           f"ściany: {walls} ({grid.mean() * 100:.1f}%), piętro {self.canvas.floor}.")
        if walls == 0:
            self.log("warn", "Parser nie znalazł żadnej ściany - podnieś 'Próg czerni' albo sprawdź 'Piksele na pole'.")
        elif walls == grid.size:
            self.log("warn", "Cała mapa jest ścianą - obniż 'Próg czerni' albo ustaw 'Piksele na pole' na auto/1.")
        return True

    def load_map_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj mapę", PROJECT_DIR,
                                              "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load_map_file(path)

    def reparse_map(self):
        if not self.canvas.map_path:
            self.log("warn", "Brak wczytanej mapy do sparsowania.")
            return
        self.load_map_file(self.canvas.map_path, fit=False)

    def _populate_palette(self, rgba):
        threshold = self.tolerance_input.value()
        self.palette_list.blockSignals(True)
        self.palette_list.clear()
        for rgb, share in palette(rgba)[:48]:
            dark = all(v < threshold for v in rgb)
            item = QListWidgetItem(f"{color_label(rgb)}  rgb{rgb}  -  {share * 100:.1f}%"
                                   + ("  (próg czerni)" if dark else ""))
            swatch = QPixmap(14, 14)
            swatch.fill(QColor(*rgb))
            item.setIcon(QIcon(swatch))
            item.setData(Qt.UserRole, rgb)
            item.setFlags((item.flags() | Qt.ItemIsUserCheckable) & ~(Qt.ItemIsEnabled if dark else Qt.NoItemFlags))
            item.setCheckState(Qt.Checked if dark or rgb in self.wall_colors else Qt.Unchecked)
            self.palette_list.addItem(item)
        self.palette_list.blockSignals(False)

    def _on_palette_changed(self, item):
        rgb = tuple(item.data(Qt.UserRole))
        if item.checkState() == Qt.Checked:
            self.wall_colors.add(rgb)
        else:
            self.wall_colors.discard(rgb)
        if self._map_rgba is None or self.canvas.map_floor is None:
            return
        grid = self._parser().to_grid(self._map_rgba)
        if grid.shape != (self.canvas.rows, self.canvas.cols):
            self.reparse_map()
            return
        self.canvas.set_collision(self.canvas.map_floor, grid)
        state = "ściana" if rgb in self.wall_colors else "przechodnie"
        self.log("system", f"{color_label(rgb)} rgb{rgb} = {state}. Ściany: {int(grid.sum())} "
                           f"({grid.mean() * 100:.1f}%).")

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
        data["floor_maps"] = {str(z): entry for z, entry in self.canvas.floor_maps.items()}
        data["client_minimap"] = self._client_minimap
        data["map_wall_colors"] = [list(c) for c in sorted(self.wall_colors)]
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
        if data.get("map_wall_colors") is not None:
            self.wall_colors = {tuple(c) for c in data["map_wall_colors"]}
        floor_maps = data.get("floor_maps")
        if floor_maps is None and data.get("map_image"):            # scenarios saved before multi-floor maps
            legacy_floor = data.get("map_floor") if data.get("map_floor") is not None else config.floor
            floor_maps = {str(legacy_floor): {"path": data["map_image"], "cell_px": data.get("map_cell_px", CELL),
                                              "offset": data.get("map_offset", [0, 0]), "client": False}}
        floor_maps = floor_maps or {}
        self.canvas.clear_floor_maps()
        self.canvas.stairs = {}
        client = data.get("client_minimap")
        self._client_minimap = None
        if client and os.path.isdir(client.get("folder", "")):
            self._run_client_import(client, log_prefix="Scenariusz: ")
        for z_text, entry in floor_maps.items():
            if entry.get("client") or not entry.get("path") or not os.path.isfile(entry["path"]):
                continue
            self.canvas.set_floor(int(z_text))
            self.canvas.load_map(entry["path"], int(entry.get("cell_px", CELL)),
                                 offset=tuple(entry.get("offset", (0, 0))), fit=False)
            try:
                rgba = load_rgba(entry["path"])
                stairs = mask_to_cells(stairs_mask(rgba), int(entry.get("cell_px", CELL)),
                                       tuple(entry.get("offset", (0, 0))), 0.3)
                if stairs.shape == (self.canvas.rows, self.canvas.cols):
                    self.canvas.stairs[int(z_text)] = stairs.astype(bool)
                self._map_rgba = rgba
            except OSError:
                self._map_rgba = None
        self.canvas.apply_config(config)
        for z_text, entry in floor_maps.items():
            if entry.get("path") and os.path.isfile(entry["path"]) and z_text not in config.collision:
                self.log("warn", f"Scenariusz nie ma kolizji dla piętra {z_text} - parsuję {os.path.basename(entry['path'])}.")
                self.canvas.set_floor(int(z_text))
                waypoints, spawns = list(self.canvas.waypoints), dict(self.canvas.spawns)
                self.load_map_file(entry["path"], fit=False)
                self.canvas.waypoints, self.canvas.spawns = waypoints, spawns
                self.canvas.refresh()
        if self._map_rgba is not None:
            self._populate_palette(self._map_rgba)
        self.canvas.fit_map()
        self.floor_input.setValue(config.floor)
        self.level_input.setValue(config.level)
        self.magic_input.setValue(config.magic_level)
        self.armor_input.setValue(config.armor)
        self.potions_input.setValue(config.mana_potions)
        self.max_steps_input.setValue(config.max_steps)
        self.utamo_checkbox.setChecked(config.use_utamo)
        self.rod_combo.setCurrentIndex(max(0, self.rod_combo.findData(config.rod)))
        self.sd_input.setValue(config.sd_runes)
        self.avalanche_input.setValue(config.avalanche_runes)
        self.exp_rate_input.setValue(config.exp_rate)
        self.random_monsters_input.setValue(config.random_monster_count)
        self.diagonal_input.setValue(config.diagonal_multiplier)
        self.ground_speed_input.setValue(config.ground_speed)
        index = self.monster_combo.findData(config.random_monster_type)
        if index >= 0:
            self.monster_combo.setCurrentIndex(index)
        self.log("system", f"Wczytano scenariusz: {os.path.basename(path)}")

    def import_client_minimap(self):
        """Ask for the client minimap folder and the area to crop, then build walls/speeds/stairs per floor."""
        previous = self._client_minimap or {}
        dialog = QDialog(self)
        dialog.setWindowTitle("Minimapa klienta Tibii")
        form = QFormLayout(dialog)
        folder = QLineEdit(previous.get("folder", ""))
        browse = QPushButton("…")
        browse.clicked.connect(lambda *_: folder.setText(
            QFileDialog.getExistingDirectory(dialog, "Folder minimapy klienta", folder.text() or PROJECT_DIR)
            or folder.text()))
        row = QHBoxLayout()
        row.addWidget(folder, 1)
        row.addWidget(browse)
        center_x, center_y = self._spin(0, 65_535, previous.get("x", 32_000)), self._spin(0, 65_535, previous.get("y", 32_000))
        radius = self._spin(8, 256, previous.get("radius", 48))
        floors = QLineEdit(",".join(str(z) for z in previous.get("floors", [self.canvas.floor])))
        hint = QLabel("Folder: .../Tibia/packages/Tibia/minimap (pliki Minimap_Color_X_Y_Z.png i "
                      "Minimap_WaypointCost_X_Y_Z.png). Współrzędne środka i piętra odczytasz np. z tibiamaps.io.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        form.addRow(hint)
        form.addRow("Folder:", row)
        form.addRow("Środek X:", center_x)
        form.addRow("Środek Y:", center_y)
        form.addRow("Promień (pola):", radius)
        form.addRow("Piętra (np. 7,8,9):", floors)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        try:
            floor_list = [int(f) for f in floors.text().replace(" ", "").split(",") if f]
        except ValueError:
            QMessageBox.warning(self, "Błąd", "Piętra podaj jako liczby oddzielone przecinkami, np. 7,8")
            return
        params = {"folder": folder.text(), "x": center_x.value(), "y": center_y.value(),
                  "radius": radius.value(), "floors": floor_list}
        self.canvas.waypoints, self.canvas.spawns = [], {}
        self._run_client_import(params)

    def _run_client_import(self, params, log_prefix=""):
        try:
            terrain = load_client_minimap(params["folder"], params["x"], params["y"], params["radius"], params["floors"])
        except Exception as exc:  # missing files, wrong folder, decoding problems - report, keep the old map
            self.log("warn", f"{log_prefix}Import minimapy klienta nieudany: {exc}")
            QMessageBox.warning(self, "Minimapa klienta", f"Import nieudany:\n{exc}")
            return False
        self.canvas.apply_terrain(terrain)
        self._client_minimap = params
        self._map_rgba = None
        self.floor_input.setValue(min(terrain.layers))
        for z, layer in sorted(terrain.layers.items()):
            slow = int((layer.speed > 150).sum())
            self.log("system", f"{log_prefix}Piętro {z}: ściany {int(layer.walls.sum())}, wolne pola {slow}, "
                               f"schody/dziury {int(layer.stairs.sum())}.")
        self.log("system", f"{log_prefix}Minimapa klienta: obszar {terrain.layers[min(terrain.layers)].walls.shape[1]} pól, "
                           f"początek świata {terrain.layers[min(terrain.layers)].origin}, "
                           f"przejścia między piętrami: {len(terrain.transitions)}.")
        return True

    def _on_floor_changed(self, z):
        self.canvas.set_floor(z)
        c = self.canvas
        if c.floor_maps and z not in c.floor_maps:
            self.log("warn", f"Piętro {z} nie ma mapy (wczytane piętra: {sorted(c.floor_maps)}). "
                             f"Kolizje: {int(c.grid().sum())} - wczytaj PNG albo minimapę klienta dla tego piętra.")

    def _warn_if_no_walls(self, config):
        """Called before an episode: a loaded map whose start floor has no walls is almost always a mistake."""
        start_floor = int(config.waypoints[0][2]) if config.waypoints else config.floor
        walls = sum(row.count("1") for row in config.collision.get(str(start_floor), []))
        if self.canvas.floor_maps and walls == 0:
            self.log("warn", f"UWAGA: piętro {start_floor} nie ma ani jednej ściany, a mapa jest wczytana "
                             f"(piętra z mapą: {sorted(self.canvas.floor_maps)}). Druid będzie chodził po pustce.")

    # ----- sprites --------------------------------------------------------- #

    def sync_sprites(self, force=False):
        """Download missing sprites for the druid and every monster in the database (background thread)."""
        if self.asset_worker is not None and self.asset_worker.isRunning():
            self._sprite_resync = True
            return
        entries = sprite_entries(self.monster_db)
        if force:
            entries = [e for e in entries if not self.asset_manager.local_path(e[0])]
        if not entries or (not force and all(self.asset_manager.local_path(name) for name, _ in entries)):
            self.log("system", f"Sprite'y: komplet {len(entries)} na dysku ({os.path.basename(SPRITES_DIR)}/).")
            return
        self.log("system", "Sprite'y: pobieram brakujące obrazki z TibiaWiki w tle…")
        self.asset_worker = AssetWorker(entries, self, force=force)
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
        sd = SPELLBOOK["sudden_death"]
        sd_low = int(druid.level * sd.level_factor + druid.magic_level * sd.min_mlvl_factor + sd.min_base)
        sd_high = int(druid.level * sd.level_factor + druid.magic_level * sd.max_mlvl_factor + sd.max_base)
        sd_text = (f"SD: {sd_low}-{sd_high} (śmierć)" if druid.magic_level >= sd.min_magic_level
                   else f"SD: wymaga ML {sd.min_magic_level}")
        ava = SPELLBOOK["avalanche"]
        ava_low = int(druid.level * ava.level_factor + druid.magic_level * ava.min_mlvl_factor + ava.min_base)
        ava_high = int(druid.level * ava.level_factor + druid.magic_level * ava.max_mlvl_factor + ava.max_base)
        sd_text += (f"  |  Avalanche: {ava_low}-{ava_high} (lód, 37 pól)" if druid.magic_level >= ava.min_magic_level
                    else f"  |  Avalanche: wymaga ML {ava.min_magic_level}")
        rod_name = self.rod_combo.currentData()
        if rod_name is None:
            rod_text = "Różdżka: brak"
        else:
            rod = self.item_db.rod(rod_name)
            rod_text = (f"{rod.name}: {ELEMENT_LABELS_PL[rod.element]} {rod.damage_min}-{rod.damage_max}, "
                        f"{rod.mana} many, zasięg {rod.range}" if druid.level >= rod.level
                        else f"Różdżka: {rod.name} wymaga poziomu {rod.level}!")
        self.preview_label.setText(f"HP {druid.max_hp}  |  Mana {druid.max_mana}  |  Prędkość {druid.speed}  |  "
                                   f"Krok {straight} ms, ukos {diagonal} ms\n{rod_text}\n{sd_text}")
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
