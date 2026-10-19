"""
PyQt5 digital twin: grid map editor, manual / agent visualizer (100 ms ticks)
and background DQN training in a QThread.
"""

import html
import json
import os
import sys

from PyQt5.QtCore import QEvent, QPoint, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath,
                         QPalette, QPen, QPixmap)
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
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
from .env import (
    ACTION_ATTACK,
    ACTION_DOWN,
    ACTION_HEAL,
    ACTION_LEFT,
    ACTION_MANA_POTION,
    ACTION_RIGHT,
    ACTION_UP,
    MONSTER_TEMPLATES,
    TICK_SECONDS,
    Druid,
    DruidHuntEnv,
    EnvConfig,
)
from .observation import ACTION_LABELS_PL, ATTACK_RANGE
from .trainer import run_training

CELL = 32
VISUAL_TICK_MS = 100
DEFAULT_GRID = (32, 24)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP_FILE = os.path.join(PROJECT_DIR, "map.png")

TOOL_WAYPOINT = "waypoint"
TOOL_MONSTER = "monster"
TOOL_OBSTACLE = "obstacle"


def format_game_time(steps):
    seconds = steps * TICK_SECONDS
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:04.1f}"


# --------------------------------------------------------------------------- #
# Map canvas
# --------------------------------------------------------------------------- #

class GridCanvas(QGraphicsView):
    """LMB: use tool, RMB: erase cell, MMB drag: pan, wheel: zoom."""

    scenario_changed = pyqtSignal()
    edit_blocked = pyqtSignal()
    cell_hovered = pyqtSignal(int, int)

    MIN_ZOOM = 0.15
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
        self.tool = TOOL_WAYPOINT
        self.monster_type = "Giant Spider"
        self.editing_enabled = True
        self.map_path = None

        self.obstacles = set()
        self.waypoints = []
        self.spawns = {}
        self._obstacle_items = {}
        self._spawn_items = {}
        self._waypoint_items = []
        self._map_item = None
        self._zoom = 1.0
        self._panning = False
        self._pan_last = QPoint()
        self._paint_button = None
        self._druid_item = None
        self._range_item = None
        self._monster_items = []

        self.load_map(MAP_FILE if os.path.isfile(MAP_FILE) else None)

    # ----- map & grid ------------------------------------------------------ #

    def load_map(self, path):
        if self._map_item is not None:
            self._scene.removeItem(self._map_item)
            self._map_item = None
        pixmap = QPixmap(path) if path else QPixmap()
        if pixmap.isNull():
            self.map_path = None
            self.cols, self.rows = DEFAULT_GRID
        else:
            self.map_path = path
            self._map_item = QGraphicsPixmapItem(pixmap)
            self._map_item.setZValue(-10)
            self._scene.addItem(self._map_item)
            self.cols = max(4, pixmap.width() // CELL)
            self.rows = max(4, pixmap.height() // CELL)
        self._scene.setSceneRect(0, 0, self.cols * CELL, self.rows * CELL)
        self._drop_out_of_bounds()
        self.centerOn(self._scene.sceneRect().center())
        return self.map_path is not None

    def _drop_out_of_bounds(self):
        inside = self._in_bounds
        for cell in [c for c in self.obstacles if not inside(*c)]:
            self._remove_obstacle(cell)
        for cell in [c for c in self.spawns if not inside(*c)]:
            self._remove_spawn(cell)
        self.waypoints = [w for w in self.waypoints if inside(*w)]
        self._redraw_waypoints()

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

    # ----- scenario editing ------------------------------------------------ #

    def _add_obstacle(self, cell):
        if cell in self.obstacles:
            return
        self._remove_spawn(cell)
        self.waypoints = [w for w in self.waypoints if w != cell]
        self._redraw_waypoints()
        item = QGraphicsRectItem(cell[0] * CELL, cell[1] * CELL, CELL, CELL)
        item.setBrush(QBrush(QColor(20, 20, 20, 200)))
        item.setPen(QPen(QColor("#795548"), 1))
        item.setZValue(1)
        self._scene.addItem(item)
        self.obstacles.add(cell)
        self._obstacle_items[cell] = item

    def _remove_obstacle(self, cell):
        if cell in self.obstacles:
            self.obstacles.discard(cell)
            self._scene.removeItem(self._obstacle_items.pop(cell))

    def _set_spawn(self, cell, name):
        self._remove_spawn(cell)
        x, y = cell
        item = QGraphicsRectItem(x * CELL + 4, y * CELL + 4, CELL - 8, CELL - 8)
        item.setBrush(QBrush(QColor(213, 0, 0, 90)))
        pen = QPen(QColor("#ff5252"), 2, Qt.DashLine)
        item.setPen(pen)
        item.setZValue(2)
        label = QGraphicsSimpleTextItem(name[0], item)
        label.setFont(QFont("Sans", 9, QFont.Bold))
        label.setBrush(QBrush(QColor("#ffffff")))
        label.setPos(x * CELL + 11, y * CELL + 7)
        item.setToolTip(f"Spawn: {name}")
        self._scene.addItem(item)
        self.spawns[cell] = name
        self._spawn_items[cell] = item

    def _remove_spawn(self, cell):
        if cell in self.spawns:
            del self.spawns[cell]
            self._scene.removeItem(self._spawn_items.pop(cell))

    def _redraw_waypoints(self):
        for item in self._waypoint_items:
            self._scene.removeItem(item)
        self._waypoint_items = []
        centers = [(x * CELL + CELL / 2, y * CELL + CELL / 2) for x, y in self.waypoints]
        pen = QPen(QColor("#ffc400"), 2)
        pen.setCosmetic(True)
        for (ax, ay), (bx, by) in zip(centers, centers[1:]):
            line = QGraphicsLineItem(ax, ay, bx, by)
            line.setPen(pen)
            line.setZValue(3)
            self._scene.addItem(line)
            self._waypoint_items.append(line)
        if len(centers) >= 3:
            loop = QGraphicsLineItem(*centers[-1], *centers[0])
            loop_pen = QPen(QColor("#ffc400"), 1.5, Qt.DashLine)
            loop_pen.setCosmetic(True)
            loop.setPen(loop_pen)
            loop.setZValue(3)
            self._scene.addItem(loop)
            self._waypoint_items.append(loop)
        for i, (cx, cy) in enumerate(centers):
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
        if not self._in_bounds(*cell):
            return
        if self.tool == TOOL_OBSTACLE:
            self._add_obstacle(cell)
        elif dragging:
            return
        elif self.tool == TOOL_WAYPOINT:
            if cell in self.obstacles or (self.waypoints and self.waypoints[-1] == cell):
                return
            self.waypoints.append(cell)
            self._redraw_waypoints()
        elif self.tool == TOOL_MONSTER:
            if cell in self.obstacles:
                return
            self._set_spawn(cell, self.monster_type)
        self.scenario_changed.emit()

    def _erase(self, cell, dragging):
        if not self._in_bounds(*cell):
            return
        had_content = cell in self.obstacles or cell in self.spawns or cell in self.waypoints
        self._remove_obstacle(cell)
        self._remove_spawn(cell)
        if cell in self.waypoints:
            self.waypoints = [w for w in self.waypoints if w != cell]
            self._redraw_waypoints()
        elif not had_content and not dragging and self.tool == TOOL_WAYPOINT and self.waypoints:
            self.waypoints.pop()
            self._redraw_waypoints()
        self.scenario_changed.emit()

    def clear_scenario(self):
        for cell in list(self.obstacles):
            self._remove_obstacle(cell)
        for cell in list(self.spawns):
            self._remove_spawn(cell)
        self.waypoints = []
        self._redraw_waypoints()
        self.scenario_changed.emit()

    def apply_scenario(self, obstacles, waypoints, spawns):
        self.clear_scenario()
        for x, y in obstacles:
            if self._in_bounds(x, y):
                self._add_obstacle((x, y))
        for x, y, name in spawns:
            if self._in_bounds(x, y) and name in MONSTER_TEMPLATES:
                self._set_spawn((x, y), name)
        self.waypoints = [(x, y) for x, y in waypoints if self._in_bounds(x, y)]
        self._redraw_waypoints()
        self.scenario_changed.emit()

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
            self._monster_items = [self._make_entity(QColor("#d50000"), m.name, 15)
                                   for m in env.monsters]

        druid = env.druid
        span = ATTACK_RANGE * 2 + 1
        self._range_item.setRect((druid.x - ATTACK_RANGE) * CELL, (druid.y - ATTACK_RANGE) * CELL,
                                 span * CELL, span * CELL)
        self._update_entity(self._druid_item, druid.x, druid.y, druid.hp / druid.max_hp,
                            druid.mana / druid.max_mana, druid.utamo_active, druid.alive)
        for monster, item in zip(env.monsters, self._monster_items):
            self._update_entity(item, monster.x, monster.y, monster.hp / monster.max_hp,
                                None, False, monster.alive)

    def _make_entity(self, color, label, z, mana_bar=False):
        r = CELL / 2 - 4
        body = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        body.setBrush(QBrush(color))
        body.setPen(QPen(color.darker(170), 2))
        body.setZValue(z)
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
        return {"body": body, "hp": hp, "mana": mana, "shield": shield, "r": r}

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
        except Exception as exc:  # surfaced in the GUI log instead of killing the thread silently
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
QSpinBox, QComboBox { background-color: #2b2d31; border: 1px solid #44474d; border-radius: 4px;
                      padding: 3px 6px; color: #ffffff; }
QSpinBox:disabled, QComboBox:disabled { color: #777; }
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
        self.resize(1560, 940)

        self.agent = DQNAgent()
        self.env = None
        self.obs = None
        self.worker = None
        self.pending_action = None
        self.manual_running = False

        self.visual_timer = QTimer(self)
        self.visual_timer.setTimerType(Qt.PreciseTimer)
        self.visual_timer.setInterval(VISUAL_TICK_MS)
        self.visual_timer.timeout.connect(self.on_visual_tick)

        self._build_ui()
        self._bind_keys()
        self._update_preview()
        self._update_model_label()
        self._refresh_controls()

        if self.canvas.map_path:
            self.log("system", f"Wczytano mapę {os.path.basename(MAP_FILE)} "
                               f"({self.canvas.cols}x{self.canvas.rows} pól).")
        else:
            self.log("system", f"Brak map.png - pusta siatka {self.canvas.cols}x{self.canvas.rows} pól.")
        self.log("system", "Narysuj trasę, przeszkody i spawny potworów, potem wybierz tryb pracy.")

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
            lambda x, y: self.cell_label.setText(f"Pole: ({x}, {y})"))
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
        right_split.setSizes([680, 240])
        right_split.setMinimumWidth(560)
        right_split.setMaximumWidth(720)
        splitter.addWidget(right_split)
        splitter.setStretchFactor(0, 1)
        splitter.setSizes([980, 580])
        self.setCentralWidget(splitter)

        self.cell_label = QLabel("Pole: -")
        self.scenario_label = QLabel()
        self.statusBar().addWidget(self.cell_label)
        self.statusBar().addWidget(self.scenario_label)
        self._on_scenario_changed()

    def _build_character_group(self):
        group = QGroupBox("Postać - Druid")
        form = QFormLayout(group)
        self.level_input = self._spin(8, 2000, 100)
        self.magic_input = self._spin(0, 150, 60)
        self.potions_input = self._spin(0, 5000, 100)
        self.max_steps_input = self._spin(50, 10000, 600)
        self.level_input.valueChanged.connect(self._update_preview)
        self.magic_input.valueChanged.connect(self._update_preview)
        self.utamo_checkbox = QCheckBox("Utamo Vita (Mana Shield, auto)")
        self.utamo_checkbox.setChecked(True)
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #9ccc65;")
        self.preview_label.setWordWrap(True)
        form.addRow("Poziom:", self.level_input)
        form.addRow("Poziom magii:", self.magic_input)
        form.addRow("Great Mana Potion:", self.potions_input)
        form.addRow("Limit kroków epizodu:", self.max_steps_input)
        form.addRow("", self.utamo_checkbox)
        form.addRow("", self.preview_label)
        return group

    def _build_editor_group(self):
        group = QGroupBox("Edycja mapy")
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
        self.tool_group.buttonClicked.connect(
            lambda b, *_: setattr(self.canvas, "tool", b.property("tool")))
        layout.addLayout(tools)

        form = QFormLayout()
        self.monster_combo = QComboBox()
        for name, t in MONSTER_TEMPLATES.items():
            self.monster_combo.addItem(name, name)
            self.monster_combo.setItemData(
                self.monster_combo.count() - 1,
                f"HP {t['max_hp']}, obrażenia {t['damage_range'][0]}-{t['damage_range'][1]}, "
                f"prędkość {t['speed']}, exp {t['experience']}", Qt.ToolTipRole)
        self.monster_combo.currentIndexChanged.connect(
            lambda *_: setattr(self.canvas, "monster_type", self.monster_combo.currentData()))
        self.random_monsters_input = self._spin(1, 30, 3)
        form.addRow("Typ potwora:", self.monster_combo)
        form.addRow("Losowe potwory:", self.random_monsters_input)
        self.random_monsters_input.setToolTip("Używane tylko, gdy na mapie nie ma spawnów")
        layout.addLayout(form)

        row = QGridLayout()
        self.clear_map_button = QPushButton("Wyczyść mapę")
        self.clear_map_button.clicked.connect(lambda *_: self.canvas.clear_scenario())
        self.save_scenario_button = QPushButton("Zapisz scenariusz")
        self.save_scenario_button.clicked.connect(self.save_scenario)
        self.load_scenario_button = QPushButton("Wczytaj scenariusz")
        self.load_scenario_button.clicked.connect(self.load_scenario)
        self.load_map_button = QPushButton("Wczytaj mapę…")
        self.load_map_button.clicked.connect(self.load_map_dialog)
        for i, button in enumerate((self.clear_map_button, self.load_map_button,
                                    self.save_scenario_button, self.load_scenario_button)):
            row.addWidget(button, i // 2, i % 2)
        layout.addLayout(row)
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
        self.agent_control_checkbox = QCheckBox("Steruje agent (bieżąca polityka, zachłannie)")
        layout.addWidget(self.agent_control_checkbox)
        keys = QLabel("Klawisze: W/A/S/D lub strzałki - ruch, 1 - atak, 2 - leczenie, 3 - mikstura many")
        keys.setObjectName("hint")
        keys.setWordWrap(True)
        layout.addWidget(keys)

        row = QHBoxLayout()
        self.manual_start_button = QPushButton("▶ Start")
        self.manual_start_button.setObjectName("primary")
        self.manual_start_button.clicked.connect(self.toggle_manual)
        self.manual_reset_button = QPushButton("⟲ Reset epizodu")
        self.manual_reset_button.setObjectName("danger")
        self.manual_reset_button.clicked.connect(lambda *_: self.reset_manual())
        row.addWidget(self.manual_start_button)
        row.addWidget(self.manual_reset_button)
        layout.addLayout(row)

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
            ("epsilon", "Epsilon:"), ("speed", "Kroki/s:"),
            ("loss", "Strata (Huber):"), ("steps", "Kroki łącznie:"),
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
            Qt.Key_1: ACTION_ATTACK, Qt.Key_2: ACTION_HEAL, Qt.Key_3: ACTION_MANA_POTION,
        }
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress and self.manual_running
                and not self.agent_control_checkbox.isChecked()
                and event.key() in self.key_actions):
            self.pending_action = self.key_actions[event.key()]
            return True
        return super().eventFilter(obj, event)

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
        for widget in (self.level_input, self.magic_input, self.potions_input,
                       self.max_steps_input, self.utamo_checkbox, self.random_monsters_input,
                       self.clear_map_button, self.load_scenario_button, self.load_map_button):
            widget.setEnabled(editable)
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
            use_utamo=self.utamo_checkbox.isChecked(),
            mana_potions=self.potions_input.value(),
            obstacles=[list(c) for c in sorted(canvas.obstacles)],
            waypoints=[list(w) for w in canvas.waypoints],
            spawns=[[x, y, name] for (x, y), name in sorted(canvas.spawns.items())],
            random_monster_count=self.random_monsters_input.value(),
            random_monster_type=self.monster_combo.currentData(),
            max_steps=self.max_steps_input.value(),
        )

    # ----- manual / visual mode -------------------------------------------- #

    def toggle_manual(self):
        if self.manual_running:
            self.visual_timer.stop()
            self.manual_running = False
            self.log("system", "Wstrzymano.")
        else:
            if self.env is None:
                self._new_manual_episode()
            self.manual_running = True
            self.visual_timer.start()
            who = "agent" if self.agent_control_checkbox.isChecked() else "gracz (klawiatura)"
            self.log("system", f"Symulacja działa ({VISUAL_TICK_MS} ms/tick), steruje: {who}.")
        self._refresh_controls()

    def _new_manual_episode(self):
        config = self.build_config()
        self.env = DruidHuntEnv(config, record_events=True)
        self.obs, info = self.env.reset()
        druid = self.env.druid
        self.log("system", "=" * 44)
        self.log("system", f"Nowy epizod: Druid lvl {druid.level}, ML {druid.magic_level}, "
                           f"HP {druid.max_hp}, Mana {druid.max_mana}, potwory: {len(self.env.monsters)}.")
        self._log_env_events()
        self.canvas.sync_entities(self.env)
        self._update_manual_status(info, 0.0, None)

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
        env = self.env
        if env is None:
            return
        if self.agent_control_checkbox.isChecked():
            action = self.agent.act(self.obs, greedy=True)
        else:
            action, self.pending_action = self.pending_action, None
        if action is None:
            self.obs, reward, terminated, truncated, info = env.idle_step()
        else:
            self.obs, reward, terminated, truncated, info = env.step(action)
        self._log_env_events()
        self.canvas.sync_entities(env)
        self._update_manual_status(info, reward, action)

        if terminated or truncated:
            self.log("system", f"Koniec epizodu: nagroda {info['episode_reward']:.1f}, "
                               f"zabite {info['kills']}, kroki {info['steps']}.")
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
        v["time"].setText(format_game_time(info["steps"]))
        v["reward"].setText(f"{info['episode_reward']:.1f}")
        v["kills"].setText(f"{info['kills']} / {len(env.monsters)}")
        v["last_reward"].setText(f"{reward:+.1f}")
        v["utamo"].setText(f"Aktywne ({druid.utamo_ticks * TICK_SECONDS:.0f} s)"
                           if druid.utamo_active else "Nieaktywne")
        v["action"].setText(ACTION_LABELS_PL[action] if action is not None else "Czekanie")

    def _log_env_events(self):
        if self.env is None:
            return
        for kind, message in self.env.events:
            self.log(kind, message, steps=self.env.steps)

    # ----- training mode --------------------------------------------------- #

    def start_training(self):
        config = self.build_config()
        episodes = self.episodes_input.value()
        self.reward_plot.clear()
        self.train_progress.setRange(0, episodes)
        self.train_progress.setValue(0)
        self.worker = TrainingWorker(config, self.agent, episodes, self)
        self.worker.progress.connect(self._on_training_progress)
        self.worker.failed.connect(lambda msg: self.log("death", f"Błąd treningu: {msg}"))
        self.worker.finished.connect(self._on_training_finished)
        self.log("train", f"Start treningu: {episodes} epizodów, siatka {config.cols}x{config.rows}, "
                          f"przeszkody {len(config.obstacles)}, spawny {len(config.spawns) or 'losowe'}.")
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
                          f"kroki: {self.agent.total_steps}.")
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
        config = self.build_config()
        written = self.agent.export_policy(path, {"env_config": config.to_dict()})
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
            f"Epizody: {self.agent.episodes_trained}   |   kroki: {self.agent.total_steps}   |   "
            f"epsilon: {self.agent.epsilon:.3f}")

    # ----- scenario files -------------------------------------------------- #

    def save_scenario(self):
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz scenariusz",
                                              os.path.join(PROJECT_DIR, "scenario.json"), "JSON (*.json)")
        if not path:
            return
        data = self.build_config().to_dict()
        data["map_image"] = self.canvas.map_path
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
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Błąd", f"Nie udało się wczytać scenariusza:\n{exc}")
            return
        map_image = data.get("map_image")
        self.canvas.load_map(map_image if map_image and os.path.isfile(map_image) else None)
        self.canvas.apply_scenario(config.obstacles, config.waypoints, config.spawns)
        self.level_input.setValue(config.level)
        self.magic_input.setValue(config.magic_level)
        self.potions_input.setValue(config.mana_potions)
        self.max_steps_input.setValue(config.max_steps)
        self.utamo_checkbox.setChecked(config.use_utamo)
        self.random_monsters_input.setValue(config.random_monster_count)
        index = self.monster_combo.findData(config.random_monster_type)
        if index >= 0:
            self.monster_combo.setCurrentIndex(index)
        self.log("system", f"Wczytano scenariusz: {os.path.basename(path)}")

    def load_map_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Wczytaj mapę", PROJECT_DIR,
                                              "Obrazy (*.png *.jpg *.jpeg *.bmp)")
        if path:
            loaded = self.canvas.load_map(path)
            self.log("system", f"Wczytano mapę: {os.path.basename(path)} ({self.canvas.cols}x{self.canvas.rows} pól)."
                     if loaded else f"Nie udało się wczytać mapy: {os.path.basename(path)}")
            self._on_scenario_changed()

    # ----- misc ------------------------------------------------------------ #

    def _on_scenario_changed(self):
        c = self.canvas
        self.scenario_label.setText(f"Siatka {c.cols}x{c.rows}   |   trasa: {len(c.waypoints)}   |   "
                                    f"spawny: {len(c.spawns)}   |   przeszkody: {len(c.obstacles)}")

    def _update_preview(self):
        druid = Druid(self.level_input.value(), self.magic_input.value(), 0)
        self.preview_label.setText(f"Maks. HP: {druid.max_hp}   |   Maks. mana: {druid.max_mana}   |   "
                                   f"Prędkość: {druid.speed}")
        if self.env is None:
            self._set_bar(self.hp_bar, druid.max_hp, druid.max_hp)
            self._set_bar(self.mana_bar, druid.max_mana, druid.max_mana)

    @staticmethod
    def _set_bar(bar, value, maximum):
        bar.setMaximum(max(1, maximum))
        bar.setValue(int(value))
        bar.setFormat(f"{int(value)} / {maximum}")

    def log(self, kind, message, steps=None):
        color = LOG_COLORS.get(kind, LOG_COLORS["info"])
        weight = "bold" if kind in ("death", "kill") else "normal"
        stamp = format_game_time(steps if steps is not None else (self.env.steps if self.env else 0))
        self.event_log.append(f'<span style="color:#6d6d6d;">[{stamp}]</span> '
                              f'<span style="color:{color}; font-weight:{weight};">{html.escape(message)}</span>')
        bar = self.event_log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def closeEvent(self, event):
        if self.training:
            self.worker.stop()
            self.worker.wait(5000)
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
