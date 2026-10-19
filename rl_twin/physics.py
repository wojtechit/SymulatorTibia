"""
Grid physics: Tibia step timing, 8-way movement with diagonal cost, A*, line of sight.

Grids are indexed grid[y][x]; 1 = blocked, 0 = walkable. Pure Python/numpy, no Qt.
"""

import hashlib
import heapq
import math
from collections import OrderedDict

import numpy as np

SERVER_TICK_MS = 50
DEFAULT_GROUND_SPEED = 150          # grass; ground speed of the tile being entered
DEFAULT_DIAGONAL_MULTIPLIER = 1.5   # a diagonal step takes 1.5x an orthogonal one

# Tibia speed formula (10.x+): effective = A * ln(speed + B) + C
SPEED_A = 857.36
SPEED_B = 261.29
SPEED_C = -4795.01

# Visible area: 15 x 11 tiles centred on the player
VIEW_RANGE_X = 7
VIEW_RANGE_Y = 5

ORTHOGONAL_DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
DIAGONAL_DIRECTIONS = ((-1, -1), (1, -1), (-1, 1), (1, 1))
DIRECTIONS_8 = ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS


def chebyshev(ax, ay, bx, by):
    return max(abs(ax - bx), abs(ay - by))


def is_diagonal(dx, dy):
    return dx != 0 and dy != 0


def in_view(ax, ay, bx, by):
    return abs(ax - bx) <= VIEW_RANGE_X and abs(ay - by) <= VIEW_RANGE_Y


def effective_speed(speed):
    if speed <= 0:
        return 0
    return max(1, math.floor(SPEED_A * math.log(speed + SPEED_B) + SPEED_C))


def step_duration_ms(speed, diagonal=False, ground_speed=DEFAULT_GROUND_SPEED,
                     diagonal_multiplier=DEFAULT_DIAGONAL_MULTIPLIER):
    """Time to complete one step, rounded up to the 50 ms server tick. None if immobile."""
    eff = effective_speed(speed)
    if eff <= 0:
        return None
    duration = 1000.0 * ground_speed / eff
    if diagonal:
        duration *= diagonal_multiplier
    return int(math.ceil(duration / SERVER_TICK_MS) * SERVER_TICK_MS)


def is_walkable(grid, x, y):
    return 0 <= y < len(grid) and 0 <= x < len(grid[0]) and not grid[y][x]


def has_line_of_sight(grid, x0, y0, x1, y1):
    """Bresenham line; blocked if any tile strictly between the endpoints is an obstacle."""
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
        if (x, y) != (x1, y1) and not is_walkable(grid, x, y):
            return False
    return True


_field_cache = OrderedDict()
FIELD_CACHE_SIZE = 64


def distance_field(collision, goal, diagonal_cost=DEFAULT_DIAGONAL_MULTIPLIER):
    """
    Dijkstra travel cost from every tile to `goal` (x, y) on an 8-connected grid, walls = inf.
    Cached per (map contents, goal, cost), so each waypoint is solved once per map.
    """
    collision = np.ascontiguousarray(collision, dtype=np.uint8)
    key = (hashlib.blake2b(collision.tobytes(), digest_size=16).hexdigest(), collision.shape,
           tuple(goal), float(diagonal_cost))
    if key in _field_cache:
        _field_cache.move_to_end(key)
        return _field_cache[key]
    rows, cols = collision.shape
    walls = collision.tolist()
    field = [[math.inf] * cols for _ in range(rows)]
    gx, gy = goal
    if 0 <= gx < cols and 0 <= gy < rows and not walls[gy][gx]:
        field[gy][gx] = 0.0
        heap = [(0.0, gx, gy)]
        while heap:
            cost, x, y = heapq.heappop(heap)
            if cost > field[y][x]:
                continue
            for dx, dy in DIRECTIONS_8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and not walls[ny][nx]:
                    new_cost = cost + (diagonal_cost if dx and dy else 1.0)
                    if new_cost < field[ny][nx]:
                        field[ny][nx] = new_cost
                        heapq.heappush(heap, (new_cost, nx, ny))
    result = np.array(field, dtype=np.float32)
    _field_cache[key] = result
    if len(_field_cache) > FIELD_CACHE_SIZE:
        _field_cache.popitem(last=False)
    return result


def path_direction(field, x, y, diagonal_cost=DEFAULT_DIAGONAL_MULTIPLIER):
    """(dx, dy, cost): first step of the shortest path toward the field's goal; (0, 0, inf) if unreachable."""
    rows, cols = field.shape
    here = float(field[y, x]) if 0 <= x < cols and 0 <= y < rows else math.inf
    if here == 0.0 or math.isinf(here):
        return 0, 0, here
    best = (0, 0, here)
    best_total = math.inf
    for dx, dy in DIRECTIONS_8:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            total = float(field[ny, nx]) + (diagonal_cost if dx and dy else 1.0)
            if total < best_total:
                best_total, best = total, (dx, dy, here)
    return best


def a_star(grid, start, target, occupied=frozenset(), adjacent_goal=True,
           diagonal_cost=DEFAULT_DIAGONAL_MULTIPLIER, max_nodes=2500):
    """
    Shortest path on an 8-connected grid where diagonal steps cost `diagonal_cost`.

    With adjacent_goal=True the search stops on any tile touching `target`
    (how melee monsters chase). Returns the list of tiles to walk (excluding
    start) or None when unreachable within max_nodes expansions.
    """
    sx, sy = start
    tx, ty = target

    def reached(x, y):
        return chebyshev(x, y, tx, ty) <= 1 if adjacent_goal else (x, y) == (tx, ty)

    def heuristic(x, y):
        # Octile distance with the configured diagonal cost (admissible lower bound)
        ddx, ddy = abs(x - tx), abs(y - ty)
        if adjacent_goal:
            ddx, ddy = max(0, ddx - 1), max(0, ddy - 1)
        straight, diag = abs(ddx - ddy), min(ddx, ddy)
        return straight + diag * min(diagonal_cost, 2.0)

    if reached(sx, sy):
        return []
    open_heap = [(heuristic(sx, sy), 0.0, sx, sy)]
    came_from = {(sx, sy): None}
    cost_so_far = {(sx, sy): 0.0}
    expanded = 0
    while open_heap:
        _, cost, x, y = heapq.heappop(open_heap)
        if cost > cost_so_far[(x, y)]:
            continue
        if reached(x, y):
            path = []
            node = (x, y)
            while node != (sx, sy):
                path.append(node)
                node = came_from[node]
            return path[::-1]
        expanded += 1
        if expanded > max_nodes:
            return None
        for ddx, ddy in DIRECTIONS_8:
            nx, ny = x + ddx, y + ddy
            if not is_walkable(grid, nx, ny) or (nx, ny) in occupied or (nx, ny) == (tx, ty):
                continue
            new_cost = cost + (diagonal_cost if ddx and ddy else 1.0)
            if new_cost < cost_so_far.get((nx, ny), float("inf")):
                cost_so_far[(nx, ny)] = new_cost
                came_from[(nx, ny)] = (x, y)
                heapq.heappush(open_heap, (new_cost + heuristic(nx, ny), new_cost, nx, ny))
    return None
