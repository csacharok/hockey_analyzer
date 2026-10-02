"""Static image-space geometry, independent of tracking and player identity."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


ON_ICE = "ON ICE"
OFF_ICE = "OFF ICE"
UNKNOWN = "UNKNOWN"
STATES = (ON_ICE, OFF_ICE, UNKNOWN)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def contact_point(box):
    """Estimated feet position from an xyxy box, retaining fractional pixels."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _distance(point, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_squared = dx * dx + dy * dy
    t = max(0, min(1, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length_squared))
    return math.hypot(point[0] - a[0] - t * dx, point[1] - a[1] - t * dy)


def _intersects(a, b, c, d):
    crosses = (_cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b))
    if crosses[0] * crosses[1] < 0 and crosses[2] * crosses[3] < 0:
        return True
    return any(value == 0 and _distance(p, u, v) < 1e-9 for value, p, u, v in (
        (crosses[0], c, a, b), (crosses[1], d, a, b),
        (crosses[2], a, c, d), (crosses[3], b, c, d),
    ))


@dataclass(frozen=True)
class RinkGeometry:
    image_width: int
    image_height: int
    polygon: tuple[tuple[float, float], ...]
    boundary_margin_px: float = 8.0

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.image_width, self.image_height)):
            raise ValueError("Rink image_width and image_height must be positive integers.")
        if not _number(self.boundary_margin_px) or self.boundary_margin_px < 0:
            raise ValueError("boundary_margin_px must be finite and nonnegative.")
        if not isinstance(self.polygon, (list, tuple)) or len(self.polygon) < 3:
            raise ValueError("Define at least three polygon vertices in boundary order.")
        for point in self.polygon:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(_number(v) for v in point)):
                raise ValueError("Polygon vertices must be finite [x, y] pairs.")
            if not (0 <= point[0] <= self.image_width - 1 and 0 <= point[1] <= self.image_height - 1):
                raise ValueError("Polygon vertices must be inside the calibration image.")
        object.__setattr__(self, "polygon", tuple(tuple(p) for p in self.polygon))
        if len(set(self.polygon)) != len(self.polygon):
            raise ValueError("Polygon vertices must be distinct; do not repeat the first vertex.")
        edges = list(self.edges())
        if abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) < 1e-9:
            raise ValueError("Polygon must enclose a nonzero area.")
        for i, (a, b) in enumerate(edges):
            for j, (c, d) in enumerate(edges[i + 1:], start=i + 1):
                if j == i + 1 or (i == 0 and j == len(edges) - 1):
                    continue
                if _intersects(a, b, c, d):
                    raise ValueError("Polygon edges must not cross or touch nonadjacent edges.")

    def edges(self):
        return zip(self.polygon, self.polygon[1:] + self.polygon[:1])

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Rink configuration must be a JSON object.")
        try:
            return cls(data["image_width"], data["image_height"], data["polygon"], data.get("boundary_margin_px", 8.0))
        except KeyError as error:
            raise ValueError(f"Missing rink configuration field: {error.args[0]}") from error

    def validate_resolution(self, width, height):
        if (width, height) != (self.image_width, self.image_height):
            raise ValueError("Rink calibration resolution does not match video; recalibrate instead of silently scaling.")

    def classify(self, box):
        if len(box) != 4 or not all(_number(v) for v in box):
            return UNKNOWN
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return UNKNOWN
        x, y = contact_point(box)
        # Feet cannot be located reliably when the box reaches the bottom edge.
        if not (0 <= x <= self.image_width - 1 and 0 <= y < self.image_height - 1):
            return UNKNOWN
        inside = False
        for a, b in self.edges():
            if _distance((x, y), a, b) <= self.boundary_margin_px:
                return UNKNOWN
            if (a[1] > y) != (b[1] > y):
                crossing_x = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                if x < crossing_x:
                    inside = not inside
        return ON_ICE if inside else OFF_ICE
