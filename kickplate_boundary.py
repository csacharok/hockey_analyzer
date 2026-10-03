"""Camera-relative board/ice boundary from yellow kickplate segments."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundaryEstimate:
    points: tuple[tuple[float, float], ...]
    coverage: float
    bin_width: float
    margin_px: float = 6

    def y_at(self, x):
        if not self.points:
            return None
        nearest = min(self.points, key=lambda point: abs(point[0] - x))
        return nearest[1] if abs(nearest[0] - x) <= self.bin_width * 1.6 else None


class KickplateBoundaryDetector:
    """Estimate a piecewise boundary without assuming a fixed image row."""

    def __init__(self, configuration=None):
        configuration = configuration or {}
        self.hue = configuration.get("hue", [15, 40])
        self.minimum_saturation = configuration.get("minimum_saturation", 90)
        self.minimum_value = configuration.get("minimum_value", 70)
        self.bins = configuration.get("bins", 48)
        self.maximum_gap_bins = configuration.get("maximum_gap_bins", 3)
        self.minimum_coverage = configuration.get("minimum_coverage", .30)
        self.margin_px = configuration.get("contact_margin_px", 6)
        if (not isinstance(self.hue, list) or len(self.hue) != 2
                or not 0 <= self.hue[0] <= self.hue[1] <= 179):
            raise ValueError("Kickplate hue must be an increasing OpenCV HSV range.")
        if type(self.bins) is not int or self.bins < 8:
            raise ValueError("Kickplate bins must be an integer of at least 8.")
        if type(self.maximum_gap_bins) is not int or self.maximum_gap_bins < 0:
            raise ValueError("maximum_gap_bins must be a nonnegative integer.")
        if not isinstance(self.minimum_coverage, (int, float)) or not 0 < self.minimum_coverage <= 1:
            raise ValueError("minimum_coverage must be in (0, 1].")

    def estimate(self, frame, cv2, hsv=None):
        import numpy as np
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if hsv is None else hsv
        low = np.array([self.hue[0], self.minimum_saturation, self.minimum_value])
        high = np.array([self.hue[1], 255, 255])
        mask = cv2.inRange(hsv, low, high)
        # Kickplates are long horizontal structures. Excluding the extreme top
        # and bottom suppresses lights/score graphics and near-camera railings.
        mask[:round(.20 * height)] = 0
        mask[round(.80 * height):] = 0
        link = max(9, width // 80)
        linked = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                  np.ones((3, link), dtype=np.uint8))
        linked = cv2.morphologyEx(linked, cv2.MORPH_OPEN,
                                  np.ones((2, max(5, link // 2)), dtype=np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(linked)
        accepted_components = np.zeros(count, dtype=bool)
        for index in range(1, count):
            x, y, component_width, component_height, area = stats[index]
            if (component_width >= .025 * width and component_width >= 3 * component_height
                    and area >= max(40, width // 24)):
                accepted_components[index] = True
        # Apply component decisions in one indexed pass instead of scanning the
        # full label image once for every connected component.
        selected = accepted_components[labels]

        edges = np.linspace(0, width, self.bins + 1, dtype=int)
        values = [None] * self.bins
        selected_y, selected_x = np.nonzero(selected)
        selected_bins = np.searchsorted(edges[1:], selected_x, side="right")
        for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            ys = selected_y[selected_bins == index]
            if len(ys) >= max(3, (right - left) // 8):
                values[index] = float(np.median(ys))
        observed = sum(value is not None for value in values)
        coverage = observed / self.bins
        if coverage < self.minimum_coverage:
            return BoundaryEstimate((), coverage, width / self.bins, self.margin_px)
        # Only bridge short occlusions bracketed by direct segment evidence.
        index = 0
        while index < self.bins:
            if values[index] is not None:
                index += 1
                continue
            start = index
            while index < self.bins and values[index] is None:
                index += 1
            gap = index - start
            if start and index < self.bins and gap <= self.maximum_gap_bins:
                before, after = values[start - 1], values[index]
                for offset in range(gap):
                    values[start + offset] = before + (after - before) * (offset + 1) / (gap + 1)
        points = tuple(((edges[i] + edges[i + 1]) / 2, value)
                       for i, value in enumerate(values) if value is not None and math.isfinite(value))
        return BoundaryEstimate(points, coverage, width / self.bins, self.margin_px)
