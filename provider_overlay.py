"""Configured normalized provider-overlay exclusion masks."""

from __future__ import annotations

import math


class ProviderOverlayMask:
    def __init__(self, configuration=None):
        configuration = configuration or {}
        if not isinstance(configuration, dict):
            raise ValueError("provider must be a JSON object.")
        regions = configuration.get("excluded_regions", [])
        if not isinstance(regions, list):
            raise ValueError("provider.excluded_regions must be a list.")
        self.name = configuration.get("name")
        self.post_detection_overlap_fraction = configuration.get(
            "post_detection_overlap_fraction", .60)
        if (not isinstance(self.post_detection_overlap_fraction, (int, float))
                or isinstance(self.post_detection_overlap_fraction, bool)
                or not 0 <= self.post_detection_overlap_fraction <= 1):
            raise ValueError("provider.post_detection_overlap_fraction must be in [0, 1].")
        self.regions = []
        for region in regions:
            if (not isinstance(region, (list, tuple)) or len(region) != 4
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in region)):
                raise ValueError("Each provider exclusion region requires four finite normalized numbers.")
            x1, y1, x2, y2 = map(float, region)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Provider exclusion regions must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.")
            self.regions.append((x1, y1, x2, y2))

    @property
    def enabled(self):
        return bool(self.regions)

    def pixel_regions(self, width, height):
        return [(round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height))
                for x1, y1, x2, y2 in self.regions]

    def apply(self, frame):
        if not self.regions:
            return frame
        masked = frame.copy()
        height, width = masked.shape[:2]
        for x1, y1, x2, y2 in self.pixel_regions(width, height):
            masked[y1:y2, x1:x2] = 0
        return masked

    def filter_detections(self, boxes, width, height):
        """Return kept indices and auditable generic overlay decisions.

        A detection is excluded when its center is masked or most of its own
        area is covered. A large person box that only clips a corner remains.
        """
        regions = self.pixel_regions(width, height)
        kept = []
        removed = []
        retained_overlaps = []
        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            maximum_overlap = 0.0
            center_inside = False
            for rx1, ry1, rx2, ry2 in regions:
                intersection = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(
                    0.0, min(y2, ry2) - max(y1, ry1))
                maximum_overlap = max(maximum_overlap, intersection / area if area else 0.0)
                center_inside |= rx1 <= center_x <= rx2 and ry1 <= center_y <= ry2
            decision = {"index": index, "center_inside": center_inside,
                        "maximum_detection_overlap": maximum_overlap}
            if center_inside or maximum_overlap >= self.post_detection_overlap_fraction:
                removed.append(decision)
            else:
                kept.append(index)
                if maximum_overlap > 0:
                    retained_overlaps.append(decision)
        return kept, {"removed": removed, "retained_overlaps": retained_overlaps}
