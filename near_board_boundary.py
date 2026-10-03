"""Camera-relative near-board boundary from edge and surface evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math

from kickplate_boundary import BoundaryEstimate


@dataclass(frozen=True)
class BoundaryCandidate:
    points: tuple[tuple[float, float], ...]
    ice_above_fraction: float
    ice_below_fraction: float
    contrast: float
    accepted: bool


class NearBoardBoundaryDetector:
    """Find a long edge with ice above and non-ice foreground below it.

    Colour is deliberately absent from the boundary definition.  Hough edges
    propose geometry; paired surface samples decide whether an edge separates
    the playing surface from the camera-side foreground.
    """

    def __init__(self, configuration=None):
        configuration = configuration or {}
        self.bins = configuration.get("bins", 48)
        self.maximum_gap_bins = configuration.get("maximum_gap_bins", 3)
        self.minimum_coverage = configuration.get("minimum_coverage", .18)
        self.maximum_saturation = configuration.get("maximum_ice_saturation", 75)
        self.minimum_value = configuration.get("minimum_ice_value", 105)
        self.minimum_ice_above = configuration.get("minimum_ice_above", .52)
        self.maximum_ice_below = configuration.get("maximum_ice_below", .38)
        self.minimum_ice_contrast = configuration.get("minimum_ice_contrast", .22)
        self.sample_offset_fraction = configuration.get("sample_offset_fraction", .025)
        self.contact_margin_px = configuration.get("contact_margin_px", 8)
        self.temporal_alpha = configuration.get("temporal_alpha", .65)
        self.detection_scale = configuration.get("detection_scale", .25)
        self.maximum_temporal_gap_frames = configuration.get("maximum_temporal_gap_frames", 2)
        self.previous = None
        self.missed_frames = 0
        if type(self.bins) is not int or self.bins < 8:
            raise ValueError("Near-board bins must be an integer of at least 8.")
        for name in ("minimum_coverage", "minimum_ice_above", "maximum_ice_below",
                     "minimum_ice_contrast", "sample_offset_fraction", "temporal_alpha"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")
        if not isinstance(self.detection_scale, (int, float)) or not 0 < self.detection_scale <= 1:
            raise ValueError("detection_scale must be in (0, 1].")
        if type(self.maximum_temporal_gap_frames) is not int or self.maximum_temporal_gap_frames < 0:
            raise ValueError("maximum_temporal_gap_frames must be a nonnegative integer.")

    def _ice_mask(self, frame, cv2, hsv=None):
        import numpy as np
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if hsv is None else hsv
        return ((hsv[:, :, 1] <= self.maximum_saturation)
                & (hsv[:, :, 2] >= self.minimum_value)).astype(np.uint8)

    @staticmethod
    def _surface_fraction(mask, points, offset, side):
        height, width = mask.shape
        values = []
        for x, y in points:
            cx, cy = round(x), round(y + side * offset)
            radius = max(2, offset // 3)
            crop = mask[max(0, cy - radius):min(height, cy + radius + 1),
                        max(0, cx - radius):min(width, cx + radius + 1)]
            if crop.size:
                values.append(float(crop.mean()))
        return sum(values) / len(values) if values else 0.0

    def evaluate_polyline(self, frame, points, cv2, ice_mask=None):
        """Score supplied edge geometry; public for deterministic diagnostics."""
        offset = max(4, round(frame.shape[0] * self.sample_offset_fraction))
        ice = self._ice_mask(frame, cv2) if ice_mask is None else ice_mask
        above = self._surface_fraction(ice, points, offset, -1)
        below = self._surface_fraction(ice, points, offset, 1)
        contrast = above - below
        accepted = (above >= self.minimum_ice_above and below <= self.maximum_ice_below
                    and contrast >= self.minimum_ice_contrast)
        return BoundaryCandidate(tuple(points), above, below, contrast, accepted)

    def estimate(self, frame, cv2, hsv=None):
        import numpy as np
        height, width = frame.shape[:2]
        proposal_frame = (frame if self.detection_scale == 1 else
                          cv2.resize(frame, None, fx=self.detection_scale,
                                     fy=self.detection_scale, interpolation=cv2.INTER_AREA))
        proposal_height, proposal_width = proposal_frame.shape[:2]
        gray = cv2.cvtColor(proposal_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 45, 130)
        edges[:round(.42 * proposal_height)] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=max(20, proposal_width // 35),
                                minLineLength=max(20, proposal_width // 18),
                                maxLineGap=max(6, proposal_width // 80))
        proposals = []
        ice_mask = None
        if lines is not None:
            for raw in lines.reshape(-1, 4):
                x1, y1, x2, y2 = (float(value) / self.detection_scale for value in raw)
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                # Perspective rails can be diagonal, but near-vertical glass
                # supports and player silhouettes are not boundary proposals.
                if abs(slope) > .75:
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                samples = max(5, round(length / max(12, width / self.bins)))
                points = tuple((x1 + (x2 - x1) * i / (samples - 1),
                                y1 + (y2 - y1) * i / (samples - 1))
                               for i in range(samples))
                # Surface validation is full-resolution, but its HSV ice mask
                # depends only on the frame, not on an individual line.
                if ice_mask is None:
                    ice_mask = (self._ice_mask(frame, cv2) if hsv is None
                                else self._ice_mask(frame, cv2, hsv))
                evidence = self.evaluate_polyline(frame, points, cv2, ice_mask)
                if evidence.accepted:
                    intercept = y1 - slope * x1
                    proposals.append((length * evidence.contrast, points, slope, intercept))

        # A single coherent rail family is safer than independently choosing
        # the lowest edge in each bin: people and open doors can also have a
        # valid local ice/non-ice transition but do not form a long boundary.
        if proposals:
            seed = max(proposals, key=lambda item: item[0])
            _, seed_points, seed_slope, seed_intercept = seed
            seed_mid_x = sum(point[0] for point in seed_points) / len(seed_points)
            coherent = []
            for proposal in proposals:
                score, points, slope, intercept = proposal
                mid_x = sum(point[0] for point in points) / len(points)
                seed_y = seed_slope * mid_x + seed_intercept
                candidate_y = slope * mid_x + intercept
                if abs(slope - seed_slope) <= .22 and abs(candidate_y - seed_y) <= .065 * height:
                    coherent.append(proposal)
            proposals = coherent

        bin_width = width / self.bins
        values = [[] for _ in range(self.bins)]
        for score, points, _, _ in proposals:
            for x, y in points:
                index = min(self.bins - 1, max(0, int(x / bin_width)))
                values[index].append((y, score))
        rows = [None] * self.bins
        for index, candidates in enumerate(values):
            if candidates:
                rows[index] = float(np.median([y for y, _ in candidates]))
        observed = sum(value is not None for value in rows)
        coverage = observed / self.bins
        if coverage < self.minimum_coverage:
            self.missed_frames += 1
            if (self.previous is not None and self.previous.points
                    and self.missed_frames <= self.maximum_temporal_gap_frames):
                return BoundaryEstimate(self.previous.points, coverage, bin_width,
                                        self.contact_margin_px)
            result = BoundaryEstimate((), coverage, bin_width, self.contact_margin_px)
            self.previous = result
            return result
        self.missed_frames = 0
        self._interpolate(rows)
        if self.previous is not None and self.previous.points:
            for index, value in enumerate(rows):
                if value is None:
                    continue
                old = self.previous.y_at((index + .5) * bin_width)
                if old is not None:
                    rows[index] = self.temporal_alpha * value + (1 - self.temporal_alpha) * old
        points = tuple(((index + .5) * bin_width, value)
                       for index, value in enumerate(rows) if value is not None)
        result = BoundaryEstimate(points, coverage, bin_width, self.contact_margin_px)
        self.previous = result
        return result

    def _interpolate(self, values):
        index = 0
        while index < len(values):
            if values[index] is not None:
                index += 1
                continue
            start = index
            while index < len(values) and values[index] is None:
                index += 1
            gap = index - start
            if start and index < len(values) and gap <= self.maximum_gap_bins:
                before, after = values[start - 1], values[index]
                for offset in range(gap):
                    values[start + offset] = before + (after - before) * (offset + 1) / (gap + 1)
