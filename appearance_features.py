"""Semantic-free appearance measurements for person observations and tracklets."""

from __future__ import annotations

from collections import defaultdict
import math


SCHEMA_VERSION = 1
MEASUREMENT_QUALITY_MAP = {"GOOD": "MEASURABLE", "LOW": "PARTIAL", "UNUSABLE": "UNUSABLE"}


def measurement_quality(quality):
    """Return measurability terminology without implying training suitability."""
    if quality in MEASUREMENT_QUALITY_MAP.values():
        return quality
    return MEASUREMENT_QUALITY_MAP.get(quality, "UNUSABLE")
BODY_REGIONS = {
    # Central crops reduce neighboring-player and background contamination.
    "torso": {"x": (0.18, 0.82), "y": (0.12, 0.50)},
    "pants": {"x": (0.20, 0.80), "y": (0.50, 0.68)},
    # The gap omits the stick/crotch area and measures the two sock/leg columns.
    "lower_legs_left": {"x": (0.12, 0.43), "y": (0.62, 0.93)},
    "lower_legs_right": {"x": (0.57, 0.88), "y": (0.62, 0.93)},
}


def normalized_region_bounds(box, x_range, y_range):
    """Return un-clipped integer bounds for a normalized bbox region."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return (round(x1 + x_range[0] * width), round(y1 + y_range[0] * height),
            round(x1 + x_range[1] * width), round(y1 + y_range[1] * height))


def _finite_box(box):
    return (isinstance(box, (list, tuple)) and len(box) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) for value in box)
            and box[2] > box[0] and box[3] > box[1])


def _extract_region(frame, box, definition):
    raw = normalized_region_bounds(box, definition["x"], definition["y"])
    left, top, right, bottom = raw
    expected = max(0, right - left) * max(0, bottom - top)
    clipped = (max(0, left), max(0, top), min(frame.shape[1], right), min(frame.shape[0], bottom))
    left, top, right, bottom = clipped
    crop = frame[top:bottom, left:right] if right > left and bottom > top else None
    actual = 0 if crop is None else crop.shape[0] * crop.shape[1]
    return crop, list(clipped), (actual / expected if expected else 0.0)


def _measure(crop, coverage, cv2, np, minimum_pixels):
    pixels = 0 if crop is None else int(crop.shape[0] * crop.shape[1])
    usable = pixels >= minimum_pixels and coverage >= 0.80
    result = {"usable": usable, "pixels": pixels, "usable_pixel_fraction": float(coverage)}
    if not usable:
        return result
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0].astype(float) / 255.0
    saturation = hsv[:, :, 1].astype(float) / 255.0
    value = hsv[:, :, 2].astype(float) / 255.0
    result.update({
        "lab_lightness_median": float(np.median(lightness)),
        "lab_lightness_q25": float(np.quantile(lightness, .25)),
        "lab_lightness_q75": float(np.quantile(lightness, .75)),
        "hsv_saturation_median": float(np.median(saturation)),
        "hsv_value_median": float(np.median(value)),
        "dark_fraction": float(np.mean(hsv[:, :, 2] <= 80)),
        "light_fraction": float(np.mean((hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 110))),
        "saturated_fraction": float(np.mean(hsv[:, :, 1] >= 65)),
    })
    return result


def _structure(torso, torso_measurement, cv2, np):
    result = {"available": False, "vertical_stripe_score": None}
    if torso is None or not torso_measurement["usable"]:
        return result
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY).astype(np.int16)
    x_edges = np.abs(np.diff(gray, axis=1)) >= 35
    y_edges = np.abs(np.diff(gray, axis=0)) >= 35
    vertical = float(np.mean(x_edges)) if x_edges.size else 0.0
    horizontal = float(np.mean(y_edges)) if y_edges.size else 0.0
    row_counts = np.sum(x_edges, axis=1) if x_edges.size else np.zeros(gray.shape[0])
    supporting_rows = int(np.sum(row_counts >= 3))
    row_fraction = supporting_rows / max(1, gray.shape[0])
    orientation = vertical / max(horizontal, 1 / max(1, gray.size))
    requirements = [torso_measurement["light_fraction"] / .25,
                    torso_measurement["dark_fraction"] / .15,
                    vertical / .025, orientation / 1.25, row_fraction / .35]
    return {
        "available": True,
        "vertical_stripe_score": float(min(requirements)),
        "vertical_edge_fraction": vertical,
        "horizontal_edge_fraction": horizontal,
        "vertical_orientation_ratio": float(orientation),
        "stripe_supporting_rows": supporting_rows,
        "stripe_total_rows": int(gray.shape[0]),
        "stripe_supporting_pixel_edges": int(np.sum(x_edges)),
        "striped_row_fraction": float(row_fraction),
    }


class AppearanceFeatureExtractor:
    """Extract normalized, interpretable visual evidence without team labels."""

    def __init__(self, minimum_region_pixels=64, minimum_bbox_height_fraction=.035):
        self.minimum_region_pixels = minimum_region_pixels
        self.minimum_bbox_height_fraction = minimum_bbox_height_fraction

    def extract(self, frame, box, cv2, detection_confidence=None):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2 or not _finite_box(box):
            return {"schema_version": SCHEMA_VERSION, "usable": False,
                    "reason": "invalid_frame_or_box", "regions": {}, "structure": {},
                    "quality": {"crop_quality": "UNUSABLE", "measurement_quality": "UNUSABLE"}}
        import numpy as np
        crops, bounds, measurements = {}, {}, {}
        for name, definition in BODY_REGIONS.items():
            crop, region_bounds, coverage = _extract_region(frame, box, definition)
            crops[name], bounds[name] = crop, region_bounds
            measurements[name] = _measure(crop, coverage, cv2, np, self.minimum_region_pixels)
        left = crops["lower_legs_left"]
        right = crops["lower_legs_right"]
        lower_coverage = min(measurements["lower_legs_left"]["usable_pixel_fraction"],
                             measurements["lower_legs_right"]["usable_pixel_fraction"])
        lower = np.concatenate((left, right), axis=1) if (left is not None and right is not None
                                                           and left.shape[0] == right.shape[0]) else None
        measurements["lower_legs"] = _measure(lower, lower_coverage, cv2, np,
                                                self.minimum_region_pixels)
        del measurements["lower_legs_left"], measurements["lower_legs_right"]
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        edge_truncated = box[0] < 0 or box[1] < 0 or box[2] > frame.shape[1] or box[3] > frame.shape[0]
        usable_regions = sum(measurements[name]["usable"] for name in ("torso", "pants", "lower_legs"))
        quality = {
            "detection_confidence": detection_confidence,
            "normalized_bbox_height": float(box_height / frame.shape[0]),
            "bbox_aspect_ratio": float(box_width / box_height),
            "frame_edge_truncated": edge_truncated,
            "usable_region_count": usable_regions,
        }
        quality["crop_quality"] = ("GOOD" if usable_regions == 3 and not edge_truncated
                                   and quality["normalized_bbox_height"] >= self.minimum_bbox_height_fraction
                                   and (detection_confidence is None or detection_confidence >= .5)
                                   else "LOW" if usable_regions else "UNUSABLE")
        # crop_quality is retained for schema-v1 compatibility.  This field says
        # only whether measurements can be extracted, never whether a sample is
        # suitable for training or what semantic class it belongs to.
        quality["measurement_quality"] = measurement_quality(quality["crop_quality"])
        return {
            "schema_version": SCHEMA_VERSION,
            "usable": usable_regions > 0,
            "regions": {name: measurements[name] for name in ("torso", "pants", "lower_legs")},
            "region_bounds": {"torso": bounds["torso"], "pants": bounds["pants"],
                              "lower_legs_left": bounds["lower_legs_left"],
                              "lower_legs_right": bounds["lower_legs_right"]},
            "structure": _structure(crops["torso"], measurements["torso"], cv2, np),
            "quality": quality,
        }


class TrackletAppearanceAggregator:
    """Robustly aggregate only usable measurements within temporary track IDs."""

    def __init__(self):
        self._tracks = defaultdict(list)

    def add(self, track_id, features):
        if features.get("usable"):
            self._tracks[track_id].append(features)

    def summaries(self):
        import numpy as np
        summaries = []
        fields = (("torso", "lab_lightness_median"), ("torso", "dark_fraction"),
                  ("pants", "dark_fraction"), ("lower_legs", "lab_lightness_median"),
                  ("lower_legs", "dark_fraction"))
        for track_id, observations in sorted(self._tracks.items()):
            result = {"track_id": track_id, "usable_observations": len(observations), "medians": {},
                      "usable_region_fractions": {}}
            for region in ("torso", "pants", "lower_legs"):
                result["usable_region_fractions"][region] = (sum(
                    item["regions"][region]["usable"] for item in observations) / len(observations))
            for region, field in fields:
                values = [item["regions"][region][field] for item in observations
                          if item["regions"][region].get("usable") and field in item["regions"][region]]
                result["medians"][f"{region}_{field}"] = float(np.median(values)) if values else None
            stripe = [item["structure"]["vertical_stripe_score"] for item in observations
                      if item["structure"].get("available")]
            result["medians"]["vertical_stripe_score"] = float(np.median(stripe)) if stripe else None
            summaries.append(result)
        return summaries
