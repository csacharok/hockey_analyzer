"""Interpretable per-observation and per-tracklet team/role classification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path


HOME = "HOME"
AWAY = "AWAY"
OFFICIAL = "OFFICIAL"
UNKNOWN = "UNKNOWN"
TEAM_STATES = (HOME, AWAY, OFFICIAL, UNKNOWN)


@dataclass(frozen=True)
class TeamResult:
    label: str
    confidence: float
    observation_label: str
    observation_confidence: float
    evidence: dict


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class TeamClassifier:
    """Classify a normalized torso crop, then accumulate evidence by track ID.

    Color profiles are deliberately external configuration: HOME and AWAY colors
    vary by game. Track history is local to a ByteTrack ID and is never treated as
    persistent player identity.
    """

    def __init__(self, configuration):
        if not isinstance(configuration, dict):
            raise ValueError("Team configuration must be a JSON object.")
        self.crop = configuration.get("torso_crop", {"x": [.18, .82], "y": [.12, .62]})
        self.min_pixels = configuration.get("minimum_pixels", 300)
        self.minimum_confidence = configuration.get("minimum_confidence", .55)
        self.ambiguity_margin = configuration.get("ambiguity_margin", .15)
        self.decay = configuration.get("track_evidence_decay", .98)
        self.profiles = configuration.get("profiles")
        self.stripe = configuration.get("official_stripe_cue", {})
        self._validate()
        self.scores = defaultdict(lambda: {state: 0.0 for state in TEAM_STATES[:-1]})
        self.observations = defaultdict(int)

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Cannot load team configuration {path}: {error}") from error

    def _validate(self):
        try:
            x, y = self.crop["x"], self.crop["y"]
        except (KeyError, TypeError):
            raise ValueError("torso_crop requires x and y normalized ranges.") from None
        if not (len(x) == len(y) == 2 and 0 <= x[0] < x[1] <= 1 and 0 <= y[0] < y[1] <= 1):
            raise ValueError("torso_crop ranges must be increasing values from zero to one.")
        if type(self.min_pixels) is not int or self.min_pixels <= 0:
            raise ValueError("minimum_pixels must be a positive integer.")
        for value, name in ((self.minimum_confidence, "minimum_confidence"),
                            (self.ambiguity_margin, "ambiguity_margin"),
                            (self.decay, "track_evidence_decay")):
            if not _finite_number(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one.")
        if not isinstance(self.profiles, dict) or set(self.profiles) != {HOME, AWAY, OFFICIAL}:
            raise ValueError("profiles must define exactly HOME, AWAY, and OFFICIAL.")
        for label, profile in self.profiles.items():
            if not isinstance(profile.get("masks"), list) or not profile["masks"]:
                raise ValueError(f"{label} requires at least one HSV mask.")
            threshold = profile.get("minimum_fraction")
            if not _finite_number(threshold) or not 0 < threshold <= 1:
                raise ValueError(f"{label} minimum_fraction must be in (0, 1].")
            for mask in profile["masks"]:
                if not isinstance(mask.get("hue"), list) or not mask["hue"]:
                    raise ValueError(f"{label} masks require hue ranges.")
                for low, high in mask["hue"]:
                    if not (0 <= low <= high <= 179):
                        raise ValueError("OpenCV hue ranges must lie between 0 and 179.")
                for channel in ("saturation", "value"):
                    bounds = mask.get(channel)
                    if not (isinstance(bounds, list) and len(bounds) == 2
                            and 0 <= bounds[0] <= bounds[1] <= 255):
                        raise ValueError(f"HSV {channel} range must lie between 0 and 255.")

    def _crop(self, frame, box):
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return None
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite_number(v) for v in box):
            return None
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        width, height = x2 - x1, y2 - y1
        left = max(0, round(x1 + self.crop["x"][0] * width))
        right = min(frame.shape[1], round(x1 + self.crop["x"][1] * width))
        top = max(0, round(y1 + self.crop["y"][0] * height))
        bottom = min(frame.shape[0], round(y1 + self.crop["y"][1] * height))
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    @staticmethod
    def _profile_fraction(hsv, profile, np):
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        selected = np.zeros(hue.shape, dtype=bool)
        for mask in profile["masks"]:
            hue_selected = np.zeros(hue.shape, dtype=bool)
            for low, high in mask["hue"]:
                hue_selected |= (hue >= low) & (hue <= high)
            selected |= (hue_selected & (saturation >= mask["saturation"][0])
                         & (saturation <= mask["saturation"][1])
                         & (value >= mask["value"][0]) & (value <= mask["value"][1]))
        return float(np.mean(selected))

    def observe(self, frame, box, cv2):
        crop = self._crop(frame, box)
        if crop is None or crop.size < self.min_pixels * crop.shape[2]:
            return UNKNOWN, 0.0, {"reason": "empty_or_small_crop", "pixels": 0 if crop is None else int(crop.shape[0] * crop.shape[1])}
        import numpy as np
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        fractions = {label: self._profile_fraction(hsv, profile, np)
                     for label, profile in self.profiles.items()}
        stripe_score = 0.0
        if self.stripe:
            value = hsv[:, :, 2]
            saturation = hsv[:, :, 1]
            bright = float(np.mean((saturation <= self.stripe.get("maximum_saturation", 70))
                                   & (value >= self.stripe.get("minimum_bright_value", 110))))
            dark = float(np.mean(value <= self.stripe.get("maximum_dark_value", 80)))
            edges = float(np.mean(np.abs(np.diff(gray.astype(np.int16), axis=1))
                                  >= self.stripe.get("minimum_edge_delta", 35)))
            requirements = (bright / self.stripe.get("minimum_bright_fraction", .25),
                            dark / self.stripe.get("minimum_dark_fraction", .15),
                            edges / self.stripe.get("minimum_edge_fraction", .025))
            stripe_score = min(requirements)
        ratios = {label: fractions[label] / self.profiles[label]["minimum_fraction"]
                  for label in TEAM_STATES[:-1]}
        ratios[OFFICIAL] = max(ratios[OFFICIAL], stripe_score)
        ordered = sorted(ratios.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_ratio = ordered[0]
        second_ratio = ordered[1][1]
        confidence = min(1.0, best_ratio / 2.0)
        evidence = {"pixels": int(crop.shape[0] * crop.shape[1]), "profile_fractions": fractions,
                    "profile_ratios": ratios, "stripe_ratio": stripe_score,
                    "margin": best_ratio - second_ratio}
        if (best_ratio < 1 or confidence < self.minimum_confidence
                or best_ratio - second_ratio < self.ambiguity_margin):
            return UNKNOWN, confidence, evidence
        return best_label, confidence, evidence

    def classify(self, track_id, frame, box, cv2):
        observation_label, observation_confidence, evidence = self.observe(frame, box, cv2)
        track_scores = self.scores[int(track_id)]
        for label in track_scores:
            track_scores[label] *= self.decay
        if observation_label != UNKNOWN:
            track_scores[observation_label] += observation_confidence
        self.observations[int(track_id)] += 1
        ordered = sorted(track_scores.items(), key=lambda item: (-item[1], item[0]))
        best_label, best_score = ordered[0]
        second_score = ordered[1][1]
        total = sum(track_scores.values())
        confidence = best_score / total if total else 0.0
        label = best_label if (best_score >= self.minimum_confidence
                               and confidence >= self.minimum_confidence
                               and confidence - (second_score / total if total else 0) >= self.ambiguity_margin) else UNKNOWN
        evidence = {**evidence, "track_scores": dict(track_scores), "track_observations": self.observations[int(track_id)]}
        return TeamResult(label, confidence, observation_label, observation_confidence, evidence)
