"""Interpretable per-observation and per-tracklet team/role classification."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

from provider_overlay import ProviderOverlayMask
from team_profiles import GameTeamProfiles


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


class OfficialTemporalState:
    """Reversible structural-role evidence scoped to one temporary tracklet."""

    def __init__(self, configuration=None):
        configuration = configuration or {}
        self.establish_window = configuration.get("establish_window", 12)
        self.establish_count = configuration.get("establish_count", 4)
        self.minimum_stripe_ratio = configuration.get("minimum_stripe_ratio", 1.2)
        self.reverse_window = configuration.get("reverse_window", 15)
        self.reverse_count = configuration.get("reverse_count", 12)
        self.contradiction_confidence = configuration.get("contradiction_confidence", .75)
        self.maximum_contradiction_stripe_ratio = configuration.get("maximum_contradiction_stripe_ratio", .75)
        self.maximum_player_lower_body_dark_fraction = configuration.get(
            "maximum_player_lower_body_dark_fraction", .25)
        for value, name in ((self.establish_window, "establish_window"),
                            (self.establish_count, "establish_count"),
                            (self.reverse_window, "reverse_window"),
                            (self.reverse_count, "reverse_count")):
            if type(value) is not int or value <= 0:
                raise ValueError(f"official_temporal.{name} must be a positive integer.")
        if self.establish_count > self.establish_window or self.reverse_count > self.reverse_window:
            raise ValueError("Official temporal counts cannot exceed their windows.")
        for value, name in ((self.minimum_stripe_ratio, "minimum_stripe_ratio"),
                            (self.contradiction_confidence, "contradiction_confidence"),
                            (self.maximum_contradiction_stripe_ratio, "maximum_contradiction_stripe_ratio"),
                            (self.maximum_player_lower_body_dark_fraction,
                             "maximum_player_lower_body_dark_fraction")):
            if not _finite_number(value) or value < 0:
                raise ValueError(f"official_temporal.{name} must be a nonnegative finite number.")
        self.official = deque(maxlen=self.establish_window)
        self.contradictions = deque(maxlen=self.reverse_window)
        self.confirmed = False
        self.established_count = 0
        self.reverted_count = 0

    def update(self, observation_label, observation_confidence, evidence):
        stripe_ratio = float(evidence.get("stripe_ratio", 0.0))
        lower_body_dark = evidence.get("stripe_evidence", {}).get("lower_body_dark_fraction")
        establishment_usable = evidence.get("official_establishment_usable", True)
        structural = (observation_label == OFFICIAL
                      and stripe_ratio >= self.minimum_stripe_ratio
                      and establishment_usable)
        # HOME/AWAY colour alone is not a role contradiction: an official's
        # white sweater commonly satisfies a generic HOME profile whenever
        # stripes are blurred or occluded. A clearly non-dark lower body is
        # positive player structure because officials are established using
        # referee stripes together with dark lower-body evidence.
        positive_player_structure = (
            _finite_number(lower_body_dark)
            and lower_body_dark <= self.maximum_player_lower_body_dark_fraction)
        contradictory = (observation_label in (HOME, AWAY)
                         and observation_confidence >= self.contradiction_confidence
                         and stripe_ratio <= self.maximum_contradiction_stripe_ratio
                         and positive_player_structure)
        transition = None
        if not self.confirmed:
            self.official.append(structural)
            if sum(self.official) >= self.establish_count:
                self.confirmed = True
                self.established_count += 1
                self.contradictions.clear()
                transition = "ESTABLISHED"
        else:
            # UNKNOWN and weak/unstructured observations are deliberately inert.
            self.contradictions.append(contradictory)
            if sum(self.contradictions) >= self.reverse_count:
                self.confirmed = False
                self.reverted_count += 1
                self.official.clear()
                transition = "REVERTED"
        return dict(confirmed=self.confirmed, transition=transition,
                    structural_official=structural, strong_contradiction=contradictory,
                    establishment_usable=establishment_usable,
                    positive_player_structure=positive_player_structure,
                    establish_support=sum(self.official), establish_window=len(self.official),
                    contradiction_support=sum(self.contradictions),
                    contradiction_window=len(self.contradictions))


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
        game_profiles = configuration.get("game_team_profiles")
        self.game_profiles = GameTeamProfiles(game_profiles) if game_profiles else None
        self.stripe = configuration.get("official_stripe_cue", {})
        self.participant_filter = configuration.get("participant_filter", {})
        self.kickplate_boundary = configuration.get("kickplate_boundary", {})
        self.near_board_boundary = configuration.get("near_board_boundary", {})
        self.provider = configuration.get("provider", {})
        ProviderOverlayMask(self.provider)  # Validate configuration before a long video run.
        self.official_temporal_configuration = configuration.get("official_temporal", {})
        self._validate()
        self.scores = defaultdict(lambda: {state: 0.0 for state in TEAM_STATES[:-1]})
        self.observations = defaultdict(int)
        self.official_states = defaultdict(lambda: OfficialTemporalState(self.official_temporal_configuration))

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
    def _region(frame, box, x_range, y_range):
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        left = max(0, round(x1 + x_range[0] * width))
        right = min(frame.shape[1], round(x1 + x_range[1] * width))
        top = max(0, round(y1 + y_range[0] * height))
        bottom = min(frame.shape[0], round(y1 + y_range[1] * height))
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

    def observe(self, frame, box, cv2, detection_confidence=None):
        crop = self._crop(frame, box)
        if crop is None or crop.size < self.min_pixels * crop.shape[2]:
            return UNKNOWN, 0.0, {"reason": "empty_or_small_crop", "pixels": 0 if crop is None else int(crop.shape[0] * crop.shape[1])}
        import numpy as np
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        fractions = {label: self._profile_fraction(hsv, profile, np)
                     for label, profile in self.profiles.items()}
        stripe_score = 0.0
        stripe_evidence = {}
        pants = self._region(frame, box, (.20, .80), (.50, .75))
        sock_parts = [self._region(frame, box, x_range, (.62, .93))
                      for x_range in ((.12, .43), (.57, .88))]
        socks = np.concatenate(sock_parts, axis=1) if all(part.size for part in sock_parts) else None
        lower = self._region(frame, box, (.12, .88), (.50, .93))
        lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV) if lower.size else None
        pants_hsv = cv2.cvtColor(pants, cv2.COLOR_BGR2HSV) if pants.size else None
        socks_hsv = cv2.cvtColor(socks, cv2.COLOR_BGR2HSV) if socks is not None else None
        lower_dark = float(np.mean(lower_hsv[:, :, 2] <= self.stripe.get("maximum_dark_value", 80))) if lower_hsv is not None else 0.0
        pants_dark = float(np.mean(pants_hsv[:, :, 2] <= self.stripe.get("maximum_dark_value", 80))) if pants_hsv is not None else 0.0
        sock_evidence = ({"available": True, "pixels": int(socks.shape[0] * socks.shape[1]),
                          "profile_fractions": {label: self._profile_fraction(socks_hsv, self.profiles[label], np)
                                                for label in (HOME, AWAY)},
                          "dark_fraction": float(np.mean(socks_hsv[:, :, 2] <= self.stripe.get("maximum_dark_value", 80))),
                          "saturated_fraction": float(np.mean(socks_hsv[:, :, 1] >= 65))}
                         if socks_hsv is not None and socks.shape[0] * socks.shape[1] >= self.min_pixels
                         else {"available": False, "pixels": 0 if socks is None else int(socks.shape[0] * socks.shape[1])})
        if self.stripe:
            value = hsv[:, :, 2]
            saturation = hsv[:, :, 1]
            bright = float(np.mean((saturation <= self.stripe.get("maximum_saturation", 70))
                                   & (value >= self.stripe.get("minimum_bright_value", 110))))
            dark = float(np.mean(value <= self.stripe.get("maximum_dark_value", 80)))
            # An x-direction gradient measures left/right intensity changes: the
            # edges produced by vertical jersey stripes.  The y-gradient is kept
            # as a competing orientation so horizontal bands/equipment do not
            # masquerade as referee stripes.
            vertical_edges = float(np.mean(np.abs(np.diff(gray.astype(np.int16), axis=1))
                                           >= self.stripe.get("minimum_edge_delta", 35)))
            horizontal_edges = float(np.mean(np.abs(np.diff(gray.astype(np.int16), axis=0))
                                             >= self.stripe.get("minimum_edge_delta", 35)))
            orientation_ratio = vertical_edges / max(horizontal_edges, 1 / gray.size)
            edges_by_row = np.sum(np.abs(np.diff(gray.astype(np.int16), axis=1))
                                  >= self.stripe.get("minimum_edge_delta", 35), axis=1)
            striped_row_fraction = float(np.mean(edges_by_row >= self.stripe.get("minimum_edges_per_row", 3)))
            requirements = (bright / self.stripe.get("minimum_bright_fraction", .25),
                            dark / self.stripe.get("minimum_dark_fraction", .15),
                            vertical_edges / self.stripe.get("minimum_vertical_edge_fraction", .025),
                            orientation_ratio / self.stripe.get("minimum_vertical_orientation_ratio", 1.25),
                            striped_row_fraction / self.stripe.get("minimum_striped_row_fraction", .35),
                            lower_dark / self.stripe.get("minimum_lower_body_dark_fraction", .30))
            stripe_score = min(requirements)
            stripe_evidence = {"bright_fraction": bright, "dark_fraction": dark,
                               "vertical_edge_fraction": vertical_edges,
                               "horizontal_edge_fraction": horizontal_edges,
                               "vertical_orientation_ratio": orientation_ratio,
                               "striped_row_fraction": striped_row_fraction,
                               "lower_body_dark_fraction": lower_dark,
                               "pants_dark_fraction": pants_dark,
                               "requirements": list(requirements)}
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        minimum_height_fraction = self.stripe.get("minimum_establishment_bbox_height_fraction", .06)
        official_quality = {
            "detection_confidence": detection_confidence,
            "bbox_height_fraction": box_height / frame.shape[0],
            "bbox_aspect_ratio": box_width / box_height,
            "edge_truncated": (box[0] <= 1 or box[1] <= 1
                               or box[2] >= frame.shape[1] - 1
                               or box[3] >= frame.shape[0] - 1),
        }
        official_establishment_usable = (
            True if detection_confidence is None else
            detection_confidence >= self.stripe.get("minimum_establishment_detection_confidence", .5)
            and official_quality["bbox_height_fraction"] >= minimum_height_fraction
            and .20 <= official_quality["bbox_aspect_ratio"] <= self.stripe.get(
                "maximum_establishment_bbox_aspect_ratio", 1.20)
            and not official_quality["edge_truncated"])
        ratios = {label: fractions[label] / self.profiles[label]["minimum_fraction"]
                  for label in (HOME, AWAY)}
        # An OFFICIAL profile is supporting diagnostic evidence only.  A hue can
        # describe a league's referee sweater, but it cannot establish the role:
        # solid practice jerseys can share that hue.  Referee-like stripe
        # structure is therefore the sole gate for OFFICIAL.
        ratios[OFFICIAL] = stripe_score * (self.stripe.get("qualified_role_boost", 1.2)
                                           if stripe_score >= 1 else 1)
        if not official_establishment_usable:
            ratios[OFFICIAL] = 0.0
        if self.game_profiles is not None and (stripe_score < 1 or not official_establishment_usable):
            lightness = lab[:, :, 0].astype(float) / 255
            uniform_feature = [float(np.median(lightness)), float(np.quantile(lightness, .25)),
                               float(np.quantile(lightness, .75))]
            best_label, confidence, game_evidence = self.game_profiles.classify(uniform_feature)
            best_ratio = confidence * 2
            second_ratio = 0.0
        else:
            ordered = sorted(ratios.items(), key=lambda item: (-item[1], item[0]))
            best_label, best_ratio = ordered[0]
            second_ratio = ordered[1][1]
            confidence = min(1.0, best_ratio / 2.0)
            game_evidence = None
        evidence = {"pixels": int(crop.shape[0] * crop.shape[1]), "profile_fractions": fractions,
                    "profile_ratios": ratios, "stripe_ratio": stripe_score,
                    "stripe_evidence": stripe_evidence,
                    "sock_evidence": sock_evidence,
                    "game_team_profile": game_evidence,
                    "official_quality": official_quality,
                    "official_establishment_usable": official_establishment_usable,
                    "margin": best_ratio - second_ratio}
        if (best_label == UNKNOWN or confidence < self.minimum_confidence
                or (self.game_profiles is None and
                    (best_ratio < 1 or best_ratio - second_ratio < self.ambiguity_margin))):
            return UNKNOWN, confidence, evidence
        return best_label, confidence, evidence

    def classify(self, track_id, frame, box, cv2, detection_confidence=None):
        observation_label, observation_confidence, evidence = self.observe(
            frame, box, cv2, detection_confidence)
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
        role = self.official_states[int(track_id)].update(
            observation_label, observation_confidence, evidence)
        if role["confirmed"]:
            label = OFFICIAL
            confidence = max(confidence, min(1.0, role["establish_support"] / self.official_states[int(track_id)].establish_count))
        evidence = {**evidence, "track_scores": dict(track_scores),
                    "track_observations": self.observations[int(track_id)],
                    "official_temporal": role}
        return TeamResult(label, confidence, observation_label, observation_confidence, evidence)
