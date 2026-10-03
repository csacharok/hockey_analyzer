"""Conservative participant filtering, separate from team classification."""

from __future__ import annotations

from dataclasses import dataclass
import math

from rink_geometry import OFF_ICE, ON_ICE


ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"
UNCERTAIN = "UNCERTAIN"
PARTICIPANT_STATES = (ACCEPTED, REJECTED, UNCERTAIN)


@dataclass(frozen=True)
class ParticipantResult:
    state: str
    accepted: bool
    reason: str
    evidence: dict


class ParticipantFilter:
    """Check whether a tracked person's foot neighborhood resembles ice.

    Geometry remains the first cue.  The appearance check is deliberately
    conservative: ambiguous observations remain in the participant pipeline.
    """

    def __init__(self, configuration=None):
        configuration = configuration or {}
        self.maximum_saturation = configuration.get("maximum_ice_saturation", 75)
        self.minimum_value = configuration.get("minimum_ice_value", 105)
        self.reject_below = configuration.get("reject_below_ice_fraction", .12)
        self.accept_at = configuration.get("accept_at_ice_fraction", .35)
        self.minimum_pixels = configuration.get("minimum_contact_pixels", 40)
        values = (self.maximum_saturation, self.minimum_value, self.minimum_pixels)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Participant pixel thresholds must be nonnegative integers.")
        if not (isinstance(self.reject_below, (int, float))
                and isinstance(self.accept_at, (int, float))
                and math.isfinite(self.reject_below) and math.isfinite(self.accept_at)
                and 0 <= self.reject_below < self.accept_at <= 1):
            raise ValueError("Participant ice fractions must satisfy 0 <= reject < accept <= 1.")

    @staticmethod
    def _contact_crop(frame, box):
        x1, y1, x2, y2 = box
        height, width = frame.shape[:2]
        box_width, box_height = x2 - x1, y2 - y1
        left = max(0, round(x1 - .25 * box_width))
        right = min(width, round(x2 + .25 * box_width))
        top = max(0, round(y2 - .08 * box_height))
        bottom = min(height, round(y2 + .12 * box_height))
        return frame[top:bottom, left:right], (left, top, right, bottom)

    def classify(self, frame, box, rink_state, cv2, boundary=None, near_boundary=None,
                 use_polygon=True):
        base = {"rink_state": rink_state}
        contact_x = (box[0] + box[2]) / 2
        contact_y = box[3]
        boundary_y = boundary.y_at(contact_x) if boundary is not None else None
        near_boundary_y = near_boundary.y_at(contact_x) if near_boundary is not None else None
        base.update(boundary_coverage=None if boundary is None else boundary.coverage,
                    boundary_y=boundary_y,
                    boundary_signed_distance=None if boundary_y is None else contact_y - boundary_y,
                    near_boundary_coverage=None if near_boundary is None else near_boundary.coverage,
                    near_boundary_y=near_boundary_y,
                    near_boundary_signed_distance=(None if near_boundary_y is None
                                                   else near_boundary_y - contact_y))
        margin = boundary.margin_px if boundary is not None else 6
        if boundary_y is not None and contact_y < boundary_y - margin:
            return ParticipantResult(REJECTED, False, "contact_point_behind_kickplate", base)
        near_margin = near_boundary.margin_px if near_boundary is not None else 8
        if near_boundary_y is not None and contact_y > near_boundary_y + near_margin:
            return ParticipantResult(REJECTED, False, "contact_point_beyond_near_boards", base)
        if boundary_y is None and near_boundary_y is None and use_polygon and rink_state == OFF_ICE:
            return ParticipantResult(REJECTED, False, "contact_point_outside_rink_polygon", base)
        if boundary_y is None and near_boundary_y is None and (not use_polygon or rink_state != ON_ICE):
            return ParticipantResult(UNCERTAIN, True, "rink_geometry_uncertain", base)
        crop, region = self._contact_crop(frame, box)
        pixels = int(crop.shape[0] * crop.shape[1]) if crop.size else 0
        evidence = {**base, "contact_region": region, "contact_pixels": pixels}
        if pixels < self.minimum_pixels:
            return ParticipantResult(UNCERTAIN, True, "insufficient_contact_pixels", evidence)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        import numpy as np
        ice_fraction = float(np.mean((hsv[:, :, 1] <= self.maximum_saturation)
                                     & (hsv[:, :, 2] >= self.minimum_value)))
        evidence["ice_like_fraction"] = ice_fraction
        if boundary_y is not None or near_boundary_y is not None:
            if ice_fraction < self.reject_below:
                return ParticipantResult(UNCERTAIN, True, "kickplate_ice_conflict", evidence)
            if ice_fraction < self.accept_at:
                return ParticipantResult(UNCERTAIN, True, "kickplate_playing_side_weak_ice", evidence)
            reason = ("visible_rink_corridor_and_ice" if boundary_y is not None and near_boundary_y is not None
                      else "kickplate_playing_side_and_ice" if boundary_y is not None
                      else "near_boundary_playing_side_and_ice")
            return ParticipantResult(ACCEPTED, True, reason, evidence)
        if ice_fraction < self.reject_below:
            return ParticipantResult(REJECTED, False, "strong_non_ice_contact_evidence", evidence)
        if ice_fraction < self.accept_at:
            return ParticipantResult(UNCERTAIN, True, "weak_ice_contact_evidence", evidence)
        return ParticipantResult(ACCEPTED, True, "ice_contact_evidence", evidence)
