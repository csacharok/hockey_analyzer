import unittest

import cv2
import numpy as np

from kickplate_boundary import BoundaryEstimate
from near_board_boundary import NearBoardBoundaryDetector
from participant_filter import ACCEPTED, REJECTED, UNCERTAIN, ParticipantFilter
from rink_geometry import OFF_ICE, ON_ICE


class NearBoardBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.detector = NearBoardBoundaryDetector({"bins": 20, "minimum_coverage": .1,
                                                   "sample_offset_fraction": .08})

    @staticmethod
    def boundary_frame(below=(30, 40, 50)):
        frame = np.full((120, 240, 3), 210, dtype=np.uint8)
        frame[70:] = below
        cv2.line(frame, (0, 70), (239, 70), (20, 20, 20), 3)
        return frame

    def test_ice_above_and_non_ice_below_is_accepted(self):
        candidate = self.detector.evaluate_polyline(
            self.boundary_frame(), ((20, 70), (120, 70), (220, 70)), cv2)
        self.assertTrue(candidate.accepted)
        self.assertGreater(candidate.ice_above_fraction, candidate.ice_below_fraction)

    def test_painted_line_with_ice_on_both_sides_is_rejected(self):
        frame = np.full((120, 240, 3), 210, dtype=np.uint8)
        cv2.line(frame, (0, 70), (239, 70), (180, 70, 20), 5)
        candidate = self.detector.evaluate_polyline(
            frame, ((20, 70), (120, 70), (220, 70)), cv2)
        self.assertFalse(candidate.accepted)
        self.assertLess(candidate.contrast, .22)

    def test_short_occlusion_is_interpolated(self):
        values = [50., 51., None, None, 54., None, None, None, None, 60.]
        self.detector.maximum_gap_bins = 2
        self.detector._interpolate(values)
        self.assertIsNotNone(values[2])
        self.assertIsNotNone(values[3])
        self.assertIsNone(values[5])

    def test_short_temporal_occlusion_holds_neighboring_boundary(self):
        detector = NearBoardBoundaryDetector({"bins": 20, "minimum_coverage": .1,
                                              "maximum_temporal_gap_frames": 1,
                                              "detection_scale": 1})
        detected = detector.estimate(self.boundary_frame(), cv2)
        self.assertTrue(detected.points)
        occluded = detector.estimate(np.zeros((120, 240, 3), dtype=np.uint8), cv2)
        self.assertEqual(occluded.points, detected.points)
        expired = detector.estimate(np.zeros((120, 240, 3), dtype=np.uint8), cv2)
        self.assertFalse(expired.points)

    def test_full_resolution_ice_mask_is_computed_once_per_frame(self):
        class CountingDetector(NearBoardBoundaryDetector):
            ice_mask_calls = 0

            def _ice_mask(self, frame, cv2_module):
                self.ice_mask_calls += 1
                return super()._ice_mask(frame, cv2_module)

        detector = CountingDetector({"bins": 20, "minimum_coverage": .1,
                                     "sample_offset_fraction": .08,
                                     "detection_scale": 1})
        detector.estimate(self.boundary_frame(), cv2)
        self.assertEqual(detector.ice_mask_calls, 1)

    def test_corridor_and_beyond_boundary_decisions(self):
        far = BoundaryEstimate(((50, 25),), 1, 100, 4)
        near = BoundaryEstimate(((50, 80),), 1, 100, 4)
        frame = np.full((120, 100, 3), 210, dtype=np.uint8)
        participant = ParticipantFilter()
        inside = participant.classify(frame, (40, 30, 60, 60), OFF_ICE, cv2,
                                      far, near, False)
        self.assertEqual(inside.state, ACCEPTED)
        self.assertEqual(inside.reason, "visible_rink_corridor_and_ice")
        beyond_near = participant.classify(frame, (40, 50, 60, 90), ON_ICE, cv2,
                                           far, near, False)
        self.assertEqual(beyond_near.state, REJECTED)
        beyond_far = participant.classify(frame, (40, 5, 60, 15), ON_ICE, cv2,
                                          far, near, False)
        self.assertEqual(beyond_far.state, REJECTED)

    def test_missing_near_boundary_does_not_reject_skater(self):
        far = BoundaryEstimate(((50, 25),), 1, 100, 4)
        frame = np.full((120, 100, 3), 210, dtype=np.uint8)
        result = ParticipantFilter().classify(frame, (40, 30, 60, 60), OFF_ICE,
                                              cv2, far, None, False)
        self.assertEqual(result.state, ACCEPTED)

    def test_disabling_polygon_does_not_change_unrelated_appearance_result(self):
        frame = np.full((120, 100, 3), 210, dtype=np.uint8)
        enabled = ParticipantFilter().classify(frame, (40, 30, 60, 60), ON_ICE,
                                               cv2, use_polygon=True)
        disabled = ParticipantFilter().classify(frame, (40, 30, 60, 60), ON_ICE,
                                                cv2, use_polygon=False)
        self.assertEqual(enabled.state, ACCEPTED)
        self.assertEqual(disabled.state, UNCERTAIN)
        self.assertTrue(disabled.accepted)


if __name__ == "__main__":
    unittest.main()
