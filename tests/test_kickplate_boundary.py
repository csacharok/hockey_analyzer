import unittest

import cv2
import numpy as np

from kickplate_boundary import BoundaryEstimate, KickplateBoundaryDetector
from participant_filter import ACCEPTED, REJECTED, UNCERTAIN, ParticipantFilter
from rink_geometry import OFF_ICE, ON_ICE


class KickplateBoundaryTests(unittest.TestCase):
    def test_precomputed_hsv_preserves_boundary(self):
        frame = np.zeros((120, 240, 3), dtype=np.uint8)
        cv2.line(frame, (0, 60), (239, 60), (0, 255, 255), 5)
        detector = KickplateBoundaryDetector({"minimum_coverage": .2})
        expected = detector.estimate(frame, cv2)
        actual = detector.estimate(frame, cv2, cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        self.assertEqual(actual, expected)

    def test_detects_piecewise_line_and_bridges_only_short_gap(self):
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        yellow = (0, 210, 210)
        cv2.line(frame, (0, 100), (170, 110), yellow, 5)
        cv2.line(frame, (195, 112), (399, 125), yellow, 5)
        result = KickplateBoundaryDetector({"bins": 20, "minimum_coverage": .3,
                                            "maximum_gap_bins": 2}).estimate(frame, cv2)
        self.assertGreater(result.coverage, .7)
        self.assertIsNotNone(result.y_at(180))
        self.assertLess(result.y_at(20), result.y_at(380))

    def test_isolated_yellow_blob_is_not_a_boundary(self):
        frame = np.zeros((240, 400, 3), dtype=np.uint8)
        frame[100:130, 190:210] = (0, 210, 210)
        result = KickplateBoundaryDetector({"bins": 20}).estimate(frame, cv2)
        self.assertEqual(result.points, ())

    def test_boundary_rejects_non_playing_side_despite_stale_polygon(self):
        boundary = BoundaryEstimate(((50, 50),), 1, 100, 4)
        frame = np.full((100, 100, 3), 210, dtype=np.uint8)
        result = ParticipantFilter().classify(frame, (40, 10, 60, 40), ON_ICE, cv2, boundary)
        self.assertEqual(result.state, REJECTED)
        self.assertEqual(result.reason, "contact_point_behind_kickplate")

    def test_boundary_and_ice_override_stale_polygon(self):
        boundary = BoundaryEstimate(((50, 30),), 1, 100, 4)
        frame = np.full((100, 100, 3), 210, dtype=np.uint8)
        result = ParticipantFilter().classify(frame, (40, 20, 60, 70), OFF_ICE, cv2, boundary)
        self.assertEqual(result.state, ACCEPTED)
        self.assertEqual(result.reason, "kickplate_playing_side_and_ice")

    def test_boundary_local_appearance_conflict_is_uncertain(self):
        boundary = BoundaryEstimate(((50, 30),), 1, 100, 4)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = ParticipantFilter().classify(frame, (40, 20, 60, 70), ON_ICE, cv2, boundary)
        self.assertEqual(result.state, UNCERTAIN)
        self.assertTrue(result.accepted)


if __name__ == "__main__":
    unittest.main()
