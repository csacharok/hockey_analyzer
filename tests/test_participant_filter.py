import unittest

import cv2
import numpy as np

from participant_filter import ACCEPTED, REJECTED, UNCERTAIN, ParticipantFilter
from rink_geometry import OFF_ICE, ON_ICE, UNKNOWN


class ParticipantFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = ParticipantFilter()
        self.box = (30, 10, 70, 70)

    def test_clear_ice_contact_is_accepted(self):
        frame = np.full((100, 100, 3), 210, dtype=np.uint8)
        result = self.filter.classify(frame, self.box, ON_ICE, cv2)
        self.assertEqual(result.state, ACCEPTED)
        self.assertTrue(result.accepted)

    def test_strong_non_ice_contact_is_rejected(self):
        frame = np.full((100, 100, 3), (20, 20, 80), dtype=np.uint8)
        result = self.filter.classify(frame, self.box, ON_ICE, cv2)
        self.assertEqual(result.state, REJECTED)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "strong_non_ice_contact_evidence")

    def test_geometry_off_ice_is_rejected_with_reason(self):
        frame = np.full((100, 100, 3), 210, dtype=np.uint8)
        result = self.filter.classify(frame, self.box, OFF_ICE, cv2)
        self.assertEqual(result.state, REJECTED)
        self.assertEqual(result.reason, "contact_point_outside_rink_polygon")

    def test_uncertain_geometry_preserves_detection(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.filter.classify(frame, self.box, UNKNOWN, cv2)
        self.assertEqual(result.state, UNCERTAIN)
        self.assertTrue(result.accepted)

    def test_weak_ice_evidence_is_uncertain_not_deleted(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[66:78, 20:80] = 130
        frame[66:78, 20:60] = 20
        result = self.filter.classify(frame, self.box, ON_ICE, cv2)
        self.assertEqual(result.state, UNCERTAIN)
        self.assertTrue(result.accepted)

    def test_invalid_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            ParticipantFilter({"reject_below_ice_fraction": .5,
                               "accept_at_ice_fraction": .4})


if __name__ == "__main__":
    unittest.main()
