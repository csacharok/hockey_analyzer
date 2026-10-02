import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from team_classifier import AWAY, HOME, OFFICIAL, UNKNOWN, TeamClassifier


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "57-test.example.team.json"


def hsv_frame(hue, saturation, value, width=80, height=160):
    hsv = np.full((height, width, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TeamClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TeamClassifier.load(CONFIG)
        self.box = (0, 0, 80, 160)

    def test_clear_home_evidence(self):
        label, confidence, evidence = self.classifier.observe(hsv_frame(175, 190, 105), self.box, cv2)
        self.assertEqual(label, HOME)
        self.assertGreaterEqual(confidence, .9)
        self.assertGreater(evidence["profile_fractions"][HOME], .9)

    def test_clear_away_evidence(self):
        label, confidence, _ = self.classifier.observe(hsv_frame(0, 20, 200), self.box, cv2)
        self.assertEqual(label, AWAY)
        self.assertGreaterEqual(confidence, .9)

    def test_bright_pink_official_evidence(self):
        label, confidence, _ = self.classifier.observe(hsv_frame(175, 200, 220), self.box, cv2)
        self.assertEqual(label, OFFICIAL)
        self.assertGreaterEqual(confidence, .9)

    def test_striped_official_evidence(self):
        frame = np.zeros((160, 80, 3), dtype=np.uint8)
        for x in range(0, 80, 8):
            frame[:, x:x + 4] = 220
        label, confidence, evidence = self.classifier.observe(frame, self.box, cv2)
        self.assertEqual(label, OFFICIAL)
        self.assertGreaterEqual(confidence, .55)
        self.assertGreaterEqual(evidence["stripe_ratio"], 1)

    def test_ambiguous_color_is_unknown(self):
        label, _, _ = self.classifier.observe(hsv_frame(60, 180, 140), self.box, cv2)
        self.assertEqual(label, UNKNOWN)

    def test_temporal_evidence_resists_one_conflicting_frame(self):
        for _ in range(3):
            result = self.classifier.classify(42, hsv_frame(175, 190, 105), self.box, cv2)
        self.assertEqual(result.label, HOME)
        result = self.classifier.classify(42, hsv_frame(0, 20, 200), self.box, cv2)
        self.assertEqual(result.label, HOME)
        self.assertLess(result.confidence, 1)

    def test_balanced_conflicting_evidence_becomes_unknown(self):
        configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
        configuration["track_evidence_decay"] = 1
        classifier = TeamClassifier(configuration)
        classifier.classify(9, hsv_frame(175, 190, 105), self.box, cv2)
        result = classifier.classify(9, hsv_frame(0, 20, 200), self.box, cv2)
        self.assertEqual(result.label, UNKNOWN)

    def test_malformed_and_empty_crops_are_unknown(self):
        for frame, box in ((None, self.box), (np.zeros((10, 10, 3), dtype=np.uint8), (5, 5, 5, 9)),
                           (np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 1, 1)),
                           (np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, float("nan"), 5))):
            with self.subTest(box=box):
                self.assertEqual(self.classifier.observe(frame, box, cv2)[0], UNKNOWN)

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            TeamClassifier({})


if __name__ == "__main__":
    unittest.main()
