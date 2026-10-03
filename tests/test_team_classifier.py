import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from team_classifier import AWAY, HOME, OFFICIAL, UNKNOWN, OfficialTemporalState, TeamClassifier


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "57-test.example.team.json"


def hsv_frame(hue, saturation, value, width=80, height=160):
    hsv = np.full((height, width, 3), (hue, saturation, value), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TeamClassifierTests(unittest.TestCase):
    def setUp(self):
        self.classifier = TeamClassifier.load(CONFIG)
        self.box = (0, 0, 80, 160)

    def test_clear_home_evidence(self):
        label, confidence, evidence = self.classifier.observe(hsv_frame(0, 20, 200), self.box, cv2)
        self.assertEqual(label, HOME)
        self.assertGreaterEqual(confidence, .9)
        self.assertGreater(evidence["profile_fractions"][HOME], .9)

    def test_clear_away_evidence(self):
        label, confidence, _ = self.classifier.observe(hsv_frame(175, 190, 105), self.box, cv2)
        self.assertEqual(label, AWAY)
        self.assertGreaterEqual(confidence, .9)

    def test_solid_bright_pink_is_not_official_evidence(self):
        label, confidence, _ = self.classifier.observe(hsv_frame(175, 200, 220), self.box, cv2)
        self.assertNotEqual(label, OFFICIAL)

    def test_striped_official_evidence(self):
        frame = np.zeros((160, 80, 3), dtype=np.uint8)
        for x in range(0, 80, 8):
            frame[:, x:x + 4] = 220
        label, confidence, evidence = self.classifier.observe(frame, self.box, cv2)
        self.assertEqual(label, OFFICIAL)
        self.assertGreaterEqual(confidence, .55)
        self.assertGreaterEqual(evidence["stripe_ratio"], 1)
        self.assertGreaterEqual(evidence["stripe_evidence"]["striped_row_fraction"], .35)
        self.assertTrue(evidence["sock_evidence"]["available"])

    def test_stripe_like_torso_without_dark_lower_body_is_not_official(self):
        frame = np.full((160, 80, 3), 220, dtype=np.uint8)
        for x in range(0, 80, 8):
            frame[:100, x:x + 4] = 20
        label, _, evidence = self.classifier.observe(frame, self.box, cv2)
        self.assertNotEqual(label, OFFICIAL)
        self.assertLess(evidence["stripe_evidence"]["lower_body_dark_fraction"], .30)

    def test_white_with_horizontal_dark_detail_is_not_official(self):
        frame = np.full((160, 80, 3), 220, dtype=np.uint8)
        frame[55:90, :] = 20
        label, _, evidence = self.classifier.observe(frame, self.box, cv2)
        self.assertNotEqual(label, OFFICIAL)
        self.assertLess(evidence["stripe_evidence"]["vertical_orientation_ratio"], 1)

    def test_partial_weak_stripes_can_remain_unknown(self):
        frame = hsv_frame(60, 180, 140)
        frame[:, 36:39] = 220
        frame[:, 39:42] = 20
        label, _, evidence = self.classifier.observe(frame, self.box, cv2)
        self.assertEqual(label, UNKNOWN)
        self.assertLess(evidence["stripe_ratio"], 1)

    def test_benchmark_configuration_explicitly_maps_white_home_maroon_away(self):
        configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
        home_mask = configuration["profiles"][HOME]["masks"][0]
        away_masks = configuration["profiles"][AWAY]["masks"]
        self.assertEqual(home_mask["saturation"], [0, 65])
        self.assertTrue(any(mask["saturation"][0] >= 65 for mask in away_masks))

    def test_ambiguous_color_is_unknown(self):
        label, _, _ = self.classifier.observe(hsv_frame(60, 180, 140), self.box, cv2)
        self.assertEqual(label, UNKNOWN)

    def test_temporal_evidence_resists_one_conflicting_frame(self):
        for _ in range(3):
            result = self.classifier.classify(42, hsv_frame(0, 20, 200), self.box, cv2)
        self.assertEqual(result.label, HOME)
        result = self.classifier.classify(42, hsv_frame(175, 190, 105), self.box, cv2)
        self.assertEqual(result.label, HOME)
        self.assertLess(result.confidence, 1)

    def test_balanced_conflicting_evidence_becomes_unknown(self):
        configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
        configuration["track_evidence_decay"] = 1
        classifier = TeamClassifier(configuration)
        classifier.classify(9, hsv_frame(0, 20, 200), self.box, cv2)
        result = classifier.classify(9, hsv_frame(175, 190, 105), self.box, cv2)
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

    def test_malformed_provider_mask_is_rejected_at_configuration_load(self):
        configuration = json.loads(CONFIG.read_text(encoding="utf-8"))
        configuration["provider"] = {"excluded_regions": [[.8, 0, .2, .1]]}
        with self.assertRaises(ValueError):
            TeamClassifier(configuration)

    def test_official_role_requires_repeated_structural_evidence(self):
        state = OfficialTemporalState({"establish_window": 4, "establish_count": 3})
        evidence = {"stripe_ratio": 1.4}
        self.assertFalse(state.update(OFFICIAL, .8, evidence)["confirmed"])
        self.assertFalse(state.update(OFFICIAL, .8, evidence)["confirmed"])
        result = state.update(OFFICIAL, .8, evidence)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["transition"], "ESTABLISHED")

    def test_confirmed_official_survives_weak_home_and_unknown(self):
        state = OfficialTemporalState({"establish_window": 2, "establish_count": 2,
                                       "reverse_window": 3, "reverse_count": 3})
        for _ in range(2):
            state.update(OFFICIAL, .8, {"stripe_ratio": 1.4})
        self.assertTrue(state.update(HOME, .5, {"stripe_ratio": .4})["confirmed"])
        self.assertTrue(state.update(UNKNOWN, 0, {"stripe_ratio": 0})["confirmed"])

    def test_confirmed_official_is_reversible_after_sustained_strong_player_evidence(self):
        state = OfficialTemporalState({"establish_window": 2, "establish_count": 2,
                                       "reverse_window": 4, "reverse_count": 3})
        for _ in range(2):
            state.update(OFFICIAL, .8, {"stripe_ratio": 1.4})
        for _ in range(2):
            self.assertTrue(state.update(AWAY, .9, {"stripe_ratio": .2,
                                                    "stripe_evidence": {"lower_body_dark_fraction": .1}})["confirmed"])
        result = state.update(AWAY, .9, {"stripe_ratio": .2,
                                         "stripe_evidence": {"lower_body_dark_fraction": .1}})
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["transition"], "REVERTED")

    def test_confirmed_official_survives_strong_generic_home_without_player_structure(self):
        state = OfficialTemporalState({"establish_window": 3, "establish_count": 3,
                                       "reverse_window": 3, "reverse_count": 3})
        for _ in range(3):
            state.update(OFFICIAL, .8, {"stripe_ratio": 1.4})
        for _ in range(6):
            result = state.update(HOME, 1.0, {"stripe_ratio": 0.0,
                                              "stripe_evidence": {"lower_body_dark_fraction": .6}})
        self.assertTrue(result["confirmed"])
        self.assertFalse(result["strong_contradiction"])

    def test_absent_stripes_alone_do_not_reverse_official(self):
        state = OfficialTemporalState({"establish_window": 2, "establish_count": 2,
                                       "reverse_window": 2, "reverse_count": 2})
        for _ in range(2):
            state.update(OFFICIAL, .8, {"stripe_ratio": 1.4})
        for _ in range(4):
            result = state.update(UNKNOWN, 0.0, {"stripe_ratio": 0.0})
        self.assertTrue(result["confirmed"])

    def test_contaminated_low_quality_crop_cannot_establish_official(self):
        state = OfficialTemporalState({"establish_window": 4, "establish_count": 2})
        evidence = {"stripe_ratio": 1.5, "official_establishment_usable": False}
        for _ in range(4):
            result = state.update(OFFICIAL, .9, evidence)
        self.assertFalse(result["confirmed"])

    def test_established_official_survives_unusable_crop(self):
        state = OfficialTemporalState({"establish_window": 2, "establish_count": 2})
        for _ in range(2):
            state.update(OFFICIAL, .9, {"stripe_ratio": 1.5,
                                        "official_establishment_usable": True})
        result = state.update(HOME, 1.0, {"stripe_ratio": 0,
                                          "official_establishment_usable": False})
        self.assertTrue(result["confirmed"])

    def test_black_pants_support_structural_official_evidence(self):
        state = OfficialTemporalState({"establish_window": 1, "establish_count": 1})
        result = state.update(OFFICIAL, .9, {"stripe_ratio": 1.5,
                                             "stripe_evidence": {
                                                 "lower_body_dark_fraction": .6}})
        self.assertTrue(result["confirmed"])

    def test_generic_dark_or_white_evidence_cannot_establish_official(self):
        state = OfficialTemporalState({"establish_window": 4, "establish_count": 2})
        for label, confidence, stripe in ((HOME, 1, .1), (UNKNOWN, 0, .9), (AWAY, 1, .3)):
            self.assertFalse(state.update(label, confidence, {"stripe_ratio": stripe})["confirmed"])


if __name__ == "__main__":
    unittest.main()
