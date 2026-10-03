import unittest

import cv2
import numpy as np

from appearance_features import (AppearanceFeatureExtractor, TrackletAppearanceAggregator,
                                 measurement_quality, normalized_region_bounds)
from export_appearance_dataset import dataset_record


def synthetic(torso=180, pants=80, legs=160, width=100, height=200):
    frame = np.full((height, width, 3), 120, dtype=np.uint8)
    frame[round(.12 * height):round(.50 * height), round(.18 * width):round(.82 * width)] = torso
    frame[round(.50 * height):round(.68 * height), round(.20 * width):round(.80 * width)] = pants
    for left, right in ((.12, .43), (.57, .88)):
        frame[round(.62 * height):round(.93 * height), round(left * width):round(right * width)] = legs
    return frame


class AppearanceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.extractor = AppearanceFeatureExtractor(minimum_region_pixels=16)
        self.box = (0, 0, 100, 200)

    def test_normalized_regions_scale_with_bbox(self):
        small = normalized_region_bounds((0, 0, 100, 200), (.2, .8), (.1, .5))
        large = normalized_region_bounds((0, 0, 200, 400), (.2, .8), (.1, .5))
        self.assertEqual(tuple(value * 2 for value in small), large)

    def test_regions_stay_inside_frame_bounds(self):
        result = self.extractor.extract(synthetic(), (-20, -20, 80, 180), cv2)
        for bounds in result["region_bounds"].values():
            self.assertGreaterEqual(min(bounds), 0)
            self.assertLessEqual(bounds[2], 100)
            self.assertLessEqual(bounds[3], 200)

    def test_truncated_crop_reports_quality(self):
        result = self.extractor.extract(synthetic(), (-50, 0, 50, 200), cv2)
        self.assertTrue(result["quality"]["frame_edge_truncated"])
        self.assertNotEqual(result["quality"]["crop_quality"], "GOOD")
        self.assertTrue(any(not region["usable"] for region in result["regions"].values()))

    def test_measurement_quality_uses_clear_backward_compatible_mapping(self):
        self.assertEqual(measurement_quality("GOOD"), "MEASURABLE")
        self.assertEqual(measurement_quality("LOW"), "PARTIAL")
        result = self.extractor.extract(synthetic(), self.box, cv2)
        self.assertEqual(result["quality"]["crop_quality"], "GOOD")
        self.assertEqual(result["quality"]["measurement_quality"], "MEASURABLE")
        self.assertNotIn("training_suitability", result["quality"])

    def test_light_torso_has_higher_lightness(self):
        light = self.extractor.extract(synthetic(torso=230), self.box, cv2)
        dark = self.extractor.extract(synthetic(torso=25), self.box, cv2)
        self.assertGreater(light["regions"]["torso"]["lab_lightness_median"],
                           dark["regions"]["torso"]["lab_lightness_median"])

    def test_dark_pants_have_higher_dark_fraction(self):
        dark = self.extractor.extract(synthetic(pants=20), self.box, cv2)
        light = self.extractor.extract(synthetic(pants=220), self.box, cv2)
        self.assertGreater(dark["regions"]["pants"]["dark_fraction"],
                           light["regions"]["pants"]["dark_fraction"])

    def test_light_legs_have_higher_lightness(self):
        light = self.extractor.extract(synthetic(legs=230), self.box, cv2)
        dark = self.extractor.extract(synthetic(legs=20), self.box, cv2)
        self.assertGreater(light["regions"]["lower_legs"]["lab_lightness_median"],
                           dark["regions"]["lower_legs"]["lab_lightness_median"])

    def test_dark_legs_have_higher_dark_fraction(self):
        dark = self.extractor.extract(synthetic(legs=20), self.box, cv2)
        light = self.extractor.extract(synthetic(legs=230), self.box, cv2)
        self.assertGreater(dark["regions"]["lower_legs"]["dark_fraction"],
                           light["regions"]["lower_legs"]["dark_fraction"])

    def test_stripe_feature_is_exposed(self):
        frame = synthetic()
        for x in range(18, 82, 8):
            frame[24:100, x:x + 4] = 240 if x % 16 else 10
        result = self.extractor.extract(frame, self.box, cv2)
        self.assertIn("vertical_stripe_score", result["structure"])
        self.assertIn("stripe_supporting_rows", result["structure"])

    def test_raw_measurements_remain_with_structure_descriptor(self):
        result = self.extractor.extract(synthetic(), self.box, cv2)
        self.assertIn("lab_lightness_median", result["regions"]["torso"])
        self.assertIn("dark_fraction", result["regions"]["lower_legs"])
        self.assertIn("vertical_stripe_score", result["structure"])

    def test_aggregation_uses_only_usable_observations(self):
        aggregator = TrackletAppearanceAggregator()
        aggregator.add(1, self.extractor.extract(None, self.box, cv2))
        aggregator.add(1, self.extractor.extract(synthetic(), self.box, cv2))
        self.assertEqual(aggregator.summaries()[0]["usable_observations"], 1)

    def test_aggregation_median_is_robust_to_outlier(self):
        aggregator = TrackletAppearanceAggregator()
        for value in (100, 100, 255):
            aggregator.add(1, self.extractor.extract(synthetic(torso=value), self.box, cv2))
        expected = self.extractor.extract(synthetic(torso=100), self.box, cv2)
        self.assertAlmostEqual(aggregator.summaries()[0]["medians"]["torso_lab_lightness_median"],
                               expected["regions"]["torso"]["lab_lightness_median"])

    def test_dataset_record_has_required_provenance(self):
        record = {"frame": 3, "timestamp": .1, "track_id": 7, "box": list(self.box),
                  "participant_state": "ACCEPTED", "participant_accepted": True,
                  "participant_reason": "test", "participant_evidence": {}}
        exported = dataset_record("game", "video.mp4", record,
                                  self.extractor.extract(synthetic(), self.box, cv2))
        for key in ("source_game", "frame", "temporary_track_id", "person_bbox",
                    "crop_reference", "appearance_features", "participant_evidence"):
            self.assertIn(key, exported)

    def test_legacy_labels_are_explicitly_not_ground_truth(self):
        record = {"frame": 3, "timestamp": .1, "track_id": 7, "box": list(self.box),
                  "label": "HOME", "confidence": .8}
        exported = dataset_record("game", "video.mp4", record,
                                  self.extractor.extract(synthetic(), self.box, cv2))
        self.assertFalse(exported["legacy_metadata"]["is_human_ground_truth"])
        self.assertIn("pseudo_label", exported["legacy_metadata"]["provenance"])

    def test_extraction_requires_no_semantic_classifier(self):
        result = AppearanceFeatureExtractor().extract(synthetic(), self.box, cv2)
        self.assertTrue(result["usable"])
        self.assertNotIn("label", result)


if __name__ == "__main__":
    unittest.main()
