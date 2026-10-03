import unittest

import numpy as np
import json
from pathlib import Path

from provider_overlay import ProviderOverlayMask


class ProviderOverlayMaskTests(unittest.TestCase):
    def test_configured_region_is_excluded_and_outside_is_unchanged(self):
        frame = np.full((100, 200, 3), 127, dtype=np.uint8)
        masked = ProviderOverlayMask({"excluded_regions": [[.1, .2, .3, .4]]}).apply(frame)
        self.assertTrue(np.all(masked[20:40, 20:60] == 0))
        self.assertTrue(np.all(masked[50:, 100:] == 127))
        self.assertTrue(np.all(frame == 127), "Masking must not modify the annotation/source frame")

    def test_normalized_region_scales_with_resolution(self):
        mask = ProviderOverlayMask({"excluded_regions": [[.25, .25, .75, .75]]})
        self.assertEqual(mask.pixel_regions(200, 100), [(50, 25, 150, 75)])
        self.assertEqual(mask.pixel_regions(400, 200), [(100, 50, 300, 150)])

    def test_no_provider_configuration_returns_original_frame(self):
        frame = np.ones((10, 10, 3), dtype=np.uint8)
        self.assertIs(ProviderOverlayMask().apply(frame), frame)

    def test_malformed_regions_fail_safely(self):
        malformed = [
            {"excluded_regions": "top-left"},
            {"excluded_regions": [[0, 0, 1]]},
            {"excluded_regions": [[-.1, 0, .2, .2]]},
            {"excluded_regions": [[.5, 0, .2, .2]]},
            {"excluded_regions": [[0, 0, float("nan"), .2]]},
        ]
        for configuration in malformed:
            with self.subTest(configuration=configuration), self.assertRaises(ValueError):
                ProviderOverlayMask(configuration)

    def test_benchmark_config_masks_black_bear_logo_with_margin(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "57-test.example.team.json"
        provider = json.loads(config_path.read_text(encoding="utf-8"))["provider"]
        mask = ProviderOverlayMask(provider)
        self.assertEqual(provider["name"], "blackbear")
        self.assertEqual(mask.pixel_regions(1920, 1080), [(1718, 923, 1910, 1053)])
        # Observed logo extent is approximately x=1750..1890, y=944..1035.
        x1, y1, x2, y2 = mask.pixel_regions(1920, 1080)[0]
        self.assertLess(x1, 1750)
        self.assertLess(y1, 944)
        self.assertGreater(x2, 1890)
        self.assertGreater(y2, 1035)

    def test_center_or_strong_overlap_removes_mask_hugging_detection(self):
        mask = ProviderOverlayMask({"excluded_regions": [[.8, .8, 1, 1]],
                                    "post_detection_overlap_fraction": .6})
        kept, evidence = mask.filter_detections(
            [[82, 79, 99, 99], [75, 75, 95, 95]], 100, 100)
        self.assertEqual(kept, [])
        self.assertEqual(len(evidence["removed"]), 2)

    def test_legitimate_large_person_merely_overlapping_mask_remains(self):
        mask = ProviderOverlayMask({"excluded_regions": [[.8, .8, 1, 1]],
                                    "post_detection_overlap_fraction": .6})
        kept, evidence = mask.filter_detections([[40, 30, 90, 90]], 100, 100)
        self.assertEqual(kept, [0])
        self.assertEqual(len(evidence["retained_overlaps"]), 1)


if __name__ == "__main__":
    unittest.main()
