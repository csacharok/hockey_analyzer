import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from rink_geometry import RinkGeometry, ON_ICE, OFF_ICE, UNKNOWN, contact_point


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.polygon = ((20, 20), (80, 20), (80, 80), (20, 80))
        self.rink = RinkGeometry(100, 100, self.polygon, 2)

    def test_bottom_center_retains_fractional_pixels(self):
        self.assertEqual(contact_point((10.5, 5, 30, 60.25)), (20.25, 60.25))

    def test_uses_contact_point_not_box_center_or_overlap(self):
        self.assertEqual(self.rink.classify((40, 0, 60, 50)), ON_ICE)
        self.assertEqual(self.rink.classify((40, 30, 60, 90)), OFF_ICE)

    def test_edge_vertex_and_margin_are_unknown_on_both_sides(self):
        for x, y in [(20, 50), (20, 20), (18, 50), (22, 50), (50, 78), (50, 82)]:
            with self.subTest(x=x, y=y):
                self.assertEqual(self.rink.classify((x - 1, 0, x + 1, y)), UNKNOWN)
        self.assertEqual(self.rink.classify((22, 0, 24, 50)), ON_ICE)
        self.assertEqual(self.rink.classify((16, 0, 18, 50)), OFF_ICE)

    def test_zero_margin_still_marks_exact_boundary_unknown(self):
        rink = RinkGeometry(100, 100, self.polygon, 0)
        self.assertEqual(rink.classify((19, 0, 21, 50)), UNKNOWN)

    def test_bottom_clipping_and_invalid_boxes_are_unknown(self):
        for box in [(40, 0, 60, 99), (40, 0, 60, 100), (-20, 0, -10, 50),
                    (40, 40, 40, 50), (40, 60, 60, 50), (40, 0, math.nan, 50)]:
            with self.subTest(box=box):
                self.assertEqual(self.rink.classify(box), UNKNOWN)

    def test_concave_polygon_and_both_vertex_orders(self):
        polygon = ((10, 10), (90, 10), (90, 40), (40, 40), (40, 90), (10, 90))
        for points in (polygon, tuple(reversed(polygon))):
            rink = RinkGeometry(100, 100, points, 0)
            self.assertEqual(rink.classify((20, 0, 30, 60)), ON_ICE)
            self.assertEqual(rink.classify((50, 0, 70, 60)), OFF_ICE)

    def test_rejects_wrong_resolution(self):
        self.rink.validate_resolution(100, 100)
        with self.assertRaises(ValueError):
            self.rink.validate_resolution(200, 100)

    def test_rejects_invalid_polygons(self):
        invalid = [[], [(1, 1), (2, 2)], [(1, 1), (2, 2), (3, 3)],
                   [(0, 0), (100, 0), (0, 20)], [(0, 0), (20, 0), (0, 0)],
                   [(0, 0), (20, 0), (math.inf, 20)], [(0, 0), (20, 0), ("x", 20)],
                   [(0, 0), (80, 80), (0, 70), (80, 0)],
                   [(0, 0), (80, 0), (80, 80), (40, 0), (0, 80)]]
        for polygon in invalid:
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                RinkGeometry(100, 100, polygon)

    def test_rejects_invalid_margin(self):
        for margin in (-1, math.nan, math.inf, True, "8"):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                RinkGeometry(100, 100, self.polygon, margin)

    def test_json_loading_and_missing_fields(self):
        data = dict(image_width=100, image_height=100, polygon=self.polygon, boundary_margin_px=2)
        with patch.object(Path, "read_text", return_value=json.dumps(data)):
            self.assertEqual(RinkGeometry.load(Path("rink.json")), self.rink)
        for invalid in ({}, [], {**data, "polygon": []}):
            with patch.object(Path, "read_text", return_value=json.dumps(invalid)):
                with self.assertRaises(ValueError):
                    RinkGeometry.load(Path("rink.json"))

    def test_example_configuration_loads(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "57-test.example.rink.json"
        RinkGeometry.load(path).validate_resolution(1920, 1080)


if __name__ == "__main__":
    unittest.main()
