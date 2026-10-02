import math
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from analyze_tracks import VideoSpec, draw_annotations, output_path, summary, track_label, validate_input
from rink_geometry import RinkGeometry, ON_ICE, OFF_ICE, UNKNOWN


class BaselineTests(unittest.TestCase):
    def test_fractional_fps_is_preserved(self):
        spec = VideoSpec(1920, 1080, 30000 / 1001)
        self.assertEqual(spec.fps, 30000 / 1001)

    def test_invalid_metadata_is_rejected(self):
        for width, height, fps in [(0, 1080, 30), (1920, -1, 30), (1920, 1080, 0), (1920, 1080, -30), (1920, 1080, math.nan), (1920, 1080, math.inf)]:
            with self.subTest(width=width, height=height, fps=fps):
                with self.assertRaises(ValueError):
                    VideoSpec(width, height, fps)

    def test_input_validation_does_not_modify_file(self):
        scratch = Path(__file__).resolve().parents[1] / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=scratch, suffix=".mp4", delete=False) as fixture:
            fixture.write(b"source data")
            source = Path(fixture.name)
        try:
            self.assertEqual(validate_input(source), source.resolve())
            self.assertEqual(source.read_bytes(), b"source data")
            for invalid in (scratch, source.with_suffix(".missing")):
                with self.assertRaises(ValueError):
                    validate_input(invalid)
        finally:
            source.unlink()

    def test_output_is_fresh_and_inside_requested_directory(self):
        directory = Path("output")
        source = directory / "game.mp4"
        first = output_path(source, directory)
        second = output_path(source, directory)
        self.assertEqual(first.parent, directory.resolve())
        self.assertEqual(first.suffix, ".mp4")
        self.assertNotEqual(first, source.resolve())
        self.assertNotEqual(first, second)

    def test_label_explicitly_uses_track_not_player(self):
        self.assertEqual(track_label(57, 0.856), "Track 57 | 0.86")

    def test_unassigned_detection_does_not_invent_id(self):
        cv2 = Mock()
        cv2.getTextSize.return_value = ((100, 12), 3)
        frame = SimpleNamespace(shape=(1080, 1920, 3))
        ids = draw_annotations(frame, [[10, 20, 30, 40]], SimpleNamespace(is_track=False), cv2)
        self.assertEqual(ids, set())
        # A raw person box and the legend's background are both drawn.
        self.assertEqual(cv2.rectangle.call_count, 2)

    def test_annotations_collect_assigned_ids_and_confidence(self):
        cv2 = Mock()
        cv2.getTextSize.return_value = ((100, 12), 3)
        frame = SimpleNamespace(shape=(1080, 1920, 3))
        boxes = Mock(is_track=True)
        boxes.cpu.return_value = boxes
        boxes.xyxy.tolist.return_value = [[10, 20, 30, 40], [50, 60, 70, 80]]
        boxes.id.tolist.return_value = [7.0, 12.0]
        boxes.conf.tolist.return_value = [0.9, 0.8]
        self.assertEqual(draw_annotations(frame, [], boxes, cv2), {7, 12})
        labels = [call.args[1] for call in cv2.putText.call_args_list]
        self.assertIn("Track 7 | 0.90", labels)
        self.assertIn("Track 12 | 0.80", labels)

    def test_summary_counts_unique_temporary_ids(self):
        report = summary(60, 2, Path("output/test.mp4"), set([1, 1, 8]))
        self.assertIn("Frames processed: 60", report)
        self.assertIn("Elapsed processing time: 2.00 s", report)
        self.assertIn("Average processing FPS: 30.00", report)
        self.assertIn("Unique short-term track IDs: 2", report)
        self.assertIn(str(Path("output/test.mp4")), report)

    def test_summary_handles_zero_elapsed(self):
        self.assertIn("Average processing FPS: 0.00", summary(0, 0, Path("test.mp4"), set()))

    def test_geometry_annotations_count_each_tracked_person_frame(self):
        cv2 = Mock()
        cv2.getTextSize.return_value = ((100, 12), 3)
        frame = SimpleNamespace(shape=(100, 100, 3))
        rink = RinkGeometry(100, 100, ((20, 20), (80, 20), (80, 80), (20, 80)), 2)
        boxes = Mock(is_track=True)
        boxes.cpu.return_value = boxes
        boxes.xyxy.tolist.return_value = [[40, 0, 60, 50], [0, 0, 10, 50], [19, 0, 21, 50]]
        boxes.id.tolist.return_value = [7.0, 12.0, 13.0]
        boxes.conf.tolist.return_value = [0.9, 0.8, 0.7]
        counts = Counter()
        for _ in range(2):
            ids = draw_annotations(frame, [[30, 0, 40, 40]], boxes, cv2, rink, counts)
            self.assertEqual(ids, {7, 12, 13})
        self.assertEqual(counts, {ON_ICE: 2, OFF_ICE: 2, UNKNOWN: 2})
        labels = [call.args[1] for call in cv2.putText.call_args_list]
        self.assertIn("Track 7 | 0.90 | ON ICE", labels)
        self.assertIn("Track 12 | 0.80 | OFF ICE", labels)
        self.assertIn("Track 13 | 0.70 | UNKNOWN", labels)
        # Contact dots distinguish all three states by color.
        colors = {call.args[3] for call in cv2.circle.call_args_list}
        self.assertEqual(len(colors), 3)
        report = summary(2, 1, Path("test.mp4"), ids, counts)
        self.assertIn("Unique short-term track IDs: 3", report)
        for state in (ON_ICE, OFF_ICE, UNKNOWN):
            self.assertIn(f"{state} detections (tracked person-frames): 2", report)

    def test_geometry_does_not_classify_unassigned_detections(self):
        cv2 = Mock()
        cv2.getTextSize.return_value = ((100, 12), 3)
        rink = RinkGeometry(100, 100, ((20, 20), (80, 20), (80, 80), (20, 80)))
        counts = Counter()
        ids = draw_annotations(SimpleNamespace(shape=(100, 100, 3)), [[30, 0, 40, 40]],
                               SimpleNamespace(is_track=False), cv2, rink, counts)
        self.assertEqual(ids, set())
        self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
