import json
from pathlib import Path
import tempfile
import unittest

from official_temporal_replay import replay


class OfficialTemporalReplayTests(unittest.TestCase):
    def test_replay_reports_establishment_and_reversal(self):
        scratch = Path(__file__).resolve().parents[1] / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=scratch, suffix=".jsonl", delete=False) as handle:
            path = Path(handle.name)
            for frame in range(2):
                handle.write(json.dumps({"event": "observation", "frame": frame, "track_id": 7,
                                         "label": "OFFICIAL", "observation_label": "OFFICIAL",
                                         "observation_confidence": .9,
                                         "evidence": {"stripe_ratio": 1.4}}) + "\n")
            for frame in range(2, 5):
                handle.write(json.dumps({"event": "observation", "frame": frame, "track_id": 7,
                                         "label": "AWAY", "observation_label": "AWAY",
                                         "observation_confidence": .9,
                                         "evidence": {"stripe_ratio": .2,
                                                      "stripe_evidence": {
                                                          "lower_body_dark_fraction": .1}}}) + "\n")
        try:
            report = replay(path, {"establish_window": 2, "establish_count": 2,
                                   "reverse_window": 3, "reverse_count": 3}, [7], [])
        finally:
            path.unlink()
        track = report["tracks"][0]
        self.assertTrue(track["official_established"])
        self.assertTrue(track["official_later_lost"])
        self.assertEqual([x["transition"] for x in track["structural_transitions"]],
                         ["ESTABLISHED", "REVERTED"])


if __name__ == "__main__":
    unittest.main()
