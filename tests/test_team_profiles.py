import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from team_profile_experiment import cluster, load_candidates
from team_profiles import AWAY, HOME, UNKNOWN, GameTeamProfiles


def profile_configuration(mapping=None):
    return {
        "clusters": [
            {"cluster": 0, "default_team": AWAY, "lightness_center": [.25, .18, .35],
             "lightness_scale": [.05, .05, .05], "maximum_distance": 4},
            {"cluster": 1, "default_team": HOME, "lightness_center": [.75, .65, .85],
             "lightness_scale": [.05, .05, .05], "maximum_distance": 4},
        ],
        "team_mapping": mapping or {},
    }


class GameTeamProfileTests(unittest.TestCase):
    def test_two_light_dark_groups_create_game_specific_profiles(self):
        samples = []
        for value in (.20, .22, .25, .72, .75, .78):
            feature = np.array([value, value - .05, value + .05, .5, .5, .1, value, .1],
                               dtype=np.float32)
            samples.append({"feature": feature, "row": {"track_id": len(samples)},
                            "crop": np.zeros((4, 4, 3), dtype=np.uint8)})
        profiles, separation = cluster(samples)
        self.assertGreater(separation, 2)
        mapped = {profile["default_team"]: profile for profile in profiles}
        self.assertGreater(mapped[HOME]["median_lightness"], mapped[AWAY]["median_lightness"])

    def test_lighter_defaults_home_and_darker_defaults_away(self):
        classifier = GameTeamProfiles(profile_configuration())
        self.assertEqual(classifier.classify([.76, .66, .86])[0], HOME)
        self.assertEqual(classifier.classify([.24, .17, .34])[0], AWAY)

    def test_mapping_can_be_swapped(self):
        classifier = GameTeamProfiles(profile_configuration({"0": HOME, "1": AWAY}))
        self.assertEqual(classifier.classify([.24, .17, .34])[0], HOME)
        self.assertEqual(classifier.classify([.76, .66, .86])[0], AWAY)

    def test_ambiguous_or_outlier_uniform_remains_unknown(self):
        classifier = GameTeamProfiles(profile_configuration())
        self.assertEqual(classifier.classify([.50, .45, .55])[0], UNKNOWN)
        self.assertEqual(classifier.classify([.98, .98, .98])[0], UNKNOWN)

    def test_structural_official_is_excluded_from_profile_candidates(self):
        scratch = Path(__file__).resolve().parents[1] / "scratch"
        scratch.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=scratch, suffix=".jsonl", delete=False) as handle:
            path = Path(handle.name)
            base = {"event": "observation", "participant_state": "ACCEPTED",
                    "observation_label": "HOME", "detection_confidence": .9,
                    "box": [100, 100, 150, 200]}
            handle.write(json.dumps({**base, "frame": 1, "track_id": 1,
                                     "evidence": {"stripe_ratio": 1.2}}) + "\n")
            handle.write(json.dumps({**base, "frame": 2, "track_id": 2,
                                     "evidence": {"stripe_ratio": .2}}) + "\n")
        try:
            rows = load_candidates(path)
        finally:
            path.unlink()
        self.assertEqual([row["track_id"] for row in rows], [2])


if __name__ == "__main__":
    unittest.main()
