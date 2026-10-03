import copy
import json
from pathlib import Path
import unittest
import uuid
from unittest import mock

from appearance_features import measurement_quality
from tools.build_labeling_dataset import (LABELS, atomic_write_labels, load_labels,
    main, padded_crop_bounds, save_label, select_candidates, select_representatives,
    select_review_candidates, track_metadata)


def observation(game, track, frame, quality="GOOD", participant="ACCEPTED", x=10, height=100):
    mapped = measurement_quality(quality)
    return {"source_game": game, "temporary_track_id": track, "frame": frame,
            "timestamp": frame / 30, "source_video": "unused.mp4",
            "person_bbox": [x, 10, x + 40, 10 + height],
            "appearance_features": {"quality": {"crop_quality": quality,
                                                   "measurement_quality": mapped,
                                                   "detection_confidence": .8},
                "regions": {"torso": {"lab_lightness_median": .5},
                            "pants": {"dark_fraction": .4},
                            "lower_legs": {"lab_lightness_median": .6}},
                "structure": {"vertical_stripe_score": .2}},
            "participant_evidence": {"decision": participant},
            "legacy_metadata": {"label": "HOME", "is_human_ground_truth": False}}


def tracks(games=("g1", "g2"), per_game=10):
    return {(game, track): [observation(game, track, frame, "LOW" if track % 3 == 0 else "GOOD",
                                        "REJECTED" if track % 4 == 0 else "ACCEPTED",
                                        x=track * 25, height=30 + track * 10)
                            for frame in range(track % 7 + 1)]
            for game in games for track in range(per_game)}


class CandidateSelectionTests(unittest.TestCase):
    def test_deterministic_for_seed(self):
        data = tracks(per_game=20)
        first = [key for key, _ in select_candidates(data, 18, 57)]
        second = [key for key, _ in select_candidates(data, 18, 57)]
        self.assertEqual(first, second)

    def test_requested_count_and_multi_game_balance(self):
        chosen = select_candidates(tracks(per_game=10), 12, 57)
        self.assertEqual(len(chosen), 12)
        counts = {game: sum(key[0] == game for key, _ in chosen) for game in ("g1", "g2")}
        self.assertEqual(counts, {"g1": 6, "g2": 6})

    def test_target_caps_at_available(self):
        self.assertEqual(len(select_candidates(tracks(per_game=3), 300, 57)), 6)

    def test_no_benchmark_specific_selection_inputs(self):
        data = tracks(games=("arbitrary-alpha", "arbitrary-beta"), per_game=4)
        chosen = select_candidates(data, 4, 9)
        self.assertEqual({key[0] for key, _ in chosen}, {"arbitrary-alpha", "arbitrary-beta"})

    def test_representatives_are_temporally_diverse(self):
        items = [observation("g", 1, frame) for frame in range(60)]
        frames = [item["frame"] for item in select_representatives(items, 6)]
        self.assertEqual(len(frames), 6)
        self.assertGreaterEqual(min(b - a for a, b in zip(frames, frames[1:])), 8)

    def test_representatives_handle_short_tracklets(self):
        items = [observation("g", 1, frame) for frame in (3, 8)]
        self.assertEqual(len(select_representatives(items, 6)), 2)

    def test_representatives_include_quality_edge_case(self):
        items = [observation("g", 1, frame) for frame in range(20)]
        items[10] = observation("g", 1, 10, "LOW")
        selected = select_representatives(items, 6)
        self.assertIn("PARTIAL", {item["appearance_features"]["quality"]["measurement_quality"] for item in selected})

    def test_crop_padding_clips_to_image(self):
        self.assertEqual(padded_crop_bounds([-10, -5, 20, 30], 100, 80, .2), (0, 0, 26, 37))
        self.assertEqual(padded_crop_bounds([80, 60, 110, 90], 100, 80, .2), (74, 54, 100, 80))

    def test_measurement_quality_not_training_or_semantics(self):
        meta = track_metadata(("g", 1), [observation("g", 1, 0)])
        self.assertEqual(meta["measurement_quality_counts"], {"MEASURABLE": 1})
        self.assertNotIn("training_suitability", meta)
        self.assertNotIn("human_label", meta)


class HumanLabelTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path("output") / f"test_labeling_{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "labels.jsonl"
        self.candidate = {"source_game": "g", "temporary_track_id": 7,
                          "representative_frames": [1, 5]}

    def tearDown(self):
        for path in self.directory.iterdir():
            path.unlink()
        self.directory.rmdir()

    def test_allowed_values_only(self):
        for label in LABELS:
            save_label(self.path, "dataset", self.candidate, label)
        with self.assertRaises(ValueError):
            save_label(self.path, "dataset", self.candidate, "UNKNOWN")

    def test_labels_persist_reload_and_change(self):
        save_label(self.path, "dataset", self.candidate, "HOME")
        self.assertEqual(load_labels(self.path)[("g", 7)]["label"], "HOME")
        save_label(self.path, "dataset", self.candidate, "AWAY")
        loaded = load_labels(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[("g", 7)]["label"], "AWAY")

    def test_each_decision_preserves_prior_labels(self):
        save_label(self.path, "dataset", self.candidate, "HOME")
        other = {"source_game": "g", "temporary_track_id": 8, "representative_frames": [2]}
        save_label(self.path, "dataset", other, "UNSURE")
        self.assertEqual(set(load_labels(self.path)), {("g", 7), ("g", 8)})

    def test_source_game_and_track_id_form_subject_key(self):
        other = {"source_game": "other", "temporary_track_id": 7, "representative_frames": [2]}
        save_label(self.path, "dataset", self.candidate, "HOME")
        save_label(self.path, "dataset", other, "AWAY")
        self.assertEqual(len(load_labels(self.path)), 2)

    def test_pseudo_label_never_initializes_human_label(self):
        source = observation("g", 7, 1)
        self.assertEqual(source["legacy_metadata"]["label"], "HOME")
        self.assertEqual(load_labels(self.path), {})

    def test_atomic_write_rejects_invalid_label(self):
        with self.assertRaises(ValueError):
            atomic_write_labels(self.path, {("g", 1): {"source_game": "g", "track_id": 1,
                                                        "label": "PSEUDO_HOME"}})

    def test_source_observations_are_not_mutated(self):
        data = tracks(per_game=3)
        before = copy.deepcopy(data)
        select_candidates(data, 4, 57)
        for observations in data.values():
            select_representatives(observations, 2)
        self.assertEqual(data, before)


class ReviewModeTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [{"source_game": "g", "temporary_track_id": track,
                            "representative_frames": [track],
                            "legacy_metadata": {"label": "HOME"}}
                           for track in range(20)]
        self.labels = {("g", track): {"label": "UNSURE" if track % 2 else "NON_PARTICIPANT"}
                       for track in range(20)}

    def test_filtering_uses_existing_human_label_and_preserves_order(self):
        selected, info = select_review_candidates(self.candidates, self.labels, "UNSURE")
        self.assertEqual([item["temporary_track_id"] for item in selected], list(range(1, 20, 2)))
        self.assertEqual(info["matching"], 10)

    def test_pseudo_label_has_no_involvement(self):
        selected, _ = select_review_candidates(self.candidates, self.labels, "HOME")
        self.assertEqual(selected, [])

    def test_sampling_is_deterministic_and_retains_candidate_order(self):
        first, _ = select_review_candidates(self.candidates, self.labels, "UNSURE", 5, 57)
        second, _ = select_review_candidates(self.candidates, self.labels, "UNSURE", 5, 57)
        ids = [item["temporary_track_id"] for item in first]
        self.assertEqual(first, second)
        self.assertEqual(ids, sorted(ids))

    def test_different_seeds_can_change_sample(self):
        first, _ = select_review_candidates(self.candidates, self.labels, "UNSURE", 5, 1)
        second, _ = select_review_candidates(self.candidates, self.labels, "UNSURE", 5, 2)
        self.assertNotEqual(first, second)

    def test_sample_larger_than_available_reviews_all(self):
        selected, info = select_review_candidates(self.candidates, self.labels, "UNSURE", 99, 57)
        self.assertEqual(len(selected), 10)
        self.assertTrue(info["sample_capped"])

    def test_loading_review_does_not_modify_labels_or_candidates(self):
        candidates_before = copy.deepcopy(self.candidates)
        labels_before = copy.deepcopy(self.labels)
        select_review_candidates(self.candidates, self.labels, "UNSURE", 3, 57)
        self.assertEqual(self.candidates, candidates_before)
        self.assertEqual(self.labels, labels_before)

    def test_label_change_persists_but_does_not_mutate_in_memory_review_set(self):
        directory = Path("output") / f"test_review_{uuid.uuid4().hex}"
        directory.mkdir(parents=True)
        path = directory / "labels.jsonl"
        try:
            records = {key: {"schema_version": 1, "dataset_id": "d", "source_game": key[0],
                             "track_id": key[1], "label": value["label"], "reviewed_at": "old",
                             "representative_frames": [key[1]], "reviewer": None}
                       for key, value in self.labels.items()}
            atomic_write_labels(path, records)
            selected, _ = select_review_candidates(self.candidates, load_labels(path), "UNSURE")
            first = selected[0]
            original_ids = [item["temporary_track_id"] for item in selected]
            save_label(path, "d", first, "HOME")
            self.assertEqual(load_labels(path)[("g", first["temporary_track_id"])]["label"], "HOME")
            self.assertEqual([item["temporary_track_id"] for item in selected], original_ids)
        finally:
            for item in directory.iterdir():
                item.unlink()
            directory.rmdir()

    def test_invalid_review_label_is_rejected(self):
        with mock.patch("sys.argv", ["tool", "--label", "project", "--review-label", "INVALID"]):
            with self.assertRaises(SystemExit):
                main()

    def test_review_sample_requires_review_label(self):
        with mock.patch("sys.argv", ["tool", "--label", "project", "--review-sample", "3"]):
            with self.assertRaises(SystemExit):
                main()

    def test_candidate_manifest_and_representative_frames_remain_unchanged(self):
        serialized = json.dumps(self.candidates, sort_keys=True)
        frames = [item["representative_frames"][:] for item in self.candidates]
        select_review_candidates(self.candidates, self.labels, "NON_PARTICIPANT", 4, 57)
        self.assertEqual(json.dumps(self.candidates, sort_keys=True), serialized)
        self.assertEqual([item["representative_frames"] for item in self.candidates], frames)


if __name__ == "__main__":
    unittest.main()
