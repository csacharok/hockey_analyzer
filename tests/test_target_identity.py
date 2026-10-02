import math
import ast
import json
from pathlib import Path
import uuid
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analyze_tracks import analyze, draw_target, target_observations
from rink_geometry import ON_ICE, OFF_ICE, UNKNOWN
from target_identity import Observation, TargetIdentity, appearance_histogram


def person(identifier=12, x=100, width=40, height=100, state=ON_ICE, appearance=(.8, .2)):
    return Observation(identifier, (x, 100, x + width, 100 + height), state, appearance)


class IdentityTests(unittest.TestCase):
    def seeded(self):
        identity = TargetIdentity(30, grace_frames=0)
        identity.update(0, [person()], (120, 150))
        return identity

    def test_seed_and_continuation(self):
        identity = self.seeded()
        self.assertEqual(identity.records[0]['event'], 'seeded')
        self.assertEqual(identity.records[0]['new_id'], 12)
        identity.update(1, [person(x=102)])
        self.assertEqual(identity.state, 'TRACKED')
        self.assertEqual(identity.current.track_id, 12)

    def test_seed_rejects_ambiguous_or_empty_selection(self):
        for observations in ([], [person(), person(19)]):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                TargetIdentity(30).update(0, observations, (120, 150))

    def test_seed_requires_finite_coordinates(self):
        with self.assertRaises(ValueError):
            TargetIdentity(30).update(0, [person()], (math.nan, 150))

    def test_transfer_requires_three_frames_and_logs_transition(self):
        identity = self.seeded()
        for frame in (1, 2):
            self.assertIsNone(identity.update(frame, [person(19)]))
            self.assertEqual(identity.state, 'LOST')
        identity.update(3, [person(19)])
        self.assertEqual(identity.state, 'REACQUIRED')
        self.assertIn('Identity transition: 12 -> 19', identity.events)
        self.assertEqual(identity.ids, [12, 19])
        identity.update(4, [person(19)])
        self.assertEqual(identity.state, 'TRACKED')

    def test_returning_same_id_requires_confirmation(self):
        identity = self.seeded()
        identity.update(1, [])
        self.assertEqual(identity.records[0]['event'], 'lost')
        self.assertEqual(identity.records[0]['previous_id'], 12)
        for frame in (2, 3, 4):
            identity.update(frame, [person()])
        self.assertEqual(identity.state, 'REACQUIRED')
        self.assertEqual(identity.ids, [12])
        self.assertFalse(any('transition' in event for event in identity.events))

    def test_bound_id_is_authoritative_despite_weak_evidence(self):
        for candidate in (person(x=400), person(width=100), person(height=200),
                          person(state=OFF_ICE), person(appearance=(.1, .9)), person(appearance=())):
            with self.subTest(candidate=candidate):
                identity = self.seeded()
                identity.update(1, [candidate])
                self.assertEqual(identity.state, 'TRACKED')
                self.assertEqual(identity.current.track_id, 12)
                self.assertEqual(identity.candidates, [])

    def test_competing_candidates_cannot_override_bound_id(self):
        for observations in ([person(), person(19, x=101)], [person(19, x=101), person()]):
            identity = self.seeded()
            identity.update(1, observations)
            self.assertEqual(identity.state, 'TRACKED')
            self.assertEqual(identity.current.track_id, 12)

    def test_unknown_rink_is_not_a_conflict(self):
        identity = self.seeded()
        identity.update(1, [person(state=UNKNOWN)])
        self.assertEqual(identity.state, 'TRACKED')

    def test_grace_then_lost_only_on_absence(self):
        identity = TargetIdentity(30, grace_frames=2)
        identity.update(0, [person()], (120, 150))
        for frame in (1, 2):
            identity.update(frame, [person(19)])
            self.assertEqual(identity.state, 'GRACE')
            self.assertIsNone(identity.current)
            self.assertEqual(identity.records, [])
            self.assertEqual(identity.candidates, [])
        identity.update(3, [person(19)])
        self.assertEqual(identity.state, 'LOST')
        event = identity.records[0]
        self.assertEqual(event['missing_since_frame'], 1)
        self.assertAlmostEqual(event['gap_seconds'], .1)
        self.assertEqual(identity.pending_count, 1)

    def test_single_dropout_returns_without_reacquisition_or_scoring(self):
        identity = TargetIdentity(30)
        identity.update(0, [person()], (120, 150))
        identity.update(1, [])
        with patch.object(identity, '_evaluate', side_effect=AssertionError('Bound ID must not be scored')):
            identity.update(2, [person(x=500, appearance=(.1, .9))])
        self.assertEqual(identity.state, 'TRACKED')
        self.assertEqual(identity.transitions, [])
        self.assertEqual(identity.counts['GRACE'], 1)

    def test_reacquired_id_becomes_authoritative(self):
        identity = self.seeded()
        for frame in (1, 2, 3):
            identity.update(frame, [person(19)])
        self.assertEqual(identity.state, 'REACQUIRED')
        with patch.object(identity, '_evaluate', side_effect=AssertionError('Reacquired ID must not be scored')):
            identity.update(4, [person(19, x=500, appearance=()), person(20)])
        self.assertEqual(identity.current.track_id, 19)
        self.assertEqual(identity.state, 'TRACKED')
        identity.update(5, [])
        self.assertEqual(identity.state, 'LOST')
        self.assertEqual(identity.records[0]['previous_id'], 19)

    def test_weak_or_ambiguous_reacquisition_stays_lost(self):
        for observations, reason in (([person(19, appearance=(.1, .9))], 'appearance_mismatch'),
                                     ([person(19), person(20)], 'ambiguous_margin'),
                                     ([person(19, x=800)], 'unreachable_displacement'),
                                     ([person(19, width=100)], 'box_size_change'),
                                     ([person(19, state=OFF_ICE)], 'rink_state_conflict'),
                                     ([person(19, appearance=())], 'appearance_unavailable')):
            with self.subTest(reason=reason):
                identity = self.seeded()
                for frame in range(1, 5):
                    identity.update(frame, observations)
                self.assertEqual(identity.state, 'LOST')
                self.assertIsNone(identity.current)
                self.assertIn(reason, identity.candidates[0]['reasons'])

    def test_existing_other_skater_is_not_a_new_fragment(self):
        identity = TargetIdentity(30, grace_frames=0)
        identity.update(0, [person(), person(19, x=300)], (120, 150))
        for frame in range(1, 20):
            identity.update(frame, [person(), person(19, x=300)])
        for frame in range(20, 24):
            identity.update(frame, [person(19)])
        self.assertEqual(identity.state, 'LOST')
        self.assertIn('track_predates_disappearance', identity.candidates[0]['reasons'])

    def test_two_short_occlusions_stitch_generic_ids(self):
        identity = self.seeded()
        for frame in range(1, 34):
            identity.update(frame, [])
        for frame in range(34, 37):
            identity.update(frame, [person(19)])
        self.assertEqual(identity.state, 'REACQUIRED')
        for frame in range(37, 50):
            identity.update(frame, [person(19)])
        for frame in range(50, 104):
            identity.update(frame, [])
        for frame in range(104, 107):
            identity.update(frame, [person(27)])
        self.assertEqual(identity.state, 'REACQUIRED')
        self.assertEqual(identity.ids, [12, 19, 27])
        self.assertAlmostEqual(identity.transitions[0]['gap_seconds'], 1.2)
        self.assertAlmostEqual(identity.transitions[1]['gap_seconds'], 1.9)
        self.assertIn('12 -> 19 -> 27', identity.summary())

    def test_timeout_does_not_reassign_even_original_id(self):
        identity = self.seeded()
        for frame in range(1, 100):
            identity.update(frame, [])
        for frame in range(100, 105):
            identity.update(frame, [person()])
        self.assertEqual(identity.state, 'LOST')

    def test_confirmation_resets_on_missing_or_different_candidate(self):
        identity = self.seeded()
        for frame, observations in enumerate(([person(19)], [], [person(19)], [person(20)], [person(19)]), 1):
            identity.update(frame, observations)
            self.assertEqual(identity.state, 'LOST')

    def test_motion_prediction_reacquires_moving_player(self):
        identity = self.seeded()
        for frame in range(1, 6):
            identity.update(frame, [person(x=100 + frame * 5)])
        identity.update(6, [])
        for frame in (7, 8, 9):
            identity.update(frame, [person(19, x=100 + frame * 5)])
        self.assertEqual(identity.state, 'REACQUIRED')

    def test_seed_anchor_does_not_drift(self):
        identity = self.seeded()
        identity.update(1, [person(appearance=(.6, .4))])
        identity.update(2, [person(appearance=(.4, .6))])
        self.assertEqual(identity.state, 'TRACKED')
        self.assertEqual(identity.anchor, (.8, .2))

    def test_short_gap_wrong_velocity_uses_reachable_position(self):
        identity = self.seeded()
        for frame in range(1, 6):
            identity.update(frame, [person(x=100 + frame * 6)])
        for frame in range(6, 20):
            identity.update(frame, [])
        confidences = []
        for frame in range(20, 23):
            identity.update(frame, [person(19, x=130)])
            evidence = identity.candidates[0]
            self.assertGreater(evidence['prediction_error_heights'], evidence['prediction_limit_heights'])
            self.assertEqual(evidence['prediction_authority'], 'blended')
            self.assertEqual(evidence['motion_decision'], 'passed')
            confidences.append(evidence['velocity_confidence'])
            if frame < 22:
                self.assertIsNone(identity.current)
        self.assertEqual(identity.state, 'REACQUIRED')
        self.assertGreater(confidences[0], confidences[-1])

    def gallery_identity(self, matching_samples):
        identity = TargetIdentity(30, grace_frames=0)
        for frame in range(15):
            view = (1., 0.) if frame < matching_samples else (0., 1.)
            identity.update(frame, [person(appearance=view)], (120, 150) if frame == 0 else None)
        return identity

    def test_high_confidence_prediction_remains_authoritative(self):
        identity = self.seeded()
        identity.velocity = (100., 0.)
        identity.update(1, [person(19)])
        evidence = identity.candidates[0]
        self.assertEqual(evidence['prediction_authority'], 'authoritative')
        self.assertIn('prediction_distance', evidence['reasons'])
        self.assertNotIn('unreachable_displacement', evidence['reasons'])

    def test_medium_confidence_blends_normalized_errors(self):
        identity = self.seeded()
        identity.velocity = (5., 0.)
        identity.first_seen[19] = 15
        evidence = identity._evaluate(person(19, x=125), 15)
        weight = evidence['prediction_weight']
        self.assertGreater(weight, 0)
        self.assertLess(weight, 1)
        expected = weight * .5 / .675 + (1 - weight) * .25 / 1.75
        self.assertAlmostEqual(evidence['effective_motion_fraction'], expected)
        self.assertAlmostEqual(expected, evidence['prediction_contribution'] + evidence['displacement_contribution'])

    def test_low_confidence_gallery_confirmation_and_safety(self):
        for scenario in ('valid', 'unreachable', 'ambiguous', 'size'):
            with self.subTest(scenario=scenario):
                identity = self.gallery_identity(3)
                identity.velocity = (10000., 0.)
                for frame in range(15, 66):
                    identity.update(frame, [])
                # Rejected appearance must not count toward confirmation.
                identity.update(66, [person(19, appearance=())])
                for frame in range(67, 70):
                    observations = [person(19, x=10000 if scenario == 'unreachable' else 120,
                                           width=100 if scenario == 'size' else 40, appearance=(1., 0.))]
                    if scenario == 'ambiguous':
                        observations.append(person(20, x=120, appearance=(1., 0.)))
                    identity.update(frame, observations)
                    evidence = identity.candidates[0]
                    self.assertTrue(evidence['gallery_supported'])
                    self.assertEqual(evidence['prediction_authority'], 'diagnostic_only')
                    self.assertEqual(evidence['prediction_contribution'], 0)
                    if scenario != 'unreachable':
                        self.assertNotIn('prediction_distance', evidence['reasons'])
                    if scenario == 'valid':
                        self.assertEqual(evidence['confirmation_frames'], frame - 66)
                        self.assertEqual(identity.state, 'REACQUIRED' if frame == 69 else 'LOST')
                    else:
                        self.assertEqual(identity.state, 'LOST')
                        self.assertIn(dict(unreachable='unreachable_displacement', ambiguous='ambiguous_margin',
                                           size='box_size_change')[scenario], evidence['reasons'])

    def test_motion_is_continuous_at_old_cutoff_and_confidence_boundaries(self):
        identity = self.seeded()
        identity.velocity = (5., 0.)
        identity.first_seen[19] = 0
        for gap in (1., -.5 * math.log2(.90), -.5 * math.log2(.10)):
            evidence = [identity._evaluate(person(19, x=125), (gap + delta) * 30)
                        for delta in (-1e-7, 0, 1e-7)]
            errors = [e['effective_motion_fraction'] for e in evidence]
            self.assertLess(max(errors) - min(errors), 1e-5)
            self.assertEqual(len({e['motion_decision'] for e in evidence}), 1)

    def test_gallery_preserves_view_and_requires_confirmation(self):
        identity = self.gallery_identity(3)
        for frame in range(15, 18):
            identity.update(frame, [person(19, appearance=(1., 0.))])
            evidence = identity.candidates[0]
            self.assertAlmostEqual(evidence['appearance_similarity'], .2)
            self.assertEqual(evidence['seed_similarity'], 1.)
            self.assertEqual(evidence['gallery_best_similarity'], 1.)
            self.assertEqual(evidence['gallery_top_k_mean_similarity'], 1.)
            self.assertEqual(evidence['appearance_model'], 'gallery_top3')
            if frame < 17:
                self.assertEqual(identity.state, 'LOST')
        self.assertEqual(identity.state, 'REACQUIRED')

    def test_single_lucky_gallery_sample_cannot_reacquire(self):
        identity = self.gallery_identity(1)
        gallery = list(identity.appearances)
        for frame in range(15, 19):
            identity.update(frame, [person(19, appearance=(1., 0.))])
            self.assertEqual(identity.candidates[0]['gallery_best_similarity'], 1.)
            self.assertFalse(identity.candidates[0]['gallery_supported'])
            self.assertIn('appearance_mismatch', identity.candidates[0]['reasons'])
            self.assertEqual(identity.state, 'LOST')
        self.assertEqual(list(identity.appearances), gallery)

    def test_gallery_confirmation_resets_after_weak_frame(self):
        identity = self.gallery_identity(3)
        for frame, view in enumerate(((1., 0.), (), (1., 0.), (1., 0.)), 15):
            identity.update(frame, [person(19, appearance=view)])
            self.assertEqual(identity.state, 'LOST')
        identity.update(19, [person(19, appearance=(1., 0.))])
        self.assertEqual(identity.state, 'REACQUIRED')

    def test_gallery_does_not_override_safety_gates(self):
        for scenario in ('unreachable', 'ambiguous', 'expired'):
            with self.subTest(scenario=scenario):
                identity = self.gallery_identity(3)
                start = 15
                if scenario == 'expired':
                    for frame in range(15, 106):
                        identity.update(frame, [])
                    start = 106
                for frame in range(start, start + 3):
                    candidates = [person(19, x=10000 if scenario == 'unreachable' else 100,
                                         appearance=(1., 0.))]
                    if scenario == 'ambiguous':
                        candidates.append(person(20, appearance=(1., 0.)))
                    identity.update(frame, candidates)
                    reason = dict(unreachable='unreachable_displacement', ambiguous='ambiguous_margin',
                                  expired='search_window_expired')[scenario]
                    self.assertIn(reason, identity.candidates[0]['reasons'])
                    self.assertEqual(identity.state, 'LOST')

    def test_benchmark_track_ids_are_not_used_in_production_branches(self):
        source = Path(__file__).resolve().parents[1] / 'target_identity.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                self.assertNotIn(node.value, (669, 840))
            if isinstance(node, ast.Compare) and 'track_id' in ast.unparse(node):
                self.assertFalse(any(isinstance(child, ast.Constant) and isinstance(child.value, int)
                                     for child in ast.walk(node)))

    def test_summary_accounts_for_all_frames(self):
        identity = TargetIdentity(30, grace_frames=0)
        identity.update(0, [])
        identity.update(1, [person()], (120, 150))
        for frame in (2, 3, 4):
            identity.update(frame, [person(19)])
        self.assertEqual(identity.counts, dict(UNSEEDED=1, TRACKED=1, LOST=2, REACQUIRED=1, GRACE=0))
        self.assertIn('40.00% of 5', identity.summary())
        self.assertIn('reacquisitions: 1', identity.summary())

    def test_invalid_api_inputs(self):
        for fps in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                TargetIdentity(fps)
        for options in (dict(grace_frames=-1), dict(grace_frames=1.5), dict(max_gap_seconds=0),
                        dict(max_gap_seconds=math.nan), dict(max_gap_seconds=math.inf)):
            with self.assertRaises(ValueError):
                TargetIdentity(30, **options)
        with self.assertRaises(ValueError):
            TargetIdentity(30).update(1, [])
        with self.assertRaises(ValueError):
            TargetIdentity(30).update(0, [person(), person()])
        for options in (dict(target_frame=-1), dict(target_point=(1, 2)),
                        dict(target_frame=0, target_point=(math.nan, 2))):
            with self.assertRaises(ValueError):
                analyze(None, **options)

    def test_lost_overlay_never_draws_target_box(self):
        identity = self.seeded()
        identity.update(1, [])
        cv2 = Mock()
        draw_target(SimpleNamespace(shape=(1080, 1920, 3)), identity, cv2)
        self.assertEqual(cv2.rectangle.call_count, 1)  # banner only
        self.assertIn('LOST', cv2.putText.call_args.args[1])

    def test_grace_overlay_has_no_stale_box_or_score(self):
        identity = TargetIdentity(30)
        identity.update(0, [person()], (120, 150))
        identity.update(1, [])
        cv2 = Mock()
        draw_target(SimpleNamespace(shape=(1080, 1920, 3)), identity, cv2)
        self.assertEqual(cv2.rectangle.call_count, 1)
        self.assertIn('GRACE (not observed)', cv2.putText.call_args.args[1])
        self.assertEqual(identity.debug_frame()['observed_id'], None)
        self.assertEqual(identity.debug_frame()['bound_id'], 12)

    def test_bound_overlay_does_not_imply_scored_identity(self):
        identity = self.seeded()
        cv2 = Mock()
        draw_target(SimpleNamespace(shape=(1080, 1920, 3)), identity, cv2)
        banner = cv2.putText.call_args.args[1]
        self.assertIn('Track 12 | TRACKED', banner)
        self.assertNotIn('score', banner.lower())

    def test_observation_adapter_ignores_unconfirmed_detections(self):
        self.assertEqual(target_observations(None, SimpleNamespace(is_track=False), Mock()), [])

    def test_histogram_uses_actual_pixels_and_is_normalized(self):
        import cv2
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :] = (0, 255, 0)
        histogram = appearance_histogram(frame, (0, 0, 100, 100), cv2)
        self.assertAlmostEqual(sum(histogram), 1)
        self.assertEqual(len(histogram), 128)
        self.assertEqual(appearance_histogram(frame, (0, 0, 2, 2), cv2), ())

    def test_video_workflow_with_controlled_id_change(self):
        """Real OpenCV decode/encode, deterministic detector: no GPU or downloads."""
        import cv2
        import numpy as np
        scratch = Path(__file__).resolve().parents[1] / 'scratch'
        scratch.mkdir(exist_ok=True)
        # mkdir inherits workspace ACLs; Windows mkdtemp's private ACL can deny
        # access to the sandbox runner's subsequent file operations.
        root = scratch / f'identity_test_{uuid.uuid4().hex}'
        root.mkdir()
        try:
            source = root / 'fixture.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.full((128, 128, 3), (50, 150, 30), dtype=np.uint8))
            writer.release()
            original = source.read_bytes()
            (root / 'yolo11s.pt').write_bytes(b'local mock weights')
            model = Mock()
            frame_number = 0

            def track(frame, **options):
                nonlocal frame_number
                boxes = Mock(is_track=True)
                boxes.cpu.return_value = boxes
                boxes.xyxy.tolist.return_value = [[30, 20, 70, 110]]
                boxes.xyxy.cpu.return_value = boxes.xyxy
                boxes.id.tolist.return_value = [12 if frame_number == 0 else 19]
                boxes.conf.tolist.return_value = [.9]
                result = SimpleNamespace(boxes=boxes)
                model.add_callback.call_args.args[1](SimpleNamespace(results=[result]))
                frame_number += 1
                return [result]

            model.track.side_effect = track
            with patch('analyze_tracks.ROOT', root), patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=Mock(return_value=model))}), patch('builtins.print') as printed:
                output = analyze(source, target_frame=0, target_point=(50, 50), target_grace_frames=0)
                messages = '\n'.join(str(call.args[0]) for call in printed.call_args_list)
                self.assertIn('Identity transition: 12 -> 19', messages)
                self.assertIn('60.00% of 5', messages)
                records = [json.loads(line) for line in output.with_suffix('.identity.jsonl').read_text().splitlines()]
                self.assertEqual(records[0]['event'], 'configuration')
                self.assertTrue(records[-1]['completed'])
                self.assertEqual(records[-1]['associated_ids'], [12, 19])
                reacquired = next(record for record in records if record['event'] == 'reacquired')
                self.assertEqual(reacquired['previous_id'], 12)
                self.assertEqual(reacquired['new_id'], 19)
                self.assertAlmostEqual(reacquired['gap_seconds'], .1)
                with patch('analyze_tracks.TargetIdentity') as identity_class:
                    baseline = analyze(source)
                    identity_class.assert_not_called()
                self.assertFalse(baseline.with_suffix('.identity.jsonl').exists())
                self.assertNotEqual(output, baseline)
            self.assertEqual(source.read_bytes(), original)
            capture = cv2.VideoCapture(str(output))
            decoded = 0
            while capture.read()[0]:
                decoded += 1
            capture.release()
            self.assertEqual(decoded, 5)
        finally:
            if (root / 'output').exists():
                for artifact in (root / 'output').iterdir():
                    artifact.unlink()
                (root / 'output').rmdir()
            for artifact in root.iterdir():
                artifact.unlink()
            root.rmdir()


if __name__ == '__main__':
    unittest.main()
