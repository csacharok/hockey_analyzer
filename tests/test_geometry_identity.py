import unittest
from unittest.mock import patch

from target_identity import Observation, TargetIdentity


def player(identifier=1, width=40, height=100, appearance=(1., 0.)):
    return Observation(identifier, (120-width/2, 150-height/2,
                                    120+width/2, 150+height/2), appearance=appearance)


class GeometryIdentityTests(unittest.TestCase):
    def seeded(self):
        identity = TargetIdentity(30, grace_frames=0)
        identity.update(0, [player()], (120, 150))
        for frame in range(1, 10):
            identity.update(frame, [player()])
        return identity

    def evidence(self, observation):
        identity = self.seeded()
        identity.update(10, [observation])
        return identity, identity.candidates[0]

    def test_uniform_scale(self):
        _, c = self.evidence(player(2, 48, 120))
        self.assertEqual(c['geometry_decision'], 'passed')

    def test_pose_height_is_soft_and_starts_confirmation(self):
        identity, c = self.evidence(player(2, height=60))
        self.assertEqual(c['geometry_decision'], 'soft_mismatch')
        self.assertNotIn('box_size_change', c['reasons'])
        self.assertEqual(identity.pending_count, 1)
        self.assertEqual(identity.state, 'LOST')
        for frame in (11, 12):
            identity.update(frame, [player(2, height=60)])
        self.assertEqual(identity.state, 'REACQUIRED')

    def test_width_only_is_soft(self):
        _, c = self.evidence(player(2, width=64))
        self.assertEqual(c['geometry_decision'], 'soft_mismatch')
        self.assertFalse(c['geometry_hard_reasons'])

    def test_extreme_scale(self):
        _, c = self.evidence(player(2, 160, 400))
        self.assertEqual(c['geometry_decision'], 'hard_rejected')
        self.assertIn('area_outside_robust_factor_4', c['geometry_hard_reasons'])

    def test_extreme_aspect_with_unchanged_area(self):
        _, c = self.evidence(player(2, 72, 100/1.8))
        self.assertAlmostEqual(c['area_ratio'], 1.)
        self.assertEqual(c['geometry_hard_reasons'], ['aspect_outside_robust_factor_2'])

    def test_anomalous_last_box_does_not_poison_history(self):
        identity = self.seeded()
        identity.update(10, [player(height=250)])
        identity.update(11, [player(2)])
        c = identity.candidates[0]
        self.assertAlmostEqual(c['height_ratio'], .4)
        self.assertEqual(c['geometry_decision'], 'passed')
        self.assertEqual(identity.pending_count, 1)

    def test_geometry_alone_cannot_accept(self):
        identity = self.seeded()
        for frame in range(10, 15):
            identity.update(frame, [player(2, appearance=())])
            self.assertEqual(identity.state, 'LOST')
            self.assertIsNone(identity.pending)

    def test_weak_appearance_with_soft_mismatch_rejected(self):
        identity, c = self.evidence(player(2, height=60, appearance=(.6, .4)))
        self.assertEqual(c['geometry_decision'], 'soft_mismatch')
        self.assertIn('appearance_mismatch', c['reasons'])
        self.assertIsNone(identity.pending)

    def test_incompatible_competitor_remains_rejected(self):
        identity = self.seeded()
        for frame in range(10, 13):
            identity.update(frame, [player(2), player(3, 160, 400)])
            self.assertIn('box_size_change', identity.candidates[1]['reasons'])
        self.assertEqual(identity.current.track_id, 2)

    def test_pending_and_rejected_observations_do_not_train(self):
        identity = self.seeded()
        history = list(identity.geometry_history)
        identity.update(10, [player(2, height=60)])
        self.assertEqual(list(identity.geometry_history), history)
        identity.update(11, [player(2, height=10)])
        self.assertEqual(list(identity.geometry_history), history)
        self.assertIsNone(identity.pending)

    def test_history_ages_by_observation_time_and_is_bounded(self):
        identity = self.seeded()
        for frame in range(10, 40):
            identity.update(frame, [player()])
        self.assertEqual(len(identity.geometry_history), 30)
        self.assertEqual(identity.geometry_history[0][0], 10)
        for frame in range(40, 72):
            identity.update(frame, [])
        for frame in range(72, 75):
            identity.update(frame, [player(2)])
        self.assertEqual(list(identity.geometry_history), [(74, (40., 100.))])

    def test_authoritative_binding_skips_geometry_evaluation(self):
        identity = self.seeded()
        with patch.object(identity, '_geometry', side_effect=AssertionError('scored bound ID')):
            identity.update(10, [player(height=1000)])
        self.assertEqual(identity.state, 'TRACKED')

    def test_diagnostics_preserve_old_ratios_and_score(self):
        _, c = self.evidence(player(2, height=60))
        self.assertEqual(c['size_ratios'], [1., .6])
        self.assertAlmostEqual(c['legacy_size_score'], .6)
        self.assertEqual(c['geometry_sample_count'], 10)
        self.assertAlmostEqual(c['geometry_score_contribution'], .2*c['geometry_similarity'])
        self.assertEqual(set(c['geometry_statistics']), {'width', 'height', 'area', 'aspect'})


if __name__ == '__main__':
    unittest.main()
