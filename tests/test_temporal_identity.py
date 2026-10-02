import unittest
from unittest.mock import patch

from rink_geometry import ON_ICE, OFF_ICE
from target_identity import Observation, TargetIdentity


def candidate(track_id=20, similarity=.6, x=100, width=40, state=ON_ICE):
    return Observation(track_id, (x, 100, x + width, 200), state,
                       (similarity, 1 - similarity))


class TemporalIdentityTests(unittest.TestCase):
    def seeded(self):
        identity = TargetIdentity(30, grace_frames=0)
        identity.update(0, [candidate(10, 1.)], (120, 150))
        return identity

    def started(self):
        identity = self.seeded()
        identity.update(1, [candidate(similarity=.7)])
        self.assertEqual(identity.pending_count, 1)
        return identity

    def reset_reason(self, identity):
        return next(r['reason'] for r in identity.records
                    if r['event'] == 'REACQUISITION_PENDING_RESET')

    def test_support_only_cannot_start(self):
        identity = self.seeded()
        for frame in range(1, 8):
            identity.update(frame, [candidate()])
            self.assertIsNone(identity.pending)
            self.assertEqual(identity.state, 'LOST')

    def test_strong_then_support_confirms_and_diagnoses(self):
        identity = self.started()
        identity.update(2, [candidate()])
        self.assertEqual(identity.records[0]['event'], 'REACQUISITION_PENDING_CONTINUED')
        identity.update(3, [candidate()])
        self.assertEqual(identity.state, 'REACQUIRED')
        record = identity.records[0]
        self.assertEqual(record['event'], 'TARGET_REACQUIRED')
        self.assertEqual(record['observation_count'], 3)
        self.assertAlmostEqual(record['temporal_mean_appearance'], 1.9 / 3)
        self.assertEqual(record['supporting_frame_count'], 3)

    def test_strong_contradiction_resets(self):
        identity = self.started()
        identity.update(2, [candidate(similarity=.1)])
        self.assertIsNone(identity.pending)
        self.assertEqual(self.reset_reason(identity), 'strongly_contradictory_appearance')

    def test_motion_failure_resets(self):
        identity = self.started()
        identity.velocity = (100., 0.)
        identity.update(2, [candidate()])
        self.assertIn('prediction_distance', self.reset_reason(identity))

    def test_unreachable_resets(self):
        identity = self.started()
        identity.update(2, [candidate(x=1000)])
        self.assertIn('unreachable_displacement', self.reset_reason(identity))

    def test_new_id_cannot_inherit(self):
        identity = self.started()
        identity.update(2, [candidate(21)])
        self.assertEqual(self.reset_reason(identity), 'candidate_missing')
        identity.update(3, [candidate(21)])
        self.assertIsNone(identity.pending)
        identity.update(4, [candidate(21, .7)])
        self.assertEqual(identity.pending_count, 1)
        self.assertEqual(identity.pending.track_id, 21)

    def test_support_level_competitor_resets(self):
        identity = self.started()
        identity.update(2, [candidate(), candidate(21)])
        self.assertEqual(self.reset_reason(identity), 'ambiguous_margin')
        self.assertEqual(identity.state, 'LOST')

    def test_search_expiry_resets_without_candidate(self):
        identity = self.seeded()
        identity.max_gap_seconds = 2 / 30
        identity.update(1, [candidate(similarity=.7)])
        identity.update(2, [candidate()])
        identity.update(3, [])
        self.assertEqual(self.reset_reason(identity), 'search_window_expired')

    def test_isolated_spike_cannot_confirm_and_window_expires(self):
        identity = self.started()
        for frame in range(2, 17):
            identity.update(frame, [candidate(similarity=.5)])
            self.assertEqual(identity.state, 'LOST')
        self.assertEqual(self.reset_reason(identity), 'pending_window_expired')

    def test_authoritative_binding_after_acceptance(self):
        identity = self.started()
        for frame in (2, 3):
            identity.update(frame, [candidate()])
        with patch.object(identity, '_evaluate', side_effect=AssertionError('bound ID scored')):
            identity.update(4, [candidate(similarity=.01, x=1000, state=OFF_ICE)])
        self.assertEqual(identity.state, 'TRACKED')
        self.assertEqual(identity.current.track_id, 20)

    def test_size_and_rink_conflicts_reset(self):
        for observation, reason in ((candidate(width=100), 'box_size_change'),
                                    (candidate(state=OFF_ICE), 'rink_state_conflict')):
            with self.subTest(reason=reason):
                identity = self.started()
                identity.update(2, [observation])
                self.assertEqual(self.reset_reason(identity), reason)

    def test_mean_and_fraction_prevent_mediocre_confirmation(self):
        identity = self.seeded()
        identity.update(1, [candidate(similarity=.65)])
        for frame in range(2, 6):
            identity.update(frame, [candidate(similarity=.586)])
            self.assertEqual(identity.state, 'LOST')
        self.assertEqual(identity.pending_count, 5)

    def test_pending_does_not_train_gallery(self):
        identity = self.started()
        gallery = list(identity.appearances)
        identity.update(2, [candidate()])
        self.assertEqual(list(identity.appearances), gallery)


if __name__ == '__main__':
    unittest.main()
