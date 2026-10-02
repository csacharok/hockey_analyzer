"""Conservative experimental identity association, independent of ByteTrack.

Scores are heuristic evidence, not calibrated identity probabilities. No OCR.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from statistics import median

from rink_geometry import UNKNOWN


@dataclass(frozen=True)
class Observation:
    track_id: int
    box: tuple[float, float, float, float]
    rink_state: str = UNKNOWN
    appearance: tuple[float, ...] = ()

    def __post_init__(self):
        if (len(self.box) != 4 or not all(math.isfinite(v) for v in self.box)
                or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]):
            raise ValueError("Observation requires a finite, positive xyxy box.")
        if self.appearance and (not all(math.isfinite(v) and v >= 0 for v in self.appearance)
                                or not math.isclose(sum(self.appearance), 1, abs_tol=1e-5)):
            raise ValueError("Appearance must be a normalized nonnegative histogram.")

    @property
    def center(self):
        a, b, c, d = self.box
        return ((a + c) / 2, (b + d) / 2)

    @property
    def size(self):
        return (self.box[2] - self.box[0], self.box[3] - self.box[1])


def appearance_histogram(frame, box, cv2):
    """Central upper-body HSV histogram; computed only on unannotated pixels."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    height, width = frame.shape[:2]
    left, right = max(0, int(x1 + .2 * w)), min(width, int(x2 - .2 * w))
    top, bottom = max(0, int(y1 + .15 * h)), min(height, int(y1 + .65 * h))
    if right - left < 4 or bottom - top < 4:
        return ()
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    return tuple(float(v) for v in hist / hist.sum())


@dataclass
class PendingReacquisition:
    track_id: int
    start_frame: int
    last_frame: int
    observations: list = field(default_factory=list)


class TargetIdentity:
    """Trust a bound ByteTrack ID; score candidates only after its disappearance.

    GRACE holds the binding without drawing a stale box or counting identity as
    visible. LOST matching is bounded in time and requires temporal evidence.
    All thresholds are generic experimental defaults, not jersey/track rules.
    """

    STRONG_APPEARANCE = .65
    SUPPORT_APPEARANCE = STRONG_APPEARANCE * .9
    TEMPORAL_APPEARANCE = (STRONG_APPEARANCE + SUPPORT_APPEARANCE) / 2
    MINIMUM_SCORE = .65
    PENDING_SECONDS = .5
    GEOMETRY_HISTORY_SECONDS = 1.0

    def __init__(self, fps, grace_frames=2, max_gap_seconds=3.0):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Identity FPS must be finite and positive.")
        if type(grace_frames) is not int or grace_frames < 0:
            raise ValueError("Target grace frames must be a nonnegative integer.")
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("Target max gap seconds must be finite and positive.")
        self.fps = fps
        self.grace_frames = grace_frames
        self.max_gap_seconds = max_gap_seconds
        self.state = "UNSEEDED"
        self.current = self.last = None
        self.last_frame = None
        self.frame = -1
        self.velocity = (0.0, 0.0)
        self.anchor = ()
        self.appearances = deque(maxlen=max(1, round(.5 * fps)))
        self.geometry_history = deque(maxlen=max(1, round(self.GEOMETRY_HISTORY_SECONDS * fps)))
        self.first_seen = {}
        self.pending = None
        self.pending_count = 0
        self.score = None
        self.events = []
        self.records = []
        self.candidates = []
        self.ids = []
        self.transitions = []
        self.counts = dict(TRACKED=0, LOST=0, REACQUIRED=0, UNSEEDED=0, GRACE=0)

    def _event(self, kind, message, **details):
        self.events.append(message)
        self.records.append(dict(event=kind, frame=self.frame, timestamp=self.frame / self.fps, **details))

    def _geometry(self, candidate):
        """Pose-aware safety evidence, trained only by authoritative acceptance.

        Log medians and MAD resist a single truncated/enlarged detector box.
        Independent dimensions allow articulation; area and aspect constrain it.
        The robust band is capped so noisy history cannot erase contradictions.
        Sparse history falls back to its median, without invented variability.
        """
        def values(size):
            w, h = size
            return dict(width=w, height=h, area=w * h, aspect=w / h)

        current, previous = values(candidate.size), values(self.last.size)
        samples = [values(size) for _, size in self.geometry_history] or [previous]
        statistics, errors, hard = {}, {}, []
        for name, value in current.items():
            logs = [math.log(s[name]) for s in samples]
            center = median(logs)
            mad = median(abs(v - center) for v in logs)
            # Area is quadratic in scale; normalize it to a linear scale error.
            power = 2 if name == 'area' else 1
            band = min(1.4826 * mad, power * math.log(1.43)) if len(samples) >= 3 else 0.
            error = max(0., abs(math.log(value) - center) - band) / power
            errors[name] = error
            statistics[name] = dict(median=math.exp(center), log_mad=mad,
                                    lower=math.exp(center - band), upper=math.exp(center + band),
                                    min=min(s[name] for s in samples), max=max(s[name] for s in samples))
            if error > math.log(2):
                hard.append(name + '_outside_robust_factor_' + ('4' if power == 2 else '2'))
        worst = max(errors.values())
        similarity = math.exp(-worst)
        decision = 'hard_rejected' if hard else 'soft_mismatch' if worst > math.log(1.43) else 'passed'
        return dict(current_width=current['width'], current_height=current['height'],
                    previous_width=previous['width'], previous_height=previous['height'],
                    width_ratio=current['width'] / previous['width'],
                    height_ratio=current['height'] / previous['height'],
                    area_ratio=current['area'] / previous['area'], aspect_ratio=current['aspect'],
                    aspect_ratio_change=current['aspect'] / previous['aspect'],
                    geometry_sample_count=len(self.geometry_history), geometry_statistics=statistics,
                    geometry_normalized_log_errors=errors, geometry_similarity=similarity,
                    geometry_decision=decision, geometry_hard_reasons=hard,
                    geometry_score_contribution=.2 * similarity,
                    legacy_size_score=min(min(current[k] / previous[k], previous[k] / current[k])
                                          for k in ('width', 'height')))

    def _evaluate(self, candidate, frame):
        """Return inspectable evidence and ALL failed gates, only used while LOST."""
        gap = (frame - self.last_frame) / self.fps
        old = self.last
        ratios = [a / b for a, b in zip(candidate.size, old.size)]
        geometry = self._geometry(candidate)
        predicted = tuple(p + v * (frame - self.last_frame) for p, v in zip(old.center, self.velocity))
        distance = math.dist(predicted, candidate.center) / old.size[1]
        # Uncertainty grows with elapsed time, not with the number of retries.
        limit = .35 + .65 * gap
        displacement = math.dist(old.center, candidate.center) / old.size[1]
        reachable = .5 + 2.5 * gap
        # Half-second velocity half-life across the entire search window.
        # Smoothstep gives prediction full authority at confidence >= .90 and
        # none at <= .10, with continuous values/slopes at both boundaries.
        # Even an arbitrarily bad stale prediction cannot veto at low confidence.
        velocity_confidence = 2 ** (-gap / .5)
        prediction_fraction = distance / limit
        confidence_position = min(1.0, max(0.0, (velocity_confidence - .10) / (.90 - .10)))
        prediction_weight = confidence_position ** 2 * (3 - 2 * confidence_position)
        prediction_contribution = prediction_weight * prediction_fraction
        displacement_contribution = (1 - prediction_weight) * displacement / reachable
        motion_fraction = prediction_contribution + displacement_contribution
        prediction_authority = ('authoritative' if prediction_weight == 1 else
                                'diagnostic_only' if prediction_weight == 0 else 'blended')
        motion_model = 'confidence_weighted'
        recent = None
        gallery_matches = []
        if candidate.appearance and self.appearances:
            template = tuple(sum(values) / len(self.appearances) for values in zip(*self.appearances))
            if len(template) == len(candidate.appearance):
                recent = sum(min(a, b) for a, b in zip(template, candidate.appearance))
                gallery_matches = sorted((sum(min(a, b) for a, b in zip(sample, candidate.appearance))
                                          for sample in self.appearances), reverse=True)
        top_matches = gallery_matches[:3]
        gallery_best = top_matches[0] if top_matches else None
        gallery_top_mean = sum(top_matches) / len(top_matches) if top_matches else None
        # A lone historical outlier is insufficient. Require three stored views
        # individually above the unchanged appearance gate, then confirmation
        # with temporal evidence in _reacquire. LOST never trains it.
        gallery_supported = len(top_matches) == 3 and min(top_matches) >= .65
        appearance_evidence = max(recent or 0.0, gallery_top_mean if gallery_supported else 0.0)
        appearance_model = ('gallery_top3' if gallery_supported and gallery_top_mean > (recent or 0.0)
                            else 'averaged_template')
        seed_similarity = None
        if self.anchor and len(self.anchor) == len(candidate.appearance):
            seed_similarity = sum(min(a, b) for a, b in zip(self.anchor, candidate.appearance))
        reasons = []
        if gap > self.max_gap_seconds:
            reasons.append("search_window_expired")
        # A person already tracked well before the occlusion is not a new fragment.
        if candidate.track_id != old.track_id and self.first_seen[candidate.track_id] < self.last_frame - round(.2 * self.fps):
            reasons.append("track_predates_disappearance")
        if old.rink_state != UNKNOWN and candidate.rink_state != UNKNOWN and old.rink_state != candidate.rink_state:
            reasons.append("rink_state_conflict")
        if geometry['geometry_decision'] == 'hard_rejected':
            reasons.append("box_size_change")
        if motion_fraction > 1:
            reasons.append("prediction_distance")
        if displacement > reachable:
            reasons.append("unreachable_displacement")
        if recent is None:
            reasons.append("appearance_unavailable")
        elif appearance_evidence < .65:
            reasons.append("appearance_mismatch")
        size_score = geometry['geometry_similarity']
        score = .5 * max(0.0, 1 - motion_fraction) + .3 * appearance_evidence + .2 * size_score
        if score < .65:
            reasons.append("low_score")
        return dict(track_id=candidate.track_id, box=candidate.box, first_seen_frame=self.first_seen[candidate.track_id],
                    gap_seconds=gap, score=score, reasons=reasons, decision="rejected" if reasons else "eligible",
                    prediction_error_heights=distance, prediction_limit_heights=limit,
                    displacement_heights=displacement, reachable_limit_heights=reachable,
                    displacement_pixels=math.dist(old.center, candidate.center),
                    prediction_error_pixels=math.dist(predicted, candidate.center),
                    velocity_confidence=velocity_confidence, velocity_half_life_seconds=.5,
                    short_gap_limit_seconds=1.0, motion_model=motion_model,
                    short_gap_limit_active=False, prediction_authority=prediction_authority,
                    prediction_weight=prediction_weight, displacement_weight=1 - prediction_weight,
                    prediction_contribution=prediction_contribution,
                    displacement_contribution=displacement_contribution,
                    prediction_authoritative_confidence=.90, prediction_diagnostic_confidence=.10,
                    effective_blended_motion_error=motion_fraction,
                    effective_motion_fraction=motion_fraction, effective_motion_limit=1.0,
                    motion_decision='rejected' if motion_fraction > 1 or displacement > reachable else 'passed',
                    gallery_sample_count=len(gallery_matches), gallery_best_similarity=gallery_best,
                    gallery_top_k=len(top_matches), gallery_top_k_mean_similarity=gallery_top_mean,
                    gallery_top_k_min_similarity=min(top_matches) if top_matches else None,
                    gallery_supported=gallery_supported, appearance_model=appearance_model,
                    effective_appearance_similarity=appearance_evidence,
                    size_ratios=ratios, appearance_similarity=recent, seed_similarity=seed_similarity,
                    rink_state=candidate.rink_state, **geometry)

    def update(self, frame, observations, seed_point=None):
        if type(frame) is not int or frame != self.frame + 1:
            raise ValueError("Identity updates require consecutive zero-based frames.")
        observations = list(observations)
        by_id = {o.track_id: o for o in observations}
        if len(by_id) != len(observations):
            raise ValueError("Duplicate temporary IDs in one frame.")
        self.frame = frame
        self.events, self.records, self.candidates = [], [], []
        self.current = None
        self.score = None
        for observation in observations:
            self.first_seen.setdefault(observation.track_id, frame)
        if seed_point is not None:
            if self.last is not None:
                raise ValueError("Target is already seeded.")
            if len(seed_point) != 2 or not all(math.isfinite(v) for v in seed_point):
                raise ValueError("Seed point must contain two finite pixel coordinates.")
            x, y = seed_point
            matches = [o for o in observations if o.box[0] <= x <= o.box[2] and o.box[1] <= y <= o.box[3]]
            if len(matches) != 1:
                raise ValueError(f"Seed must select exactly one confirmed track; selected {len(matches)}. Choose another point/frame.")
            self.anchor = matches[0].appearance
            self._accept(matches[0], frame, "TRACKED", None)
            self._event("seeded", f"TARGET #57 seeded -> Track {self.last.track_id} | frame {frame} | time {frame / self.fps:.3f}s",
                        previous_id=None, new_id=self.last.track_id, seed_point=seed_point)
        elif self.last is not None:
            bound = by_id.get(self.last.track_id)
            # The architectural boundary: no similarity scoring while bound.
            if self.state != "LOST" and bound is not None:
                self._accept(bound, frame, "TRACKED", None)
            elif self.state != "LOST" and frame - self.last_frame <= self.grace_frames:
                self.state = "GRACE"
            else:
                if self.state != "LOST":
                    self._event("lost", f"TARGET LOST | previous Track {self.last.track_id} | frame {frame} | time {frame / self.fps:.3f}s",
                                previous_id=self.last.track_id, new_id=None, last_seen_frame=self.last_frame,
                                missing_since_frame=self.last_frame + 1, gap_seconds=(frame - self.last_frame) / self.fps)
                self.state = "LOST"
                self._reacquire(frame, observations)
        self.counts[self.state] += 1
        return self.current

    def _reacquire(self, frame, observations):
        self.candidates = [self._evaluate(o, frame) for o in sorted(observations, key=lambda o: o.track_id)]
        # Support evidence is diagnostic for all candidates, but can only be
        # accumulated after an unchanged strong-start gate has passed.
        for c in self.candidates:
            gallery_ok = (c['gallery_top_k'] == 3 and
                          c['gallery_top_k_min_similarity'] >= self.SUPPORT_APPEARANCE)
            c['support_effective_appearance'] = max(c['appearance_similarity'] or 0.,
                c['gallery_top_k_mean_similarity'] if gallery_ok else 0.)
            c['support_score'] = c['score'] + .3 * (c['support_effective_appearance'] -
                                                       c['effective_appearance_similarity'])
        soft = {'appearance_mismatch', 'low_score'}
        rivals = [c for c in self.candidates if not set(c['reasons']) - soft
                  and c['support_effective_appearance'] >= self.SUPPORT_APPEARANCE]

        def ambiguous(candidate):
            return any(c['track_id'] != candidate['track_id'] and
                       candidate['support_score'] - c['support_score'] < .15 for c in rivals)

        winner = None
        if self.pending is not None:
            winner = next((c for c in self.candidates if c['track_id'] == self.pending.track_id), None)
            reason = None
            if (frame - self.last_frame) / self.fps > self.max_gap_seconds:
                reason = 'search_window_expired'
            elif winner is None:
                reason = 'candidate_missing'  # zero-frame interruption allowance
            elif set(winner['reasons']) - soft:
                reason = ','.join(r for r in winner['reasons'] if r not in soft)
            elif ambiguous(winner):
                reason = 'ambiguous_margin'
            elif winner['support_effective_appearance'] < self.SUPPORT_APPEARANCE ** 2:
                reason = 'strongly_contradictory_appearance'
            elif frame - self.pending.start_frame >= max(3, round(self.PENDING_SECONDS * self.fps)):
                reason = 'pending_window_expired'
            if reason:
                self._pending_event('REACQUISITION_PENDING_RESET', reason, winner)
                self.pending, self.pending_count = None, 0
                winner = None
                # A failed observation cannot restart confirmation in this frame.
                self._candidate_record(frame)
                return
        else:
            ranked = sorted((c for c in self.candidates if not c['reasons']),
                            key=lambda c: (-c['score'], c['track_id']))
            winner = ranked[0] if ranked else None
            if winner and ambiguous(winner):
                for c in ranked:
                    c['decision'] = 'ambiguous'
                    c['reasons'].append('ambiguous_margin')
                winner = None
            if winner:
                self.pending = PendingReacquisition(winner['track_id'], frame, frame)

        accepted = False
        if winner:
            self.pending.observations.append(dict(winner))
            self.pending.last_frame = frame
            self.pending_count = len(self.pending.observations)
            values = [c['support_effective_appearance'] for c in self.pending.observations]
            scores = [c['support_score'] for c in self.pending.observations]
            supporting = sum(v >= self.SUPPORT_APPEARANCE for v in values)
            accepted = (supporting >= 3 and supporting / len(values) >= .8
                        and values[-1] >= self.SUPPORT_APPEARANCE
                        and sum(values) / len(values) >= self.TEMPORAL_APPEARANCE
                        and sum(scores) / len(scores) >= self.MINIMUM_SCORE)
            winner['decision'] = 'accepted' if accepted else 'confirming'
            winner['confirmation_frames'] = self.pending_count
            self._pending_event('TARGET_REACQUIRED' if accepted else
                                'REACQUISITION_PENDING_STARTED' if self.pending_count == 1 else
                                'REACQUISITION_PENDING_CONTINUED',
                                'temporal_evidence_sufficient' if accepted else
                                'strong_start' if self.pending_count == 1 else
                                'awaiting_sufficient_temporal_evidence', winner)
        self._candidate_record(frame)
        if accepted:
            candidate_id = winner['track_id']
            previous = self.last.track_id
            gap = (frame - self.last_frame) / self.fps
            observation = next(o for o in observations if o.track_id == candidate_id)
            self._accept(observation, frame, "REACQUIRED", winner['support_score'])
            transition = dict(previous_id=previous, new_id=candidate_id, gap_seconds=gap, score=winner['support_score'])
            self.transitions.append(dict(frame=frame, timestamp=frame / self.fps, **transition))
            self._event("reacquired", f"TARGET REACQUIRED | {previous} -> {candidate_id} | frame {frame} | time {frame / self.fps:.3f}s | gap {gap:.3f}s | score {winner['support_score']:.3f}", **transition)
            if previous != candidate_id:
                self.events.append(f"Identity transition: {previous} -> {candidate_id}")

    def _pending_event(self, event, reason, candidate):
        pending = self.pending
        values = [c['support_effective_appearance'] for c in pending.observations]
        scores = [c['support_score'] for c in pending.observations]
        supporting = sum(v >= self.SUPPORT_APPEARANCE for v in values)
        top = sorted(values, reverse=True)[:3]
        details = dict(pending_track_id=pending.track_id, pending_age=self.frame - pending.start_frame + 1,
                       start_frame=pending.start_frame, last_frame=pending.last_frame,
                       observation_count=len(values), strong_appearance_threshold=self.STRONG_APPEARANCE,
                       support_appearance_threshold=self.SUPPORT_APPEARANCE,
                       temporal_appearance_threshold=self.TEMPORAL_APPEARANCE,
                       temporal_mean_appearance=sum(values) / len(values), temporal_median_appearance=median(values),
                       temporal_best_appearance=max(values), temporal_top_k_mean=sum(top) / len(top),
                       supporting_frame_count=supporting, supporting_frame_fraction=supporting / len(values),
                       mean_score=sum(scores) / len(scores), cumulative_score=sum(scores),
                       valid_motion_count=len(values), hard_failure_count=int(event.endswith('RESET')),
                       confirmation_state=event, reason=reason, current_evidence=dict(candidate) if candidate else None)
        if candidate is not None:
            candidate['pending_confirmation'] = {k: v for k, v in details.items() if k != 'current_evidence'}
        self._event(event, f"{event} | Track {pending.track_id} | frame {self.frame} | {reason}", **details)

    def _candidate_record(self, frame):
        self.records.append(dict(event="candidates", frame=frame, timestamp=frame / self.fps,
                                 previous_id=self.last.track_id, last_seen_frame=self.last_frame,
                                 last_box=self.last.box, velocity_pixels_per_frame=self.velocity,
                                 candidates=self.candidates))

    def _accept(self, observation, frame, state, score):
        if self.last is not None:
            gap = frame - self.last_frame
            measured = tuple((a - b) / gap for a, b in zip(observation.center, self.last.center))
            self.velocity = tuple(.5 * v + .5 * m for v, m in zip(self.velocity, measured))
        self.last = self.current = observation
        self.last_frame = frame
        while self.geometry_history and frame - self.geometry_history[0][0] >= self.GEOMETRY_HISTORY_SECONDS * self.fps:
            self.geometry_history.popleft()
        self.geometry_history.append((frame, observation.size))
        self.state, self.score = state, score
        self.pending, self.pending_count = None, 0
        if observation.appearance:
            if self.appearances and len(self.appearances[0]) != len(observation.appearance):
                self.appearances.clear()
            self.appearances.append(observation.appearance)
        if observation.track_id not in self.ids:
            self.ids.append(observation.track_id)

    def debug_frame(self):
        return dict(event="frame", frame=self.frame, timestamp=self.frame / self.fps,
                    state=self.state, bound_id=self.last.track_id if self.last else None,
                    observed_id=self.current.track_id if self.current else None,
                    observed_box=self.current.box if self.current else None,
                    last_seen_frame=self.last_frame, score=self.score)

    def summary(self):
        total = sum(self.counts.values())
        known = self.counts['TRACKED'] + self.counts['REACQUIRED']
        chain = [self.ids[0]] + [t['new_id'] for t in self.transitions] if self.ids else []
        return (f"Target identity summary (manual #57; experimental)\n"
                f"TRACKED frames: {self.counts['TRACKED']}\n"
                f"LOST frames: {self.counts['LOST']}\n"
                f"GRACE frames (bound but not observed): {self.counts['GRACE']}\n"
                f"REACQUIRED frames / reacquisitions: {self.counts['REACQUIRED']}\n"
                f"UNSEEDED frames: {self.counts['UNSEEDED']}\n"
                f"Associated temporary ByteTrack IDs: {self.ids}\n"
                f"Binding sequence: {' -> '.join(map(str, chain))}\n"
                f"Identity known: {100 * known / total if total else 0:.2f}% of {total} processed video frames")
