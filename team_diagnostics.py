"""Structured diagnostics and compact review gallery for opt-in team analysis."""

from __future__ import annotations

from collections import Counter
import json

from team_classifier import TEAM_STATES, UNKNOWN
from participant_filter import PARTICIPANT_STATES, REJECTED, UNCERTAIN


class TeamDiagnostics:
    def __init__(self, path, configuration):
        self.path = path
        self.handle = path.open("x", encoding="utf-8")
        self.handle.write(json.dumps(configuration) + "\n")
        self.counts = Counter({state: 0 for state in TEAM_STATES})
        self.participant_counts = Counter({state: 0 for state in PARTICIPANT_STATES})
        self.labels = {}
        self.stats = {}
        self.samples = {state: [] for state in TEAM_STATES}
        self.samples["UNSTABLE"] = []
        self.samples[REJECTED] = []
        self.samples[UNCERTAIN] = []

    def add(self, frame_number, fps, identifier, box, detection_confidence,
            participant, result, frame, cv2):
        self.participant_counts[participant.state] += 1
        record = dict(event="observation", frame=frame_number,
                      timestamp=frame_number / fps, track_id=identifier,
                      box=box, detection_confidence=detection_confidence,
                      participant_state=participant.state,
                      participant_accepted=participant.accepted,
                      participant_reason=participant.reason,
                      participant_evidence=participant.evidence)
        if result is None:
            self.handle.write(json.dumps(record, allow_nan=False) + "\n")
            self._sample(REJECTED, frame_number, identifier, 1.0, box, frame, cv2)
            return
        self.counts[result.label] += 1
        previous = self.labels.get(identifier)
        stats = self.stats.setdefault(identifier, dict(observations=0, unknown=0,
                                                       label_changes=0, confidence_sum=0.0))
        stats["observations"] += 1
        stats["unknown"] += result.label == UNKNOWN
        stats["confidence_sum"] += result.confidence
        if previous is not None and previous != result.label:
            stats["label_changes"] += 1
        self.labels[identifier] = result.label
        self.handle.write(json.dumps(dict(**record, label=result.label,
                                          confidence=result.confidence,
                                          observation_label=result.observation_label,
                                          observation_confidence=result.observation_confidence,
                                          evidence=result.evidence), allow_nan=False) + "\n")
        category = (UNCERTAIN if participant.state == UNCERTAIN else
                    "UNSTABLE" if previous is not None and previous != result.label else result.label)
        score = 1 - result.confidence if category in (UNKNOWN, "UNSTABLE", UNCERTAIN) else result.confidence
        self._sample(category, frame_number, identifier, score, box, frame, cv2,
                     result.confidence)

    def _sample(self, category, frame_number, identifier, score, box, frame, cv2,
                confidence=0.0):
        x1, y1, x2, y2 = map(round, box)
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if crop.size:
            tile = cv2.resize(crop, (120, 160), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (119, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"f{frame_number} t{identifier} {confidence:.2f}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
            self.samples[category].append((score, tile.copy()))
            self.samples[category].sort(key=lambda item: item[0], reverse=True)
            del self.samples[category][6:]

    def flush(self):
        self.handle.flush()

    def finish(self, frames, cv2, provider_exclusions=None):
        import numpy as np
        track_counts = Counter(self.labels.values())
        tracks = []
        for identifier, stats in sorted(self.stats.items()):
            tracks.append(dict(track_id=identifier, final_label=self.labels[identifier], **stats,
                               unknown_rate=stats["unknown"] / stats["observations"],
                               mean_confidence=stats["confidence_sum"] / stats["observations"]))
        included = sum(self.counts.values())
        report = dict(frames=frames, participant_observations_accepted=included,
                      participant_decision_counts=dict(self.participant_counts),
                      off_ice_detections_rejected=self.participant_counts[REJECTED],
                      relevant_observations=included,
                      observation_counts=dict(self.counts), relevant_tracks=len(self.labels),
                      track_counts={state: track_counts[state] for state in TEAM_STATES},
                      provider_exclusions=provider_exclusions or {},
                      tracks=tracks, completed=True)
        self.handle.write(json.dumps(dict(event="summary", **report)) + "\n")
        self.handle.flush()
        (self.path.parent / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows = []
        for category, samples in self.samples.items():
            tiles = [tile for _, tile in samples]
            while len(tiles) < 6:
                tiles.append(np.zeros((160, 120, 3), dtype=np.uint8))
            row = np.hstack(tiles)
            cv2.putText(row, category, (4, 156), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            rows.append(row)
        cv2.imwrite(str(self.path.parent / "review_contact_sheet.jpg"), np.vstack(rows))
        return report

    def close(self):
        self.handle.close()
