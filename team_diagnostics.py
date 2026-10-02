"""Structured diagnostics and compact review gallery for opt-in team analysis."""

from __future__ import annotations

from collections import Counter
import json

from team_classifier import TEAM_STATES, UNKNOWN


class TeamDiagnostics:
    def __init__(self, path, configuration):
        self.path = path
        self.handle = path.open("x", encoding="utf-8")
        self.handle.write(json.dumps(configuration) + "\n")
        self.counts = Counter({state: 0 for state in TEAM_STATES})
        self.labels = {}
        self.stats = {}
        self.samples = {state: [] for state in TEAM_STATES}
        self.samples["UNSTABLE"] = []

    def add(self, frame_number, fps, identifier, box, detection_confidence,
            result, frame, cv2):
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
        self.handle.write(json.dumps(dict(event="observation", frame=frame_number,
                                          timestamp=frame_number / fps, track_id=identifier,
                                          box=box, detection_confidence=detection_confidence,
                                          rink_state="ON ICE", label=result.label,
                                          confidence=result.confidence,
                                          observation_label=result.observation_label,
                                          observation_confidence=result.observation_confidence,
                                          evidence=result.evidence), allow_nan=False) + "\n")
        category = "UNSTABLE" if previous is not None and previous != result.label else result.label
        x1, y1, x2, y2 = map(round, box)
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if crop.size:
            tile = cv2.resize(crop, (120, 160), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (119, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"f{frame_number} t{identifier} {result.confidence:.2f}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
            score = 1 - result.confidence if category in (UNKNOWN, "UNSTABLE") else result.confidence
            self.samples[category].append((score, tile.copy()))
            self.samples[category].sort(key=lambda item: item[0], reverse=True)
            del self.samples[category][6:]

    def flush(self):
        self.handle.flush()

    def finish(self, frames, cv2):
        import numpy as np
        track_counts = Counter(self.labels.values())
        tracks = []
        for identifier, stats in sorted(self.stats.items()):
            tracks.append(dict(track_id=identifier, final_label=self.labels[identifier], **stats,
                               unknown_rate=stats["unknown"] / stats["observations"],
                               mean_confidence=stats["confidence_sum"] / stats["observations"]))
        report = dict(frames=frames, relevant_observations=sum(self.counts.values()),
                      observation_counts=dict(self.counts), relevant_tracks=len(self.labels),
                      track_counts={state: track_counts[state] for state in TEAM_STATES},
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
