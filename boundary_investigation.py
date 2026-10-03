"""Replay stored observations through participant-boundary alternatives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from kickplate_boundary import KickplateBoundaryDetector
from near_board_boundary import NearBoardBoundaryDetector
from participant_filter import ParticipantFilter
from rink_geometry import RinkGeometry, contact_point


def observations_by_frame(path):
    grouped = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "observation":
                grouped.setdefault(record["frame"], []).append(record)
    return grouped


def draw_boundary(frame, boundary, color, cv2):
    if boundary is None:
        return
    for left, right in zip(boundary.points, boundary.points[1:]):
        if right[0] - left[0] <= boundary.bin_width * 1.6:
            cv2.line(frame, tuple(map(round, left)), tuple(map(round, right)), color, 3)


def run(video, observations, rink, configuration, output):
    import cv2
    output.mkdir(parents=True, exist_ok=True)
    diagnostic_directory = output / "boundary_diagnostics"
    diagnostic_directory.mkdir(exist_ok=True)
    records = observations_by_frame(observations)
    participant = ParticipantFilter(configuration.get("participant_filter"))
    far_detector = KickplateBoundaryDetector(configuration.get("kickplate_boundary"))
    near_detector = NearBoardBoundaryDetector(configuration.get("near_board_boundary"))
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or max(records) + 1
    sample_interval = max(1, total // 6)
    variants = ("v3_current", "v3_without_polygon", "far_near_without_polygon")
    counts = {name: Counter() for name in variants}
    transitions = Counter()
    reason_counts = {name: Counter() for name in variants}
    coverage = {"far_frames": 0, "near_frames": 0, "decoded_frames": 0,
                "near_coverage_sum": 0.0}
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        far = far_detector.estimate(frame, cv2)
        near = near_detector.estimate(frame, cv2)
        coverage["decoded_frames"] += 1
        coverage["far_frames"] += bool(far.points)
        coverage["near_frames"] += bool(near.points)
        coverage["near_coverage_sum"] += near.coverage
        annotated = frame.copy() if frame_number % sample_interval == 0 else None
        for record in records.get(frame_number, ()):
            box = record["box"]
            rink_state = rink.classify(box)
            results = {
                "v3_current": participant.classify(frame, box, rink_state, cv2, far, None, True),
                "v3_without_polygon": participant.classify(frame, box, rink_state, cv2, far, None, False),
                "far_near_without_polygon": participant.classify(frame, box, rink_state, cv2, far, near, False),
            }
            for name, result in results.items():
                counts[name][result.state] += 1
                reason_counts[name][result.reason] += 1
            before, after = results["v3_current"], results["far_near_without_polygon"]
            if before.state != after.state:
                transitions[(before.state, after.state)] += 1
            if annotated is not None:
                x, y = map(round, contact_point(box))
                state = after.state
                color = {"ACCEPTED": (0, 255, 0), "REJECTED": (0, 0, 255),
                         "UNCERTAIN": (0, 220, 255)}[state]
                cv2.circle(annotated, (x, y), 5, color, -1)
        if annotated is not None:
            draw_boundary(annotated, far, (0, 255, 255), cv2)
            draw_boundary(annotated, near, (255, 0, 255), cv2)
            cv2.putText(annotated, "yellow=far magenta=near green/red/yellow=A/R/U",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            cv2.imwrite(str(diagnostic_directory / f"frame_{frame_number}.jpg"), annotated)
        frame_number += 1
    capture.release()
    comparison = {
        "protocol": "Fixed v3 detections replayed through participant filters; not ground-truth accuracy.",
        "variants": {name: {"decision_counts": dict(counts[name]),
                            "reason_counts": dict(reason_counts[name])} for name in variants},
        "transitions_current_to_proposed": {
            f"{before}->{after}": count for (before, after), count in transitions.items()},
        "boundary_availability": {**coverage,
            "near_frame_fraction": coverage["near_frames"] / max(1, coverage["decoded_frames"]),
            "mean_near_coverage": coverage["near_coverage_sum"] / max(1, coverage["decoded_frames"])},
    }
    (output / "prebenchmark_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")
    polygon = {
        "protocol": comparison["protocol"],
        "current_with_polygon": dict(counts["v3_current"]),
        "far_only_without_polygon": dict(counts["v3_without_polygon"]),
        "far_near_without_polygon": dict(counts["far_near_without_polygon"]),
        "polygon_unique_decision_changes": sum(
            count for (before, after), count in transitions.items() if before != after),
        "recommendation_basis": "Review transitions and diagnostics before production deprecation.",
    }
    (output / "polygon_ablation.json").write_text(json.dumps(polygon, indent=2), encoding="utf-8")
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("rink", type=Path)
    parser.add_argument("team_config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    configuration = json.loads(args.team_config.read_text(encoding="utf-8"))
    result = run(args.video, args.observations, RinkGeometry.load(args.rink),
                 configuration, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
