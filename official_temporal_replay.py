"""Replay role-specific temporal evidence from an existing observations JSONL.

Track/frame selectors supplied on the command line are human-review metadata;
they never affect production classification.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from team_classifier import OFFICIAL, OfficialTemporalState


def replay(path, configuration, reviewed_officials=(), negative_controls=()):
    rows = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "observation" and "label" in record:
                rows[record["track_id"]].append(record)
    reviewed = set(reviewed_officials) | set(negative_controls)
    tracks = []
    aggregate = Counter()
    for track_id, records in sorted(rows.items()):
        state = OfficialTemporalState(configuration)
        raw = Counter()
        emitted = Counter()
        transitions = []
        previous = None
        label_changes = 0
        establishment_count = 0
        reversal_count = 0
        for record in records:
            raw[record["observation_label"]] += 1
            role = state.update(record["observation_label"], record["observation_confidence"],
                                record.get("evidence", {}))
            label = OFFICIAL if role["confirmed"] else record["label"]
            emitted[label] += 1
            label_changes += previous is not None and previous != label
            previous = label
            if role["transition"]:
                establishment_count += role["transition"] == "ESTABLISHED"
                reversal_count += role["transition"] == "REVERTED"
                stripe_evidence = record.get("evidence", {}).get("stripe_evidence", {})
                transitions.append(dict(frame=record["frame"], transition=role["transition"],
                                        observation_label=record["observation_label"],
                                        observation_confidence=record["observation_confidence"],
                                        stripe_ratio=record.get("evidence", {}).get("stripe_ratio", 0),
                                        lower_body_dark_fraction=stripe_evidence.get(
                                            "lower_body_dark_fraction"),
                                        reason=("repeated_positive_referee_structure"
                                                if role["transition"] == "ESTABLISHED" else
                                                "repeated_team_color_with_non_dark_lower_body"),
                                        role_evidence=role))
        if state.established_count:
            aggregate["tracklets_establishing_official"] += 1
        if state.reverted_count:
            aggregate["tracklets_reverting_from_official"] += 1
        aggregate["role_transitions"] += len(transitions)
        if track_id in negative_controls and state.established_count:
            aggregate["negative_controls_falsely_establishing_official"] += 1
        if track_id in reviewed or state.established_count:
            tracks.append(dict(track_id=track_id,
                               review_label=("OFFICIAL" if track_id in reviewed_officials else
                                             "NEGATIVE_CONTROL" if track_id in negative_controls else "UNREVIEWED"),
                               raw_observations=dict(raw), temporal_emitted_labels=dict(emitted),
                               emitted_label_transitions=label_changes,
                               official_established=bool(state.established_count),
                               official_later_lost=bool(state.reverted_count),
                               establishments=establishment_count,
                               reversals=reversal_count,
                               re_establishments=max(0, establishment_count - 1),
                               structural_transitions=transitions))
    aggregate.setdefault("negative_controls_falsely_establishing_official", 0)
    aggregate["rapid_oscillations"] = sum(
        len(track["structural_transitions"]) > 2 for track in tracks)
    return dict(method="Offline replay of fixed observations; behavior measurement, not accuracy.",
                configuration=configuration, summary=dict(aggregate), tracks=tracks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--team-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-official", type=int, action="append", default=[])
    parser.add_argument("--negative-control", type=int, action="append", default=[])
    parser.add_argument("--video", type=Path, help="Optional source video for a transition contact sheet")
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()
    config = json.loads(args.team_config.read_text(encoding="utf-8"))["official_temporal"]
    report = replay(args.observations, config, args.reviewed_official, args.negative_control)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.video is not None and args.contact_sheet is not None:
        write_contact_sheet(args.video, args.observations, report, args.contact_sheet)


def write_contact_sheet(video, observations, report, output):
    """Render representative reviewed/transition crops; review metadata only."""
    import cv2
    import numpy as np
    wanted = []
    for track in report["tracks"]:
        for transition in track["structural_transitions"][:2]:
            wanted.append((transition["frame"], track["track_id"], transition["transition"]))
    boxes = {}
    with Path(observations).open(encoding="utf-8") as handle:
        keys = {(frame, track) for frame, track, _ in wanted}
        for line in handle:
            row = json.loads(line)
            key = (row.get("frame"), row.get("track_id"))
            if key in keys:
                boxes[key] = row["box"]
    capture = cv2.VideoCapture(str(video))
    tiles = []
    try:
        for frame_number, track_id, transition in wanted[:18]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            box = boxes.get((frame_number, track_id))
            if not ok or box is None:
                continue
            x1, y1, x2, y2 = map(round, box)
            crop = frame[max(0, y1):min(frame.shape[0], y2),
                         max(0, x1):min(frame.shape[1], x2)]
            if not crop.size:
                continue
            tile = cv2.resize(crop, (160, 200), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (159, 25), (0, 0, 0), -1)
            cv2.putText(tile, f"t{track_id} {transition}", (3, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            tiles.append(tile)
    finally:
        capture.release()
    while len(tiles) < 18:
        tiles.append(np.zeros((200, 160, 3), dtype=np.uint8))
    sheet = np.vstack([np.hstack(tiles[index:index + 6]) for index in range(0, 18, 6)])
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Could not write contact sheet: {output}")


if __name__ == "__main__":
    main()
