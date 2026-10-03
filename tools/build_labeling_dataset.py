"""Build and review a human-ground-truth tracklet labeling project.

The selector uses observation metadata for diversity only.  It never assigns a
semantic label, and legacy semantic output is deliberately omitted from the UI.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from appearance_features import measurement_quality

LABELS = ("HOME", "AWAY", "OFFICIAL", "NON_PARTICIPANT", "MIXED_TRACK", "UNSURE")
KEY_LABELS = {ord("h"): "HOME", ord("a"): "AWAY", ord("o"): "OFFICIAL",
              ord("n"): "NON_PARTICIPANT", ord("m"): "MIXED_TRACK", ord("u"): "UNSURE"}
SCHEMA_VERSION = 1


def _quality(obs):
    quality = obs.get("appearance_features", {}).get("quality", {})
    return measurement_quality(quality.get("measurement_quality", quality.get("crop_quality")))


def read_observations(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize_exports(export_dirs):
    """Load observations grouped by the collision-safe (source_game, track_id) key."""
    tracks = defaultdict(list)
    sources = []
    for directory in map(Path, export_dirs):
        obs_path = directory / "appearance_observations.jsonl"
        track_path = directory / "appearance_tracklets.json"
        if not obs_path.is_file() or not track_path.is_file():
            raise FileNotFoundError(f"Expected appearance exports in {directory}")
        digest = hashlib.sha256()
        with obs_path.open("rb") as raw:
            for chunk in iter(lambda: raw.read(1024 * 1024), b""):
                digest.update(chunk)
        source_hash = digest.hexdigest()
        sources.append({"directory": str(directory.resolve()), "observations": str(obs_path.resolve()),
                        "tracklets": str(track_path.resolve()), "observations_sha256": source_hash})
        for obs in read_observations(obs_path):
            tracks[(str(obs["source_game"]), obs["temporary_track_id"])].append(obs)
    return tracks, sources


def _median(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def track_metadata(key, observations):
    qualities = Counter(_quality(o) for o in observations)
    participants = Counter((o.get("participant_evidence", {}).get("decision") or "UNKNOWN")
                           for o in observations)
    boxes = [o["person_bbox"] for o in observations]
    heights = [b[3] - b[1] for b in boxes]
    centers_x = [(b[0] + b[2]) / 2 for b in boxes]
    feature = lambda region, field: [o.get("appearance_features", {}).get("regions", {})
                                     .get(region, {}).get(field) for o in observations]
    stripes = [o.get("appearance_features", {}).get("structure", {}).get("vertical_stripe_score")
               for o in observations]
    confidence = [o.get("appearance_features", {}).get("quality", {}).get("detection_confidence")
                  for o in observations]
    return {"source_game": key[0], "temporary_track_id": key[1], "observation_count": len(observations),
            "first_frame": min(o["frame"] for o in observations),
            "last_frame": max(o["frame"] for o in observations),
            "measurement_quality_counts": dict(qualities), "participant_decision_counts": dict(participants),
            "median_bbox_height": _median(heights), "median_image_x": _median(centers_x),
            "median_torso_lightness": _median(feature("torso", "lab_lightness_median")),
            "median_pants_dark_fraction": _median(feature("pants", "dark_fraction")),
            "median_lower_leg_lightness": _median(feature("lower_legs", "lab_lightness_median")),
            "median_stripe_score": _median(stripes), "median_detection_confidence": _median(confidence)}


def _bucket(meta):
    count = meta["observation_count"]
    length_band = 0 if count < 6 else 1 if count < 30 else 2 if count < 150 else 3
    q = meta["measurement_quality_counts"]
    dominant_q = max(("MEASURABLE", "PARTIAL", "UNUSABLE"), key=lambda value: (q.get(value, 0), value))
    dominant_p = max(meta["participant_decision_counts"], key=meta["participant_decision_counts"].get)
    bbox_band = min(3, int((meta["median_bbox_height"] or 0) / 80))
    x_band = min(3, int((meta["median_image_x"] or 0) / 480))
    torso_band = min(3, int((meta["median_torso_lightness"] or 0) * 4))
    return dominant_q, dominant_p, length_band, bbox_band, x_band, torso_band


def select_candidates(tracks, target, seed):
    """Balanced per-game, deterministic round-robin sampling over metadata strata."""
    metadata = {key: track_metadata(key, observations) for key, observations in tracks.items()}
    by_game = defaultdict(list)
    for key, meta in metadata.items():
        by_game[key[0]].append((key, meta))
    games = sorted(by_game)
    if not games or target <= 0:
        return []
    base, extra = divmod(min(target, len(metadata)), len(games))
    selected = []
    for game_index, game in enumerate(games):
        quota = min(len(by_game[game]), base + (game_index < extra))
        rng = random.Random(f"{seed}:{game}")
        strata = defaultdict(list)
        for item in by_game[game]:
            strata[_bucket(item[1])].append(item)
        for values in strata.values():
            rng.shuffle(values)
        keys = list(strata)
        rng.shuffle(keys)
        while quota and keys:
            next_keys = []
            for signature in keys:
                if strata[signature] and quota:
                    selected.append(strata[signature].pop())
                    quota -= 1
                if strata[signature]:
                    next_keys.append(signature)
            keys = next_keys
    # Redistribute unfilled game quota when a game has fewer candidates.
    chosen = {key for key, _ in selected}
    remainder = [(key, meta) for key, meta in metadata.items() if key not in chosen]
    random.Random(seed).shuffle(remainder)
    selected.extend(remainder[:max(0, min(target, len(metadata)) - len(selected))])
    return selected


def select_representatives(observations, requested):
    """Choose one strong observation per temporal bin, preserving one quality edge case."""
    ordered = sorted(observations, key=lambda item: item["frame"])
    count = min(requested, len(ordered))
    if count <= 0:
        return []
    quality_rank = {"MEASURABLE": 2, "PARTIAL": 1, "UNUSABLE": 0}
    chosen = []
    for index in range(count):
        start, end = index * len(ordered) // count, (index + 1) * len(ordered) // count
        candidates = ordered[start:end]
        def score(observation):
            confidence = (observation.get("appearance_features", {}).get("quality", {})
                          .get("detection_confidence"))
            midpoint = (candidates[0]["frame"] + candidates[-1]["frame"]) / 2
            return quality_rank[_quality(observation)], confidence if confidence is not None else -1, -abs(observation["frame"] - midpoint)
        chosen.append(max(candidates, key=score))
    worst = min(ordered, key=lambda o: (quality_rank[_quality(o)],
                                        o.get("appearance_features", {}).get("quality", {})
                                        .get("detection_confidence") or -1))
    if count > 1 and quality_rank[_quality(worst)] < min(quality_rank[_quality(o)] for o in chosen):
        chosen[-1] = worst
    return sorted({o["frame"]: o for o in chosen}.values(), key=lambda o: o["frame"])


def padded_crop_bounds(box, frame_width, frame_height, margin=.20):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(frame_width, math.ceil(x2 + dx)), min(frame_height, math.ceil(y2 + dy)))


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def atomic_write_labels(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in sorted(records.values(), key=lambda r: (r["source_game"], str(r["track_id"]))):
            if record["label"] not in LABELS:
                raise ValueError(f"Invalid human label: {record['label']}")
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(path)


def load_labels(path):
    result = {}
    if Path(path).exists():
        for record in read_observations(path):
            if record.get("label") not in LABELS:
                raise ValueError(f"Invalid stored human label: {record.get('label')}")
            result[(record["source_game"], record["track_id"])] = record
    return result


def save_label(path, dataset_id, candidate, label, reviewer=None):
    if label not in LABELS:
        raise ValueError(f"Invalid human label: {label}")
    labels = load_labels(path)
    key = (candidate["source_game"], candidate["temporary_track_id"])
    labels[key] = {"schema_version": SCHEMA_VERSION, "dataset_id": dataset_id,
                   "source_game": key[0], "track_id": key[1], "label": label,
                   "reviewed_at": datetime.now(timezone.utc).isoformat(),
                   "representative_frames": candidate["representative_frames"], "reviewer": reviewer}
    atomic_write_labels(path, labels)


def select_review_candidates(candidates, labels, review_label=None, review_sample=None, seed=57):
    """Filter only by current human labels, then optionally sample deterministically.

    Sampling chooses positions with a seeded RNG and restores candidate order so
    navigation remains consistent with the original project.
    """
    if review_label is None:
        if review_sample is not None:
            raise ValueError("review_sample requires review_label")
        return list(candidates), {"matching": len(candidates), "selected": len(candidates),
                                  "sample_capped": False}
    if review_label not in LABELS:
        raise ValueError(f"Invalid review label: {review_label}")
    matching = [candidate for candidate in candidates
                if labels.get((candidate["source_game"], candidate["temporary_track_id"]), {})
                .get("label") == review_label]
    if review_sample is None:
        selected = matching
    else:
        if review_sample <= 0:
            raise ValueError("review_sample must be positive")
        count = min(review_sample, len(matching))
        positions = sorted(random.Random(seed).sample(range(len(matching)), count))
        selected = [matching[position] for position in positions]
    return selected, {"matching": len(matching), "selected": len(selected),
                      "sample_capped": review_sample is not None and review_sample > len(matching)}


def generate_crops(project, candidates, tracks, margin):
    import cv2
    requests = defaultdict(list)
    for candidate in candidates:
        key = (candidate["source_game"], candidate["temporary_track_id"])
        for obs in select_representatives(tracks[key], candidate["representatives_requested"]):
            requests[Path(obs["source_video"])].append((obs, candidate))
    for video, items in requests.items():
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"Cannot open source video: {video}")
        try:
            for obs, candidate in sorted(items, key=lambda pair: pair[0]["frame"]):
                capture.set(cv2.CAP_PROP_POS_FRAMES, obs["frame"])
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot read {video} frame {obs['frame']}")
                bounds = padded_crop_bounds(obs["person_bbox"], frame.shape[1], frame.shape[0], margin)
                x1, y1, x2, y2 = bounds
                directory = project / "crops" / candidate["source_game"] / str(candidate["temporary_track_id"])
                directory.mkdir(parents=True, exist_ok=True)
                relative = Path("crops") / candidate["source_game"] / str(candidate["temporary_track_id"]) / f"frame_{obs['frame']:08d}.jpg"
                if not cv2.imwrite(str(project / relative), frame[y1:y2, x1:x2]):
                    raise RuntimeError(f"Cannot write crop {relative}")
                candidate["representative_frames"].append(obs["frame"])
                candidate["representative_crops"].append(str(relative))
        finally:
            capture.release()


def _distribution(values, boundaries):
    counts = [0] * (len(boundaries) + 1)
    for value in values:
        counts[sum(value > boundary for boundary in boundaries)] += 1
    return {f"bin_{i}": count for i, count in enumerate(counts)}


def _five_number(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not values:
        return {"count": 0, "min": None, "q25": None, "median": None, "q75": None, "max": None}
    at = lambda fraction: values[round(fraction * (len(values) - 1))]
    return {"count": len(values), "min": values[0], "q25": at(.25), "median": at(.5),
            "q75": at(.75), "max": values[-1]}


def build_manifest(dataset_id, sources, candidates, target, representatives, seed, margin):
    quality, participant = Counter(), Counter()
    for item in candidates:
        quality.update(item["measurement_quality_counts"])
        participant.update(item["participant_decision_counts"])
    return {"schema_version": SCHEMA_VERSION, "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(), "source_artifacts": sources,
            "source_games": sorted({c["source_game"] for c in candidates}), "random_seed": seed,
            "target_tracklet_count": target, "actual_selected_tracklet_count": len(candidates),
            "representatives_requested": representatives, "crop_margin_fraction": margin,
            "selection_strategy": "equal per-game quotas, then seeded round-robin over metadata strata: measurement quality, participant decision, track length, bbox size, image x-position, and torso lightness; unused quota is redistributed",
            "representative_strategy": "one highest-measurability/confidence observation per equal temporal bin; one lower-quality edge observation replaces the final choice when it adds variation",
            "counts_by_source_game": dict(Counter(c["source_game"] for c in candidates)),
            "measurement_quality_observation_counts": dict(quality),
            "tracklet_length_distribution": _distribution([c["observation_count"] for c in candidates], [5, 29, 149]),
            "bbox_height_distribution": _distribution([c["median_bbox_height"] or 0 for c in candidates], [40, 80, 160]),
            "participant_decision_observation_counts": dict(participant),
            "diversity_summaries": {name: _five_number([c[name] for c in candidates]) for name in
                                    ("first_frame", "median_image_x", "median_bbox_height",
                                     "median_torso_lightness", "median_pants_dark_fraction",
                                     "median_lower_leg_lightness", "median_stripe_score",
                                     "median_detection_confidence")},
            "labeling": {"number_labeled": 0, "label_counts": {}, "number_unlabeled": len(candidates)},
            "leakage_note": "Future splits must group by (source_game, temporary_track_id); prefer cross-game/rink evaluation when enough games exist."}


def write_report(path, manifest):
    lines = [f"# Labeling project {manifest['dataset_id']}", "",
             "Human labels are separate from measurements and legacy pseudo-labels. MEASURABLE means only that features can be extracted; it does not mean training-suitable.", "",
             f"- Selected: {manifest['actual_selected_tracklet_count']} / target {manifest['target_tracklet_count']}",
             f"- Seed: {manifest['random_seed']}", f"- Representatives requested: {manifest['representatives_requested']}",
             f"- Games: {manifest['counts_by_source_game']}",
             f"- Measurement-quality observations: {manifest['measurement_quality_observation_counts']}",
             f"- Tracklet lengths: {manifest['tracklet_length_distribution']}",
             f"- Bbox heights: {manifest['bbox_height_distribution']}",
             f"- Participant decisions: {manifest['participant_decision_observation_counts']}", "",
             f"- Human labels: {manifest['labeling']}",
             f"- Diversity summaries: {manifest['diversity_summaries']}", "",
             "## Selection", "", manifest["selection_strategy"], "", "## Representatives", "",
             manifest["representative_strategy"], "", "## Leakage", "", manifest["leakage_note"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_labeling_guide(path, dataset_id):
    path.write_text(f"""# Human labeling guide

Launch from the repository root:

```powershell
.venv\\Scripts\\python.exe -m tools.build_labeling_dataset --label output\\{dataset_id}
```

The UI deliberately does not display legacy V7 pseudo-labels. Labels are saved atomically after every decision in `labels.jsonl`; reopening the UI loads existing decisions and permits changes.

| Key | Label | Definition |
|---|---|---|
| H | HOME | Consistently a home-team player. |
| A | AWAY | Consistently an away-team player. |
| O | OFFICIAL | Consistently an on-ice official/referee. |
| N | NON_PARTICIPANT | Spectator, coach, bench/foreground person, scorekeeper, or another off-ice non-participant. |
| M | MIXED_TRACK | Identity or semantic subject changes, making one class unsafe. |
| U | UNSURE | Insufficient visual evidence for a reliable decision. |

Navigation: `S`, `]`, or Right Arrow skips/advances without labeling; `[` or Left Arrow goes back; `Q` or Escape saves and quits. Track IDs are temporary operational identifiers, not player identities.

## Review existing labels

```powershell
.venv\\Scripts\\python.exe -m tools.build_labeling_dataset --label output\\{dataset_id} --review-label UNSURE
.venv\\Scripts\\python.exe -m tools.build_labeling_dataset --label output\\{dataset_id} --review-label MIXED_TRACK
.venv\\Scripts\\python.exe -m tools.build_labeling_dataset --label output\\{dataset_id} --review-label NON_PARTICIPANT --review-sample 25 --seed 57
```

Review filtering uses only current human labels in `labels.jsonl`. Opening, navigating, or quitting review mode does not change a label. If a changed label no longer matches the filter, it remains in the current in-memory sequence but is excluded next time.

## Future annotation considerations

Goalies remain `HOME` or `AWAY`; goalie status is not a primary semantic class. If later error analysis demonstrates value, a future dataset may add an orthogonal human-reviewed `player_role = SKATER | GOALIE` annotation.

`NON_PARTICIPANT` intentionally includes spectators, coaches/bench staff, off-rink scorekeepers, off-ice players, non-human false detections, and other irrelevant detections. A future dataset may add optional human-reviewed subtypes if error analysis demonstrates value; they must not be inferred automatically.
""", encoding="utf-8")


def build(export_dirs, output, target=300, representatives=6, seed=57, margin=.20):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite labeling project: {output}")
    tracks, sources = summarize_exports(export_dirs)
    selected = select_candidates(tracks, target, seed)
    candidates = []
    for _, meta in selected:
        candidates.append({**meta, "representatives_requested": representatives,
                           "representative_frames": [], "representative_crops": []})
    output.mkdir(parents=True)
    generate_crops(output, candidates, tracks, margin)
    dataset_id = output.name
    with (output / "candidates.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, separators=(",", ":"), allow_nan=False) + "\n")
    (output / "labels.jsonl").touch()
    manifest = build_manifest(dataset_id, sources, candidates, target, representatives, seed, margin)
    _write_json(output / "manifest.json", manifest)
    write_report(output / "report.md", manifest)
    write_labeling_guide(output / "LABELING_GUIDE.md", dataset_id)
    return manifest


def label_project(project, reviewer=None, review_label=None, review_sample=None, seed=57):
    import cv2
    import numpy as np
    project = Path(project)
    all_candidates = list(read_observations(project / "candidates.jsonl"))
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    labels_path = project / "labels.jsonl"
    labels = load_labels(labels_path)
    candidates, review = select_review_candidates(all_candidates, labels, review_label,
                                                   review_sample, seed)
    if review_label:
        message = (f"Reviewing {review['selected']} of {review['matching']} current "
                   f"{review_label} labels")
        if review["sample_capped"]:
            message += f"; requested sample {review_sample} exceeds available count, so all are included"
        print(message)
    if not candidates:
        print("No candidates match the requested review.")
        return
    index = 0
    while 0 <= index < len(candidates):
        candidate = candidates[index]
        tiles = []
        for relative in candidate["representative_crops"]:
            image = cv2.imread(str(project / relative))
            if image is not None:
                scale = min(260 / image.shape[1], 300 / image.shape[0])
                tiles.append(cv2.resize(image, None, fx=scale, fy=scale))
        canvas = np.zeros((700, 900, 3), dtype=np.uint8)
        x = y = 10
        for tile in tiles:
            if x + tile.shape[1] > 890: x, y = 10, y + 310
            canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
            x += tile.shape[1] + 10
        existing = load_labels(labels_path).get((candidate["source_game"], candidate["temporary_track_id"]))
        title = f"{index + 1}/{len(candidates)}  {candidate['source_game']}  track {candidate['temporary_track_id']}  current: {existing['label'] if existing else 'UNLABELED'}"
        cv2.putText(canvas, title, (10, 640), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        cv2.putText(canvas, "H home  A away  O official  N non-participant  M mixed  U unsure", (10, 670), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1)
        cv2.imshow("Hockey tracklet labeling | arrows/[,]: previous/next | S skip | Q save+quit", canvas)
        key = cv2.waitKey(0) & 0xff
        if key in KEY_LABELS:
            save_label(labels_path, manifest["dataset_id"], candidate, KEY_LABELS[key], reviewer)
            labels = load_labels(labels_path)
            counts = Counter(record["label"] for record in labels.values())
            manifest["labeling"] = {"number_labeled": len(labels), "label_counts": dict(counts),
                                    "number_unlabeled": len(all_candidates) - len(labels)}
            _write_json(project / "manifest.json", manifest)
            write_report(project / "report.md", manifest)
            index += 1
        elif key in (ord("q"), 27): break
        elif key in (ord("s"), ord("]"), 83): index += 1
        elif key in (ord("["), 81): index = max(0, index - 1)
    cv2.destroyAllWindows()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports", nargs="*", type=Path, help="Directories containing appearance exports")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-tracklets", type=int, default=300)
    parser.add_argument("--representatives", type=int, default=6)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--crop-margin", type=float, default=.20)
    parser.add_argument("--label", type=Path, metavar="PROJECT")
    parser.add_argument("--reviewer")
    parser.add_argument("--review-label", choices=LABELS,
                        help="Review only candidates with this current human label")
    parser.add_argument("--review-sample", type=int, metavar="N",
                        help="Deterministically sample N matching review candidates")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.review_sample is not None and args.review_label is None:
        parser.error("--review-sample requires --review-label")
    if args.review_sample is not None and args.review_sample <= 0:
        parser.error("--review-sample must be positive")
    if (args.review_label is not None or args.review_sample is not None) and args.label is None:
        parser.error("review options require --label PROJECT")
    if args.label:
        label_project(args.label, args.reviewer, args.review_label, args.review_sample, args.seed)
    else:
        if not args.exports or not args.output:
            parser.error("exports and --output are required when building")
        print(json.dumps(build(args.exports, args.output, args.target_tracklets,
                               args.representatives, args.seed, args.crop_margin), indent=2))


if __name__ == "__main__":
    main()
