"""Replay V7 observation logs into semantic-free appearance datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from appearance_features import AppearanceFeatureExtractor, BODY_REGIONS, SCHEMA_VERSION, TrackletAppearanceAggregator


def _legacy(record):
    if "label" not in record:
        return None
    return {"provenance": "v7_semantic_classifier_pseudo_label",
            "is_human_ground_truth": False, "label": record.get("label"),
            "confidence": record.get("confidence"),
            "observation_label": record.get("observation_label"),
            "observation_confidence": record.get("observation_confidence")}


def dataset_record(game, source, record, features):
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_game": game,
        "source_video": str(source),
        "frame": record["frame"],
        "timestamp": record["timestamp"],
        "temporary_track_id": record["track_id"],
        "person_bbox": record["box"],
        "crop_reference": {"source_video": str(source), "frame": record["frame"],
                           "bbox": record["box"]},
        "appearance_features": features,
        "participant_evidence": {"decision": record.get("participant_state"),
                                 "accepted": record.get("participant_accepted"),
                                 "reason": record.get("participant_reason"),
                                 "evidence": record.get("participant_evidence", {})},
        "legacy_metadata": _legacy(record),
    }
    return result


class ContactSheets:
    def __init__(self, limit=20):
        self.limit = limit
        self.items = {name: [] for name in (
            "highest_torso_lightness", "lowest_torso_lightness",
            "highest_pants_dark_fraction", "lowest_pants_dark_fraction",
            "highest_lower_leg_lightness", "lowest_lower_leg_lightness",
            "highest_stripe_score", "low_quality")}

    def add(self, record, features, frame, cv2):
        if not record.get("participant_accepted"):
            return
        candidates = []
        regions = features["regions"]
        if regions["torso"].get("usable"):
            value = regions["torso"]["lab_lightness_median"]
            candidates += [("highest_torso_lightness", value, value),
                           ("lowest_torso_lightness", -value, value)]
        if regions["pants"].get("usable"):
            value = regions["pants"]["dark_fraction"]
            candidates += [("highest_pants_dark_fraction", value, value),
                           ("lowest_pants_dark_fraction", -value, value)]
        if regions["lower_legs"].get("usable"):
            value = regions["lower_legs"]["lab_lightness_median"]
            candidates += [("highest_lower_leg_lightness", value, value),
                           ("lowest_lower_leg_lightness", -value, value)]
        if features["structure"].get("available"):
            value = features["structure"]["vertical_stripe_score"]
            candidates.append(("highest_stripe_score", value, value))
        if features["quality"]["crop_quality"] != "GOOD":
            candidates.append(("low_quality", -features["quality"]["usable_region_count"],
                               features["quality"]["usable_region_count"]))
        if not candidates:
            return
        x1, y1, x2, y2 = map(round, record["box"])
        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
        if not crop.size:
            return
        tile = cv2.resize(crop, (140, 190), interpolation=cv2.INTER_AREA)
        for category, rank, displayed in candidates:
            annotated = tile.copy()
            cv2.rectangle(annotated, (0, 0), (139, 27), (0, 0, 0), -1)
            text = f"f{record['frame']} t{record['track_id']} {displayed:.3f}"
            cv2.putText(annotated, text, (3, 17), cv2.FONT_HERSHEY_SIMPLEX, .34,
                        (255, 255, 255), 1)
            self.items[category].append((rank, annotated))
            self.items[category].sort(key=lambda item: item[0], reverse=True)
            del self.items[category][self.limit:]

    def write(self, directory, cv2):
        import numpy as np
        paths = []
        for category, ranked in self.items.items():
            if not ranked:
                continue
            tiles = [tile for _, tile in ranked]
            rows = []
            for start in range(0, len(tiles), 5):
                row = tiles[start:start + 5]
                while len(row) < 5:
                    row.append(np.zeros_like(tiles[0]))
                rows.append(np.hstack(row))
            path = directory / f"appearance_contact_sheet_{category}.jpg"
            cv2.imwrite(str(path), np.vstack(rows))
            paths.append(path)
        return paths


def export(game, video, observations, output):
    import cv2
    output.mkdir(parents=True, exist_ok=False)
    source = video.resolve()
    extractor = AppearanceFeatureExtractor()
    aggregator = TrackletAppearanceAggregator()
    sheets = ContactSheets()
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_number, frame = -1, None
    counts, extraction_seconds, started = Counter(), 0.0, time.perf_counter()
    output_path = output / "appearance_observations.jsonl"
    try:
        with observations.open("r", encoding="utf-8") as source_log, output_path.open("x", encoding="utf-8") as target:
            for line in source_log:
                record = json.loads(line)
                if record.get("event") != "observation":
                    continue
                requested = record["frame"]
                while frame_number < requested:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Video ended before logged frame {requested}")
                    frame_number += 1
                tick = time.perf_counter()
                features = extractor.extract(frame, record["box"], cv2, record.get("detection_confidence"))
                extraction_seconds += time.perf_counter() - tick
                exported = dataset_record(game, source, record, features)
                target.write(json.dumps(exported, allow_nan=False, separators=(",", ":")) + "\n")
                aggregator.add(record["track_id"], features)
                sheets.add(record, features, frame, cv2)
                counts["observations"] += 1
                counts[f"quality_{features['quality']['crop_quality'].lower()}"] += 1
                if record.get("participant_accepted"):
                    counts["participant_accepted"] += 1
    finally:
        capture.release()
    tracklets = aggregator.summaries()
    (output / "appearance_tracklets.json").write_text(json.dumps({
        "schema_version": SCHEMA_VERSION, "source_game": game,
        "temporary_tracklets_only": True, "tracklets": tracklets}, indent=2), encoding="utf-8")
    contact_paths = sheets.write(output, cv2)
    elapsed = time.perf_counter() - started
    performance = {
        "source_game": game, **counts, "tracklets": len(tracklets),
        "appearance_extraction_seconds": extraction_seconds,
        "extraction_ms_per_observation": 1000 * extraction_seconds / max(1, counts["observations"]),
        "export_wall_seconds": elapsed,
        "replay_fps": (frame_number + 1) / max(elapsed, 1e-9),
        "source_fps": fps,
        "note": "Replay FPS excludes YOLO/boundary inference; V7 full-pipeline FPS is reported separately.",
    }
    (output / "performance.json").write_text(json.dumps(performance, indent=2), encoding="utf-8")
    (output / "performance_report.md").write_text(
        f"# {game} appearance extraction performance\n\n"
        f"- Observations: {counts['observations']}\n"
        f"- Extraction: {performance['extraction_ms_per_observation']:.3f} ms/observation "
        f"({extraction_seconds:.3f} s total)\n"
        f"- Offline replay: {performance['replay_fps']:.2f} decoded frames/s over {elapsed:.2f} s\n"
        "- Added work: four normalized array views, two lower-leg concatenation/copies, and "
        "HSV/Lab conversions for each usable region. Contact-sheet tiles are the only retained images.\n"
        "- The replay rate is not end-to-end analyzer FPS because detector, tracker, and boundaries were reused from V7.\n",
        encoding="utf-8")
    return performance, contact_paths


def write_schema(path):
    schema = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "Semantic-free visual observations; no field is a human team/role label.",
        "normalized_body_regions": BODY_REGIONS,
        "appearance_features": {
            "region_measurements": ["lab_lightness_median", "lab_lightness_q25",
                                    "lab_lightness_q75", "hsv_saturation_median",
                                    "hsv_value_median", "dark_fraction", "light_fraction",
                                    "saturated_fraction", "pixels", "usable_pixel_fraction", "usable"],
            "structure": ["vertical_stripe_score", "vertical_edge_fraction",
                          "horizontal_edge_fraction", "vertical_orientation_ratio",
                          "stripe_supporting_rows", "stripe_total_rows",
                          "stripe_supporting_pixel_edges", "striped_row_fraction"],
            "quality": ["detection_confidence", "normalized_bbox_height", "bbox_aspect_ratio",
                        "frame_edge_truncated", "usable_region_count", "crop_quality"],
        },
        "legacy_metadata": {"is_human_ground_truth": False,
                            "warning": "V7 semantic outputs are pseudo-label metadata only."},
    }
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game")
    parser.add_argument("video", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    performance, paths = export(args.game, args.video, args.observations, args.output)
    write_schema(args.output.parent / "appearance_schema.json")
    print(json.dumps({"performance": performance, "contact_sheets": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
