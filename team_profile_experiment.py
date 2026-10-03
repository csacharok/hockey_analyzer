"""Offline two-cluster uniform experiment using generic colour statistics.

Track IDs only cap repeated samples from one temporary tracklet; they never
define either cluster or provide labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


def load_candidates(observations, per_track=3):
    rows = defaultdict(list)
    with Path(observations).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("event") != "observation" or row.get("participant_state") != "ACCEPTED"
                    or "observation_label" not in row):
                continue
            box = row["box"]
            width, height = box[2] - box[0], box[3] - box[1]
            evidence = row.get("evidence", {})
            # Positive referee structure and weak/partial boxes are not team
            # profile samples. No existing HOME/AWAY label is consulted.
            aspect = width / height
            if (evidence.get("stripe_ratio", 0) >= 1 or row["detection_confidence"] < .5
                    or width < 25 or height < 70 or not .20 <= aspect <= .90
                    or box[0] <= 1 or box[1] <= 1 or box[2] >= 1919 or box[3] >= 1079):
                continue
            rows[row["track_id"]].append(row)
    selected = []
    for track_rows in rows.values():
        if len(track_rows) <= per_track:
            selected.extend(track_rows)
            continue
        positions = [round(index * (len(track_rows) - 1) / (per_track - 1))
                     for index in range(per_track)]
        selected.extend(track_rows[index] for index in positions)
    return selected


def torso_crop(frame, box):
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left, right = round(x1 + .18 * width), round(x1 + .82 * width)
    top, bottom = round(y1 + .12 * height), round(y1 + .62 * height)
    return frame[max(0, top):min(frame.shape[0], bottom),
                 max(0, left):min(frame.shape[1], right)]


def extract(video, rows):
    import cv2
    import numpy as np
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame"]].append(row)
    capture = cv2.VideoCapture(str(video))
    samples = []
    try:
        for frame_number in sorted(by_frame):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = capture.read()
            if not ok:
                continue
            for row in by_frame[frame_number]:
                crop = torso_crop(frame, row["box"])
                if crop.size < 900:
                    continue
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lightness = lab[:, :, 0].astype(float) / 255
                feature = np.array([
                    np.median(lightness), np.quantile(lightness, .25),
                    np.quantile(lightness, .75), np.median(lab[:, :, 1]) / 255,
                    np.median(lab[:, :, 2]) / 255, np.median(hsv[:, :, 1]) / 255,
                    np.median(hsv[:, :, 2]) / 255,
                    np.mean((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 45)),
                ], dtype=np.float32)
                samples.append(dict(row=row, feature=feature, crop=crop.copy()))
    finally:
        capture.release()
    return samples


def cluster(samples):
    import cv2
    import numpy as np
    values = np.vstack([sample["feature"] for sample in samples])
    median = np.median(values, axis=0)
    scale = np.median(np.abs(values - median), axis=0)
    scale[scale < .02] = .02
    # Lightness distribution is the generic setup prior. Chroma statistics are
    # retained for auditing, but letting each colour dimension vote equally
    # caused saturated minority jerseys/contamination to form a cluster while
    # light and dark uniforms remained mixed.
    clustering_columns = (0, 1, 2)
    columns = list(clustering_columns)
    normalized = ((values[:, columns] - median[columns])
                  / scale[columns]).astype(np.float32)
    cv2.setRNGSeed(7)
    _, labels, centers = cv2.kmeans(
        normalized, 2, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        20, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    lightness_centers = centers * scale[list(clustering_columns)] + median[list(clustering_columns)]
    light = int(lightness_centers[0, 0] < lightness_centers[1, 0])
    mapping = {light: "HOME", 1 - light: "AWAY"}
    distances = np.linalg.norm(normalized - centers[labels], axis=1)
    center_separation = float(np.linalg.norm(centers[0] - centers[1]))
    for sample, label, distance in zip(samples, labels, distances):
        sample["cluster"] = int(label)
        sample["team"] = mapping[int(label)]
        sample["distance"] = float(distance)
    result = []
    for label in range(2):
        chosen = [sample for sample in samples if sample["cluster"] == label]
        matrix = np.vstack([sample["feature"] for sample in chosen])
        lightness = matrix[:, :3]
        lightness_center = np.median(lightness, axis=0)
        lightness_scale = np.maximum(
            np.median(np.abs(lightness - lightness_center), axis=0), .03)
        profile_distances = np.linalg.norm(
            (lightness - lightness_center) / lightness_scale, axis=1)
        result.append(dict(
            cluster=label, default_team=mapping[label], observations=len(chosen),
            tracklets=len({sample["row"]["track_id"] for sample in chosen}),
            median_feature=np.median(matrix, axis=0).tolist(),
            lightness_center=lightness_center.tolist(),
            lightness_scale=lightness_scale.tolist(),
            maximum_distance=float(np.quantile(profile_distances, .95)),
            median_lightness=float(np.median(matrix[:, 0])),
            median_saturation=float(np.median(matrix[:, 5])),
            median_value=float(np.median(matrix[:, 6])),
            within_cluster_median_distance=float(np.median(
                [sample["distance"] for sample in chosen])),
            within_cluster_p90_distance=float(np.quantile(
                [sample["distance"] for sample in chosen], .9))))
    return result, center_separation


def contact_sheet(samples, output):
    import cv2
    import numpy as np
    tiles = []
    for label in range(2):
        chosen = sorted((sample for sample in samples if sample["cluster"] == label),
                        key=lambda sample: sample["distance"])
        positions = [round(index * (len(chosen) - 1) / 11) for index in range(12)]
        row = []
        for index in positions:
            sample = chosen[index]
            tile = cv2.resize(sample["crop"], (120, 160), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (119, 22), (0, 0, 0), -1)
            cv2.putText(tile, f"{sample['team']} c{label}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1)
            row.append(tile)
        tiles.append(np.hstack(row))
    cv2.imwrite(str(output), np.vstack(tiles))


def run(video, observations, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = load_candidates(observations)
    samples = extract(video, rows)
    clusters, separation = cluster(samples)
    contact_sheet(samples, output_directory / "cluster_contact_sheet.jpg")
    report = dict(
        method="Unsupervised two-cluster experiment; no existing team labels or known IDs used.",
        feature_names=["median_L", "q25_L", "q75_L", "median_Lab_a",
                       "median_Lab_b", "median_saturation", "median_value",
                       "saturated_fraction"],
        candidate_observations=len(samples), center_separation_robust_units=separation,
        clusters=clusters)
    (output_directory / "inferred_team_profiles.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.video, arguments.observations,
                         arguments.output_directory), indent=2))
