"""Local temporary ByteTrack IDs with optional experimental manual target identity."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
import time
import uuid

from rink_geometry import ON_ICE, OFF_ICE, UNKNOWN, STATES, RinkGeometry, contact_point
from target_identity import Observation, TargetIdentity, appearance_histogram
from team_classifier import TeamClassifier
from team_diagnostics import TeamDiagnostics
from participant_filter import ParticipantFilter, REJECTED
from kickplate_boundary import KickplateBoundaryDetector
from near_board_boundary import NearBoardBoundaryDetector
from pipeline_profiler import PipelineProfiler
from provider_overlay import ProviderOverlayMask


ROOT = Path(__file__).resolve().parent
TRACK_OPTIONS = dict(
    persist=True, tracker="bytetrack.yaml", classes=[0], imgsz=1280,
    conf=0.15, device=0, verbose=False, save=False,
)


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float

    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video dimensions must be positive.")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Video must report a finite, positive frame rate.")


def validate_input(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"Input video does not exist: {path}")
    return path


def output_path(source: Path, directory: Path) -> Path:
    """Use a fresh name, including when a generated video is used as input."""
    return directory.resolve() / f"{source.stem}_tracks_{uuid.uuid4().hex[:12]}.mp4"


def track_label(track_id: int, confidence: float, state: str | None = None) -> str:
    return f"Track {track_id} | {confidence:.2f}" + (f" | {state}" if state else "")


def summary(frames: int, elapsed: float, output: Path, unique_ids: set[int], counts=None) -> str:
    fps = frames / elapsed if elapsed > 0 else 0.0
    report = (
        f"Frames processed: {frames}\n"
        f"Elapsed processing time: {elapsed:.2f} s\n"
        f"Average processing FPS: {fps:.2f}\n"
        f"Output: {output}\n"
        f"Unique short-term track IDs: {len(unique_ids)}"
    )
    if counts is not None:
        report += "\n" + "\n".join(f"{state} detections (tracked person-frames): {counts[state]}" for state in STATES)
    return report


def draw_annotations(frame, detections, tracked_boxes, cv2, rink=None, counts=None,
                     team_results=None, participant_results=None, boundary=None,
                     near_boundary=None, use_static_polygon=False) -> set[int]:
    """Classify and draw after inference; never feed annotations to the tracker."""
    height, width = frame.shape[:2]
    scale = max(0.45, height / 1800)
    thickness = max(1, round(height / 720))
    colors = {ON_ICE: (70, 255, 70), OFF_ICE: (70, 70, 255), UNKNOWN: (0, 220, 255)}
    team_colors = {"HOME": (40, 80, 255), "AWAY": (255, 180, 40),
                   "OFFICIAL": (255, 80, 255), "UNKNOWN": (0, 220, 255)}
    if rink is not None and use_static_polygon:
        for a, b in rink.edges():
            cv2.line(frame, tuple(round(v) for v in a), tuple(round(v) for v in b), (255, 220, 0), thickness)
    if boundary is not None and boundary.points:
        for a, b in zip(boundary.points, boundary.points[1:]):
            if b[0] - a[0] <= boundary.bin_width * 1.6:
                cv2.line(frame, tuple(round(v) for v in a), tuple(round(v) for v in b),
                         (0, 255, 255), thickness)
    if near_boundary is not None and near_boundary.points:
        for a, b in zip(near_boundary.points, near_boundary.points[1:]):
            if b[0] - a[0] <= near_boundary.bin_width * 1.6:
                cv2.line(frame, tuple(round(v) for v in a), tuple(round(v) for v in b),
                         (255, 0, 255), thickness)

    def label(text, x, y, color):
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max(0, min(x, width - tw - 8))
        y = max(th + 8, min(y, height - baseline - 4))
        cv2.rectangle(frame, (x, y - th - 6), (min(width - 1, x + tw + 6), y + baseline + 3), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    for x1, y1, x2, y2 in detections:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (170, 170, 170), 1)

    ids = set()
    if tracked_boxes is not None and tracked_boxes.is_track:
        boxes = tracked_boxes.cpu()
        for xyxy, identifier, confidence in zip(boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist()):
            identifier = int(identifier)
            ids.add(identifier)
            state = rink.classify(xyxy) if rink is not None else None
            team = team_results.get(identifier) if team_results else None
            participant = participant_results.get(identifier) if participant_results else None
            color = ((70, 70, 255) if participant is not None and participant.state == REJECTED
                     else team_colors[team.label] if team is not None
                     else (colors[state] if state else (70, 255, 70)))
            if state is not None:
                if counts is not None:
                    counts[state] += 1
                point = tuple(round(v) for v in contact_point(xyxy))
                cv2.circle(frame, point, 5, color, -1)
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            text = (f"NON-PARTICIPANT {identifier} | {participant.reason}" 
                    if participant is not None and participant.state == REJECTED
                    else f"{team.label} {identifier} | {team.confidence:.2f}" if team is not None
                    else track_label(identifier, confidence, state))
            label(text, x1, y1 - 4, color)
    legend = "Temporary tracks - NOT player/jersey IDs | Gray: person detection"
    if rink is not None:
        legend += " | Green: ON ICE | Red: OFF ICE | Yellow: UNKNOWN"
    else:
        legend += " | Green: tracked"
    label(legend, 8, 26, (255, 255, 255))
    if rink is not None and use_static_polygon:
        label("STATIC polygon experiment - camera pans can invalidate geometry", 8, 54, (255, 255, 255))
    if team_results is not None:
        label("Team mode: visual evidence; track IDs remain temporary", 8, 82, (255, 255, 255))
    return ids


def tracked_box_rows(boxes):
    if boxes is None or not boxes.is_track:
        return []
    boxes = boxes.cpu()
    return [(tuple(box), int(identifier), float(confidence))
            for box, identifier, confidence in zip(boxes.xyxy.tolist(), boxes.id.tolist(), boxes.conf.tolist())]


def target_observations(frame, boxes, cv2, rink=None):
    if boxes is None or not boxes.is_track:
        return []
    boxes = boxes.cpu()
    return [Observation(int(identifier), tuple(box), rink.classify(box) if rink else UNKNOWN,
                        appearance_histogram(frame, box, cv2))
            for box, identifier in zip(boxes.xyxy.tolist(), boxes.id.tolist())]


def select_target_point(frame, observations, cv2):
    """Fit preview to desktop; return click in original-resolution coordinates."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / width, 720 / height)
    preview = cv2.resize(frame, (round(width * scale), round(height * scale)))
    sx, sy = preview.shape[1] / width, preview.shape[0] / height
    for observation in observations:
        x1, y1, x2, y2 = observation.box
        cv2.rectangle(preview, (round(x1 * sx), round(y1 * sy)),
                      (round(x2 * sx), round(y2 * sy)), (0, 255, 255), 2)
    cv2.putText(preview, "Click #57 inside ONE yellow box | Esc cancels", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    title = "Seed TARGET #57"
    point = []

    def clicked(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [x / sx, y / sy]

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(title, clicked)
        cv2.imshow(title, preview)
        while not point:
            if cv2.waitKey(30) & 0xFF == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Target seeding cancelled.")
    finally:
        cv2.destroyWindow(title)
    return tuple(point)


def draw_target(frame, identity, cv2):
    color = (255, 0, 255) if identity.state != "LOST" else (0, 165, 255)
    text = f"TARGET #57 | {identity.state} | Frame {identity.frame}"
    if identity.current is not None:
        x1, y1, x2, y2 = map(round, identity.current.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        text = (f"TARGET #57 | Track {identity.current.track_id} | {identity.state}"
                f" | Frame {identity.frame}")
        if identity.score is not None:
            text += f" | Reacquisition score {identity.score:.2f}"
        cv2.putText(frame, "TARGET #57", (max(0, x1), max(20, y1 - 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)
    elif identity.state == "GRACE":
        text = f"TARGET #57 | Track {identity.last.track_id} | GRACE (not observed) | Frame {identity.frame}"
    cv2.rectangle(frame, (5, 65), (min(frame.shape[1] - 1, 1120), 100), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2, cv2.LINE_AA)


def analyze(source: Path, rink: RinkGeometry | None = None,
            target_frame: int | None = None, target_point=None,
            target_grace_frames=2, target_max_gap_seconds=3.0,
            team_classifier: TeamClassifier | None = None,
            team_output_directory: Path | None = None,
            performance_profile: Path | None = None) -> Path:
    if target_frame is not None and (type(target_frame) is not int or target_frame < 0):
        raise ValueError("Target frame must be a nonnegative zero-based integer.")
    if target_point is not None and (target_frame is None or len(target_point) != 2
                                     or not all(math.isfinite(v) for v in target_point)):
        raise ValueError("--target-point requires --target-frame and two finite pixel coordinates.")
    if (team_classifier is not None and rink is None
            and team_classifier.participant_filter.get("use_static_polygon", False)):
        raise ValueError("Legacy static-polygon filtering requires --rink.")
    source = validate_input(source)
    weights = ROOT / "yolo11s.pt"
    if not weights.is_file():
        raise ValueError(f"Local model missing: {weights}. No model will be downloaded.")

    # Keep deterministic helpers importable without importing the ML stack.
    import os
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import cv2
    from ultralytics import YOLO

    started = time.perf_counter()
    profiler = PipelineProfiler() if performance_profile is not None else None
    capture = cv2.VideoCapture(str(source))
    writer = None
    debug = None
    team_debug = None
    partial = None
    frames = 0
    unique_ids: set[int] = set()
    counts = Counter({state: 0 for state in STATES}) if rink is not None else None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open input video: {source}")
        spec = VideoSpec(
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
        if rink is not None:
            rink.validate_resolution(spec.width, spec.height)
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        identity = TargetIdentity(spec.fps, target_grace_frames, target_max_gap_seconds) if target_frame is not None else None
        participant_filter = ParticipantFilter(team_classifier.participant_filter) if team_classifier is not None else None
        boundary_detector = KickplateBoundaryDetector(team_classifier.kickplate_boundary) if team_classifier is not None else None
        near_boundary_detector = NearBoardBoundaryDetector(team_classifier.near_board_boundary) if team_classifier is not None else None
        overlay_mask = ProviderOverlayMask(team_classifier.provider) if team_classifier is not None else ProviderOverlayMask()
        provider_exclusions = Counter()
        use_polygon = team_classifier.participant_filter.get("use_static_polygon", False) if team_classifier is not None else True
        if target_point is not None and not (0 <= target_point[0] < spec.width and 0 <= target_point[1] < spec.height):
            raise ValueError("Target point must be inside the original video resolution.")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode the first frame: {source}")

        model = YOLO(str(weights))
        raw_detections = []

        def retain_detections(predictor):
            # Registered BEFORE model.track's callback: tracking can discard
            # detections that have not yet acquired a confirmed ID.
            boxes = predictor.results[0].boxes
            coordinates = boxes.xyxy.cpu().tolist()
            if overlay_mask.enabled:
                kept, decisions = overlay_mask.filter_detections(
                    coordinates, spec.width, spec.height)
                provider_exclusions["post_yolo_removed"] += len(decisions["removed"])
                provider_exclusions["retained_partial_overlaps"] += len(
                    decisions["retained_overlaps"])
                boxes = boxes[kept]
                predictor.results[0].boxes = boxes
                coordinates = boxes.xyxy.cpu().tolist()
            raw_detections[:] = coordinates

        model.add_callback("on_predict_postprocess_end", retain_detections)
        output = ((team_output_directory or ROOT / "output" / "team_baseline").resolve() / "annotated.mp4"
                  if team_classifier is not None else output_path(source, ROOT / "output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ValueError(f"Output already exists; move or remove it before rerunning: {output}")
        if team_classifier is not None:
            diagnostics_path = output.parent / "observations.jsonl"
            if diagnostics_path.exists():
                raise ValueError(f"Diagnostics already exist; move or remove them before rerunning: {diagnostics_path}")
            team_debug = TeamDiagnostics(diagnostics_path, dict(
                event="configuration", schema_version=1, source=str(source), output=str(output),
                fps=spec.fps, width=spec.width, height=spec.height,
                track_options=TRACK_OPTIONS, rink=vars(rink)))
        if identity is not None:
            debug_path = output.with_suffix(".identity.jsonl")
            debug = debug_path.open("x", encoding="utf-8")
            debug.write(json.dumps(dict(event="configuration", schema_version=5, source=str(source),
                                       output=str(output), fps=spec.fps, width=spec.width, height=spec.height,
                                       seed_frame=target_frame, seed_point=target_point,
                                       grace_frames=target_grace_frames, max_gap_seconds=target_max_gap_seconds,
                                       confirmation_frames=3, minimum_score=.65, ambiguity_margin=.15,
                                       confirmation_model='temporal_hysteresis',
                                       geometry_model='robust_log_history',
                                       geometry_history_seconds=identity.GEOMETRY_HISTORY_SECONDS,
                                       geometry_soft_factor=1.43, geometry_hard_factor=2.,
                                       strong_appearance_threshold=identity.STRONG_APPEARANCE,
                                       support_appearance_threshold=identity.SUPPORT_APPEARANCE,
                                       temporal_appearance_threshold=identity.TEMPORAL_APPEARANCE,
                                       pending_window_seconds=identity.PENDING_SECONDS,
                                       minimum_support_fraction=.8, allowed_missing_frames=0,
                                       track_options=TRACK_OPTIONS,
                                       rink=vars(rink) if rink else None)) + "\n")
            debug.flush()
            print(f"Target diagnostics: {debug_path}", flush=True)
        partial = output.with_suffix(".partial.mp4")
        # Exclusive creation avoids overwriting anything already on disk.
        with partial.open("xb"):
            pass
        writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps, (spec.width, spec.height))
        if not writer.isOpened():
            raise RuntimeError("Cannot initialize the MP4 video writer (mp4v).")

        while ok:
            if frame.shape[:2] != (spec.height, spec.width):
                raise RuntimeError("Decoded frame dimensions changed unexpectedly.")
            with profiler.measure("provider overlay masking") if profiler else nullcontext():
                analysis_frame = overlay_mask.apply(frame)
            with profiler.measure("shared frame preprocessing") if profiler else nullcontext():
                analysis_hsv = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2HSV)
            track_started = time.perf_counter()
            result = model.track(analysis_frame, **TRACK_OPTIONS)[0]
            track_elapsed = time.perf_counter() - track_started
            if profiler:
                speed = result.speed or {}
                yolo_seconds = sum(float(speed.get(key, 0.0)) for key in ("preprocess", "inference", "postprocess")) / 1000
                profiler.add("YOLO person inference", min(track_elapsed, yolo_seconds))
                profiler.add("ByteTrack/tracking and framework overhead", max(0.0, track_elapsed - yolo_seconds))
            team_results = None
            participant_results = None
            with profiler.measure("far kickplate detection") if profiler else nullcontext():
                boundary = (boundary_detector.estimate(analysis_frame, cv2, analysis_hsv)
                            if boundary_detector is not None else None)
            with profiler.measure("near-board detection") if profiler else nullcontext():
                near_boundary = (near_boundary_detector.estimate(analysis_frame, cv2, analysis_hsv)
                                 if near_boundary_detector is not None else None)
            if team_classifier is not None:
                team_results = {}
                participant_results = {}
                for box, identifier, detection_confidence in tracked_box_rows(result.boxes):
                    filter_started = time.perf_counter()
                    participant = participant_filter.classify(
                        analysis_frame, box, rink.classify(box) if rink is not None else UNKNOWN, cv2,
                                                              boundary, near_boundary, use_polygon)
                    if profiler:
                        elapsed = time.perf_counter() - filter_started
                        profiler.add("participant filtering", elapsed, "detection")
                        profiler.add("local surface/ice validation", elapsed, "detection")
                    participant_results[identifier] = participant
                    classified = None
                    if participant.accepted:
                        classify_started = time.perf_counter()
                        classified = team_classifier.classify(
                            identifier, analysis_frame, box, cv2, detection_confidence)
                        if profiler:
                            profiler.add("team/role feature extraction and temporal classification",
                                         time.perf_counter() - classify_started, "detection")
                        team_results[identifier] = classified
                    diagnostic_started = time.perf_counter()
                    team_debug.add(frames, spec.fps, identifier, box, detection_confidence,
                                   participant, classified, analysis_frame, cv2)
                    if profiler:
                        profiler.add("JSONL/diagnostic output", time.perf_counter() - diagnostic_started,
                                     "detection")
            if identity is not None:
                observations = target_observations(frame, result.boxes, cv2, rink)
                point = None
                if frames == target_frame:
                    if not observations:
                        raise ValueError("No confirmed tracks at seed frame; choose another --target-frame.")
                    point = target_point if target_point is not None else select_target_point(frame, observations, cv2)
                identity.update(frames, observations, point)
                for event in identity.events:
                    print(event, flush=True)
                for record in [*identity.records, identity.debug_frame()]:
                    debug.write(json.dumps(record, allow_nan=False) + "\n")
                if identity.events or frames % 30 == 0:
                    debug.flush()
                # Full per-frame evidence is saved; console samples it to stay readable.
                if identity.candidates and (identity.events or identity.pending_count
                                            or frames % max(1, round(spec.fps / 2)) == 0):
                    print(f"REACQUISITION CANDIDATES | frame {frames} | time {frames / spec.fps:.3f}s", flush=True)
                    for candidate in identity.candidates:
                        reasons = ', '.join(candidate['reasons']) or candidate['decision']
                        print(f"  Track {candidate['track_id']}: score {candidate['score']:.3f} | {reasons}", flush=True)
            annotation_started = time.perf_counter()
            unique_ids.update(draw_annotations(frame, raw_detections, result.boxes, cv2, rink, counts,
                                               team_results, participant_results, boundary, near_boundary,
                                               use_static_polygon=use_polygon))
            if identity is not None:
                draw_target(frame, identity, cv2)
            if profiler:
                profiler.add("annotation rendering", time.perf_counter() - annotation_started)
            write_started = time.perf_counter()
            writer.write(frame)
            if profiler:
                profiler.add("video encoding/writing", time.perf_counter() - write_started)
            frames += 1
            if frames % 300 == 0:
                print(f"Processed {frames}/{expected_frames or '?'} frames", flush=True)
                if team_debug is not None:
                    team_debug.flush()
            decode_started = time.perf_counter()
            ok, frame = capture.read()
            if profiler:
                profiler.add("video decode / frame acquisition", time.perf_counter() - decode_started)

        if identity is not None and identity.last is None:
            raise ValueError(f"Seed frame {target_frame} was not decoded; only {frames} frames available.")
        if expected_frames > 0 and frames != expected_frames:
            print(
                f"Frame-count discrepancy: decoded {frames}; container reports {expected_frames}. "
                "Output includes every successfully decoded frame. "
                "This can indicate inaccurate metadata or an early decode failure.",
                flush=True,
            )
        writer.release()
        writer = None
        if partial.stat().st_size == 0:
            raise RuntimeError("Video writer produced an empty file.")
        partial.rename(output)
        if debug is not None:
            debug.write(json.dumps(dict(event="summary", frames=frames, counts=identity.counts,
                                       associated_ids=identity.ids, transitions=identity.transitions,
                                       report=identity.summary(), completed=True)) + "\n")
        if team_debug is not None:
            team_debug.finish(frames, cv2, dict(provider_exclusions))
    except BaseException:
        if partial is not None and partial.exists():
            print(f"Incomplete output (not a finished result): {partial}", flush=True)
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if debug is not None:
            debug.close()
        if team_debug is not None:
            team_debug.close()

    elapsed = time.perf_counter() - started
    if profiler is not None:
        profiler.write(performance_profile, frames, elapsed, notes=[
            "YOLO timing uses Ultralytics preprocess/inference/postprocess measurements; the remainder of model.track wall time is attributed to ByteTrack and framework overhead.",
            "Participant filtering includes local surface validation; the local-surface row is a nested view and must not be added to stage totals.",
            "Team/role and diagnostic stages operate per accepted detection; participant filtering operates per tracked detection.",
        ])
    print(summary(frames, elapsed, output, unique_ids, counts))
    if identity is not None:
        print(identity.summary())
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Existing local video; opened read-only")
    parser.add_argument("--rink", type=Path, help="Static playable-ice polygon JSON; omit for tracking-only baseline")
    parser.add_argument("--target-frame", type=int, help="Seed #57 on this zero-based decoded frame; opens click preview")
    parser.add_argument("--target-point", type=float, nargs=2, metavar=("X", "Y"),
                        help="Seed using original-resolution pixels instead of click preview; requires --target-frame")
    parser.add_argument("--target-grace-frames", type=int, default=2,
                        help="Missing frames tolerated before LOST (default: 2); no stale target box is drawn")
    parser.add_argument("--target-max-gap-seconds", type=float, default=3.0,
                        help="Reacquisition window since last observed target (default: 3 seconds)")
    parser.add_argument("--team-config", type=Path,
                        help="Opt-in team analysis using per-game visual profiles")
    parser.add_argument("--team-output-directory", type=Path,
                        help="Team artifact directory (default: output/team_baseline)")
    parser.add_argument("--performance-profile", type=Path,
                        help="Write stage timing JSON and a sibling Markdown report")
    args = parser.parse_args()
    try:
        rink = RinkGeometry.load(args.rink) if args.rink else None
        team_classifier = TeamClassifier.load(args.team_config) if args.team_config else None
        analyze(args.video, rink, args.target_frame, args.target_point,
                args.target_grace_frames, args.target_max_gap_seconds,
                team_classifier, args.team_output_directory, args.performance_profile)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
