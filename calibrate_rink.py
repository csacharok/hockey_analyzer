"""Export an unscaled video frame, coordinate grid, and editable rink JSON."""

import argparse
import json
from pathlib import Path
import uuid

from analyze_tracks import ROOT, validate_input


def export_calibration(source: Path, frame_index: int) -> Path:
    source = validate_input(source)
    if frame_index < 0:
        raise ValueError("Frame index must be nonnegative.")
    import cv2

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        # Sequential decoding avoids reliance on inaccurate container frame counts.
        for index in range(frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode requested frame {frame_index}; decoding ended at {index}.")
    finally:
        capture.release()

    height, width = frame.shape[:2]
    directory = ROOT / "output" / "calibration" / f"{source.stem}_frame_{frame_index}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    grid = frame.copy()
    for x in range(0, width, 100):
        cv2.line(grid, (x, 0), (x, height - 1), (255, 180, 0), 1)
        cv2.putText(grid, str(x), (min(x + 3, width - 60), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(grid, str(x), (min(x + 3, width - 60), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    for y in range(100, height, 100):
        cv2.line(grid, (0, y), (width - 1, y), (255, 180, 0), 1)
        cv2.putText(grid, str(y), (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(grid, str(y), (5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    for filename, image in (("frame.png", frame), ("grid.png", grid)):
        if not cv2.imwrite(str(directory / filename), image):
            raise RuntimeError(f"Could not write {filename}.")
    template = {
        "image_width": width, "image_height": height,
        "polygon": [], "boundary_margin_px": 8,
        "calibration": {"source": source.name, "frame_index": frame_index},
        "notes": "Fill polygon with pixel [x,y] vertices in boundary order; do not repeat first vertex. Static camera-view experiment only.",
    }
    config = directory / "rink.json"
    config.write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration files: {directory}\nImage: {width} x {height}; zero-based decoded frame {frame_index}.")
    print("Use grid.png or a pixel-coordinate image viewer to select the visible playable-ice boundary.")
    print(f"Edit polygon in {config}; origin is top-left, x increases right, y increases down.")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--frame", type=int, default=0, help="Zero-based decoded frame index (default: 0)")
    args = parser.parse_args()
    try:
        export_calibration(args.video, args.frame)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
