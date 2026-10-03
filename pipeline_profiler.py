"""Small wall-clock profiler for the opt-in analysis pipeline."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import time


class PipelineProfiler:
    def __init__(self):
        self.started = time.perf_counter()
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.units = {}

    @contextmanager
    def measure(self, stage, unit="frame"):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, time.perf_counter() - started, unit)

    def add(self, stage, seconds, unit="frame", calls=1):
        self.seconds[stage] += max(0.0, float(seconds))
        self.calls[stage] += calls
        self.units[stage] = unit

    def report(self, frames, total_seconds=None):
        total = float(total_seconds if total_seconds is not None else time.perf_counter() - self.started)
        stages = []
        for name in sorted(self.seconds, key=self.seconds.get, reverse=True):
            seconds = self.seconds[name]
            calls = self.calls[name]
            stages.append(dict(
                stage=name, total_seconds=seconds,
                mean_milliseconds_per_frame=1000 * seconds / frames if frames else 0.0,
                percentage_of_total=100 * seconds / total if total else 0.0,
                call_count=calls, call_unit=self.units[name],
                mean_milliseconds_per_call=1000 * seconds / calls if calls else 0.0,
            ))
        return dict(total_seconds=total, frames=frames,
                    processing_fps=frames / total if total else 0.0, stages=stages)

    def write(self, json_path, frames, total_seconds=None, notes=None):
        json_path = Path(json_path)
        report = self.report(frames, total_seconds)
        report["notes"] = list(notes or [])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        rows = [
            "# Pipeline performance profile", "",
            f"- Frames: {frames:,}",
            f"- Wall-clock runtime: {report['total_seconds']:.2f} seconds",
            f"- Processing rate: {report['processing_fps']:.2f} FPS", "",
            "| Stage | Total seconds | Mean ms/frame | Runtime % | Calls | Unit | Mean ms/call |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
        for stage in report["stages"]:
            rows.append("| {stage} | {total_seconds:.2f} | {mean_milliseconds_per_frame:.3f} | "
                        "{percentage_of_total:.1f}% | {call_count:,} | {call_unit} | "
                        "{mean_milliseconds_per_call:.3f} |".format(**stage))
        if report["notes"]:
            rows.extend(["", "## Notes", ""] + [f"- {note}" for note in report["notes"]])
        json_path.with_suffix(".md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return report
