# Hockey Analyzer — Agent Instructions

## Project purpose

Hockey Analyzer is a local computer-vision system for analyzing hockey game video.

The current primary objective is **team-level hockey analysis**:

1. detect players and other relevant objects,
2. maintain short-term tracklets,
3. classify skaters by team/role,
4. understand team-level positioning and hockey events,
5. produce timestamped, auditable analysis.

Reliable persistent identification of a specific player remains a project goal, but it is **not a prerequisite for useful team analysis**.

The system should distinguish between:

- detection: where an object is,
- tracking: which detections belong to the same short-term tracklet,
- team/role classification: which side or role a tracklet belongs to,
- hockey-event understanding: what happened,
- player identity: which rostered player owns one or more tracklets.

Do not treat a ByteTrack track ID as a permanent player identity.

---

## Development approach

- Prefer the simplest implementation that satisfies the requirement.
- Make small, targeted changes rather than broad refactors unless necessary.
- Run relevant tests after code changes.
- Preserve existing behavior unless the task explicitly changes it.
- When debugging, inspect existing logs/tests/code before modifying architecture.
- Do not hardcode benchmark-specific IDs, timestamps, coordinates, or expected results.
- For difficult bugs, identify the root cause before implementing a workaround.

---

## Current architectural direction

The intended analysis pipeline is:

```text
video
  ↓
object detection
  ↓
short-term tracking / tracklets
  ↓
rink / on-ice classification
  ↓
team and role classification
  ↓
anonymous player/team analysis
  ↓
hockey-event understanding
  ↓
optional persistent player identity resolution
  ↓
clips, statistics, and higher-level analysis
```

Player identity is deliberately downstream.

The analyzer should be capable of producing useful results even when no player has been assigned a roster identity.

For example, a continuous tracklet may represent:

```text
team = HOME
track_id = 123
identity = UNKNOWN
```

and still contribute to positioning, zone occupancy, pressure, entries/exits, shifts, possession evidence, or other team-level analysis.

---

## Identity and tracking principles

### Track IDs are temporary

ByteTrack IDs represent short-term visual tracklets.

Never assume:

```text
track_id == player_identity
```

Track IDs may change after:

- occlusion,
- glass/stanchion obstruction,
- players crossing,
- leaving/re-entering frame,
- detection failure,
- line changes,
- camera movement.

### ByteTrack authority

When a known target is currently bound to an active ByteTrack track, ByteTrack remains authoritative.

Do not attempt unnecessary reacquisition while the bound track remains available.

### Persistent identity

Persistent player identity should eventually combine multiple independent sources of evidence, potentially including:

- track continuity,
- team membership,
- motion,
- geometry,
- temporal evidence,
- jersey number evidence,
- equipment/stick characteristics,
- appearance,
- roster/on-ice constraints.

No single weak cue should be treated as definitive identity evidence.

### Ambiguity is valid

The analyzer must be allowed to return:

```text
UNKNOWN
```

rather than force an uncertain classification or identity.

A false confident identity is generally worse than temporarily losing identity.

---

## Team classification principles

Team classification is now the active development priority.

Initial useful categories should remain simple:

```text
HOME
AWAY
OFFICIAL
UNKNOWN
```

Names may differ if the existing implementation already establishes equivalent terminology.

Do not introduce unnecessary team-specific assumptions.

Classification should be based on observable visual evidence rather than benchmark track IDs or manually encoded player locations.

Team classification should tolerate:

- changing player pose,
- partial occlusion,
- lighting variation,
- different apparent player sizes,
- temporary contamination from nearby players.

The implementation must not assume that every rink has the same camera angle, perspective, lighting, glass layout, or obstruction pattern.

Avoid pixel-coordinate rules tied to the current benchmark rink.

---

## Rink and camera portability

The current benchmark represents one camera/rink configuration.

Future footage may differ in:

- camera height,
- viewing angle,
- zoom,
- pan behavior,
- resolution,
- lighting and white balance,
- board/glass layout,
- stanchion positions,
- netting,
- score overlays,
- benches and penalty boxes,
- visible ice area.

Do not prematurely build a complete rink-calibration system, but avoid architectural choices that require the current camera geometry.

When practical, prefer normalized or learned measurements over fixed pixel-coordinate assumptions.

Camera/rink calibration may become a separate future layer.

---

## Hockey analysis principles

Team-level analysis should be developed incrementally.

Do not infer complex hockey concepts until the underlying observations are reliable.

Prefer this progression:

```text
player detection
→ team classification
→ anonymous tracklets
→ rink position
→ puck detection
→ simple observable events
→ possession evidence
→ zone transitions
→ team structure / pressure
→ higher-level hockey interpretation
```

An analysis result should remain traceable to source video.

Higher-level observations should eventually retain:

- timestamps,
- relevant track IDs,
- team assignments,
- confidence/evidence,
- source-frame or clip references where practical.

---

## Existing player-identity work

The existing player-specific tracking and reacquisition system is valuable and must not be discarded.

It currently includes work around:

- ByteTrack-authoritative binding,
- confidence-aware motion,
- temporal appearance evidence,
- geometry history,
- ambiguity handling,
- HSV appearance galleries,
- offline OSNet experiments,
- offline DINOv2 experiments.

OSNet and DINOv2 experiments are research results, not production identity components.

Do not integrate rejected embedding models into production identity behavior without new evidence.

The existing player-identity code should remain stable while team analysis is developed unless a task explicitly requires changing it.

---

## Benchmark integrity

The current benchmark video is useful for regression testing and investigation, but it must not become the specification.

Never hardcode:

- Track 3,
- Track 669,
- Track 778,
- Track 840,
- Track 814,
- Track 745,
- jersey #57,
- benchmark timestamps,
- benchmark frame numbers,
- benchmark coordinates,
- expected transition sequences,
- rink-specific locations,

into production algorithms.

These values may be used by offline evaluation or benchmark tooling where explicitly appropriate.

Production behavior must emerge from generic evidence.

---

## Experimental work

Experimental models or algorithms should normally be evaluated offline before production integration.

When testing a hypothesis:

1. define the question,
2. preserve a fixed evaluation protocol where possible,
3. measure positive and negative cases,
4. avoid tuning solely to the benchmark,
5. record failures as well as successes,
6. do not modify production behavior unless the evidence supports integration.

Negative experiments are useful results.

Keep experiment scripts/results when they provide meaningful architectural evidence.

---

## Testing

Run relevant unit tests after code changes.

For broad analyzer changes, run the complete suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

At the time of this pivot, the existing suite contains 84 passing tests.

Do not weaken or remove existing tests merely because a new implementation conflicts with them.

If behavior intentionally changes:

1. explain why the old expectation is no longer valid,
2. update or replace the relevant test deliberately,
3. add coverage for the new behavior.

For experimental scripts that should not alter production behavior, verify that protected production files remain unchanged when practical.

---

## Benchmark execution

The existing player-identity benchmark can still be used for regression testing:

```powershell
.venv\Scripts\python.exe analyze_tracks.py input/57-test.mp4 --target-frame 0 --target-point 834 561 --target-grace-frames 2 --target-max-gap-seconds 3
```

This command is a benchmark/control workflow, not the intended final product interface.

Do not change production algorithms merely to improve this benchmark.

---

## Generated artifacts

Generated benchmark and experiment artifacts belong under:

```text
output/
```

They are not source-of-truth project files.

Old generated artifacts may be deleted when no longer useful.

Retain important milestone reports or experimental results when they document architectural decisions.

Source video under `input/` should be treated as read-only.

---

## Performance

Prefer correctness and interpretable evidence before premature optimization.

However, avoid designs that obviously require expensive inference on every player in every frame when the same model could be invoked only when needed.

Measure performance before optimizing.

GPU-heavy secondary analysis should eventually be event-driven or batched where practical.

---

## Code quality

Prefer:

- clear data structures,
- deterministic behavior,
- explicit state,
- inspectable diagnostics,
- small functions,
- testable components.

Avoid:

- hidden global state,
- benchmark-specific conditionals,
- unexplained thresholds,
- duplicated classification logic,
- broad refactors unrelated to the task.

When adding thresholds, document what they represent and provide evidence for their initial value.

---

## Agent workflow

Before modifying code:

1. inspect the relevant implementation,
2. inspect existing tests,
3. inspect existing logs/results when debugging,
4. identify the smallest appropriate change.

After modifying code:

1. run targeted tests,
2. run the complete suite when appropriate,
3. inspect the actual output,
4. report what changed,
5. report failures or uncertainty,
6. avoid claiming success solely because tests pass.

For benchmark investigations, distinguish between:

- measured evidence,
- visual interpretation,
- expected continuation,
- confirmed ground truth.

Do not promote an assumption into ground truth.

---

## Current priority

The immediate project priority is:

> Build a reliable team-level player classification and anonymous-tracklet foundation without destabilizing the existing player-identity system.

The first milestone is an annotated benchmark video in which detected on-ice participants are consistently classified as:

```text
HOME / AWAY / OFFICIAL / UNKNOWN
```

while retaining their ByteTrack track IDs.

Persistent player identity, puck tracking, possession inference, and higher-level hockey analysis come later.