# Hockey Analyzer Roadmap

## Team-classification baseline

Team analysis is opt-in and does not alter the existing tracking or target-identity defaults. It requires both the existing on-ice polygon and a per-game visual profile:

```powershell
.venv\Scripts\python.exe analyze_tracks.py input\57-test.mp4 `
  --rink configs\57-test.example.rink.json `
  --team-config configs\57-test.example.team.json
```

The run writes an annotated video, observation JSONL, summary JSON, and review contact sheet to `output/team_baseline/`. Labels are `HOME`, `AWAY`, `OFFICIAL`, or `UNKNOWN`; the displayed number remains a temporary ByteTrack track ID, not a player identity. HOME/AWAY colors are configuration, because uniform colors differ between games.

## Vision

Build a hockey-video analysis system that can transform ordinary game footage into auditable team and player analysis.

The system should eventually answer questions at several levels:

```text
What happened?
Which team did it?
Which anonymous player did it?
Which rostered player was that?
Was it a good hockey play?
```

These are separate problems and should not block one another unnecessarily.

---

# Current strategy

Development originally began with persistent identification of one specific player.

That work demonstrated that:

- player detection is practical,
- short-term ByteTrack tracking is useful,
- team distinction is substantially easier than persistent individual identity,
- individual identity becomes difficult after occlusion and track fragmentation,
- generic visual ReID models do not reliably distinguish similarly uniformed hockey players.

The project is therefore pivoting to **team-level analysis first**.

Persistent player identity remains an important downstream capability.

---

# Phase 0 — Video and CV foundation

## Status: COMPLETE

Establish the basic local analysis pipeline.

Completed capabilities include:

- source-video decoding,
- YOLO player detection,
- ByteTrack tracking,
- annotated-video generation,
- structured diagnostic output,
- GPU acceleration,
- repeatable benchmark execution,
- automated tests.

The benchmark demonstrated that temporary track IDs are useful but cannot be treated as permanent player identities.

---

# Phase 1 — Player identity research

## Status: PAUSED / FOUNDATION PRESERVED

Initial goal:

> Reliably follow one known player through track fragmentation.

Implemented and investigated:

- target seeding,
- ByteTrack-authoritative binding,
- reacquisition after disappearance,
- confidence-aware motion,
- temporal appearance evidence,
- geometry history,
- ambiguity handling,
- HSV appearance galleries.

Successful benchmark continuity improved from the original Track 3 to:

```text
3 → 669 → 778
```

without benchmark-specific production hardcoding.

Track 840 became a plausible reacquisition candidate but remained unresolved because competing candidates created legitimate ambiguity.

### ReID experiments

Offline visual embedding experiments evaluated:

- OSNet,
- DINOv2 ViT-S/14.

Neither was suitable for production integration.

Both demonstrated that generic visual embeddings have difficulty distinguishing hockey teammates wearing nearly identical uniforms.

DINOv2 also demonstrated that a simple deterministic inset crop does not solve the problem.

### Decision

Do not continue tuning persistent player identity against the current benchmark at this stage.

Preserve all existing identity functionality and tests.

Return to persistent identity after the team-analysis foundation provides stronger contextual constraints.

---

# Phase 2 — Team classification

## Status: ACTIVE

## Goal

Classify detected on-ice participants into useful team/role categories while preserving anonymous track IDs.

Initial classes:

```text
HOME
AWAY
OFFICIAL
UNKNOWN
```

The implementation should not require knowledge of individual player identity.

## Milestone 2.1 — Team-classification baseline

Build an offline analysis of the existing benchmark.

For every relevant tracked participant:

- retain ByteTrack track ID,
- classify team/role,
- record confidence/evidence,
- allow UNKNOWN,
- generate diagnostics.

Produce an annotated video showing approximately:

```text
HOME  #123
AWAY  #456
OFFICIAL #88
UNKNOWN #901
```

Track IDs remain temporary identifiers.

### Acceptance criteria

The system must:

- classify teams using generic visual evidence,
- avoid benchmark-ID hardcoding,
- preserve existing player-identity behavior,
- preserve all existing tests,
- expose uncertainty rather than force classification.

---

## Milestone 2.2 — Temporal team stability

Prevent team labels from unnecessarily oscillating frame to frame.

Investigate:

- track-level evidence accumulation,
- confidence hysteresis,
- uniform-color history,
- recovery after partial occlusion.

A track's historical team evidence may be stronger than one contaminated frame.

Team identity should remain independent from permanent player identity.

---

## Milestone 2.3 — Team-classification validation

Evaluate team classification across:

1. the existing benchmark,
2. another game from the same rink,
3. footage with different conditions,
4. eventually another rink/camera configuration.

Measure:

- classification coverage,
- HOME/AWAY errors,
- OFFICIAL errors,
- UNKNOWN rate,
- label stability,
- recovery after occlusion.

Do not optimize solely for the original benchmark.

---

# Phase 3 — Anonymous player tracking and rink position

## Goal

Turn team-classified tracklets into useful hockey observations without requiring roster identity.

For each active tracklet, maintain information such as:

```text
track_id
team
bounding_box
rink_position
velocity
confidence
first_seen
last_seen
```

## Milestone 3.1 — On-ice filtering

Distinguish relevant on-ice participants from:

- bench players,
- spectators,
- coaches,
- reflections or false detections,
- other irrelevant detections.

---

## Milestone 3.2 — Rink-relative position

Move toward normalized rink position rather than relying solely on raw image pixels.

Initially use the simplest robust approach.

Future versions may detect rink landmarks such as:

- boards,
- center/red line,
- blue lines,
- goal lines,
- faceoff circles.

Full rink calibration is not required until evidence shows it is necessary.

---

## Milestone 3.3 — Anonymous tracklet analysis

Extract useful continuous-track observations such as:

- path traveled,
- speed/movement estimates,
- zone occupancy,
- spacing relative to teammates,
- proximity to opponents,
- transition through rink regions.

These statistics remain useful even when the roster identity is unknown.

---

# Phase 4 — Puck detection and tracking

## Goal

Reliably detect and track the puck sufficiently for hockey-event analysis.

Expected challenges include:

- very small object size,
- motion blur,
- boards,
- sticks/skates,
- occlusion,
- compression artifacts.

Develop and validate puck tracking independently before using it as authoritative possession evidence.

Allow puck state to become UNKNOWN when visibility is insufficient.

---

# Phase 5 — Observable hockey events

## Goal

Infer simple, auditable events from player/team/puck observations.

Begin with events closest to direct visual evidence.

Candidate events include:

- zone entry,
- zone exit,
- shot attempt,
- puck recovery,
- turnover,
- dump-in,
- clear,
- contested puck,
- rush,
- line change.

Every detected event should retain enough information to trace it back to the source video.

Avoid subjective coaching judgments at this stage.

---

# Phase 6 — Team-level hockey analysis

## Goal

Aggregate observable events and positioning into useful team analysis.

Potential outputs include:

- offensive-zone time,
- defensive-zone time,
- territorial pressure,
- shot attempts,
- entry/exit success,
- possession evidence,
- forecheck behavior,
- player spacing,
- team shape,
- rush patterns,
- line-change patterns,
- heatmaps.

Metrics should distinguish measured facts from inferred hockey concepts.

---

# Phase 7 — Persistent player identity

## Goal

Associate anonymous tracklets with rostered players.

Return to the player-identity problem with substantially more context than was available during Phase 1.

Potential evidence includes:

- team classification,
- jersey-number evidence,
- nameplate evidence,
- equipment/stick characteristics,
- track continuity,
- motion,
- geometry,
- shift timing,
- roster constraints,
- known players currently on ice,
- line combinations,
- temporal identity history.

A jersey number does not need to be readable on every frame.

A high-confidence observation may establish identity temporarily, after which continuity can propagate it until evidence becomes insufficient.

Identity should remain probabilistic/uncertain when evidence conflicts.

---

# Phase 8 — Player-level analysis

## Goal

Attach hockey observations and events to persistent player identities.

Potential outputs include:

- shifts,
- ice time,
- zone time,
- entries/exits,
- shot attempts,
- puck recoveries,
- turnovers,
- positioning,
- pressure involvement,
- clips of notable plays.

This phase reconnects the team-analysis pipeline to the project's original player-analysis objective.

---

# Phase 9 — Hockey interpretation and coaching analysis

## Goal

Build higher-level analysis on top of validated observations.

Potential questions include:

- Was the player in an appropriate position?
- Did the player support the puck?
- Was there an available passing option?
- Did the player maintain defensive-side positioning?
- Was an entry controlled or uncontrolled?
- Did the team maintain useful spacing?

Higher-level interpretations must remain traceable to observable evidence.

Avoid presenting subjective analysis as objective measurement.

---

# Phase 10 — Multi-rink robustness

## Goal

Generalize beyond the original camera and rink.

Test variation in:

- camera height,
- viewing angle,
- zoom,
- camera movement,
- lighting,
- white balance,
- rink geometry,
- glass/stanchion placement,
- netting,
- overlays,
- video resolution and compression.

Add explicit camera/rink calibration only where testing demonstrates it is necessary.

Avoid designing current algorithms around fixed benchmark coordinates.

---

# Phase 11 — Productization

Potential product capabilities include:

- automatic full-game processing,
- team reports,
- player reports,
- searchable event timelines,
- automatic clip generation,
- game comparisons,
- season trends,
- coaching review workflows.

Product decisions should follow demonstrated analysis reliability rather than precede it.

---

# Immediate next milestone

## Team classification baseline

The next development task is:

> Classify tracked on-ice participants in the existing benchmark as HOME, AWAY, OFFICIAL, or UNKNOWN and render those classifications with temporary track IDs in an annotated video.

Do not add puck tracking, event inference, roster identity, or complex rink calibration during this milestone.

The purpose is to establish a reliable team-analysis foundation while preserving the existing player-identity work.
