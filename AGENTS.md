# Hockey Analyzer — Agent Instructions

## Project purpose

Hockey Analyzer is a local computer-vision and machine-learning system for analyzing hockey game video.

The current primary objective is **reliable anonymous team-level hockey analysis**.

The project deliberately separates:

1. visual observation,
2. semantic interpretation,
3. temporal/tracklet reasoning,
4. hockey-event understanding,
5. optional persistent player identity.

Reliable persistent identification of a specific rostered player remains a project goal, but it is **not a prerequisite for useful team analysis**.

Do not treat a ByteTrack track ID as a permanent player identity.

---

## Development approach

- Prefer the simplest implementation that satisfies the requirement.
- Make small, targeted changes rather than broad refactors unless necessary.
- Run relevant tests after code changes.
- Preserve existing behavior unless the task explicitly changes it.
- When debugging, inspect existing logs, tests, code, and generated artifacts before modifying architecture.
- Do not hardcode benchmark-specific IDs, timestamps, coordinates, colors, jersey numbers, or expected results.
- For difficult bugs, identify the root cause before implementing a workaround.
- Measure behavior before tuning it.
- Preserve negative experimental results when they provide architectural evidence.
- Do not claim success solely because tests pass or aggregate metrics look stable; inspect actual outputs when the task is visual.

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
rink geometry / provider masking / on-ice participation evidence
  ↓
semantic-free appearance observations
  ↓
human ground truth for supervised development
  ↓
learned participant classification
  ↓
temporal / tracklet aggregation
  ↓
anonymous player and team analysis
  ↓
rink position / puck / observable hockey events
  ↓
possession evidence / zone transitions / team structure
  ↓
optional persistent player identity resolution
  ↓
clips, statistics, and higher-level hockey analysis
```

### Architectural boundary

The classical-computer-vision layer is responsible for **observable evidence**.

Examples include:

- person detections,
- temporary tracklets,
- provider-overlay exclusion,
- far- and near-rink boundary evidence,
- on-ice / participant evidence,
- normalized body-region measurements,
- torso appearance,
- pants appearance,
- lower-leg / sock appearance,
- stripe structure,
- crop / measurement quality,
- source-frame provenance.

The learned-model layer is responsible for **semantic interpretation** such as:

- HOME,
- AWAY,
- OFFICIAL,
- NON_PARTICIPANT,
- later hockey semantics.

Do not respond to semantic-classification failures by adding new handcrafted HOME/AWAY/OFFICIAL rule trees or thresholds unless a task explicitly calls for a classical-baseline experiment.

The existing V7 semantic classifier is retained for regression/reference only. Its outputs are **pseudo-label metadata, not ground truth**.

---

## Observation vs interpretation

Keep these concepts distinct.

### Detection

Where an object appears in a frame.

### Tracking

Which nearby detections belong to the same temporary visual tracklet.

### Rink / participation evidence

Whether geometry and local visual evidence suggest that a detection is on the playing surface or otherwise relevant.

### Appearance observation

What can be measured from the crop without assigning hockey meaning.

Examples:

```text
torso_lightness
torso_dark_fraction
pants_dark_fraction
lower_leg_lightness
lower_leg_dark_fraction
vertical_stripe_score
measurement_quality
```

### Semantic classification

What the subject actually represents:

```text
HOME
AWAY
OFFICIAL
NON_PARTICIPANT
UNKNOWN
```

This is now primarily a learned-model problem trained/evaluated against human ground truth.

### Hockey-event understanding

What happened in the game.

### Player identity

Which rostered player owns one or more tracklets.

Do not collapse these layers merely because a convenient heuristic appears to work on one benchmark.

---

## Measurement quality is not training suitability

The appearance pipeline may describe an observation as measurable even when it is a poor supervised-learning example.

A technically measurable crop may contain:

- a clean skater,
- an official,
- a spectator,
- a coach,
- a bench person,
- a foreground person,
- an occluded player,
- a contaminated bbox,
- multiple people.

Keep separate concepts for:

1. **measurement quality** — can useful features be extracted?
2. **training suitability** — should this sample be used for supervised learning?
3. **human semantic ground truth** — what is the subject?

Do not infer #2 or #3 merely from a measurement-quality label.

---

## Human ground truth

Human labels are the authoritative semantic labels for supervised model development.

Initial tracklet-level labeling categories are expected to include:

```text
HOME
AWAY
OFFICIAL
NON_PARTICIPANT
MIXED_TRACK
UNSURE
```

Definitions:

- `HOME`: tracklet consistently represents a home-team player.
- `AWAY`: tracklet consistently represents an away-team player.
- `OFFICIAL`: tracklet consistently represents an on-ice official.
- `NON_PARTICIPANT`: spectator, coach, bench person, scorekeeper, foreground person, or another person who should not be treated as an on-ice game participant.
- `MIXED_TRACK`: temporary tracklet changes identity or contains incompatible subjects.
- `UNSURE`: available visual evidence is insufficient for reliable human classification.

Human labels must be stored separately from:

- V7 pseudo-labels,
- appearance descriptors,
- participant-filter decisions,
- model predictions.

Never silently promote pseudo-labels into ground truth.

Do not expose pseudo-labels in a labeling interface in a way that biases the reviewer.

---

## Dataset and leakage principles

Supervised datasets must preserve provenance.

Each labeled subject/sample should retain enough information to recover:

- source game,
- frame/time,
- temporary track ID,
- bbox/crop reference,
- structured appearance evidence,
- measurement quality,
- participant/rink evidence,
- human label when available.

Observations from the same temporary tracklet must never be randomly split across train and validation/test sets.

When enough games are available, prefer evaluation that measures portability across:

- games,
- rinks,
- cameras,
- lighting,
- uniforms,
- providers.

Adjacent frames are highly correlated. Do not report frame-randomized validation accuracy as if it demonstrated real generalization.

Dataset sampling should seek useful diversity rather than simply maximize sample count.

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

Tracklets may also contain identity switches. This is why `MIXED_TRACK` is a valid human label.

### ByteTrack authority for legacy target tracking

When a known target is currently bound to an active ByteTrack track in the existing player-specific identity system, ByteTrack remains authoritative.

Do not attempt unnecessary reacquisition while the bound track remains available.

### Persistent identity

Persistent player identity should eventually combine multiple independent sources of evidence, potentially including:

- track continuity,
- learned team/role classification,
- motion,
- geometry,
- temporal evidence,
- jersey number evidence,
- equipment/stick characteristics,
- appearance,
- roster/on-ice constraints.

No single weak cue should be treated as definitive identity evidence.

### Ambiguity is valid

The analyzer must be allowed to return `UNKNOWN` rather than force an uncertain classification or identity.

A false confident identity is generally worse than temporarily losing identity.

---

## Classical appearance features

The current appearance extractor is intended to provide **semantic-free sensors**.

Preserve continuous/interpretable measurements where practical rather than collapsing them prematurely into categorical rules.

Current useful feature families include:

- torso lightness/darkness/color statistics,
- pants / waist darkness,
- lower-leg / sock lightness/darkness/color statistics,
- vertical stripe structure,
- normalized bbox / crop quality,
- region usability,
- detection confidence,
- participant/rink evidence.

Derived descriptors such as light/dark appearance buckets may be useful diagnostics, but they are not synonyms for HOME/AWAY/OFFICIAL.

Do not create mappings such as:

```text
LIGHT_TOP_LIGHT_LEGS = HOME
DARK_TOP = AWAY
LIGHT_TOP_DARK_LEGS = OFFICIAL
```

unless explicitly requested as an offline classical-baseline experiment.

Glove evidence has not been shown reliable enough at the current video resolution to justify a dedicated production rule.

---

## Rink and participant filtering

Camera-relative rink geometry and participant filtering are useful classical-CV components and should be preserved.

Current direction:

- far side: use visible board/kickplate evidence,
- near side: use camera-relative board-boundary evidence,
- use contact-point / lower-bbox geometry,
- allow uncertainty when boundary evidence is missing,
- avoid dependence on stale manually configured rink polygons.

The static polygon is legacy/debug behavior and should not become the normal portability solution again.

Do not redesign rink geometry merely to improve semantic team/official classification.

A boundary estimate and a participant decision are separate concepts. If a tolerance is required, preserve the physical boundary estimate and apply tolerance in the decision layer.

---

## Provider overlay handling

Known provider overlays may be masked using normalized provider/game configuration.

The analysis frame may be masked before:

- person detection,
- rink-boundary detection,
- local surface evidence,
- appearance extraction where appropriate.

The original unmasked frame should remain available for:

- final annotation,
- human review,
- crop generation where masking would hide useful subject pixels.

Post-detection provider exclusion may be used to prevent mask-edge/logo detections from reaching tracking.

Do not hardcode provider pixel coordinates into production algorithms when normalized configuration can express them.

Do not implement automatic watermark/provider detection without evidence that it is needed.

---

## Rink and camera portability

Footage may differ in:

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
- visible ice area,
- uniforms and officials.

Avoid architectural choices that require the current benchmark geometry or colors.

When practical, prefer normalized, measured, configured, or learned representations over fixed pixel-coordinate assumptions.

Camera/rink calibration may become a separate future layer if multi-video evidence demonstrates the need.

---

## Learned-model development principles

Learned models should be introduced incrementally and benchmarked against human-labeled data.

Prefer a narrow, measurable first problem over a large end-to-end model.

The first learned semantic task is participant classification from some combination of:

- person crop,
- structured appearance features,
- rink/participant evidence,
- temporal/tracklet context.

Candidate model families should be compared empirically rather than selected because they are fashionable or large.

Start with simple pretrained vision classifiers / backbones before escalating model complexity.

Do not assume a large language model, mixture-of-experts model, or general multimodal model is necessary for a narrow visual-classification task.

Report:

- dataset composition,
- split methodology,
- class counts,
- human-label policy,
- per-class metrics,
- confusion matrix,
- cross-game performance,
- important failure modes.

Accuracy on leaked adjacent frames is not meaningful evidence of portability.

---

## Hockey analysis principles

Team-level hockey analysis should be developed incrementally.

Do not infer complex hockey concepts until the underlying observations and learned participant semantics are sufficiently reliable.

Preferred progression:

```text
person detection
→ temporary tracklets
→ rink / participation evidence
→ appearance observations
→ learned participant classification
→ temporal semantic aggregation
→ anonymous rink position
→ puck detection
→ simple observable events
→ possession evidence
→ zone transitions
→ team structure / pressure
→ higher-level hockey interpretation
```

Analysis results should remain traceable to source video.

Higher-level observations should eventually retain:

- timestamps,
- relevant temporary track IDs,
- semantic assignments,
- confidence/evidence,
- source-frame or clip references where practical.

---

## Existing player-identity work

The existing player-specific tracking and reacquisition system is valuable and must not be discarded.

It includes work around:

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

The existing player-identity code should remain stable while learned participant classification and team analysis are developed unless a task explicitly requires changing it.

Persistent roster identity remains downstream and optional.

---

## Benchmark integrity

Benchmark videos are useful for regression testing and investigation, but they must not become the specification.

Never hardcode production behavior using:

- benchmark Track IDs,
- jersey numbers,
- benchmark timestamps,
- benchmark frame numbers,
- benchmark coordinates,
- expected transition sequences,
- rink-specific locations,
- Game-1 team colors,
- Game-2 team colors,
- expected human labels.

Reviewed IDs/frames may be used by offline diagnostic/evaluation tooling where explicitly appropriate.

Production behavior must emerge from generic evidence.

Multiple benchmark games should be retained when they expose meaningfully different conditions.

Do not tune sequentially until all currently reviewed benchmark videos happen to look correct.

---

## Experimental work

Experimental models or algorithms should normally be evaluated offline before production integration.

When testing a hypothesis:

1. define the question,
2. preserve a fixed evaluation protocol where possible,
3. measure positive and negative cases,
4. use human ground truth when evaluating semantic accuracy,
5. avoid tuning solely to the benchmark,
6. record failures as well as successes,
7. do not modify production behavior unless evidence supports integration.

Negative experiments are useful results.

Keep experiment scripts/results when they provide meaningful architectural evidence.

Aggregate behavior metrics are not semantic accuracy unless compared against ground truth.

---

## Testing

Run relevant unit tests after code changes.

For broad analyzer changes, run the complete suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Do not hardcode the expected total number of passing tests in this file. Report the actual result from the current repository.

Do not weaken or remove existing tests merely because a new implementation conflicts with them.

If behavior intentionally changes:

1. explain why the old expectation is no longer valid,
2. update or replace the relevant test deliberately,
3. add coverage for the new behavior.

For experimental scripts that should not alter production behavior, verify that protected production files remain unchanged when practical.

Dataset/labeling tests should additionally protect:

- deterministic sampling,
- provenance,
- label persistence,
- pseudo-label/ground-truth separation,
- leakage-resistant grouping,
- non-mutation of source exports.

---

## Legacy benchmark execution

The existing player-identity benchmark can still be used for regression testing:

```powershell
.venv\Scripts\python.exe analyze_tracks.py input/57-test.mp4 --target-frame 0 --target-point 834 561 --target-grace-frames 2 --target-max-gap-seconds 3
```

This is a legacy identity benchmark/control workflow, not the intended final product interface and not the primary acceptance test for learned participant classification.

Do not change production algorithms merely to improve this benchmark.

---

## Generated artifacts

Generated benchmark, dataset, and experiment artifacts belong under:

```text
output/
```

They are not source-of-truth project files unless a task explicitly promotes a human-labeled dataset manifest/label file as a retained project artifact.

Old generated artifacts may be deleted when no longer useful.

Retain important milestone reports, human-label definitions, dataset manifests, or experimental results when they document architectural decisions.

Source video under `input/` should be treated as read-only.

Human labels must never be silently overwritten by regenerated model outputs.

---

## Performance

Prefer correctness, trustworthy ground truth, and interpretable evidence before premature optimization.

Measure performance before optimizing.

Avoid designs that obviously require expensive inference on every player in every frame when inference can be:

- batched,
- sampled,
- cached,
- aggregated at tracklet level,
- invoked only when needed.

GPU-heavy secondary analysis should eventually be event-driven or batched where practical.

Do not optimize the appearance extractor or labeling pipeline merely because production real-time performance is a future goal; profile the actual bottleneck first.

---

## Code quality

Prefer:

- clear data structures,
- deterministic behavior,
- explicit state,
- inspectable diagnostics,
- small functions,
- testable components,
- versioned schemas for retained datasets,
- reproducible sampling.

Avoid:

- hidden global state,
- benchmark-specific conditionals,
- unexplained thresholds,
- duplicated classification logic,
- semantic leakage from pseudo-labels into ground truth,
- broad refactors unrelated to the task.

When adding thresholds, document what they represent and provide evidence for their initial value.

---

## Agent workflow

Before modifying code:

1. inspect the relevant implementation,
2. inspect existing tests,
3. inspect existing logs/results/artifacts when debugging,
4. identify whether the task belongs to observation, semantics, temporal reasoning, or hockey analysis,
5. identify the smallest appropriate change.

After modifying code:

1. run targeted tests,
2. run the complete suite when appropriate,
3. inspect actual output,
4. report what changed,
5. report failures or uncertainty,
6. avoid claiming success solely because tests pass.

For investigations, distinguish between:

- measured evidence,
- model prediction,
- pseudo-label,
- visual interpretation,
- human ground truth,
- expected continuation.

Do not promote an assumption or pseudo-label into ground truth.

---

## Current priority

The immediate project priority is:

> Build a trustworthy human-ground-truth dataset and labeling workflow for learned participant classification, using the existing detection, tracking, rink/participant, provider-mask, and semantic-free appearance infrastructure.

The current milestone is **not** to improve handcrafted HOME/AWAY/OFFICIAL rules.

The next learned-model milestone will classify participant semantics from human-labeled data and will be evaluated with leakage-resistant, tracklet-aware and eventually cross-game/cross-rink validation.

Persistent player identity, puck tracking, possession inference, and higher-level hockey analysis remain downstream.
