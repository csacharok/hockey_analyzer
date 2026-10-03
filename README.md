# Hockey Analyzer

Hockey Analyzer is a local computer-vision and machine-learning project for extracting auditable hockey observations and, eventually, team- and player-level analysis from game video.

The current priority is reliable anonymous participant and team analysis. Persistent identification of a rostered player remains a downstream capability, not a prerequisite for useful results.

## Current architecture

The project separates visual observation, semantic interpretation, temporal reasoning, hockey-event understanding, and optional player identity:

```text
video
  -> YOLO person detection
  -> ByteTrack temporary tracking
  -> provider masking / rink geometry / participation evidence
  -> semantic-free appearance observations
  -> human-ground-truth tracklet labels
  -> learned participant classifier [NEXT MILESTONE]
  -> temporal semantic aggregation
  -> anonymous hockey analysis
  -> optional persistent player identity
```

ByteTrack IDs are temporary tracklet identifiers. They must not be treated as permanent player identities or jersey numbers.

### Observation versus interpretation

Classical computer vision is the observation and sensing layer. It is responsible for evidence such as:

- person detections and temporary tracklets,
- configured provider-overlay exclusion,
- camera-relative rink boundaries,
- on-ice participation evidence,
- normalized body-region measurements,
- torso, pants, and lower-leg appearance,
- stripe structure,
- crop and measurement quality,
- source-frame provenance.

Learned models are the forward semantic interpretation architecture. The next model milestone will use human ground truth to classify participant semantics from some combination of person crops, structured appearance evidence, participation/rink evidence, and tracklet context.

Measurement quality, training suitability, and semantic ground truth are distinct concepts. A measurable crop can still contain an occluded player, multiple people, a spectator, a coach, or another unsuitable subject.

### Rink and participation evidence

Normal operation uses camera-relative evidence:

- visible far-side board/kickplate evidence,
- near-board boundary evidence,
- lower-bbox/contact-point geometry,
- local surface evidence,
- explicit accepted, rejected, or uncertain participation decisions.

Static rink polygons are retained only as legacy/debug behavior. They are not the normal portability solution and should not drive semantic team or official classification.

Configured provider masks use normalized coordinates and can be applied to the analysis path before detection and rink sensing. Original unmasked frames remain available for annotation, crop generation, and human review.

## Current milestone: learned participant classification

The human-ground-truth labeling milestone is complete. The next milestone is to benchmark a narrow learned participant classifier against that ground truth.

Initial work should:

1. preserve tracklet-aware grouping so observations from one temporary tracklet never cross train/validation/test boundaries,
2. begin with simple, well-understood pretrained vision classifiers or backbones,
3. evaluate image inputs and structured appearance evidence empirically,
4. report dataset composition, split methodology, per-class precision/recall/F1, confusion matrices, cross-game behavior, and visual failure modes,
5. preserve uncertainty rather than force unsupported classifications.

Randomly splitting adjacent frames is leakage and does not demonstrate portability. As more data becomes available, evaluation should increasingly test different games, rinks, cameras, lighting, providers, and uniforms.

Classifier implementation and training have not begun as part of this documentation update.

## Dataset v1

Dataset v1 contains 300 human-labeled temporary tracklets from the two current benchmark games.

| Human label | Tracklets |
|---|---:|
| HOME | 51 |
| AWAY | 52 |
| OFFICIAL | 13 |
| NON_PARTICIPANT | 166 |
| MIXED_TRACK | 5 |
| UNSURE | 13 |
| **Total** | **300** |

The label audit was completed with no changes.

Human labels are authoritative ground truth and remain separate from appearance measurements, participation decisions, legacy pseudo-labels, and future model predictions.

`MIXED_TRACK` and `UNSURE` are human-review outcomes; they are not automatically semantic training classes. Their treatment must be chosen explicitly when defining the first model dataset and evaluation protocol.

Goalies are currently labeled by team as `HOME` or `AWAY`. Goalie status is orthogonal to team semantics. A future `PLAYER_ROLE` annotation such as `SKATER` or `GOALIE` should be introduced only if model error analysis demonstrates that it is useful.

### Dataset preservation

The working dataset remains under ignored `output/` storage and is backed up separately. These directories are frozen project assets and must not be casually regenerated, overwritten, or modified:

```text
output/labeling_dataset_v1/
output/appearance_features_v1/
```

Dataset v1 depends on both layers:

- `output/labeling_dataset_v1/` contains representative crops, deterministic candidate metadata, the manifest, labeling guide, report, and human labels.
- `output/appearance_features_v1/` contains the semantic-free source observations, tracklet summaries, and feature schema used to construct the labeling project and preserve provenance.

[`docs/datasets/dataset_v1_checksums.json`](docs/datasets/dataset_v1_checksums.json) records SHA-256 fingerprints and byte sizes for the frozen labels, candidates, manifest, appearance schema, observation exports, tracklet summaries, and source-video assets. Use it to verify restored or transferred copies; do not use regenerated model output to overwrite human labels.

Source videos under `input/` are read-only provenance assets.

## Preserved legacy and experimental work

### V7 handcrafted semantic baseline

The existing handcrafted `HOME` / `AWAY` / `OFFICIAL` / `UNKNOWN` classifier remains preserved and tested as the V7 legacy baseline.

Its outputs are useful for regression, diagnostics, and pseudo-label metadata. They are not human ground truth and are not the forward semantic architecture. Semantic failures should not be addressed by adding increasingly complex handcrafted rule trees unless an explicitly scoped classical-baseline experiment calls for it.

### Persistent target identity

The target-specific tracking and reacquisition system remains preserved experimental/downstream work. It includes ByteTrack-authoritative binding, motion and geometry history, temporal appearance evidence, ambiguity handling, and HSV appearance galleries.

Offline OSNet and DINOv2 experiments did not justify production integration for distinguishing similarly uniformed players. Persistent identity should eventually combine multiple independent cues and must be able to return `UNKNOWN`.

Anonymous hockey analysis should remain useful when persistent player identity is unavailable.

## Development direction

After learned participant classification, the intended progression is:

```text
learned participant classification
  -> temporal semantic aggregation
  -> anonymous rink position and team occupancy
  -> puck detection
  -> simple observable events
  -> possession evidence and zone transitions
  -> team structure and pressure
  -> optional persistent roster identity
  -> player-level and higher-level hockey analysis
```

Higher-level conclusions must remain traceable to source timestamps, tracklets, semantic assignments, confidence/evidence, and reviewable frames or clips.

## Testing

Run the complete test suite from the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover current observation infrastructure, dataset and labeling integrity, the preserved V7 baseline, and persistent target-identity behavior. Do not remove legacy tests while the corresponding functionality remains intentionally preserved.

## Project guidance

See [`AGENTS.md`](AGENTS.md) for the authoritative architectural and development rules and [`ROADMAP.md`](ROADMAP.md) for milestone sequencing.

Core principles:

- classical CV observes; learned models interpret,
- human labels are ground truth; pseudo-labels are metadata,
- temporary track IDs are not player identities,
- uncertainty is preferable to a confident false classification,
- benchmark videos are evidence, not specifications,
- source video and frozen Dataset v1 assets retain provenance,
- model evaluation must prevent tracklet and adjacent-frame leakage,
- anonymous team analysis should not wait for persistent roster identity.
