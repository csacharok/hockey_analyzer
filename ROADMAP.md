# Hockey Analyzer roadmap

## Project direction

Hockey Analyzer is transitioning from a predominantly handcrafted semantic-classification approach to a hybrid computer-vision / machine-learning architecture.

The core architectural principle is:

> **Classical CV observes. Learned models interpret.**

The classical-computer-vision layer should provide reliable, inspectable evidence such as detections, temporary tracklets, rink geometry, on-ice participation evidence, provider masking, crop quality, and normalized appearance measurements.

It does not need to determine `HOME`, `AWAY`, or `OFFICIAL` through increasingly complex handcrafted rule trees.

Semantic participant classification will be learned from human-labeled ground truth.

Persistent roster/player identity remains optional and downstream.

---

## Completed foundation

The project has established useful infrastructure in several areas.

### Detection and temporary tracking

- person detection,
- ByteTrack short-term tracklets,
- explicit recognition that track IDs are temporary rather than player identities.

### Player-specific identity research

The original #57 work established useful tracking/reacquisition infrastructure and produced important negative research results.

Completed work includes:

- ByteTrack-authoritative target binding,
- motion and geometry history,
- temporal appearance evidence,
- ambiguity handling,
- HSV appearance galleries,
- offline OSNet evaluation,
- offline DINOv2 evaluation.

The embedding experiments did not provide reliable enough same-uniform player identity to justify production integration.

Persistent player identity is therefore deferred rather than serving as the foundation of team analysis.

### Rink and participant filtering

The project moved away from a stale manually configured rink polygon toward camera-relative rink evidence.

Useful components include:

- far-side board / kickplate evidence,
- near-board boundary evidence,
- contact-point reasoning,
- local surface evidence,
- accepted / rejected / uncertain participant decisions.

Across the reviewed benchmark games, rink geometry and on-ice-vs-off-ice filtering are useful enough to retain as classical-CV evidence, although they are not perfect.

The static rink polygon remains legacy/debug behavior.

### Provider overlay handling

Configured normalized provider masks are applied to the analysis path, with post-detection handling for mask-edge leakage.

The original unmasked frame remains available for annotation and human review.

### Classical semantic baseline

A handcrafted semantic classifier was developed for:

```text
HOME
AWAY
OFFICIAL
UNKNOWN
```

This work was valuable as an experiment and exposed useful visual cues, including:

- torso lightness,
- lower-body darkness,
- vertical stripe structure,
- temporal persistence,
- crop quality.

However, cross-game visual review demonstrated that aggregate stability and internally consistent labels did not equal semantic accuracy.

The classical semantic classifier remains available as a **legacy baseline / pseudo-label source**, not ground truth and not the primary future architecture.

### Semantic-free appearance extraction

The current appearance layer separates observation from interpretation.

For normalized person regions it can expose evidence such as:

- torso lightness/darkness/color statistics,
- pants / waist darkness,
- lower-leg / sock lightness/darkness/color statistics,
- vertical stripe structure,
- region usability,
- detection confidence,
- crop / measurement quality,
- participant/rink evidence.

Two-game contact-sheet review showed that these measurements contain useful visual signal.

It also showed an important limitation:

> A technically measurable crop is not necessarily a good supervised-learning example.

Foreground people, spectators, bench personnel, occluded bodies, contaminated boxes, and multiple-person crops may still produce valid pixel measurements.

Therefore the project now explicitly separates:

1. measurement quality,
2. training suitability,
3. semantic ground truth.

---

## Current milestone: human-ground-truth dataset

The immediate milestone is to build a reproducible candidate-selection and human-labeling workflow.

### Goals

1. Consume existing appearance observation / tracklet exports.
2. Select a diverse, manageable subset of temporary tracklets for review.
3. Choose several temporally diverse representative crops per selected tracklet.
4. Present those crops without exposing semantic pseudo-labels that could bias the reviewer.
5. Record human labels separately and safely.
6. Preserve full provenance.
7. Prepare for leakage-resistant model evaluation.

### Initial human labels

The first labeling protocol should support:

```text
HOME
AWAY
OFFICIAL
NON_PARTICIPANT
MIXED_TRACK
UNSURE
```

`MIXED_TRACK` is important because ByteTrack may occasionally switch subjects within one temporary tracklet.

`NON_PARTICIPANT` is a first-class class, not merely discarded noise. Spectators, coaches, bench personnel, scorekeepers, and foreground people provide valuable hard-negative examples.

### Initial dataset size

The first labeling project should be intentionally modest.

Current target:

- approximately 300 selected tracklets,
- approximately 6 representative crops per tracklet,
- sampled across both existing benchmark games,
- deterministic/reproducible candidate selection.

This is a workflow-validation and first-model dataset, not the final production corpus.

### Label integrity

Human labels must remain distinct from:

- V7 semantic outputs,
- appearance descriptors,
- participant-filter decisions,
- future model predictions.

Existing HOME/AWAY/OFFICIAL outputs are pseudo-label metadata only.

Do not automatically initialize human labels from pseudo-labels.

### Leakage prevention

Future train/validation/test splits must be tracklet-aware.

Observations from one temporary tracklet must not be randomly split between training and validation/test sets.

As the dataset grows, evaluation should increasingly test generalization across:

- games,
- rinks,
- cameras,
- lighting,
- providers,
- uniforms.

Do not rely on random adjacent-frame validation as evidence of real portability.

---

## Next milestone: learned participant classification

After the labeling workflow has been inspected and a useful first ground-truth corpus exists, benchmark learned participant classifiers.

### Initial problem

Given some combination of:

- person crop,
- structured appearance features,
- participant/rink evidence,
- temporary tracklet context,

predict participant semantics.

Initial semantic classes should follow the human-label policy where appropriate.

### Model strategy

Start with simple, well-understood pretrained vision classifiers/backbones.

Do not begin with a giant end-to-end multimodal system merely because it is available.

Benchmark a small number of reasonable alternatives and compare them using human ground truth.

Structured appearance evidence may be:

- concatenated with learned image features,
- used as auxiliary inputs,
- evaluated separately as a baseline.

Let measured validation results determine whether the structured features materially help.

### Required evaluation

Report at minimum:

- dataset composition,
- train/validation/test grouping method,
- per-class precision/recall/F1,
- confusion matrix,
- overall accuracy where meaningful,
- cross-game performance,
- important visual failure modes.

Do not call pseudo-label agreement accuracy.

---

## Following milestone: temporal semantic aggregation

Frame/crop classification should not automatically become the final tracklet decision.

Once the learned observation classifier is useful, add a small temporal layer that can aggregate evidence across a temporary tracklet.

Goals:

- stabilize predictions across blur/occlusion,
- preserve uncertainty,
- avoid one bad frame flipping an established interpretation,
- identify likely mixed/contaminated tracklets,
- expose confidence/evidence.

Do not perform cross-track player identity matching merely to stabilize semantic labels.

---

## Portability expansion

After the learned participant classifier works on the initial two-game dataset, add more games deliberately.

Prefer footage that changes meaningful conditions:

1. different rink / camera,
2. different uniforms,
3. different lighting,
4. different provider or overlay layout,
5. different player size/age where relevant.

Do not select only easy examples.

Use new games primarily as generalization tests before using them for additional training.

When failures appear, determine whether they belong to:

- detection,
- tracking,
- rink geometry,
- participation evidence,
- appearance observation,
- learned semantic classification,
- temporal aggregation.

Fix the correct layer.

---

## Anonymous team-level hockey analysis

Once participant semantics are reliable enough, begin extracting hockey information without requiring persistent player identity.

Planned progression:

```text
learned participant classification
→ temporal semantic aggregation
→ anonymous rink position / team occupancy
→ puck detection
→ simple observable events
→ possession evidence
→ zone transitions
→ team structure / pressure
→ higher-level hockey interpretation
```

Useful analysis should remain possible even when:

```text
player_identity = UNKNOWN
```

Examples of future anonymous analysis may include:

- on-ice player counts,
- team occupancy,
- zone presence,
- entries/exits,
- pressure/territorial evidence,
- shift-like intervals,
- puck proximity / possession evidence,
- timestamped clips for human review.

Do not infer complex coaching conclusions until the underlying observations are reliable.

---

## Puck and event analysis

Puck detection and hockey-event understanding remain downstream.

Develop incrementally:

1. puck candidate detection,
2. observable puck movement,
3. simple events,
4. possession evidence,
5. zone transitions,
6. team-level structure.

Keep higher-level conclusions traceable to source frames/clips.

---

## Persistent player identity

Persistent roster identity remains a future capability rather than a prerequisite.

Potential evidence may eventually include:

- learned team membership,
- track continuity,
- jersey-number evidence,
- motion,
- geometry,
- equipment/stick characteristics,
- appearance,
- roster constraints,
- on-ice lineup constraints.

Do not revive rejected OSNet/DINOv2 approaches without new evidence.

Do not equate ByteTrack IDs with player identities.

A future identity layer may join multiple temporary tracklets, but it must tolerate ambiguity and return `UNKNOWN` rather than force a guess.

---

## Human-in-the-loop runtime resolution

This is separate from the current offline supervised-data labeling workflow.

Later, during actual game analysis, persistent unusual participants may benefit from user assistance.

Examples:

- practice jerseys,
- pinnies,
- unusual goalie equipment,
- inconsistent uniforms,
- persistent classifier uncertainty.

A future runtime workflow may collect representative crops and ask the user for a game-specific classification such as:

```text
HOME
AWAY
OFFICIAL
IGNORE
```

Any grouping used for such prompts must not assume one prompt per ByteTrack ID.

Do not implement this merely as a substitute for building the supervised classifier.

---

## Provider overlay handling

Current behavior supports configured normalized exclusion masks applied to the analysis path.

A future iteration may add:

- provider profiles,
- optional provider detection,
- automatic static-overlay discovery,

but only if multi-video evidence shows that the current configured approach is insufficient.

Do not spend development time generalizing provider detection without demonstrated need.

---

## Performance

Current development does not require real-time processing.

Priorities are:

1. correctness,
2. trustworthy ground truth,
3. portability,
4. interpretable diagnostics,
5. then optimization.

Measure before optimizing.

Potential future optimizations include:

- batching learned inference,
- sampling observations rather than classifying every frame,
- caching features,
- tracklet-level aggregation,
- event-driven secondary models.

Do not sacrifice dataset integrity or semantic accuracy merely to increase FPS.

---

## Validation principles

### Classical observation layers

Validate against what they actually measure.

Examples:

- boundary location,
- participant geometry,
- light/dark appearance,
- stripe structure,
- crop measurability.

Do not call semantic pseudo-label stability accuracy.

### Learned semantic layers

Evaluate against human ground truth.

Use:

- per-class metrics,
- confusion matrices,
- grouped splits,
- cross-game/rink tests,
- visual failure review.

### Higher-level hockey analysis

Keep conclusions traceable to:

- timestamps,
- tracklets,
- semantic predictions,
- source frames/clips,
- confidence/evidence.

---

## Safety and accuracy principles

- Track IDs are temporary and must never be represented as jersey/player identities.
- Unknown/unsure is preferable to guessing.
- Source video is read-only.
- Generated output should be reproducible.
- Human ground truth must remain separate from pseudo-labels and model predictions.
- Adjacent-frame leakage must not be mistaken for model generalization.
- Benchmark videos are evidence, not specifications.
- Coaching conclusions must eventually be traceable to source video evidence.

---

## Immediate next action

Build and inspect the **human-ground-truth labeling dataset workflow**.

The first implementation should:

1. select a diverse candidate set from both current benchmark games,
2. choose representative crops per temporary tracklet,
3. provide a lightweight labeling interface,
4. save labels incrementally and separately from source exports,
5. preserve provenance,
6. generate a dataset manifest/report,
7. stop before model training.

After the generated candidate set and labeling UI have been visually reviewed, begin human labeling.

Only after that should the first learned participant classifier be trained.
