# Hockey Analyzer roadmap

## Current milestone

Diagnostic ON ICE / OFF ICE classification using a configurable static polygon.

The completed YOLO11s + ByteTrack baseline is preserved. Tracked box
bottom-centers are classified against a manually calibrated playable-ice polygon.
Boundary ambiguity and bottom-clipped boxes are UNKNOWN. A frame/grid export
utility supports calibration, and the overlay shows geometry, contact points,
temporary IDs, and diagnostic classifications. Statistics count tracked
person-frames, not people. Container frame counts remain advisory.

The sample polygon is an experimental approximation at decoded frame 1800.
Camera pans can invalidate it; this is not automatic rink segmentation or
camera-motion compensation. No player identity inference is performed.

## Next planned milestone

Review geometry diagnostics across camera pans and refine the calibration
approach based on human inspection before adding further analysis.

## Later

Team/role classification -> multi-cue player re-identification -> shift detection
-> candidate clips -> semantic hockey analysis.

## Safety and accuracy principles

- Track IDs are temporary and must never be represented as jersey/player identities.
- Unknown identity is preferable to guessing.
- Source video is read-only.
- Generated output is disposable/reproducible.
- Coaching conclusions must eventually be traceable to source video evidence.
