# ErSL Development Plan

Living plan. Milestones come from CLAUDE.md; this file breaks them into concrete steps with a
"done when" check for each. Tick boxes as we go.

---

## M0: Environment + live overlay ✅ done
- [x] venv, pinned `requirements.txt`, `.gitignore`, git repo
- [x] `src/landmarks.py`: shared Hand + Pose Landmarker wrapper
- [x] `src/live_overlay.py`: webcam preview
- [x] You run `python -m src.live_overlay` and confirm tracking looks right (2026-10-30)
- [x] Push to GitHub (github.com/IceYohannes/EriSL; confirm it is private)

---

## M1: Record -> extract -> features, end to end on 3 test signs (next)

### Step 1: Lock the feature format (`src/features.py`) ✅ done (15 tests + golden file)
This is the contract with the Android app. Decide once, test hard, then reimplement in Kotlin.

Per frame, input = raw arrays from `landmarks.py`:
`hands (2,21,3)`, `presence (2,)`, `pose (7,3)`.

Normalization (one function, `normalize_frame`):
1. Origin = midpoint of the two shoulders (pose points 11, 12).
2. Scale = shoulder width (distance between shoulders in x,y). Divide x,y by it.
3. Hand z: MediaPipe gives it relative to the wrist; scale it by the same factor.
   Pose z: dropped (too noisy from a single camera).
4. Missing hand stays all zeros (after normalization) and presence = 0.
5. No pose detected in a frame: reuse the last valid origin/scale; if none yet, skip frame.
6. Left-handed signers: flip x (`x -> -x` after normalization) and swap the hand slots, so every
   signer looks right-dominant. Needs a `dominant_hand` value per signer (see decisions below).

Frame vector = 126 (hands) + 2 (presence) + 14 (pose x,y) = **142 floats**.

Windowing (`to_window`): resample any clip to exactly **30 frames** by linear interpolation over
time (works for both shorter and longer clips). Output shape `(30, 142)`.

Done when: unit tests cover translation/scale invariance, missing hands, missing pose,
left-hand mirroring, and resampling; a fixed "golden" input/output pair is saved in `tests/` so
the Kotlin port can be checked against exactly the same numbers.

### Step 2: Recording tool (`src/collect.py`) ✅ code done, awaiting your recording
- CLI: `python -m src.collect --signer s01 --signs hello,thanks,yes --reps 10`
- Refuses to start unless the signer has `consent_ok=true` (consent recorded first, once per signer).
- Per rep: on-screen sign name, 3-2-1 countdown, records ~2 s, shows live landmark overlay,
  keys: `space` = keep, `r` = redo, `q` = quit.
- Saves `data/raw/videos/<signer>/<sign>/<rep>.mp4` and appends a row to `data/metadata.csv`.
- Sign names must exist in `docs/signs.md` (for the 3 test signs we mark them as `test_*` placeholders
  so we don't invent real ErSL labels).

Done when: you record 3 signs x 10 reps, and metadata.csv has 30 correct rows.

Alternative: `src/import_videos.py` ✅ imports clips already recorded on a phone (one file = one
rep; re-encoded to .mp4, max 1280 px; same consent + signs.md checks; originals untouched):
`python -m src.import_videos --signer s01 --sign test_letter_1 --dominant-hand right --device pixel7_back --lighting indoor_day <folder>`

### Step 3: Extraction (`src/extract.py`) ✅ code done, awaiting real clips
- Walks `data/raw/videos`, skips anything without consent, runs `LandmarkExtractor` frame by frame
  (timestamps from the video's FPS), saves raw per-frame arrays to `data/processed/<signer>/<sign>/<rep>.npz`.
- Raw (un-normalized) arrays are stored so we can change normalization later without re-running MediaPipe.
- Writes a quality report: % frames with pose, % frames with each hand, clip length.

Done when: all 30 test clips extract; quality report flags no clip with < 80 % hand detection.

### Step 4: Sanity check (`src/inspect_clips.py`) ✅ code done, awaiting real clips
- Plots normalized fingertip trajectories per sign + runs the baselines (should be near 100 %
  signer-dependent; it's just a pipeline check, not a result).

---

## M2: Dataset collection (weeks 3-6, with community partners)
- Get the sign list validated by Deaf Eritrean signers -> fill `docs/signs.md` (20-30 signs).
- Consent form (Tigrinya + English) and process; record signer id, dominant hand, device, lighting.
  Draft v0.1 (English, placeholders): https://claude.ai/artifact/MU1ywHt3wogwXjUGReE6n7. Needs partner review,
  human Tigrinya translation, ErSL explanation by an interpreter. Track choices B-F per signer before M5.
- Target: 20-30 signs x 3+ signers x 10-20 reps. Vary lighting, background, distance, device.
- Re-run extraction + quality report after each session; re-record bad clips while the signer is there.

## M3: Models + evaluation (week 7)
- ✅ `src/dataset.py`: load windows + labels + signer ids; **split by signer** (leave-one-signer-out).
- ✅ `src/baseline.py`: kNN and MLP on flattened `(30,142)` windows (scikit-learn).
- GRU (TensorFlow, install at this point): 2 layers x 64-128 units + softmax; augmentation
  (rotation, scale jitter, speed change, noise).
- `src/evaluate.py`: confusion matrix, per-sign and per-signer accuracy; report **both**
  signer-dependent and signer-independent numbers.

## M4: Android (weeks 8-10)
- `export_tflite.py`; check TFLite output == Keras output on test windows.
- Kotlin: CameraX -> Hand + Pose Landmarker -> `normalizeFrame` (port, checked against the
  golden test file) -> 30-frame ring buffer -> TFLite -> show top-3 signs.

## M5: Write-up and dataset release plan (with signer consent)

---

## Open decisions (need your answer)
1. ~~dominant_hand column~~: **decided yes** (added).
2. ~~Test sign labels for M1~~: **decided** ገ, ረ, ተ -> `test_letter_ge`, `test_letter_re`, `test_letter_te`.
3. ~~Clip length~~: **decided fixed 2 s**.
