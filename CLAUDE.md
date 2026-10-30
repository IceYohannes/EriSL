# Eritrean Sign Language (ErSL) Recognition — Project Brief

## Goal
Build the first open pipeline for recognizing Eritrean Sign Language signs from a phone camera.
v1 = isolated-sign recognition (alphabet + 20-50 common words). NOT sentence-level translation.
No public ErSL dataset exists, so data collection is as important as the model.

## Owner context
- Developer: Isaias. Strong in Kotlin/Android, learning applied ML.
- Explain ML steps briefly as you go; prefer simple, readable code over clever code.
- Final target: Android app (Kotlin, CameraX, MediaPipe Tasks, TFLite), fully on-device.

## Tech stack
- Python 3.11, virtualenv in `.venv`
- `mediapipe` (Tasks API: HandLandmarker + PoseLandmarker; do NOT use legacy `mp.solutions`)
- `opencv-python`, `numpy`, `pandas`, `scikit-learn`
- `tensorflow` / Keras for the GRU classifier, export to TFLite
- `pytest` for tests

## Repo layout
```
ersl/
  data/
    raw/videos/<signer_id>/<sign_label>/<rep>.mp4   # NEVER commit (gitignored)
    processed/                                       # landmark .npy sequences (gitignored)
    metadata.csv      # signer_id, sign_label, rep, device, lighting, date, consent_ok
  src/
    collect.py        # webcam/phone recording helper with countdown + labels
    extract.py        # video -> landmark sequence (.npy) via MediaPipe
    features.py       # normalization + windowing
    train.py          # GRU training, signer-independent split
    evaluate.py       # confusion matrix, per-sign + per-signer accuracy
    export_tflite.py
  notebooks/          # exploration only
  tests/
  docs/signs.md       # sign inventory, validated by Deaf signers
  CLAUDE.md
```

## Feature design (decide once, keep consistent with Android)
- Per frame: 21 landmarks x 2 hands x (x,y,z) + upper-body pose (shoulders, elbows, wrists, nose).
- Missing hand -> zeros plus a presence flag.
- Normalize: translate to shoulder midpoint, scale by shoulder width. Mirror left-handed signers
  consistently (document the choice).
- Window: 30 frames (~1 s) for isolated signs; pad/resample shorter clips.
- The exact same normalization must be reimplemented in Kotlin. Keep it in one small, well-tested function.

## Model
- Start: 2-layer GRU (64-128 units) + dense softmax. Tiny on purpose.
- Baseline first: simple kNN/MLP on flattened features, so GRU gains are measurable.
- Augmentation: small rotations, scale jitter, speed change, landmark noise.

## Non-negotiable rules
1. **Signer-independent evaluation**: split by signer, never randomly by clip. Always report both
   signer-dependent and signer-independent accuracy.
2. **Consent**: no recording enters `data/` without `consent_ok=true` in metadata.csv. Never commit raw
   video or faces to git. No cloud upload of recordings without explicit approval from me.
3. **Linguistic validity**: `docs/signs.md` is the source of truth for the sign list. Do not invent or
   rename signs; Deaf Eritrean signers must validate it.
4. Pin dependency versions in `requirements.txt`. Seed all randomness.
5. Ask before deleting files or running anything that downloads large data.

## Milestones
- M0 (week 1): environment, repo skeleton, MediaPipe running on webcam with live landmark overlay.
- M1 (week 2): `collect.py` + `extract.py` working end to end on 3 test signs recorded by me.
- M2 (weeks 3-6): dataset collection with community partners; target 20-30 signs x 3+ signers x 10-20 reps.
- M3 (week 7): baseline + GRU, signer-independent evaluation report.
- M4 (weeks 8-10): TFLite export, Android demo app with CameraX + Hand/Pose Landmarker.
- M5: write-up / dataset release plan (with signer consent).

## Current status
M0 in progress: skeleton, pinned deps, live landmark overlay (`src/live_overlay.py`).
