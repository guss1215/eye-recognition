# eye_recognition

Offline iris recognition in Flutter — enroll people once, identify them later with nothing but the phone camera. No cloud, no NIR hardware.

Iris recognition normally depends on near-infrared cameras. This app implements the classic Daugman pipeline (segmentation → rubber-sheet normalization → Gabor IrisCode → masked Hamming matching) with OpenCV on ordinary **visible-light** phone cameras, and tunes every stage around the noise that choice introduces.

<!-- TODO: screenshot/GIF of the scanner screen (live guidance + burst) goes here -->

## What it does

- **Enroll** — 3 guided burst captures per person; the app greedily keeps the 3 most *diverse* templates (max–min Hamming distance) so one person isn't represented by near-duplicates.
- **Identify** — one burst, matched against everyone enrolled, with a three-zone decision: confirmed match (HD ≤ 0.30), *suggested* candidates for human confirmation (≤ 0.38), or fall straight into enrollment reusing the burst just captured.
- **Local-first** — persons in SQLite, iris images on device storage. Nothing leaves the phone.

## Pipeline

1. **Preprocess** — red channel instead of luma (~620–750 nm is the closest visible band to NIR and penetrates melanin in brown irises) + CLAHE.
2. **Segment** — Hough circles for limbus and pupil, then geometric sanity gates: pupil contained in iris, concentricity ≤ 22 %, radius ratio in [0.2, 0.7].
3. **Normalize** — Daugman rubber-sheet unwrap to a 256 × 64 polar strip (bilinear `cv.remap`).
4. **Encode** — 2D Gabor bank (4 orientations × 2 wavelengths) → **8,192-bit IrisCode**, plus an occlusion mask and *fragile-bit* masking keyed to the per-filter **median** magnitude, so a single specular highlight can't shift which bits get masked between captures.
5. **Match** — masked fractional Hamming distance over mutually-valid bits, circular shifts of ±45° for head-tilt compensation, min-rule score fusion across all probe × enrolled template pairs per person.

Live preview guidance (too far / too close / off-center / blurry / **ready**) triggers the burst automatically; each frame is quality-scored (sharpness, occlusion, specular highlights, centering, resolution) and only the best frames become templates.

## Design notes — why visible light is hard

- Match thresholds are deliberately looser than the NIR literature (Daugman's 0.26–0.28): visible-light genuine distributions are inflated, so the pipeline optimizes repeatability and delegates the gray zone to a human — the *suggested* dialog is designed for a clinic check-in flow, not unattended auth.
- Normalization uses a **single polar origin** (pupil center) for both boundaries: two-center estimates are too noisy on phone captures.
- Continuous autofocus/exposure stays ON during bursts — best-frame selection tolerates exposure variation better than a stale AF lock.

This is an experimental/portfolio project, **not** a biometric security product.

## Tech

Flutter · [`opencv_dart`](https://pub.dev/packages/opencv_dart) (trimmed to `core`/`imgproc`/`imgcodecs` via Dart build hooks to cut binary size) · `camera` · `sqflite`

Android-only in practice (iOS is scaffolded but untested).

## Run it

```bash
flutter pub get
flutter run   # Android device with an autofocus camera
```

Tests exercise the encode/match core with synthetic iris strips — self-match, ±45° rotation recovery, impostor separation:

```bash
flutter test
```
