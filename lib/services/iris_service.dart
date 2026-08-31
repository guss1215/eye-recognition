import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import '../models/person.dart';
import 'database_service.dart';

/// Complete iris recognition pipeline using Daugman's IrisCode.
///
/// Pipeline:
/// 1. Preprocessing – single channel (red for visible light), resize, CLAHE
/// 2. Segmentation – COUPLED pupil/iris (limbus first, pupil constrained to it)
/// 3. Normalization – Daugman's rubber sheet, single polar origin, bilinear (remap)
/// 4. Encoding – Gabor filters → phase quantization → binary IrisCode + noise mask
/// 5. Matching – masked Hamming distance with fine rotation compensation + score fusion
///
/// Design notes for a VISIBLE-LIGHT phone camera (no near-infrared): iris texture
/// is weak, especially for brown eyes, so the pipeline is tuned for REPEATABILITY
/// (same eye → same code) rather than raw discriminative power. Consistency of
/// segmentation and rotation alignment matters more here than filter richness.
class IrisService {
  final DatabaseService _dbService;

  IrisService(this._dbService);

  // ─── IRISCODE STRUCTURE CONSTANTS ──────────────────────────────────

  // Gabor filter bank: 4 orientations × 2 wavelengths = 8 filters
  static const _orientations = [0.0, pi / 4, pi / 2, 3 * pi / 4];
  static const _wavelengths = [6.0, 12.0]; // well-separated scales
  static const _kRows = 5; // kernel height (radial) — small to avoid border contamination
  static const _kCols = 15; // kernel width (angular) — captures iris texture patterns
  static const _numFilters = 8; // _orientations.length * _wavelengths.length

  // Normalized iris image dimensions
  static const _angularRes = 256;
  static const _radialRes = 64;

  // Eyelid crop: skip top/bottom 8 rows (eyelid zone)
  static const _skipRows = 8;
  static const _cropRows = 48; // _radialRes - 2 * _skipRows

  // Sampling grid for IrisCode (on cropped image).
  // _stepTheta=4 → 64 angular columns → each rotation shift is 360/64 = 5.625°,
  // which is finer than the smallest Gabor wavelength (6px) so rotation alignment
  // no longer scrambles the fine-filter phase bits.
  static const _stepTheta = 4; // angular step → 64 columns
  static const _stepR = 6; // radial step → 8 rows
  static const _gridCols = 64; // _angularRes / _stepTheta
  static const _gridRows = 8; // _cropRows / _stepR
  static const _bitsPerSample = 2; // real + imaginary phase bits
  static const _bitsPerFilter = 1024; // _gridCols * _gridRows * _bitsPerSample
  static const _codeBits = 8192; // _numFilters * _bitsPerFilter

  // Rotation compensation: ±8 column shifts (each = 5.625°, total ±45°)
  static const _maxRotationShift = 8;

  // Minimum mutually-valid bits required for a meaningful comparison.
  // MUST stay well below (encodeMinValid)^2 so two genuine captures whose masks
  // decorrelate are never rejected outright. 15% of the code keeps HD variance
  // low (std ≈ 0.014 at n≈1229) while still refusing near-empty comparisons.
  static const _minMutualFraction = 0.15;

  // A template is only accepted if at least this fraction of its own bits are valid.
  static const _encodeMinValid = 0.50;

  // Matching thresholds. Visible-light irises have an inflated genuine
  // distribution vs. NIR, so these are looser than Daugman's 0.26–0.28. The
  // "suggested" zone surfaces borderline candidates for a human (clinic staff)
  // to confirm — ideal for returning-patient recognition.
  static const confirmThreshold = 0.30;
  static const suggestThreshold = 0.38;

  // Use the RED channel (≈620–750nm, closest to NIR) instead of luma. Melanin in
  // brown irises absorbs less red light, so the red channel exposes more stromal
  // texture. Set to false to fall back to luma if a device's color path misbehaves.
  static const useRedChannel = true;

  // ─── 1. IMAGE STORAGE ───────────────────────────────────────────────

  Future<String> saveEyeImage(File imageFile) async {
    final directory = await getApplicationDocumentsDirectory();
    final irisDir = Directory('${directory.path}/iris_images');
    if (!await irisDir.exists()) {
      await irisDir.create(recursive: true);
    }

    final uuid = const Uuid().v4();
    final savedPath = '${irisDir.path}/$uuid.png';
    await imageFile.copy(savedPath);
    return savedPath;
  }

  // ─── 2. PREPROCESSING ──────────────────────────────────────────────

  /// Preprocesses a color image: single channel (red) → resize → CLAHE.
  ({cv.Mat gray, double scale}) _preprocessImage(cv.Mat image) {
    const standardWidth = 640;
    final scale = standardWidth / image.cols;
    final newHeight = (image.rows * scale).round();
    final resized = cv.resize(image, (standardWidth, newHeight));

    // imread returns BGR; the red channel is index 2.
    final cv.Mat single;
    if (useRedChannel && resized.channels >= 3) {
      single = cv.extractChannel(resized, 2);
    } else {
      single = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
    }
    resized.dispose();

    final enhanced = _applyClahe(single);
    single.dispose();

    return (gray: enhanced, scale: scale);
  }

  /// Preprocessing variant for an already single-channel input (camera stream).
  ({cv.Mat gray, double scale}) _preprocessGray(cv.Mat grayImage) {
    const standardWidth = 640;
    final scale = standardWidth / grayImage.cols;
    final newHeight = (grayImage.rows * scale).round();
    final resized = cv.resize(grayImage, (standardWidth, newHeight));

    final enhanced = _applyClahe(resized);
    resized.dispose();

    return (gray: enhanced, scale: scale);
  }

  cv.Mat _applyClahe(cv.Mat src) {
    final clahe = cv.CLAHE.create(2.0, (8, 8));
    final out = clahe.apply(src);
    clahe.dispose();
    return out;
  }

  // ─── 3. SEGMENTATION ───────────────────────────────────────────────

  /// Segments the iris by detecting the limbus (iris/sclera boundary) first —
  /// the strong, reliable edge — then selecting a pupil that is DARK and
  /// CONCENTRIC with it. Coupling the two boundaries is what makes the
  /// normalization repeatable across captures of the same eye.
  IrisSegmentation? segmentIris(cv.Mat grayImage) {
    final blurred = cv.medianBlur(grayImage, 7);

    final irisCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5,
      100,
      param1: 80,
      param2: 30,
      minRadius: 50,
      maxRadius: 220,
    );

    final pupilCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5,
      40,
      param1: 100,
      param2: 25,
      minRadius: 8,
      maxRadius: 110,
    );

    if (irisCircles.rows == 0 || pupilCircles.rows == 0) {
      blurred.dispose();
      irisCircles.dispose();
      pupilCircles.dispose();
      return null;
    }

    final centerX = grayImage.cols / 2.0;
    final centerY = grayImage.rows / 2.0;

    final irisList = _extractCircles(irisCircles);
    final pupilList = _extractCircles(pupilCircles);
    irisCircles.dispose();
    pupilCircles.dispose();

    // Iris = limbus, which sits near the frame center: pick the most central.
    final iris = _selectIris(irisList, centerX, centerY);

    // Pupil = darkest circle that is concentric with the iris and physiologically sized.
    final pupil = _selectPupil(pupilList, blurred, iris);
    blurred.dispose();

    if (pupil == null) {
      print('[IrisService] Segmentation rejected: no concentric dark pupil found');
      return null;
    }

    if (iris.r <= pupil.r) return null;

    // Euclidean containment: the whole pupil disc must sit inside the iris.
    final dx = pupil.x - iris.x;
    final dy = pupil.y - iris.y;
    final d = sqrt(dx * dx + dy * dy);
    if (d + pupil.r > iris.r) {
      print('[IrisService] Segmentation rejected: pupil not contained in iris');
      return null;
    }

    // Concentricity: real irises are slightly non-concentric (pupil biased nasal),
    // so allow up to 22% of the iris radius of offset but reject stray pairings.
    if (d > 0.22 * iris.r) {
      print('[IrisService] Segmentation rejected: pupil/iris not concentric (d=${d.toStringAsFixed(1)}, max=${(0.22 * iris.r).toStringAsFixed(1)})');
      return null;
    }

    // Physiological pupil/iris ratio.
    final ratio = pupil.r / iris.r;
    if (ratio < 0.2 || ratio > 0.7) {
      print('[IrisService] Segmentation rejected: pupil/iris ratio ${ratio.toStringAsFixed(2)} outside range');
      return null;
    }

    if (iris.r < 40) {
      print('[IrisService] Segmentation rejected: iris too small (r=${iris.r.toStringAsFixed(1)})');
      return null;
    }

    print('[IrisService] Segmentation: pupil=(${pupil.x.toStringAsFixed(0)},${pupil.y.toStringAsFixed(0)},r=${pupil.r.toStringAsFixed(0)}) '
        'iris=(${iris.x.toStringAsFixed(0)},${iris.y.toStringAsFixed(0)},r=${iris.r.toStringAsFixed(0)}) '
        'ratio=${ratio.toStringAsFixed(2)} concentricity=${d.toStringAsFixed(1)}');

    return IrisSegmentation(
      pupilX: pupil.x,
      pupilY: pupil.y,
      pupilR: pupil.r,
      irisX: iris.x,
      irisY: iris.y,
      irisR: iris.r,
    );
  }

  /// HoughCircles returns a 1-row CV_32FC3 Mat; each circle is 3 consecutive
  /// values (x, y, r) and the number of circles is `circles.cols`.
  List<_Circle> _extractCircles(cv.Mat circles) {
    final n = circles.cols;
    final list = <_Circle>[];
    for (int i = 0; i < n; i++) {
      list.add(_Circle(
        circles.at<double>(0, i * 3),
        circles.at<double>(0, i * 3 + 1),
        circles.at<double>(0, i * 3 + 2),
      ));
    }
    return list;
  }

  /// Iris (limbus) selection: the boundary nearest the frame center.
  _Circle _selectIris(List<_Circle> circles, double cx, double cy) {
    _Circle best = circles.first;
    double bestDist = double.infinity;
    for (final c in circles) {
      final dist = (c.x - cx) * (c.x - cx) + (c.y - cy) * (c.y - cy);
      if (dist < bestDist) {
        bestDist = dist;
        best = c;
      }
    }
    return best;
  }

  /// Pupil selection: among candidates that are plausibly sized and roughly
  /// concentric with the iris, choose the one whose interior is DARKEST. In
  /// visible light the pupil is the most reliably dark region, so darkness is a
  /// far more stable cue than proximity to the frame center.
  _Circle? _selectPupil(List<_Circle> circles, cv.Mat blurred, _Circle iris) {
    _Circle? best;
    double bestScore = double.infinity;

    for (final c in circles) {
      if (c.r >= iris.r * 0.9) continue; // must be smaller than the iris
      final ratio = c.r / iris.r;
      if (ratio < 0.15 || ratio > 0.75) continue;

      final dx = c.x - iris.x;
      final dy = c.y - iris.y;
      final dist = sqrt(dx * dx + dy * dy);
      if (dist > 0.5 * iris.r) continue; // rough concentricity prefilter

      final darkness = _patchMean(blurred, c.x, c.y, c.r * 0.6);
      // Lower is better: prefer dark interior, break ties toward concentric.
      final score = darkness / 255.0 + 1.5 * (dist / iris.r);
      if (score < bestScore) {
        bestScore = score;
        best = c;
      }
    }
    return best;
  }

  /// Mean intensity of a square patch centered at (x, y).
  double _patchMean(cv.Mat img, double x, double y, double radius) {
    final rr = max(2, radius.round());
    final x0 = (x - rr).round().clamp(0, img.cols - 1);
    final y0 = (y - rr).round().clamp(0, img.rows - 1);
    final w = min(2 * rr, img.cols - x0);
    final h = min(2 * rr, img.rows - y0);
    if (w <= 0 || h <= 0) return 255.0;
    final roi = img.region(cv.Rect(x0, y0, w, h));
    final m = cv.mean(roi).val1;
    roi.dispose();
    return m;
  }

  // ─── 4. NORMALIZATION (DAUGMAN'S RUBBER SHEET, SINGLE ORIGIN) ────────

  /// Unwraps the iris annulus into a fixed-size polar strip using bilinear
  /// interpolation (cv.remap). A SINGLE polar origin (the pupil center) is used
  /// for both boundaries: on a visible-light phone the two circle centers cannot
  /// be estimated reliably enough to model non-concentricity, so a deterministic
  /// single-origin unwrap — applied identically at enrollment and verification —
  /// matches more consistently than a noisy two-center model.
  cv.Mat normalizeIris(
    cv.Mat grayImage,
    IrisSegmentation seg, {
    int angularResolution = _angularRes,
    int radialResolution = _radialRes,
  }) {
    final cx = seg.pupilX;
    final cy = seg.pupilY;
    final innerR = seg.pupilR;
    final outerR = seg.irisR;

    final n = radialResolution * angularResolution;
    final mapXData = Float32List(n);
    final mapYData = Float32List(n);

    for (int theta = 0; theta < angularResolution; theta++) {
      final angle = 2 * pi * theta / angularResolution;
      final ct = cos(angle);
      final st = sin(angle);
      for (int r = 0; r < radialResolution; r++) {
        final ratio = r / radialResolution;
        final radius = innerR + ratio * (outerR - innerR);
        final idx = r * angularResolution + theta;
        mapXData[idx] = (cx + radius * ct);
        mapYData[idx] = (cy + radius * st);
      }
    }

    final mapX = cv.Mat.fromList(radialResolution, angularResolution, cv.MatType.CV_32FC1, mapXData);
    final mapY = cv.Mat.fromList(radialResolution, angularResolution, cv.MatType.CV_32FC1, mapYData);

    final normalized = cv.remap(
      grayImage,
      mapX,
      mapY,
      cv.INTER_LINEAR,
      borderMode: cv.BORDER_REPLICATE,
    );

    mapX.dispose();
    mapY.dispose();
    return normalized;
  }

  // ─── 5. GABOR KERNEL GENERATION ─────────────────────────────────────

  cv.Mat _createGaborKernel(
    double sigma, double theta, double lambd, double psi,
  ) {
    final data = List<double>.filled(_kRows * _kCols, 0);
    final halfR = _kRows ~/ 2; // 2
    final halfC = _kCols ~/ 2; // 7
    const gamma = 0.5;

    for (int y = -halfR; y <= halfR; y++) {
      for (int x = -halfC; x <= halfC; x++) {
        final xPrime = x * cos(theta) + y * sin(theta);
        final yPrime = -x * sin(theta) + y * cos(theta);

        final gaussian = exp(
          -(xPrime * xPrime + gamma * gamma * yPrime * yPrime) /
              (2 * sigma * sigma),
        );
        final sinusoidal = cos(2 * pi * xPrime / lambd + psi);

        data[(y + halfR) * _kCols + (x + halfC)] = gaussian * sinusoidal;
      }
    }

    return cv.Mat.fromList(_kRows, _kCols, cv.MatType.CV_64FC1, data);
  }

  // ─── 6. NOISE MASK GENERATION ───────────────────────────────────────

  /// Marks low-variance or very dark/bright blocks (reflections, eyelid
  /// fragments) as occluded.
  List<bool> _generateNoiseMask(cv.Mat croppedIris) {
    final mask = List<bool>.filled(_gridRows * _gridCols, true);

    for (int r = 0; r < _gridRows; r++) {
      for (int c = 0; c < _gridCols; c++) {
        final roiX = c * _stepTheta;
        final roiY = r * _stepR;
        final roiW = min(_stepTheta, croppedIris.cols - roiX);
        final roiH = min(_stepR, croppedIris.rows - roiY);

        if (roiW <= 0 || roiH <= 0) {
          mask[r * _gridCols + c] = false;
          continue;
        }

        final roi = croppedIris.region(cv.Rect(roiX, roiY, roiW, roiH));
        final meanStd = cv.meanStdDev(roi);
        final meanVal = meanStd.$1.val1;
        final stdVal = meanStd.$2.val1;
        roi.dispose();

        if (stdVal < 8.0 || meanVal < 20.0 || meanVal > 245.0) {
          mask[r * _gridCols + c] = false;
        }
      }
    }

    return mask;
  }

  // ─── 7. FEATURE ENCODING (DAUGMAN'S IRISCODE) ──────────────────────

  /// Encodes the normalized iris into a binary IrisCode with noise mask.
  /// Returns [code_bits..., mask_bits...] or null if quality too low.
  List<double>? encodeIris(cv.Mat normalizedIris) {
    final enhanced = _applyClahe(normalizedIris);

    final cropped = enhanced.region(
      cv.Rect(0, _skipRows, _angularRes, _cropRows),
    );
    enhanced.dispose();

    final padW = _kCols ~/ 2; // 7
    final padH = _kRows ~/ 2; // 2
    final padded = cv.copyMakeBorder(
      cropped, padH, padH, padW, padW, cv.BORDER_WRAP,
    );

    final src = padded.convertTo(cv.MatType.CV_64FC1);
    padded.dispose();

    final noiseMask = _generateNoiseMask(cropped);
    cropped.dispose();

    final iriscode = List<double>.filled(_codeBits, 0.0);
    final maskBits = List<double>.filled(_codeBits, 1.0);

    // First pass: compute all filter responses + track magnitudes per filter.
    final allResponses = <List<(double, double)>>[];
    final allMagnitudes = <List<double>>[];

    for (final lambda in _wavelengths) {
      final sigma = lambda / 2;

      for (final theta in _orientations) {
        final kernelReal = _createGaborKernel(sigma, theta, lambda, 0);
        final kernelImag = _createGaborKernel(sigma, theta, lambda, pi / 2);

        final responseReal = cv.filter2D(src, cv.MatType.CV_64FC1.depth, kernelReal);
        final responseImag = cv.filter2D(src, cv.MatType.CV_64FC1.depth, kernelImag);

        final responses = <(double, double)>[];
        final magnitudes = <double>[];

        for (int r = 0; r < _cropRows; r += _stepR) {
          for (int t = 0; t < _angularRes; t += _stepTheta) {
            final rv = responseReal.at<double>(r + padH, t + padW);
            final iv = responseImag.at<double>(r + padH, t + padW);
            responses.add((rv, iv));
            magnitudes.add(sqrt(rv * rv + iv * iv));
          }
        }

        allResponses.add(responses);
        allMagnitudes.add(magnitudes);

        kernelReal.dispose();
        kernelImag.dispose();
        responseReal.dispose();
        responseImag.dispose();
      }
    }

    src.dispose();

    // Second pass: phase quantize + fragile-bit masking.
    // The dead-zone reference is the per-filter MEDIAN magnitude (robust and
    // repeatable across captures), NOT the per-capture maximum (which a single
    // specular highlight can swing, masking different bits each capture).
    int filterIdx = 0;
    int bitIdx = 0;
    const deadZoneFactor = 0.35; // mask bits below 0.35 × median magnitude

    for (final responses in allResponses) {
      final deadZone = deadZoneFactor * _median(allMagnitudes[filterIdx]);

      int sampleIdx = 0;
      for (int r = 0; r < _gridRows; r++) {
        for (int c = 0; c < _gridCols; c++) {
          final (rv, iv) = responses[sampleIdx];

          iriscode[bitIdx] = rv >= 0 ? 1.0 : 0.0;
          iriscode[bitIdx + 1] = iv >= 0 ? 1.0 : 0.0;

          final gridValid = noiseMask[r * _gridCols + c];
          final magnitude = sqrt(rv * rv + iv * iv);
          if (!gridValid || magnitude < deadZone) {
            maskBits[bitIdx] = 0.0;
            maskBits[bitIdx + 1] = 0.0;
          }

          bitIdx += 2;
          sampleIdx++;
        }
      }
      filterIdx++;
    }

    int validCount = 0;
    for (final m in maskBits) {
      if (m == 1.0) validCount++;
    }
    final validFraction = validCount / maskBits.length;

    print('[Encode] IrisCode: ${iriscode.length} bits, '
        '${(validFraction * 100).toStringAsFixed(0)}% valid');

    if (validFraction < _encodeMinValid) {
      print('[Encode] Rejected: too few valid bits (${(validFraction * 100).toStringAsFixed(0)}%)');
      return null;
    }

    return [...iriscode, ...maskBits];
  }

  double _median(List<double> values) {
    if (values.isEmpty) return 0.0;
    final sorted = List<double>.from(values)..sort();
    final mid = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
  }

  // ─── 8. IMAGE QUALITY ASSESSMENT ───────────────────────────────────

  static const _minSharpness = 60.0;

  /// Laplacian variance. Blurry < 60, acceptable 60–150, sharp > 150.
  double _measureSharpness(cv.Mat grayRegion) {
    final lap = cv.laplacian(grayRegion, cv.MatType.CV_16SC1.depth);
    final meanStd = cv.meanStdDev(lap);
    final stddev = meanStd.$2.val1;
    lap.dispose();
    return stddev * stddev;
  }

  cv.Mat _extractIrisROI(cv.Mat gray, IrisSegmentation seg) {
    final x = max(0, (seg.irisX - seg.irisR).round());
    final y = max(0, (seg.irisY - seg.irisR).round());
    final w = min((seg.irisR * 2).round(), gray.cols - x);
    final h = min((seg.irisR * 2).round(), gray.rows - y);
    return gray.region(cv.Rect(x, y, w, h));
  }

  // ─── 9. FULL PIPELINE (still image) ────────────────────────────────

  Future<IrisProcessingResult?> processIrisImage(String imagePath) async {
    final image = cv.imread(imagePath, flags: cv.IMREAD_COLOR);
    if (image.isEmpty) return null;

    final (:gray, :scale) = _preprocessImage(image);
    image.dispose();

    print('[IrisService] Preprocessed: ${gray.cols}x${gray.rows} (scale=$scale)');

    final segmentation = segmentIris(gray);
    if (segmentation == null) {
      gray.dispose();
      print('[IrisService] Segmentation failed');
      return null;
    }

    final irisROI = _extractIrisROI(gray, segmentation);
    final sharpness = _measureSharpness(irisROI);
    irisROI.dispose();
    print('[IrisService] Sharpness: ${sharpness.toStringAsFixed(1)}');

    if (sharpness < _minSharpness) {
      gray.dispose();
      print('[IrisService] Rejected: too blurry (sharpness=$sharpness)');
      return null;
    }

    final normalized = normalizeIris(gray, segmentation);
    final template = encodeIris(normalized);

    gray.dispose();
    normalized.dispose();

    if (template == null) {
      print('[Pipeline] Encoding failed (quality too low)');
      return null;
    }

    return IrisProcessingResult(
      template: template,
      segmentation: segmentation,
    );
  }

  Future<List<double>?> generateIrisTemplate(String imagePath) async {
    final result = await processIrisImage(imagePath);
    return result?.template;
  }

  // ─── 10. QUICK DETECTION (for live preview) ────────────────────────

  IrisDetectionResult quickDetectIris(cv.Mat grayFrame) {
    const previewWidth = 320;
    final scale = previewWidth / grayFrame.cols;
    final previewHeight = (grayFrame.rows * scale).round();
    final resized = cv.resize(grayFrame, (previewWidth, previewHeight));

    final blurred = cv.medianBlur(resized, 7);

    final irisCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5, 50,
      param1: 80, param2: 30,
      minRadius: 30, maxRadius: 130,
    );

    final pupilCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5, 25,
      param1: 100, param2: 22,
      minRadius: 5, maxRadius: 60,
    );

    if (irisCircles.rows == 0 || pupilCircles.rows == 0) {
      blurred.dispose();
      irisCircles.dispose();
      pupilCircles.dispose();
      resized.dispose();
      return const IrisDetectionResult(status: IrisDetectionStatus.notFound);
    }

    final centerX = previewWidth / 2.0;
    final centerY = previewHeight / 2.0;

    final irisList = _extractCircles(irisCircles);
    final pupilList = _extractCircles(pupilCircles);
    irisCircles.dispose();
    pupilCircles.dispose();

    final iris = _selectIris(irisList, centerX, centerY);
    final pupil = _selectPupil(pupilList, blurred, iris);
    blurred.dispose();

    if (pupil == null || iris.r <= pupil.r) {
      resized.dispose();
      return const IrisDetectionResult(status: IrisDetectionStatus.notFound);
    }

    // Reject non-concentric detections early so bursts don't start on bad frames.
    final dx = pupil.x - iris.x;
    final dy = pupil.y - iris.y;
    final concentricity = sqrt(dx * dx + dy * dy);
    if (concentricity > 0.22 * iris.r || pupil.r / iris.r > 0.7) {
      resized.dispose();
      return const IrisDetectionResult(status: IrisDetectionStatus.notFound);
    }

    final seg = IrisSegmentation(
      pupilX: pupil.x, pupilY: pupil.y, pupilR: pupil.r,
      irisX: iris.x, irisY: iris.y, irisR: iris.r,
    );

    IrisDetectionStatus status;
    // Encourage a large iris (more pixels → more usable texture). At 320px
    // preview, r<50 → iris fills <31% of the frame (too far).
    if (iris.r < 50) {
      status = IrisDetectionStatus.tooFar;
    } else if (iris.r > 120) {
      status = IrisDetectionStatus.tooClose;
    } else {
      final offX = (iris.x - centerX).abs();
      final offY = (iris.y - centerY).abs();
      final maxOffset = previewWidth * 0.30;
      if (offX > maxOffset || offY > maxOffset) {
        status = IrisDetectionStatus.notCentered;
      } else {
        final roiX = max(0, (iris.x - iris.r).round());
        final roiY = max(0, (iris.y - iris.r).round());
        final roiW = min((iris.r * 2).round(), previewWidth - roiX);
        final roiH = min((iris.r * 2).round(), previewHeight - roiY);
        if (roiW > 0 && roiH > 0) {
          final roi = resized.region(cv.Rect(roiX, roiY, roiW, roiH));
          final lap = cv.laplacian(roi, cv.MatType.CV_16SC1.depth);
          final meanStd = cv.meanStdDev(lap);
          final sharpness = meanStd.$2.val1 * meanStd.$2.val1;
          lap.dispose();
          roi.dispose();
          status = sharpness < 40.0
              ? IrisDetectionStatus.tooBlurry
              : IrisDetectionStatus.ready;
        } else {
          status = IrisDetectionStatus.ready;
        }
      }
    }

    resized.dispose();
    return IrisDetectionResult(status: status, segmentation: seg);
  }

  // ─── 11. MATCHING (MASKED HAMMING + ROTATION COMPENSATION) ─────────

  /// Masked fractional Hamming distance with circular bit-shifting for rotation.
  /// - Same iris: ~0.20–0.32 (visible light runs higher than NIR)
  /// - Different iris: ~0.42–0.50
  double compareTemplates(List<double> template1, List<double> template2,
      {bool verbose = false}) {
    if (template1.length != template2.length) {
      if (verbose) print('[Match] Length mismatch: ${template1.length} vs ${template2.length}');
      return 1.0;
    }
    if (template1.length < 2) return 1.0;

    final codeLen = template1.length ~/ 2;
    final code1 = template1.sublist(0, codeLen);
    final mask1 = template1.sublist(codeLen);
    final code2 = template2.sublist(0, codeLen);
    final mask2 = template2.sublist(codeLen);

    double bestHD = 1.0;
    int bestShift = 0;

    for (int shift = -_maxRotationShift; shift <= _maxRotationShift; shift++) {
      final hd = _hammingWithShift(code1, mask1, code2, mask2, shift);
      if (hd < bestHD) {
        bestHD = hd;
        bestShift = shift;
      }
    }

    if (verbose) {
      print('[Match] Best: shift=$bestShift HD=${bestHD.toStringAsFixed(4)}');
    }

    return bestHD;
  }

  /// Masked Hamming distance with a circular column shift on the angular axis.
  double _hammingWithShift(
    List<double> code1,
    List<double> mask1,
    List<double> code2,
    List<double> mask2,
    int shift,
  ) {
    int validBits = 0;
    int differences = 0;

    for (int f = 0; f < _numFilters; f++) {
      final filterBase = f * _bitsPerFilter;

      for (int r = 0; r < _gridRows; r++) {
        final rowBase = filterBase + r * _gridCols * _bitsPerSample;

        for (int c = 0; c < _gridCols; c++) {
          final c2 = ((c + shift) % _gridCols + _gridCols) % _gridCols;

          for (int b = 0; b < _bitsPerSample; b++) {
            final idx1 = rowBase + c * _bitsPerSample + b;
            final idx2 = rowBase + c2 * _bitsPerSample + b;

            if (mask1[idx1] == 1.0 && mask2[idx2] == 1.0) {
              validBits++;
              if (code1[idx1] != code2[idx2]) differences++;
            }
          }
        }
      }
    }

    if (validBits == 0) return 1.0;
    // Require enough MUTUALLY-valid bits for a statistically meaningful distance,
    // but keep the floor well below the per-template encode floor so two genuine
    // captures whose masks decorrelate are never rejected outright.
    final minRequired = (_codeBits * _minMutualFraction).round();
    if (validBits < minRequired) return 1.0;
    return differences / validBits;
  }

  /// Best (minimum) distance between a probe and a set of enrolled templates.
  double _bestDistanceToTemplates(List<double> probe, List<List<double>> enrolled) {
    double best = 1.0;
    for (final t in enrolled) {
      final d = compareTemplates(probe, t);
      if (d < best) best = d;
    }
    return best;
  }

  /// Searches all registered persons using a SINGLE probe template.
  Future<List<IrisMatchResult>> findCandidates(List<double> template) =>
      findCandidatesMulti([template]);

  /// Searches all registered persons using MULTIPLE probe templates (score
  /// fusion, min-rule): for each person the distance is the best match across
  /// all (probe × enrolled) pairs. Fusing on both sides is markedly more robust
  /// when capture conditions differ between enrollment and verification.
  Future<List<IrisMatchResult>> findCandidatesMulti(List<List<double>> probes) async {
    final persons = await _dbService.getPersonsWithIrisTemplate();
    final results = <IrisMatchResult>[];

    for (final person in persons) {
      final enrolled = person.irisTemplates;
      if (enrolled == null || enrolled.isEmpty) continue;

      double bestDistance = 1.0;
      for (final probe in probes) {
        final d = _bestDistanceToTemplates(probe, enrolled);
        if (d < bestDistance) bestDistance = d;
      }

      print('[IrisService] vs ${person.fullName}: bestHD=${bestDistance.toStringAsFixed(4)} '
          '(${probes.length} probe × ${enrolled.length} enrolled)');

      if (bestDistance <= suggestThreshold) {
        results.add(IrisMatchResult(
          person: person,
          distance: bestDistance,
          matchType: bestDistance <= confirmThreshold
              ? MatchType.confirmed
              : MatchType.suggested,
        ));
      }
    }

    results.sort((a, b) => a.distance.compareTo(b.distance));

    if (results.isNotEmpty) {
      print('[IrisService] Best: ${results.first.person.fullName} '
          'HD=${results.first.distance.toStringAsFixed(3)} '
          '(${results.first.matchType.name})');
    } else {
      print('[IrisService] No candidates found');
    }

    return results;
  }

  Future<IrisMatchResult?> findMatch(List<double> template) async {
    final candidates = await findCandidates(template);
    if (candidates.isEmpty) return null;
    final best = candidates.first;
    if (best.matchType == MatchType.confirmed) return best;
    return null;
  }

  // ─── 12. QUALITY SCORING SYSTEM ──────────────────────────────────────

  ScoredFrame? scoreFrame(cv.Mat grayFrame) {
    final (:gray, :scale) = _preprocessGray(grayFrame);

    final segmentation = segmentIris(gray);
    if (segmentation == null) {
      gray.dispose();
      return null;
    }

    final sharpness = _scoreSharpness(gray, segmentation);
    final occlusion = _scoreOcclusion(gray, segmentation);
    final specular = _scoreSpecular(gray, segmentation);
    final centering = _scoreCentering(gray, segmentation);
    final resolution = _scoreResolution(segmentation);

    final quality = FrameQualityScore(
      sharpness: sharpness,
      occlusion: occlusion,
      specular: specular,
      centering: centering,
      resolution: resolution,
    );

    print('[ScoreFrame] Quality: sharp=${sharpness.toStringAsFixed(1)} occl=${occlusion.toStringAsFixed(1)} '
        'spec=${specular.toStringAsFixed(1)} center=${centering.toStringAsFixed(1)} '
        'res=${resolution.toStringAsFixed(1)} => composite=${quality.composite.toStringAsFixed(1)}');

    return ScoredFrame(
      grayImage: gray,
      segmentation: segmentation,
      quality: quality,
    );
  }

  double _scoreSharpness(cv.Mat gray, IrisSegmentation seg) {
    final roi = _extractIrisROI(gray, seg);
    final sharpness = _measureSharpness(roi);
    roi.dispose();
    return ((sharpness - 30.0) / (200.0 - 30.0) * 100.0).clamp(0.0, 100.0);
  }

  double _scoreOcclusion(cv.Mat gray, IrisSegmentation seg) {
    final normalized = normalizeIris(gray, seg);
    final cropped = normalized.region(
      cv.Rect(0, _skipRows, _angularRes, _cropRows),
    );
    normalized.dispose();

    final mask = _generateNoiseMask(cropped);
    cropped.dispose();

    int validCount = 0;
    for (final v in mask) {
      if (v) validCount++;
    }
    return (validCount / mask.length * 100.0).clamp(0.0, 100.0);
  }

  double _scoreSpecular(cv.Mat gray, IrisSegmentation seg) {
    final roi = _extractIrisROI(gray, seg);
    final total = roi.rows * roi.cols;
    if (total == 0) {
      roi.dispose();
      return 0.0;
    }
    // Count bright (specular) pixels with a thresholded mask instead of a
    // per-pixel loop (far faster on the frame stream).
    final (_, bright) = cv.threshold(roi, 230, 255, cv.THRESH_BINARY);
    final brightPixels = cv.countNonZero(bright);
    bright.dispose();
    roi.dispose();

    final brightFraction = brightPixels / total;
    return ((0.15 - brightFraction) / (0.15 - 0.01) * 100.0).clamp(0.0, 100.0);
  }

  double _scoreCentering(cv.Mat gray, IrisSegmentation seg) {
    final imgCenterX = gray.cols / 2.0;
    final imgCenterY = gray.rows / 2.0;
    final dx = (seg.irisX - imgCenterX).abs();
    final dy = (seg.irisY - imgCenterY).abs();
    final dist = sqrt(dx * dx + dy * dy);
    final maxDist = gray.cols * 0.3;
    return ((1.0 - dist / maxDist) * 100.0).clamp(0.0, 100.0);
  }

  double _scoreResolution(IrisSegmentation seg) {
    return ((seg.irisR - 40.0) / (100.0 - 40.0) * 100.0).clamp(0.0, 100.0);
  }

  /// Selects the best frames from a burst. Requires an independent sharpness
  /// floor in ADDITION to the composite score, so a very blurry frame can never
  /// pass on the strength of good centering/resolution alone.
  List<ScoredFrame> selectBestFrames(
    List<ScoredFrame> frames, {
    int maxFrames = 5,
    double minScore = 50.0,
    double minSharpness = 35.0,
  }) {
    final filtered = frames
        .where((f) => f.quality.composite >= minScore && f.quality.sharpness >= minSharpness)
        .toList();
    filtered.sort((a, b) => b.quality.composite.compareTo(a.quality.composite));
    return filtered.take(maxFrames).toList();
  }

  /// Processes burst frames into templates.
  ///
  /// Enrollment targets 3 templates; verification now also keeps up to 3 so the
  /// matcher can fuse scores across probe templates. Templates inconsistent with
  /// the best-quality one (HD > 0.34) are discarded.
  Future<BurstResult?> processBurstFrames(
    List<ScoredFrame> frames, {
    ScanMode mode = ScanMode.verification,
  }) async {
    final targetTemplates = 3;
    final minScore = mode == ScanMode.enrollment ? 55.0 : 45.0;

    final bestFrames = selectBestFrames(frames, maxFrames: 5, minScore: minScore);
    if (bestFrames.isEmpty) {
      print('[IrisService] Burst: no frames passed quality filter');
      return null;
    }

    print('[IrisService] Burst: ${bestFrames.length} frames passed quality '
        '(best=${bestFrames.first.quality.composite.toStringAsFixed(1)})');

    final templates = <List<double>>[];
    ScoredFrame? bestSavedFrame;
    double bestScore = 0;

    for (final frame in bestFrames) {
      final normalized = normalizeIris(frame.grayImage, frame.segmentation);
      final template = encodeIris(normalized);
      normalized.dispose();

      if (template != null) {
        templates.add(template);
        if (frame.quality.composite > bestScore) {
          bestScore = frame.quality.composite;
          bestSavedFrame = frame;
        }
      }
    }

    if (templates.isEmpty) {
      print('[IrisService] Burst: no valid templates encoded');
      return null;
    }

    // Consistency check against the best-quality template (first, since
    // bestFrames is sorted by composite score).
    if (templates.length > 1) {
      final reference = templates.first;
      final consistent = <List<double>>[reference];
      for (int i = 1; i < templates.length; i++) {
        final hd = compareTemplates(reference, templates[i]);
        if (hd <= 0.34) {
          consistent.add(templates[i]);
        } else {
          print('[IrisService] Burst: template $i discarded (HD=${hd.toStringAsFixed(3)} vs reference)');
        }
      }
      templates
        ..clear()
        ..addAll(consistent);
    }

    final finalTemplates = templates.take(targetTemplates).toList();

    String? savedImagePath;
    if (bestSavedFrame != null) {
      try {
        final directory = await getApplicationDocumentsDirectory();
        final irisDir = Directory('${directory.path}/iris_images');
        if (!await irisDir.exists()) {
          await irisDir.create(recursive: true);
        }
        final uuid = const Uuid().v4();
        savedImagePath = '${irisDir.path}/$uuid.png';
        cv.imwrite(savedImagePath, bestSavedFrame.grayImage);
      } catch (e) {
        print('[IrisService] Burst: failed to save image: $e');
      }
    }

    print('[IrisService] Burst result: ${finalTemplates.length} templates, '
        'bestScore=${bestScore.toStringAsFixed(1)}');

    return BurstResult(
      templates: finalTemplates,
      savedImagePath: savedImagePath,
      bestQualityScore: bestScore,
    );
  }
}

// ─── DATA CLASSES ──────────────────────────────────────────────────────

class _Circle {
  final double x;
  final double y;
  final double r;
  const _Circle(this.x, this.y, this.r);
}

class IrisSegmentation {
  final double pupilX;
  final double pupilY;
  final double pupilR;
  final double irisX;
  final double irisY;
  final double irisR;

  const IrisSegmentation({
    required this.pupilX,
    required this.pupilY,
    required this.pupilR,
    required this.irisX,
    required this.irisY,
    required this.irisR,
  });
}

class IrisProcessingResult {
  final List<double> template;
  final IrisSegmentation segmentation;

  const IrisProcessingResult({
    required this.template,
    required this.segmentation,
  });
}

enum MatchType { confirmed, suggested }

class IrisMatchResult {
  final Person person;
  final double distance;
  final MatchType matchType;

  const IrisMatchResult({
    required this.person,
    required this.distance,
    required this.matchType,
  });

  /// Confidence as a fraction (for display). Higher = better match.
  double get confidence => 1.0 - distance;
}

enum IrisDetectionStatus { notFound, tooFar, tooClose, notCentered, tooBlurry, ready }

enum ScanMode { enrollment, verification }

class FrameQualityScore {
  final double sharpness;   // 0-100, weight 40%
  final double occlusion;   // 0-100, weight 25%
  final double specular;    // 0-100, weight 15%
  final double centering;   // 0-100, weight 10%
  final double resolution;  // 0-100, weight 10%

  const FrameQualityScore({
    required this.sharpness,
    required this.occlusion,
    required this.specular,
    required this.centering,
    required this.resolution,
  });

  double get composite =>
      sharpness * 0.40 +
      occlusion * 0.25 +
      specular * 0.15 +
      centering * 0.10 +
      resolution * 0.10;
}

class ScoredFrame {
  final cv.Mat grayImage; // 640w preprocessed single channel — caller must dispose
  final IrisSegmentation segmentation;
  final FrameQualityScore quality;

  ScoredFrame({
    required this.grayImage,
    required this.segmentation,
    required this.quality,
  });

  void dispose() => grayImage.dispose();
}

class BurstResult {
  final List<List<double>> templates;
  final String? savedImagePath;
  final double bestQualityScore;

  const BurstResult({
    required this.templates,
    this.savedImagePath,
    required this.bestQualityScore,
  });
}

class IrisDetectionResult {
  final IrisDetectionStatus status;
  final IrisSegmentation? segmentation;

  const IrisDetectionResult({required this.status, this.segmentation});
}
