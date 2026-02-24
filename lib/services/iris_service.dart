import 'dart:io';
import 'dart:math';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import '../models/person.dart';
import 'database_service.dart';

/// Complete iris recognition pipeline using Daugman's IrisCode.
///
/// Pipeline:
/// 1. Preprocessing – grayscale, resize, CLAHE
/// 2. Segmentation – pupil/iris boundaries via Hough circles
/// 3. Normalization – Daugman's rubber sheet model
/// 4. Encoding – Gabor filters → phase quantization → binary IrisCode + noise mask
/// 5. Matching – masked Hamming distance with rotation compensation
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

  // Sampling grid for IrisCode (on cropped image)
  static const _stepTheta = 8; // angular step → 32 columns
  static const _stepR = 6; // radial step → 8 rows
  static const _gridCols = 32; // _angularRes / _stepTheta
  static const _gridRows = 8; // _cropRows / _stepR
  static const _bitsPerSample = 2; // real + imaginary phase bits
  static const _bitsPerFilter = 512; // _gridCols * _gridRows * _bitsPerSample
  static const _codeBits = 4096; // _numFilters * _bitsPerFilter

  // Rotation compensation: ±4 column shifts (each = 11.25°, total ±45°)
  static const _maxRotationShift = 4;

  // Matching thresholds (Daugman recommends 0.26-0.28 for FAR ~10^-11)
  static const confirmThreshold = 0.27;
  static const suggestThreshold = 0.35;

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

  ({cv.Mat gray, double scale}) _preprocessImage(cv.Mat image) {
    const standardWidth = 640;
    final scale = standardWidth / image.cols;
    final newHeight = (image.rows * scale).round();
    final resized = cv.resize(image, (standardWidth, newHeight));

    final gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
    resized.dispose();

    final clahe = cv.CLAHE.create(2.0, (8, 8));
    final enhanced = clahe.apply(gray);
    gray.dispose();
    clahe.dispose();

    return (gray: enhanced, scale: scale);
  }

  /// Preprocessing variant for grayscale input (e.g. Y-plane from camera stream).
  /// Skips cvtColor since input is already grayscale. Resize + CLAHE.
  ({cv.Mat gray, double scale}) _preprocessGray(cv.Mat grayImage) {
    const standardWidth = 640;
    final scale = standardWidth / grayImage.cols;
    final newHeight = (grayImage.rows * scale).round();
    final resized = cv.resize(grayImage, (standardWidth, newHeight));

    final clahe = cv.CLAHE.create(2.0, (8, 8));
    final enhanced = clahe.apply(resized);
    resized.dispose();
    clahe.dispose();

    return (gray: enhanced, scale: scale);
  }

  // ─── 3. SEGMENTATION ───────────────────────────────────────────────

  IrisSegmentation? segmentIris(cv.Mat grayImage) {
    final blurred = cv.medianBlur(grayImage, 7);

    final pupilCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5,
      50,
      param1: 100,
      param2: 40,
      minRadius: 10,
      maxRadius: 80,
    );

    final irisCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5,
      100,
      param1: 80,
      param2: 35,
      minRadius: 60,
      maxRadius: 200,
    );

    blurred.dispose();

    if (pupilCircles.rows == 0 || irisCircles.rows == 0) {
      pupilCircles.dispose();
      irisCircles.dispose();
      return null;
    }

    final centerX = grayImage.cols / 2.0;
    final centerY = grayImage.rows / 2.0;

    final pupil = _selectBestCircle(pupilCircles, centerX, centerY);
    final iris = _selectBestCircle(irisCircles, centerX, centerY);

    pupilCircles.dispose();
    irisCircles.dispose();

    if (iris.radius <= pupil.radius) return null;

    // Validate pupil is inside iris
    final dx = (pupil.x - iris.x).abs();
    final dy = (pupil.y - iris.y).abs();
    if (dx + pupil.radius > iris.radius || dy + pupil.radius > iris.radius) {
      print('[IrisService] Segmentation rejected: pupil not contained in iris');
      return null;
    }

    // Validate physiological pupil/iris ratio (0.2 - 0.7)
    final ratio = pupil.radius / iris.radius;
    if (ratio < 0.2 || ratio > 0.7) {
      print('[IrisService] Segmentation rejected: pupil/iris ratio $ratio outside range');
      return null;
    }

    // Reject if iris is too small for reliable encoding
    if (iris.radius < 40) {
      print('[IrisService] Segmentation rejected: iris too small (r=${iris.radius})');
      return null;
    }

    print('[IrisService] Segmentation: pupil=(${pupil.x},${pupil.y},r=${pupil.radius}) '
        'iris=(${iris.x},${iris.y},r=${iris.radius}) ratio=${ratio.toStringAsFixed(2)}');

    return IrisSegmentation(
      pupilCenter: Point(pupil.x, pupil.y),
      pupilRadius: pupil.radius,
      irisCenter: Point(iris.x, iris.y),
      irisRadius: iris.radius,
    );
  }

  ({int x, int y, int radius}) _selectBestCircle(
    cv.Mat circles, double centerX, double centerY,
  ) {
    int bestIdx = 0;
    double bestDist = double.infinity;

    final numCircles = circles.cols;
    for (int i = 0; i < numCircles; i++) {
      final cx = circles.at<double>(0, i * 3);
      final cy = circles.at<double>(0, i * 3 + 1);
      final dist = (cx - centerX) * (cx - centerX) + (cy - centerY) * (cy - centerY);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }

    return (
      x: circles.at<double>(0, bestIdx * 3).toInt(),
      y: circles.at<double>(0, bestIdx * 3 + 1).toInt(),
      radius: circles.at<double>(0, bestIdx * 3 + 2).toInt(),
    );
  }

  // ─── 4. NORMALIZATION (DAUGMAN'S RUBBER SHEET MODEL) ────────────────

  cv.Mat normalizeIris(
    cv.Mat grayImage,
    IrisSegmentation seg, {
    int angularResolution = _angularRes,
    int radialResolution = _radialRes,
  }) {
    final normalized = cv.Mat.zeros(radialResolution, angularResolution, cv.MatType.CV_8UC1);

    for (int theta = 0; theta < angularResolution; theta++) {
      final angle = 2 * pi * theta / angularResolution;

      for (int r = 0; r < radialResolution; r++) {
        final ratio = r / radialResolution;

        final xPupil = seg.pupilCenter.x + seg.pupilRadius * cos(angle);
        final yPupil = seg.pupilCenter.y + seg.pupilRadius * sin(angle);

        final xIris = seg.irisCenter.x + seg.irisRadius * cos(angle);
        final yIris = seg.irisCenter.y + seg.irisRadius * sin(angle);

        final x = ((1 - ratio) * xPupil + ratio * xIris).round();
        final y = ((1 - ratio) * yPupil + ratio * yIris).round();

        if (x >= 0 && x < grayImage.cols && y >= 0 && y < grayImage.rows) {
          final pixel = grayImage.at<int>(y, x);
          normalized.set<int>(r, theta, pixel);
        }
      }
    }

    return normalized;
  }

  // ─── 5. GABOR KERNEL GENERATION ─────────────────────────────────────

  /// Creates an asymmetric 2D Gabor filter kernel (narrow radial, wide angular).
  ///
  /// Using a narrow radial dimension (5px) avoids border contamination on the
  /// 48-row cropped iris strip. The wide angular dimension (15px) captures
  /// the discriminative radial texture patterns (crypts, furrows).
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

  /// Generates a noise mask on the cropped iris strip.
  ///
  /// Low-variance or very dark blocks are marked as occluded (reflections,
  /// remaining eyelid fragments after crop, etc.).
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

        if (stdVal < 12.0 || meanVal < 25.0 || meanVal > 240.0) {
          mask[r * _gridCols + c] = false;
        }
      }
    }

    return mask;
  }

  // ─── 7. FEATURE ENCODING (DAUGMAN'S IRISCODE) ──────────────────────

  /// Encodes the normalized iris into a compact binary IrisCode with noise mask.
  ///
  /// Steps:
  /// 1. Apply CLAHE to normalize illumination on the iris strip
  /// 2. Crop eyelid rows (top/bottom 8 rows) → 256×48
  /// 3. Circularly pad angular dimension (BORDER_WRAP)
  /// 4. Apply 8 asymmetric Gabor filters (5×15 kernels)
  /// 5. Sample on coarse grid → 4096 bits total
  /// 6. Phase quantize + dead-zone masking
  ///
  /// Returns [code_bits..., mask_bits...] or null if quality too low.
  List<double>? encodeIris(cv.Mat normalizedIris) {
    // Step 1: CLAHE on normalized strip for consistent illumination
    final clahe = cv.CLAHE.create(2.0, (8, 8));
    final enhanced = clahe.apply(normalizedIris);
    clahe.dispose();

    // Step 2: Crop eyelid rows (top/bottom _skipRows)
    final cropped = enhanced.region(
      cv.Rect(0, _skipRows, _angularRes, _cropRows),
    );
    enhanced.dispose();

    // Step 3: Circular padding for angular dimension (avoids edge artifacts)
    final padW = _kCols ~/ 2; // 7
    final padH = _kRows ~/ 2; // 2
    final padded = cv.copyMakeBorder(
      cropped, padH, padH, padW, padW, cv.BORDER_WRAP,
    );

    // Convert to float64 for filter operations
    final src = padded.convertTo(cv.MatType.CV_64FC1);
    padded.dispose();

    // Generate noise mask on cropped (pre-padding) image
    final noiseMask = _generateNoiseMask(cropped);
    cropped.dispose();

    final iriscode = List<double>.filled(_codeBits, 0.0);
    final maskBits = List<double>.filled(_codeBits, 1.0);

    // First pass: compute all filter responses + track max magnitudes
    final allResponses = <List<(double, double)>>[];
    final maxMagnitudes = <double>[];

    for (final lambda in _wavelengths) {
      final sigma = lambda / 2;

      for (final theta in _orientations) {
        final kernelReal = _createGaborKernel(sigma, theta, lambda, 0);
        final kernelImag = _createGaborKernel(sigma, theta, lambda, pi / 2);

        final responseReal = cv.filter2D(src, cv.MatType.CV_64FC1.depth, kernelReal);
        final responseImag = cv.filter2D(src, cv.MatType.CV_64FC1.depth, kernelImag);

        final responses = <(double, double)>[];
        double maxMag = 0;

        // Sample from the valid (non-padded) region of the response
        for (int r = 0; r < _cropRows; r += _stepR) {
          for (int t = 0; t < _angularRes; t += _stepTheta) {
            final rv = responseReal.at<double>(r + padH, t + padW);
            final iv = responseImag.at<double>(r + padH, t + padW);
            responses.add((rv, iv));
            final mag = sqrt(rv * rv + iv * iv);
            if (mag > maxMag) maxMag = mag;
          }
        }

        allResponses.add(responses);
        maxMagnitudes.add(maxMag);

        kernelReal.dispose();
        kernelImag.dispose();
        responseReal.dispose();
        responseImag.dispose();
      }
    }

    src.dispose();

    // Second pass: phase quantize + dead-zone masking
    int filterIdx = 0;
    int bitIdx = 0;
    const deadZoneFraction = 0.12;

    for (final responses in allResponses) {
      final maxMag = maxMagnitudes[filterIdx];
      final deadZone = deadZoneFraction * maxMag;

      int sampleIdx = 0;
      for (int r = 0; r < _gridRows; r++) {
        for (int c = 0; c < _gridCols; c++) {
          final (rv, iv) = responses[sampleIdx];

          iriscode[bitIdx] = rv >= 0 ? 1.0 : 0.0;
          iriscode[bitIdx + 1] = iv >= 0 ? 1.0 : 0.0;

          final gridValid = noiseMask[r * _gridCols + c];
          final magnitude = sqrt(rv * rv + iv * iv);
          final magnitudeValid = magnitude >= deadZone;

          if (!gridValid || !magnitudeValid) {
            maskBits[bitIdx] = 0.0;
            maskBits[bitIdx + 1] = 0.0;
          }

          bitIdx += 2;
          sampleIdx++;
        }
      }
      filterIdx++;
    }

    // Quality check + per-filter diagnostics
    int validCount = 0;
    for (final m in maskBits) {
      if (m == 1.0) validCount++;
    }
    final validFraction = validCount / maskBits.length;

    // Per-filter valid bit breakdown
    final filterNames = <String>[];
    for (final lambda in _wavelengths) {
      for (final theta in _orientations) {
        filterNames.add('λ${lambda.toInt()}θ${(theta * 180 / pi).toInt()}');
      }
    }
    final perFilterLog = StringBuffer();
    for (int f = 0; f < _numFilters; f++) {
      int fValid = 0;
      final base = f * _bitsPerFilter;
      for (int i = base; i < base + _bitsPerFilter; i++) {
        if (maskBits[i] == 1.0) fValid++;
      }
      perFilterLog.write('${filterNames[f]}:${(fValid/_bitsPerFilter*100).toStringAsFixed(0)}% ');
    }

    // Noise mask vs dead zone breakdown
    int noiseMasked = 0, deadZoned = 0;
    for (final v in noiseMask) {
      if (!v) noiseMasked++;
    }
    // Count dead-zoned bits (valid in noise mask but masked in final)
    for (int f = 0; f < _numFilters; f++) {
      final base = f * _bitsPerFilter;
      int sIdx = 0;
      for (int r = 0; r < _gridRows; r++) {
        for (int c = 0; c < _gridCols; c++) {
          final gridValid = noiseMask[r * _gridCols + c];
          if (gridValid && maskBits[base + sIdx * 2] == 0.0) {
            deadZoned++;
          }
          sIdx++;
        }
      }
    }

    print('[Encode] IrisCode: ${iriscode.length} bits, '
        '${(validFraction * 100).toStringAsFixed(0)}% valid '
        '(noise-masked: $noiseMasked/${_gridRows * _gridCols} cells, '
        'dead-zoned: $deadZoned samples)');
    print('[Encode] Per-filter valid: $perFilterLog');

    if (validFraction < 0.55) {
      print('[Encode] Rejected: too few valid bits');
      return null;
    }

    return [...iriscode, ...maskBits];
  }

  // ─── 8. IMAGE QUALITY ASSESSMENT ───────────────────────────────────

  static const _minSharpness = 50.0;

  /// Measures image sharpness using Laplacian variance.
  /// Higher values = sharper image. Blurry < 50, acceptable 50-150, sharp > 150.
  double _measureSharpness(cv.Mat grayRegion) {
    final lap = cv.laplacian(grayRegion, cv.MatType.CV_16SC1.depth);
    final meanStd = cv.meanStdDev(lap);
    final stddev = meanStd.$2.val1;
    lap.dispose();
    return stddev * stddev;
  }

  /// Extracts a square ROI around the iris for quality assessment.
  cv.Mat _extractIrisROI(cv.Mat gray, IrisSegmentation seg) {
    final x = max(0, seg.irisCenter.x - seg.irisRadius);
    final y = max(0, seg.irisCenter.y - seg.irisRadius);
    final w = min(seg.irisRadius * 2, gray.cols - x);
    final h = min(seg.irisRadius * 2, gray.rows - y);
    return gray.region(cv.Rect(x, y, w, h));
  }

  // ─── 9. FULL PIPELINE ──────────────────────────────────────────────

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

    // Quality check: reject blurry images before encoding
    final irisROI = _extractIrisROI(gray, segmentation);
    final sharpness = _measureSharpness(irisROI);
    irisROI.dispose();
    print('[IrisService] Sharpness: ${sharpness.toStringAsFixed(1)}');

    if (sharpness < _minSharpness) {
      gray.dispose();
      print('[IrisService] Rejected: too blurry (sharpness=$sharpness)');
      return null;
    }

    print('[Pipeline] Segmentation: pupil=(${segmentation.pupilCenter.x},${segmentation.pupilCenter.y} r=${segmentation.pupilRadius}) '
        'iris=(${segmentation.irisCenter.x},${segmentation.irisCenter.y} r=${segmentation.irisRadius}) '
        'ratio=${(segmentation.pupilRadius / segmentation.irisRadius).toStringAsFixed(2)}');
    print('[Pipeline] Image size: ${gray.cols}x${gray.rows}, '
        'iris covers ${(segmentation.irisRadius * 2 / gray.cols * 100).toStringAsFixed(0)}% of width');

    final normalized = normalizeIris(gray, segmentation);
    final template = encodeIris(normalized);

    gray.dispose();
    normalized.dispose();

    if (template == null) {
      print('[Pipeline] Encoding failed (quality too low)');
      return null;
    }

    print('[Pipeline] Template: ${template.length} values '
        '(${template.length ~/ 2} code + ${template.length ~/ 2} mask)');

    return IrisProcessingResult(
      template: template,
      segmentation: segmentation,
    );
  }

  Future<List<double>?> generateIrisTemplate(String imagePath) async {
    final result = await processIrisImage(imagePath);
    return result?.template;
  }

  // ─── 9. QUICK DETECTION (for live preview) ─────────────────────────

  IrisDetectionResult quickDetectIris(cv.Mat grayFrame) {
    const previewWidth = 320;
    final scale = previewWidth / grayFrame.cols;
    final previewHeight = (grayFrame.rows * scale).round();
    final resized = cv.resize(grayFrame, (previewWidth, previewHeight));

    final blurred = cv.medianBlur(resized, 7);

    final pupilCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5, 25,
      param1: 100, param2: 40,
      minRadius: 5, maxRadius: 40,
    );

    final irisCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5, 50,
      param1: 80, param2: 35,
      minRadius: 30, maxRadius: 100,
    );

    blurred.dispose();

    if (pupilCircles.rows == 0 || irisCircles.rows == 0) {
      pupilCircles.dispose();
      irisCircles.dispose();
      resized.dispose();
      return const IrisDetectionResult(status: IrisDetectionStatus.notFound);
    }

    final centerX = previewWidth / 2.0;
    final centerY = previewHeight / 2.0;

    final pupil = _selectBestCircle(pupilCircles, centerX, centerY);
    final iris = _selectBestCircle(irisCircles, centerX, centerY);

    pupilCircles.dispose();
    irisCircles.dispose();

    if (iris.radius <= pupil.radius) {
      resized.dispose();
      return const IrisDetectionResult(status: IrisDetectionStatus.notFound);
    }

    final seg = IrisSegmentation(
      pupilCenter: Point(pupil.x, pupil.y),
      pupilRadius: pupil.radius,
      irisCenter: Point(iris.x, iris.y),
      irisRadius: iris.radius,
    );

    IrisDetectionStatus status;
    if (iris.radius < 40) {
      status = IrisDetectionStatus.tooFar;
    } else if (iris.radius > 90) {
      status = IrisDetectionStatus.tooClose;
    } else {
      final dx = (iris.x - centerX).abs();
      final dy = (iris.y - centerY).abs();
      final maxOffset = previewWidth * 0.30;
      if (dx > maxOffset || dy > maxOffset) {
        status = IrisDetectionStatus.notCentered;
      } else {
        // Size and centering OK — check sharpness before declaring ready
        final roiX = max(0, iris.x - iris.radius);
        final roiY = max(0, iris.y - iris.radius);
        final roiW = min(iris.radius * 2, previewWidth - roiX);
        final roiH = min(iris.radius * 2, previewHeight - roiY);
        if (roiW > 0 && roiH > 0) {
          final roi = resized.region(cv.Rect(roiX, roiY, roiW, roiH));
          final lap = cv.laplacian(roi, cv.MatType.CV_16SC1.depth);
          final meanStd = cv.meanStdDev(lap);
          final sharpness = meanStd.$2.val1 * meanStd.$2.val1;
          lap.dispose();
          roi.dispose();
          if (sharpness < 30.0) {
            status = IrisDetectionStatus.tooBlurry;
          } else {
            status = IrisDetectionStatus.ready;
          }
        } else {
          status = IrisDetectionStatus.ready;
        }
      }
    }

    resized.dispose();
    return IrisDetectionResult(status: status, segmentation: seg);
  }

  // ─── 10. MATCHING (MASKED HAMMING WITH ROTATION COMPENSATION) ──────

  /// Compares two IrisCodes using masked fractional Hamming distance
  /// with circular bit-shifting for rotation invariance.
  ///
  /// Template format: [code_bits..., mask_bits...] (each half = _codeBits length)
  ///
  /// Returns the minimum Hamming distance across ±_maxRotationShift column shifts.
  /// - Same iris: ~0.20–0.30
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

    if (verbose) {
      // Mask statistics
      int valid1 = 0, valid2 = 0, mutualValid = 0;
      for (int i = 0; i < codeLen; i++) {
        if (mask1[i] == 1.0) valid1++;
        if (mask2[i] == 1.0) valid2++;
        if (mask1[i] == 1.0 && mask2[i] == 1.0) mutualValid++;
      }
      print('[Match] Template1 valid: $valid1/$codeLen (${(valid1/codeLen*100).toStringAsFixed(0)}%)');
      print('[Match] Template2 valid: $valid2/$codeLen (${(valid2/codeLen*100).toStringAsFixed(0)}%)');
      print('[Match] Mutual valid (no shift): $mutualValid/$codeLen (${(mutualValid/codeLen*100).toStringAsFixed(0)}%)');

      // Bit distribution (should be ~50/50 for good templates)
      int ones1 = 0, ones2 = 0;
      for (int i = 0; i < codeLen; i++) {
        if (code1[i] == 1.0) ones1++;
        if (code2[i] == 1.0) ones2++;
      }
      print('[Match] Bit distribution T1: ${(ones1/codeLen*100).toStringAsFixed(1)}% ones');
      print('[Match] Bit distribution T2: ${(ones2/codeLen*100).toStringAsFixed(1)}% ones');
    }

    double bestHD = 1.0;
    int bestShift = 0;
    final shiftResults = <String>[];

    for (int shift = -_maxRotationShift; shift <= _maxRotationShift; shift++) {
      final hd = _hammingWithShift(code1, mask1, code2, mask2, shift);
      if (verbose) shiftResults.add('shift=$shift: HD=${hd.toStringAsFixed(4)}');
      if (hd < bestHD) {
        bestHD = hd;
        bestShift = shift;
      }
    }

    if (verbose) {
      print('[Match] All shifts: ${shiftResults.join(', ')}');
      print('[Match] Best: shift=$bestShift HD=${bestHD.toStringAsFixed(4)}');

      // Per-filter breakdown at best shift
      _logPerFilterHD(code1, mask1, code2, mask2, bestShift);
    }

    return bestHD;
  }

  /// Logs per-filter Hamming distance breakdown for diagnostics.
  void _logPerFilterHD(
    List<double> code1, List<double> mask1,
    List<double> code2, List<double> mask2,
    int shift,
  ) {
    final filterNames = <String>[];
    for (final lambda in _wavelengths) {
      for (final theta in _orientations) {
        filterNames.add('λ=${lambda.toInt()} θ=${(theta * 180 / pi).toInt()}°');
      }
    }

    for (int f = 0; f < _numFilters; f++) {
      final filterBase = f * _bitsPerFilter;
      int valid = 0, diffs = 0;

      for (int r = 0; r < _gridRows; r++) {
        final rowBase = filterBase + r * _gridCols * _bitsPerSample;
        for (int c = 0; c < _gridCols; c++) {
          final c2 = ((c + shift) % _gridCols + _gridCols) % _gridCols;
          for (int b = 0; b < _bitsPerSample; b++) {
            final idx1 = rowBase + c * _bitsPerSample + b;
            final idx2 = rowBase + c2 * _bitsPerSample + b;
            if (mask1[idx1] == 1.0 && mask2[idx2] == 1.0) {
              valid++;
              if (code1[idx1] != code2[idx2]) diffs++;
            }
          }
        }
      }

      final hd = valid > 0 ? diffs / valid : 1.0;
      print('[Match] Filter $f (${filterNames[f]}): HD=${hd.toStringAsFixed(3)} '
          'valid=$valid/$_bitsPerFilter diffs=$diffs');
    }
  }

  /// Calculates masked Hamming distance with a circular column shift applied
  /// to code2/mask2 in the angular dimension.
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
          // Circular shift on the angular (column) dimension
          final c2 = ((c + shift) % _gridCols + _gridCols) % _gridCols;

          for (int b = 0; b < _bitsPerSample; b++) {
            final idx1 = rowBase + c * _bitsPerSample + b;
            final idx2 = rowBase + c2 * _bitsPerSample + b;

            // Both bits must be valid (unmasked)
            if (mask1[idx1] == 1.0 && mask2[idx2] == 1.0) {
              validBits++;
              if (code1[idx1] != code2[idx2]) differences++;
            }
          }
        }
      }
    }

    if (validBits == 0) return 1.0;
    // Require at least 60% mutually valid bits for reliable comparison.
    // With fewer bits, HD variance is too high and impostors can match by chance.
    final minRequired = (_codeBits * 0.60).round();
    if (validBits < minRequired) return 1.0;
    return differences / validBits;
  }

  /// Searches all registered persons and returns candidates sorted by distance.
  ///
  /// Compares against ALL templates per person, using the best (minimum) HD.
  /// Three match zones:
  /// - HD ≤ 0.27: confirmed match (high confidence)
  /// - 0.27 < HD ≤ 0.35: suggested match (needs confirmation)
  /// - HD > 0.35: no match
  Future<List<IrisMatchResult>> findCandidates(List<double> template) async {
    final persons = await _dbService.getPersonsWithIrisTemplate();
    final results = <IrisMatchResult>[];

    for (final person in persons) {
      if (person.irisTemplates == null || person.irisTemplates!.isEmpty) continue;

      // Compare against all templates, keep best (minimum) HD
      double bestDistance = 1.0;
      int bestTemplateIdx = 0;
      for (int i = 0; i < person.irisTemplates!.length; i++) {
        final distance = compareTemplates(template, person.irisTemplates![i]);
        print('[IrisService] vs ${person.fullName} template[$i]: HD=${distance.toStringAsFixed(4)}');
        if (distance < bestDistance) {
          bestDistance = distance;
          bestTemplateIdx = i;
        }
      }

      // Verbose comparison for the best template
      print('[IrisService] vs ${person.fullName}: bestHD=${bestDistance.toStringAsFixed(4)} '
          '(template[$bestTemplateIdx] of ${person.irisTemplates!.length})');
      compareTemplates(template, person.irisTemplates![bestTemplateIdx], verbose: true);

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

    // Sort by distance (best match first)
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

  /// Backward-compatible wrapper: returns the best confirmed match or null.
  Future<IrisMatchResult?> findMatch(List<double> template) async {
    final candidates = await findCandidates(template);
    if (candidates.isEmpty) return null;

    final best = candidates.first;
    if (best.matchType == MatchType.confirmed) return best;

    return null;
  }

  // ─── 12. QUALITY SCORING SYSTEM ──────────────────────────────────────

  /// Scores a single grayscale frame from the camera stream.
  /// Returns null if no iris is detected in the frame.
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

    print('[ScoreFrame] Seg: pupil=(${segmentation.pupilCenter.x},${segmentation.pupilCenter.y} r=${segmentation.pupilRadius}) '
        'iris=(${segmentation.irisCenter.x},${segmentation.irisCenter.y} r=${segmentation.irisRadius}) '
        'ratio=${(segmentation.pupilRadius / segmentation.irisRadius).toStringAsFixed(2)}');
    print('[ScoreFrame] Quality: sharp=${sharpness.toStringAsFixed(1)} occl=${occlusion.toStringAsFixed(1)} '
        'spec=${specular.toStringAsFixed(1)} center=${centering.toStringAsFixed(1)} '
        'res=${resolution.toStringAsFixed(1)} => composite=${quality.composite.toStringAsFixed(1)}');

    return ScoredFrame(
      grayImage: gray, // caller must dispose
      segmentation: segmentation,
      quality: quality,
    );
  }

  /// Sharpness score (0-100). Uses Laplacian variance on iris ROI.
  /// Maps: <30 → 0, >200 → 100
  double _scoreSharpness(cv.Mat gray, IrisSegmentation seg) {
    final roi = _extractIrisROI(gray, seg);
    final sharpness = _measureSharpness(roi);
    roi.dispose();
    return ((sharpness - 30.0) / (200.0 - 30.0) * 100.0).clamp(0.0, 100.0);
  }

  /// Occlusion score (0-100). Normalizes iris and checks noise mask valid fraction.
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

  /// Specular reflection score (0-100). Counts bright pixels (>230) in iris annulus.
  /// <1% bright → 100, >15% bright → 0
  double _scoreSpecular(cv.Mat gray, IrisSegmentation seg) {
    final roi = _extractIrisROI(gray, seg);
    int totalPixels = 0;
    int brightPixels = 0;

    for (int y = 0; y < roi.rows; y++) {
      for (int x = 0; x < roi.cols; x++) {
        final pixel = roi.at<int>(y, x);
        totalPixels++;
        if (pixel > 230) brightPixels++;
      }
    }
    roi.dispose();

    if (totalPixels == 0) return 0.0;
    final brightFraction = brightPixels / totalPixels;
    // <1% → 100, >15% → 0
    return ((0.15 - brightFraction) / (0.15 - 0.01) * 100.0).clamp(0.0, 100.0);
  }

  /// Centering score (0-100). Distance of iris center to image center.
  double _scoreCentering(cv.Mat gray, IrisSegmentation seg) {
    final imgCenterX = gray.cols / 2.0;
    final imgCenterY = gray.rows / 2.0;
    final dx = (seg.irisCenter.x - imgCenterX).abs();
    final dy = (seg.irisCenter.y - imgCenterY).abs();
    final dist = sqrt(dx * dx + dy * dy);
    final maxDist = gray.cols * 0.3; // 30% of width = worst acceptable
    return ((1.0 - dist / maxDist) * 100.0).clamp(0.0, 100.0);
  }

  /// Resolution score (0-100). Based on iris radius in pixels.
  /// 40px → 0, 100px+ → 100
  double _scoreResolution(IrisSegmentation seg) {
    return ((seg.irisRadius - 40.0) / (100.0 - 40.0) * 100.0).clamp(0.0, 100.0);
  }

  /// Selects the best frames from a burst based on composite quality score.
  List<ScoredFrame> selectBestFrames(
    List<ScoredFrame> frames, {
    int maxFrames = 5,
    double minScore = 50.0,
  }) {
    final filtered = frames.where((f) => f.quality.composite >= minScore).toList();
    filtered.sort((a, b) => b.quality.composite.compareTo(a.quality.composite));
    return filtered.take(maxFrames).toList();
  }

  /// Processes burst frames into templates. Returns null if quality is too low.
  ///
  /// For enrollment: targets 3 templates. For verification: targets 1.
  /// Performs consistency check — discards templates with HD > 0.30 vs the best.
  Future<BurstResult?> processBurstFrames(
    List<ScoredFrame> frames, {
    ScanMode mode = ScanMode.verification,
  }) async {
    final targetTemplates = mode == ScanMode.enrollment ? 3 : 1;
    final minScore = mode == ScanMode.enrollment ? 60.0 : 50.0;

    final bestFrames = selectBestFrames(frames, maxFrames: 5, minScore: minScore);
    if (bestFrames.isEmpty) {
      print('[IrisService] Burst: no frames passed quality filter');
      return null;
    }

    print('[IrisService] Burst: ${bestFrames.length} frames passed quality '
        '(best=${bestFrames.first.quality.composite.toStringAsFixed(1)})');

    // Encode templates from best frames
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

    // Consistency check: compare each template against the first, discard outliers
    if (templates.length > 1) {
      final reference = templates.first;
      final consistent = <List<double>>[reference];
      for (int i = 1; i < templates.length; i++) {
        final hd = compareTemplates(reference, templates[i]);
        if (hd <= 0.30) {
          consistent.add(templates[i]);
        } else {
          print('[IrisService] Burst: template $i discarded (HD=$hd vs reference)');
        }
      }
      templates
        ..clear()
        ..addAll(consistent);
    }

    if (templates.isEmpty) {
      print('[IrisService] Burst: all templates inconsistent');
      return null;
    }

    // Keep only the target number of templates
    final finalTemplates = templates.take(targetTemplates).toList();

    // Save best frame as image
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

class Point {
  final int x;
  final int y;
  const Point(this.x, this.y);
}

class IrisSegmentation {
  final Point pupilCenter;
  final int pupilRadius;
  final Point irisCenter;
  final int irisRadius;

  const IrisSegmentation({
    required this.pupilCenter,
    required this.pupilRadius,
    required this.irisCenter,
    required this.irisRadius,
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

  /// Confidence as percentage (for display). Higher = better match.
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
  final cv.Mat grayImage; // 640w preprocessed — caller must dispose
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
