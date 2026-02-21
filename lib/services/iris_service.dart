import 'dart:io';
import 'dart:math';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import '../models/person.dart';
import 'database_service.dart';

/// Complete iris recognition pipeline using OpenCV.
///
/// The pipeline follows the standard approach for iris biometrics:
///
/// 1. **Preprocessing** – Convert to grayscale, reduce noise
/// 2. **Segmentation** – Find pupil and iris boundaries using Hough circles
/// 3. **Normalization** – Unwrap the iris ring into a rectangular strip
///    (Daugman's rubber sheet model)
/// 4. **Encoding** – Apply Gabor-like filters to extract a feature vector
/// 5. **Matching** – Compare feature vectors using normalized distance
class IrisService {
  final DatabaseService _dbService;

  IrisService(this._dbService);

  // ─── 1. IMAGE STORAGE ───────────────────────────────────────────────

  /// Saves an eye image captured from the camera to local storage.
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

  /// Resizes image to a standard width and applies CLAHE for consistent
  /// illumination across different captures.
  ///
  /// Returns the preprocessed grayscale image and the scale factor used.
  ({cv.Mat gray, double scale}) _preprocessImage(cv.Mat image) {
    // Resize to standard width so Hough circle radius parameters work
    // consistently regardless of camera resolution
    const standardWidth = 640;
    final scale = standardWidth / image.cols;
    final newHeight = (image.rows * scale).round();
    final resized = cv.resize(image, (standardWidth, newHeight));

    final gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY);
    resized.dispose();

    // CLAHE normalizes lighting so captures under different conditions
    // produce more consistent segmentation results
    final clahe = cv.CLAHE.create(2.0, (8, 8));
    final enhanced = clahe.apply(gray);
    gray.dispose();
    clahe.dispose();

    return (gray: enhanced, scale: scale);
  }

  // ─── 3. SEGMENTATION ───────────────────────────────────────────────

  /// Detects the pupil (inner circle) and iris (outer circle) boundaries.
  ///
  /// Expects a preprocessed grayscale image (already resized and CLAHE-enhanced).
  /// Returns null if detection fails (image quality too low, eye not found).
  IrisSegmentation? segmentIris(cv.Mat grayImage) {
    // Apply median blur to reduce noise while preserving edges
    final blurred = cv.medianBlur(grayImage, 7);

    // --- Detect pupil (smaller, darker circle) ---
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

    // --- Detect iris boundary (larger circle) ---
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

    // Select the best circle from candidates instead of taking the first one.
    // Prefer circles closest to the image center (the user aligns the eye there).
    final centerX = grayImage.cols / 2.0;
    final centerY = grayImage.rows / 2.0;

    final pupil = _selectBestCircle(pupilCircles, centerX, centerY);
    final iris = _selectBestCircle(irisCircles, centerX, centerY);

    pupilCircles.dispose();
    irisCircles.dispose();

    // Basic sanity check: iris must be larger than pupil
    if (iris.radius <= pupil.radius) return null;

    print('[IrisService] Segmentation: pupil=(${pupil.x},${pupil.y},r=${pupil.radius}) '
        'iris=(${iris.x},${iris.y},r=${iris.radius})');

    return IrisSegmentation(
      pupilCenter: Point(pupil.x, pupil.y),
      pupilRadius: pupil.radius,
      irisCenter: Point(iris.x, iris.y),
      irisRadius: iris.radius,
    );
  }

  /// Picks the circle closest to the given center from HoughCircles results.
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

  /// Unwraps the donut-shaped iris region into a fixed-size rectangular strip.
  ///
  /// Imagine "unrolling" the iris ring like a sheet of paper:
  /// - The angular dimension (0° to 360°) becomes the horizontal axis
  /// - The radial dimension (pupil edge to iris edge) becomes the vertical axis
  ///
  /// This normalization handles:
  /// - Different iris sizes (near/far from camera)
  /// - Pupil dilation changes
  /// - Non-concentric pupil/iris (pupil is rarely perfectly centered in iris)
  ///
  /// Output size: [angularResolution] x [radialResolution] grayscale image.
  cv.Mat normalizeIris(
    cv.Mat grayImage,
    IrisSegmentation seg, {
    int angularResolution = 256, // number of angular samples (width)
    int radialResolution = 64, // number of radial samples (height)
  }) {
    final normalized = cv.Mat.zeros(radialResolution, angularResolution, cv.MatType.CV_8UC1);

    for (int theta = 0; theta < angularResolution; theta++) {
      final angle = 2 * pi * theta / angularResolution;

      for (int r = 0; r < radialResolution; r++) {
        // Interpolate between pupil boundary and iris boundary
        final ratio = r / radialResolution;

        // Point on pupil boundary at this angle
        final xPupil = seg.pupilCenter.x + seg.pupilRadius * cos(angle);
        final yPupil = seg.pupilCenter.y + seg.pupilRadius * sin(angle);

        // Point on iris boundary at this angle
        final xIris = seg.irisCenter.x + seg.irisRadius * cos(angle);
        final yIris = seg.irisCenter.y + seg.irisRadius * sin(angle);

        // Linear interpolation between pupil and iris boundaries
        final x = ((1 - ratio) * xPupil + ratio * xIris).round();
        final y = ((1 - ratio) * yPupil + ratio * yIris).round();

        // Bounds check
        if (x >= 0 && x < grayImage.cols && y >= 0 && y < grayImage.rows) {
          final pixel = grayImage.at<int>(y, x);
          normalized.set<int>(r, theta, pixel);
        }
      }
    }

    return normalized;
  }

  // ─── 5. FEATURE ENCODING ────────────────────────────────────────────

  /// Encodes the normalized iris image into a feature vector.
  ///
  /// We apply multiple Gabor-like filters at different frequencies to capture
  /// the iris texture patterns. The response magnitudes form our feature vector.
  ///
  /// In a production system, you would use proper 2D Gabor wavelets and extract
  /// phase information (as in Daugman's original IrisCode). This implementation
  /// uses a simplified approach that still captures discriminative texture features:
  ///
  /// 1. Divide the normalized image into blocks
  /// 2. For each block, compute mean and standard deviation
  /// 3. Also apply Gaussian blur at different scales to capture multi-scale features
  /// 4. Concatenate all features into a single vector
  List<double> encodeIris(cv.Mat normalizedIris) {
    final features = <double>[];

    // --- Block-based statistics ---
    // Divide into blocks and extract mean/std per block
    // This captures the spatial distribution of iris texture
    const blockRows = 8;
    const blockCols = 16;
    final blockH = normalizedIris.rows ~/ blockRows;
    final blockW = normalizedIris.cols ~/ blockCols;

    for (int br = 0; br < blockRows; br++) {
      for (int bc = 0; bc < blockCols; bc++) {
        final roi = normalizedIris.region(
          cv.Rect(bc * blockW, br * blockH, blockW, blockH),
        );

        final meanStd = cv.meanStdDev(roi);
        final mean = meanStd.$1;
        final stddev = meanStd.$2;

        // Normalize to [0, 1] range
        features.add(mean.val1 / 255.0);
        features.add(stddev.val1 / 255.0);

        roi.dispose();
      }
    }

    // --- Multi-scale blur features ---
    // Captures texture at different scales (fine to coarse detail)
    for (final kernelSize in [3, 7, 15]) {
      final blurred = cv.gaussianBlur(normalizedIris, (kernelSize, kernelSize), 0);

      // Difference of original and blurred = band-pass filter
      // This highlights texture at a specific scale
      final diff = cv.subtract(normalizedIris, blurred);

      final meanStd = cv.meanStdDev(diff);
      features.add(meanStd.$1.val1 / 255.0);
      features.add(meanStd.$2.val1 / 255.0);

      blurred.dispose();
      diff.dispose();
    }

    return features;
  }

  // ─── 6. FULL PIPELINE ──────────────────────────────────────────────

  /// Runs the complete pipeline: load → preprocess → segment → normalize → encode.
  ///
  /// Returns null if segmentation fails (eye not found in image).
  Future<IrisProcessingResult?> processIrisImage(String imagePath) async {
    final image = cv.imread(imagePath, flags: cv.IMREAD_COLOR);
    if (image.isEmpty) return null;

    // Step 1: Preprocess (resize + CLAHE for consistent results)
    final (:gray, :scale) = _preprocessImage(image);
    image.dispose();

    print('[IrisService] Preprocessed: ${gray.cols}x${gray.rows} (scale=$scale)');

    // Step 2: Segment iris
    final segmentation = segmentIris(gray);
    if (segmentation == null) {
      gray.dispose();
      print('[IrisService] Segmentation failed - iris not found');
      return null;
    }

    // Step 3: Normalize
    final normalized = normalizeIris(gray, segmentation);

    // Step 4: Encode
    final template = encodeIris(normalized);

    print('[IrisService] Template generated: ${template.length} features');

    // Clean up
    gray.dispose();
    normalized.dispose();

    return IrisProcessingResult(
      template: template,
      segmentation: segmentation,
    );
  }

  /// Convenience method that wraps processIrisImage for the scanner flow.
  Future<List<double>?> generateIrisTemplate(String imagePath) async {
    final result = await processIrisImage(imagePath);
    return result?.template;
  }

  // ─── 7. MATCHING ───────────────────────────────────────────────────

  /// Compares two iris templates using normalized Euclidean distance.
  ///
  /// Returns a similarity score from 0.0 (completely different) to 1.0 (identical).
  ///
  /// The distance is normalized by the vector length so that templates of
  /// different sizes (shouldn't happen, but defensive) can still be compared.
  double compareTemplates(List<double> template1, List<double> template2) {
    if (template1.length != template2.length) return 0.0;
    if (template1.isEmpty) return 0.0;

    double sumSquaredDiff = 0;
    for (int i = 0; i < template1.length; i++) {
      final diff = template1[i] - template2[i];
      sumSquaredDiff += diff * diff;
    }

    // Normalized Euclidean distance
    final distance = sqrt(sumSquaredDiff / template1.length);

    // Convert distance to similarity (0 distance = 1.0 similarity)
    // Using exponential decay; multiplier of 3.0 allows more tolerance
    // for real-world capture variations from mobile cameras
    return exp(-distance * 3.0);
  }

  /// Searches all registered persons for the best iris match.
  ///
  /// Returns the best match above [threshold], or null if no match.
  /// Threshold of 0.55 accounts for real-world capture variations.
  Future<IrisMatchResult?> findMatch(
    List<double> template, {
    double threshold = 0.55,
  }) async {
    final persons = await _dbService.getPersonsWithIrisTemplate();

    Person? bestMatch;
    double bestScore = 0;

    for (final person in persons) {
      if (person.irisTemplate == null) continue;

      final score = compareTemplates(template, person.irisTemplate!);
      print('[IrisService] Match vs ${person.fullName}: ${(score * 100).toStringAsFixed(1)}%');
      if (score > bestScore) {
        bestScore = score;
        bestMatch = person;
      }
    }

    print('[IrisService] Best score: ${(bestScore * 100).toStringAsFixed(1)}% '
        '(threshold: ${(threshold * 100).toStringAsFixed(0)}%)');

    if (bestMatch != null && bestScore >= threshold) {
      return IrisMatchResult(person: bestMatch, confidence: bestScore);
    }

    return null;
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

class IrisMatchResult {
  final Person person;
  final double confidence;

  const IrisMatchResult({required this.person, required this.confidence});
}
