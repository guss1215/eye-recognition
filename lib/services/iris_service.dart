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

  // ─── 2. SEGMENTATION ───────────────────────────────────────────────

  /// Detects the pupil (inner circle) and iris (outer circle) boundaries.
  ///
  /// Returns null if detection fails (image quality too low, eye not found).
  ///
  /// How it works:
  /// - We convert to grayscale and blur to reduce noise
  /// - HoughCircles finds circular shapes in the image
  /// - The smallest detected circle is likely the pupil
  /// - The largest detected circle is likely the iris boundary
  IrisSegmentation? segmentIris(cv.Mat image) {
    // Convert to grayscale
    final gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

    // Apply median blur to reduce noise while preserving edges
    // The kernel size (7) is chosen because iris images have fine texture
    final blurred = cv.medianBlur(gray, 7);

    // --- Detect pupil (smaller, darker circle) ---
    // We use stricter parameters for the pupil since it's well-defined
    final pupilCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5, // dp: accumulator resolution ratio (1.5 = slightly coarser than input)
      50, // minDist: minimum distance between circle centers
      param1: 100, // upper Canny threshold
      param2: 40, // accumulator threshold (higher = fewer but more confident detections)
      minRadius: 10, // minimum pupil radius in pixels
      maxRadius: 80, // maximum pupil radius in pixels
    );

    // --- Detect iris boundary (larger circle) ---
    final irisCircles = cv.HoughCircles(
      blurred,
      cv.HOUGH_GRADIENT,
      1.5,
      100, // larger minDist since the iris is bigger
      param1: 80,
      param2: 35,
      minRadius: 60, // iris is always larger than pupil
      maxRadius: 200,
    );

    gray.dispose();
    blurred.dispose();

    // Need at least one of each to proceed
    if (pupilCircles.rows == 0 || irisCircles.rows == 0) {
      pupilCircles.dispose();
      irisCircles.dispose();
      return null;
    }

    // Take the first detected circle for each
    // pupilCircles is a Mat of shape (1, N, 3) where each circle is (x, y, radius)
    final pupilX = pupilCircles.at<double>(0, 0).toInt();
    final pupilY = pupilCircles.at<double>(0, 1).toInt();
    final pupilR = pupilCircles.at<double>(0, 2).toInt();

    final irisX = irisCircles.at<double>(0, 0).toInt();
    final irisY = irisCircles.at<double>(0, 1).toInt();
    final irisR = irisCircles.at<double>(0, 2).toInt();

    pupilCircles.dispose();
    irisCircles.dispose();

    // Basic sanity check: iris must be larger than pupil
    if (irisR <= pupilR) return null;

    return IrisSegmentation(
      pupilCenter: Point(pupilX, pupilY),
      pupilRadius: pupilR,
      irisCenter: Point(irisX, irisY),
      irisRadius: irisR,
    );
  }

  // ─── 3. NORMALIZATION (DAUGMAN'S RUBBER SHEET MODEL) ────────────────

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

  // ─── 4. FEATURE ENCODING ────────────────────────────────────────────

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

  // ─── 5. FULL PIPELINE ──────────────────────────────────────────────

  /// Runs the complete pipeline: load → segment → normalize → encode.
  ///
  /// Returns null if segmentation fails (eye not found in image).
  Future<IrisProcessingResult?> processIrisImage(String imagePath) async {
    final image = cv.imread(imagePath, flags: cv.IMREAD_COLOR);
    if (image.isEmpty) return null;

    final gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY);

    // Step 1: Segment iris
    final segmentation = segmentIris(image);
    if (segmentation == null) {
      image.dispose();
      gray.dispose();
      return null;
    }

    // Step 2: Normalize
    final normalized = normalizeIris(gray, segmentation);

    // Step 3: Encode
    final template = encodeIris(normalized);

    // Clean up
    image.dispose();
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

  // ─── 6. MATCHING ───────────────────────────────────────────────────

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
    // Using exponential decay so small differences still give high scores
    return exp(-distance * 5.0);
  }

  /// Searches all registered persons for the best iris match.
  ///
  /// Returns the best match above [threshold], or null if no match.
  /// Default threshold of 0.75 balances false accepts vs false rejects.
  Future<IrisMatchResult?> findMatch(
    List<double> template, {
    double threshold = 0.75,
  }) async {
    final persons = await _dbService.getPersonsWithIrisTemplate();

    Person? bestMatch;
    double bestScore = 0;

    for (final person in persons) {
      if (person.irisTemplate == null) continue;

      final score = compareTemplates(template, person.irisTemplate!);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = person;
      }
    }

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
