import 'dart:math';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:eye_recognition/services/database_service.dart';
import 'package:eye_recognition/services/iris_service.dart';

/// Builds a 256×64 synthetic "normalized iris" strip with strong multi-oriented
/// texture so Gabor responses have magnitude and most bits are valid.
Uint8List _buildStrip({double phase = 0.0, int fx = 12, int fy = 6, int fd = 9}) {
  final d = Uint8List(64 * 256);
  for (int y = 0; y < 64; y++) {
    for (int x = 0; x < 256; x++) {
      final v = 128 +
          55 * sin(2 * pi * x / fx + phase) +
          35 * sin(2 * pi * y / fy) +
          25 * sin(2 * pi * (x + 2 * y) / fd + phase);
      d[y * 256 + x] = v.clamp(0, 255).toInt();
    }
  }
  return d;
}

/// Circularly rolls the strip along the angular (column) axis.
Uint8List _roll(Uint8List src, int shift) {
  final d = Uint8List(64 * 256);
  for (int y = 0; y < 64; y++) {
    for (int x = 0; x < 256; x++) {
      d[y * 256 + ((x + shift) % 256)] = src[y * 256 + x];
    }
  }
  return d;
}

cv.Mat _strip(Uint8List data) =>
    cv.Mat.fromList(64, 256, cv.MatType.CV_8UC1, data);

void main() {
  final svc = IrisService(DatabaseService());

  test('normalizeIris (remap) produces a valid polar strip', () {
    // Gray 480×640 with a bright annulus around a dark pupil.
    final gray = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC1);
    cv.circle(gray, cv.Point(320, 240), 120, cv.Scalar.all(180), thickness: -1);
    cv.circle(gray, cv.Point(320, 240), 40, cv.Scalar.all(20), thickness: -1);

    final seg = IrisSegmentation(
      pupilX: 320, pupilY: 240, pupilR: 40,
      irisX: 320, irisY: 240, irisR: 120,
    );

    final norm = svc.normalizeIris(gray, seg);
    expect(norm.rows, 64);
    expect(norm.cols, 256);
    final meanStd = cv.meanStdDev(norm);
    expect(meanStd.$1.val1, greaterThan(0)); // not all-zero
    gray.dispose();
    norm.dispose();
  });

  test('encodeIris yields a full-length template with enough valid bits', () {
    final strip = _strip(_buildStrip());
    final t = svc.encodeIris(strip);
    strip.dispose();

    expect(t, isNotNull);
    // [code_bits..., mask_bits...] → 2 × 8192
    expect(t!.length, 2 * 8192);
    final codeLen = t.length ~/ 2;
    final valid = t.sublist(codeLen).where((m) => m == 1.0).length;
    expect(valid / codeLen, greaterThan(0.50));
  });

  test('a template matches ITSELF with HD≈0 (mutual-bit floor does not reject genuine)', () {
    final strip = _strip(_buildStrip());
    final t = svc.encodeIris(strip)!;
    strip.dispose();

    final hd = svc.compareTemplates(t, t);
    expect(hd, lessThan(0.05));
    expect(hd, isNot(1.0)); // the regression we fixed
  });

  test('rotation compensation recovers an angular shift of the same texture', () {
    final base = _buildStrip();
    final s1 = _strip(base);
    final s2 = _strip(_roll(base, 8)); // 8px ≈ 2 grid columns of angular rotation
    final t1 = svc.encodeIris(s1)!;
    final t2 = svc.encodeIris(s2)!;
    s1.dispose();
    s2.dispose();

    final hd = svc.compareTemplates(t1, t2);
    expect(hd, lessThan(0.25)); // genuine, aligned by the shift search
  });

  test('a different iris texture gives a clearly higher HD than a genuine match', () {
    final genuineA = svc.encodeIris(_strip(_buildStrip(phase: 0.0)))!;
    final genuineB = svc.encodeIris(_strip(_buildStrip(phase: 0.15)))!; // tiny change
    final impostor = svc.encodeIris(_strip(_buildStrip(phase: pi, fx: 7, fy: 5, fd: 11)))!;

    final genuineHd = svc.compareTemplates(genuineA, genuineB);
    final impostorHd = svc.compareTemplates(genuineA, impostor);

    expect(genuineHd, lessThan(IrisService.confirmThreshold));
    expect(impostorHd, greaterThan(genuineHd + 0.1));
  });
}
