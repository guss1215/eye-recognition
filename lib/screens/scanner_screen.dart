import 'dart:math';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import '../services/iris_service.dart';
import '../services/database_service.dart';
import '../models/person.dart';
import 'registration_screen.dart';
import 'person_detail_screen.dart';

enum _ScannerPhase { idle, liveDetection, bursting, processing }

class ScannerScreen extends StatefulWidget {
  final ScanMode mode;

  const ScannerScreen({super.key, this.mode = ScanMode.verification});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isAnalyzingFrame = false;
  String _statusMessage = 'Initializing camera...';

  // Phase state machine
  _ScannerPhase _phase = _ScannerPhase.idle;

  // Live detection state
  IrisDetectionStatus _detectionStatus = IrisDetectionStatus.notFound;
  DateTime? _readySince;
  DateTime? _lastFrameProcessed;
  static const _readyDelay = Duration(milliseconds: 500);
  static const _frameInterval = Duration(milliseconds: 400);

  // Burst capture state
  List<ScoredFrame> _burstFrames = [];
  DateTime? _burstStartTime;
  double _currentQualityScore = 0.0;
  double _burstProgress = 0.0;
  static const _burstTargetFrames = 20;
  static const _burstMaxDuration = Duration(seconds: 2);

  // Enrollment state (multi-burst)
  late ScanMode _scanMode;
  int _enrollmentBurstCount = 0;
  static const _enrollmentTotalBursts = 3;
  List<List<double>> _enrollmentTemplates = [];
  String? _enrollmentBestImagePath;

  final DatabaseService _dbService = DatabaseService();
  late final IrisService _irisService;

  @override
  void initState() {
    super.initState();
    _scanMode = widget.mode;
    _irisService = IrisService(_dbService);
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras == null || _cameras!.isEmpty) {
        setState(() => _statusMessage = 'No cameras available');
        return;
      }

      final camera = _cameras!.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras!.first,
      );

      _cameraController = CameraController(
        camera,
        ResolutionPreset.veryHigh,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() {
          _isInitialized = true;
          _statusMessage = 'Align eye inside circle, then press Start';
        });
      }
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  // ─── LIVE DETECTION ─────────────────────────────────────────────────

  void _startLiveDetection() {
    _cameraController?.startImageStream(_onCameraFrame);
  }

  void _onCameraFrame(CameraImage image) {
    switch (_phase) {
      case _ScannerPhase.idle:
        return; // camera preview only, no analysis
      case _ScannerPhase.liveDetection:
        _handleLiveDetectionFrame(image);
      case _ScannerPhase.bursting:
        _handleBurstFrame(image);
      case _ScannerPhase.processing:
        return; // ignore frames during processing
    }
  }

  void _startScanning() {
    setState(() {
      _phase = _ScannerPhase.liveDetection;
      _detectionStatus = IrisDetectionStatus.notFound;
      _readySince = null;
      _statusMessage = 'Scanning... align the eye';
    });
    // Start stream if not already running
    if (_cameraController != null &&
        !_cameraController!.value.isStreamingImages) {
      _startLiveDetection();
    }
  }

  void _handleLiveDetectionFrame(CameraImage image) {
    if (_isAnalyzingFrame) return;
    final now = DateTime.now();
    if (_lastFrameProcessed != null &&
        now.difference(_lastFrameProcessed!) < _frameInterval) {
      return;
    }
    _lastFrameProcessed = now;
    _isAnalyzingFrame = true;

    try {
      final grayMat = _cameraImageToGrayscale(image);
      final result = _irisService.quickDetectIris(grayMat);
      grayMat.dispose();

      if (!mounted) return;

      setState(() {
        _detectionStatus = result.status;
        _statusMessage = _messageForStatus(result.status);
      });

      if (result.status == IrisDetectionStatus.ready) {
        _readySince ??= DateTime.now();
        if (DateTime.now().difference(_readySince!) >= _readyDelay) {
          _readySince = null;
          _startBurstCapture();
        }
      } else {
        _readySince = null;
      }
    } catch (e) {
      print('[Scanner] Frame analysis error: $e');
    } finally {
      _isAnalyzingFrame = false;
    }
  }

  void _handleBurstFrame(CameraImage image) {
    if (_isAnalyzingFrame) return;

    // Check burst completion
    final elapsed = DateTime.now().difference(_burstStartTime!);
    if (_burstFrames.length >= _burstTargetFrames ||
        elapsed >= _burstMaxDuration) {
      _finishBurstCapture();
      return;
    }

    _isAnalyzingFrame = true;

    try {
      final grayMat = _cameraImageToGrayscale(image);
      final scored = _irisService.scoreFrame(grayMat);
      grayMat.dispose();

      if (scored != null) {
        _burstFrames.add(scored);

        if (mounted) {
          setState(() {
            _currentQualityScore = scored.quality.composite;
            _burstProgress = _burstFrames.length / _burstTargetFrames;
            _statusMessage =
                'Capturing ${_burstFrames.length}/$_burstTargetFrames';
          });
        }
      }
    } catch (e) {
      print('[Scanner] Burst frame error: $e');
    } finally {
      _isAnalyzingFrame = false;
    }
  }

  cv.Mat _cameraImageToGrayscale(CameraImage image) {
    final yPlane = image.planes[0];
    final width = image.width;
    final height = image.height;

    if (yPlane.bytesPerRow == width) {
      return cv.Mat.fromList(height, width, cv.MatType.CV_8UC1, yPlane.bytes);
    }

    final bytes = Uint8List(width * height);
    for (int row = 0; row < height; row++) {
      final srcOffset = row * yPlane.bytesPerRow;
      final dstOffset = row * width;
      bytes.setRange(dstOffset, dstOffset + width, yPlane.bytes, srcOffset);
    }
    return cv.Mat.fromList(height, width, cv.MatType.CV_8UC1, bytes);
  }

  String _messageForStatus(IrisDetectionStatus status) {
    return switch (status) {
      IrisDetectionStatus.notFound => 'Point the camera at an eye',
      IrisDetectionStatus.tooFar => 'Move closer to the eye',
      IrisDetectionStatus.tooClose => 'Move back a little',
      IrisDetectionStatus.notCentered => 'Center the eye inside the circle',
      IrisDetectionStatus.tooBlurry => 'Hold steady — image is blurry',
      IrisDetectionStatus.ready => 'Hold steady...',
    };
  }

  // ─── BURST CAPTURE ────────────────────────────────────────────────

  void _startBurstCapture() {
    setState(() {
      _phase = _ScannerPhase.bursting;
      _burstFrames = [];
      _burstStartTime = DateTime.now();
      _burstProgress = 0.0;
      _currentQualityScore = 0.0;
      _statusMessage = 'Hold steady... capturing';
    });

    // Lock AE/AF for consistent frames
    try {
      _cameraController?.setExposureMode(ExposureMode.locked);
    } catch (_) {}
    try {
      _cameraController?.setFocusMode(FocusMode.locked);
    } catch (_) {}
  }

  Future<void> _finishBurstCapture() async {
    _phase = _ScannerPhase.processing;

    // Stop stream and unlock AE/AF
    try {
      await _cameraController?.stopImageStream();
    } catch (_) {}
    try {
      _cameraController?.setExposureMode(ExposureMode.auto);
    } catch (_) {}
    try {
      _cameraController?.setFocusMode(FocusMode.auto);
    } catch (_) {}

    if (!mounted) return;

    setState(() {
      _statusMessage = 'Processing ${_burstFrames.length} frames...';
    });

    print('[Scanner] Burst complete: ${_burstFrames.length} frames captured');

    final burstResult = await _irisService.processBurstFrames(
      _burstFrames,
      mode: _scanMode,
    );

    // Dispose all burst frames
    for (final frame in _burstFrames) {
      frame.dispose();
    }
    _burstFrames = [];

    if (burstResult == null) {
      if (mounted) {
        setState(() {
          _statusMessage = 'Quality too low. Try again.';
        });
        await Future.delayed(const Duration(seconds: 1));
        _restartLiveDetection();
      }
      return;
    }

    if (_scanMode == ScanMode.enrollment) {
      await _handleEnrollmentBurst(burstResult);
    } else {
      await _handleVerificationResult(burstResult);
    }
  }

  // ─── ENROLLMENT FLOW ──────────────────────────────────────────────

  Future<void> _handleEnrollmentBurst(BurstResult burstResult) async {
    _enrollmentBurstCount++;
    _enrollmentTemplates.addAll(burstResult.templates);
    _enrollmentBestImagePath ??= burstResult.savedImagePath;

    if (!mounted) return;

    if (_enrollmentBurstCount < _enrollmentTotalBursts) {
      setState(() {
        _statusMessage =
            'Burst $_enrollmentBurstCount/$_enrollmentTotalBursts done. Reposition slightly.';
      });
      await Future.delayed(const Duration(seconds: 2));
      _restartLiveDetection();
      return;
    }

    // All bursts complete — select best diverse templates
    final finalTemplates = _selectDiverseTemplates(_enrollmentTemplates, 3);

    print('[Scanner] Enrollment complete: ${finalTemplates.length} templates '
        'from ${_enrollmentTemplates.length} total');

    if (mounted) {
      await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => RegistrationScreen(
            irisImagePath: _enrollmentBestImagePath,
            irisTemplates: finalTemplates,
          ),
        ),
      );

      // Reset enrollment state
      _enrollmentBurstCount = 0;
      _enrollmentTemplates = [];
      _enrollmentBestImagePath = null;
      _restartLiveDetection();
    }
  }

  /// Selects the N most diverse templates from a pool.
  /// Starts with the first, then greedily picks the most different from selected set.
  List<List<double>> _selectDiverseTemplates(
    List<List<double>> pool,
    int count,
  ) {
    if (pool.length <= count) return pool;

    final selected = <List<double>>[pool.first];
    final remaining = pool.sublist(1).toList();

    while (selected.length < count && remaining.isNotEmpty) {
      int bestIdx = 0;
      double bestMinDist = 0;

      for (int i = 0; i < remaining.length; i++) {
        double minDist = double.infinity;
        for (final sel in selected) {
          final hd = _irisService.compareTemplates(remaining[i], sel);
          if (hd < minDist) minDist = hd;
        }
        if (minDist > bestMinDist) {
          bestMinDist = minDist;
          bestIdx = i;
        }
      }

      selected.add(remaining.removeAt(bestIdx));
    }

    return selected;
  }

  // ─── VERIFICATION FLOW ────────────────────────────────────────────

  Future<void> _handleVerificationResult(BurstResult burstResult) async {
    if (!mounted) return;

    setState(() => _statusMessage = 'Searching for match...');

    // Use the best template for matching
    final template = burstResult.templates.first;
    final candidates = await _irisService.findCandidates(template);

    if (!mounted) return;

    // Zone 1: Confirmed match
    final confirmed =
        candidates.where((c) => c.matchType == MatchType.confirmed).toList();
    if (confirmed.isNotEmpty) {
      final best = confirmed.first;
      setState(() => _statusMessage =
          'Match: ${best.person.fullName} (${(best.confidence * 100).toStringAsFixed(0)}%)');

      await Future.delayed(const Duration(milliseconds: 500));

      if (mounted) {
        await Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => PersonDetailScreen(person: best.person),
          ),
        );
        _restartLiveDetection();
      }
      return;
    }

    // Zone 2: Suggested matches
    final suggested =
        candidates.where((c) => c.matchType == MatchType.suggested).toList();
    if (suggested.isNotEmpty) {
      final dialogResult = await showDialog<_CandidateAction>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Possible matches'),
          content: SizedBox(
            width: double.maxFinite,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'No confirmed match. These persons have similar irises:',
                  style: TextStyle(fontSize: 14),
                ),
                const SizedBox(height: 12),
                ...suggested.map((c) => ListTile(
                      leading: const Icon(Icons.person_outline),
                      title: Text(c.person.fullName),
                      subtitle: Text(
                        'Similarity: ${(c.confidence * 100).toStringAsFixed(0)}% '
                        '(HD: ${c.distance.toStringAsFixed(3)})',
                      ),
                      trailing: FilledButton(
                        onPressed: () => Navigator.pop(
                          ctx,
                          _CandidateAction.select(c.person),
                        ),
                        child: const Text('Select'),
                      ),
                      dense: true,
                      contentPadding: EdgeInsets.zero,
                    )),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, _CandidateAction.cancel()),
              child: const Text('Cancel'),
            ),
            OutlinedButton(
              onPressed: () => Navigator.pop(ctx, _CandidateAction.register()),
              child: const Text('Register new'),
            ),
          ],
        ),
      );

      if (!mounted) return;

      if (dialogResult != null) {
        if (dialogResult.person != null) {
          await Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) =>
                  PersonDetailScreen(person: dialogResult.person!),
            ),
          );
        } else if (dialogResult.shouldRegister) {
          // Switch to enrollment mode for proper multi-template registration
          _scanMode = ScanMode.enrollment;
          _enrollmentBurstCount = 0;
          _enrollmentTemplates = [];
          _enrollmentBestImagePath = burstResult.savedImagePath;
          // Add current burst templates as first enrollment burst
          _enrollmentTemplates.addAll(burstResult.templates);
          _enrollmentBurstCount = 1;
          setState(() {
            _statusMessage =
                'Enrollment mode. Burst 1/$_enrollmentTotalBursts done. Reposition.';
          });
          await Future.delayed(const Duration(seconds: 2));
          _restartLiveDetection();
          return;
        }
      }
      _restartLiveDetection();
      return;
    }

    // Zone 3: No match — offer to re-scan in enrollment mode
    final shouldEnroll = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('No match found'),
        content: const Text(
          'This iris is not registered. Would you like to enroll a new person?\n\n'
          'Enrollment captures 3 scans for better accuracy.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Enroll'),
          ),
        ],
      ),
    );

    if (shouldEnroll == true && mounted) {
      // Switch to enrollment mode, count current burst as first
      _scanMode = ScanMode.enrollment;
      _enrollmentBurstCount = 1;
      _enrollmentTemplates = List.from(burstResult.templates);
      _enrollmentBestImagePath = burstResult.savedImagePath;
      setState(() {
        _statusMessage =
            'Enrollment mode. Burst 1/$_enrollmentTotalBursts done. Reposition.';
      });
      await Future.delayed(const Duration(seconds: 2));
      _restartLiveDetection();
      return;
    }

    _restartLiveDetection();
  }

  // ─── RESTART ──────────────────────────────────────────────────────

  void _restartLiveDetection() {
    if (!mounted) return;
    // For enrollment mid-flow (between bursts), auto-start next scan
    final autoStart = _scanMode == ScanMode.enrollment &&
        _enrollmentBurstCount > 0 &&
        _enrollmentBurstCount < _enrollmentTotalBursts;

    setState(() {
      _phase = autoStart ? _ScannerPhase.liveDetection : _ScannerPhase.idle;
      _detectionStatus = IrisDetectionStatus.notFound;
      _readySince = null;
      _burstProgress = 0.0;
      _currentQualityScore = 0.0;
      _statusMessage = autoStart
          ? 'Scanning... align the eye'
          : 'Align eye inside circle, then press Start';
    });
    if (autoStart) {
      _startLiveDetection();
    }
  }

  @override
  void dispose() {
    for (final frame in _burstFrames) {
      frame.dispose();
    }
    _cameraController?.dispose();
    super.dispose();
  }

  // ─── BUILD ────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final isProcessing = _phase == _ScannerPhase.processing;

    return Scaffold(
      appBar: AppBar(
        title: Text(_scanMode == ScanMode.enrollment
            ? 'Iris Enrollment'
            : 'Iris Scanner'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          if (_scanMode == ScanMode.enrollment && _enrollmentBurstCount > 0)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Center(
                child: Text(
                  'Burst $_enrollmentBurstCount/$_enrollmentTotalBursts',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _isInitialized
                ? Stack(
                    alignment: Alignment.center,
                    children: [
                      CameraPreview(_cameraController!),
                      CustomPaint(
                        size: Size.infinite,
                        painter: _EyeGuidePainter(
                          status: _detectionStatus,
                          phase: _phase,
                          qualityScore: _currentQualityScore,
                          burstProgress: _burstProgress,
                        ),
                      ),
                      if (isProcessing)
                        Container(
                          color: Colors.black45,
                          child: const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                        ),
                      // Start scan button (only in idle phase)
                      if (_phase == _ScannerPhase.idle)
                        Positioned(
                          bottom: 32,
                          child: FilledButton.icon(
                            onPressed: _startScanning,
                            icon: const Icon(Icons.play_arrow, size: 28),
                            label: Text(
                              _scanMode == ScanMode.enrollment
                                  ? 'Start Enrollment'
                                  : 'Start Scan',
                              style: const TextStyle(fontSize: 16),
                            ),
                            style: FilledButton.styleFrom(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 24, vertical: 14),
                            ),
                          ),
                        ),
                    ],
                  )
                : Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 16),
                        Text(_statusMessage),
                      ],
                    ),
                  ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            color: _statusBarColor,
            child: Row(
              children: [
                Icon(
                  _statusBarIcon,
                  color: _statusBarTextColor,
                  size: 28,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    _statusMessage,
                    style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                          color: _statusBarTextColor,
                          fontWeight: FontWeight.w500,
                        ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color get _statusBarColor {
    return Theme.of(context).colorScheme.surface;
  }

  Color get _statusBarTextColor {
    return switch (_phase) {
      _ScannerPhase.idle => Colors.white70,
      _ScannerPhase.bursting => Colors.cyan,
      _ScannerPhase.processing => Colors.blue,
      _ScannerPhase.liveDetection => switch (_detectionStatus) {
          IrisDetectionStatus.notFound => Colors.red.shade300,
          IrisDetectionStatus.tooFar => Colors.orange.shade300,
          IrisDetectionStatus.tooClose => Colors.orange.shade300,
          IrisDetectionStatus.notCentered => Colors.orange.shade300,
          IrisDetectionStatus.tooBlurry => Colors.orange.shade300,
          IrisDetectionStatus.ready => Colors.green.shade300,
        },
    };
  }

  IconData get _statusBarIcon {
    return switch (_phase) {
      _ScannerPhase.idle => Icons.play_circle_outline,
      _ScannerPhase.bursting => Icons.camera,
      _ScannerPhase.processing => Icons.hourglass_top,
      _ScannerPhase.liveDetection => switch (_detectionStatus) {
          IrisDetectionStatus.notFound => Icons.visibility_off,
          IrisDetectionStatus.tooFar => Icons.zoom_in,
          IrisDetectionStatus.tooClose => Icons.zoom_out,
          IrisDetectionStatus.notCentered => Icons.open_with,
          IrisDetectionStatus.tooBlurry => Icons.blur_on,
          IrisDetectionStatus.ready => Icons.check_circle,
        },
    };
  }
}

/// Result from the candidate selection dialog.
class _CandidateAction {
  final Person? person;
  final bool shouldRegister;

  _CandidateAction._({this.person, this.shouldRegister = false});

  factory _CandidateAction.select(Person person) =>
      _CandidateAction._(person: person);
  factory _CandidateAction.register() =>
      _CandidateAction._(shouldRegister: true);
  factory _CandidateAction.cancel() => _CandidateAction._();
}

/// Custom painter that draws the eye alignment guide with quality arc,
/// burst progress ring, and status-based colors.
class _EyeGuidePainter extends CustomPainter {
  final IrisDetectionStatus status;
  final _ScannerPhase phase;
  final double qualityScore; // 0-100
  final double burstProgress; // 0-1

  _EyeGuidePainter({
    required this.status,
    required this.phase,
    required this.qualityScore,
    required this.burstProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final guideRadius = size.width * 0.22;

    final color = switch (phase) {
      _ScannerPhase.idle => Colors.white.withValues(alpha: 0.5),
      _ScannerPhase.bursting => Colors.cyan,
      _ScannerPhase.processing => Colors.blue,
      _ScannerPhase.liveDetection => switch (status) {
          IrisDetectionStatus.notFound => Colors.white.withValues(alpha: 0.5),
          IrisDetectionStatus.tooFar => Colors.orange,
          IrisDetectionStatus.tooClose => Colors.orange,
          IrisDetectionStatus.notCentered => Colors.orange,
          IrisDetectionStatus.tooBlurry => Colors.orange,
          IrisDetectionStatus.ready => Colors.greenAccent,
        },
    };

    // Outer guide ring
    final guidePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = (status == IrisDetectionStatus.ready ||
              phase == _ScannerPhase.bursting)
          ? 4.0
          : 2.5;

    canvas.drawCircle(center, guideRadius, guidePaint);

    // Inner pupil guide
    final pupilGuide = Paint()
      ..color = color.withValues(alpha: 0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    canvas.drawCircle(center, guideRadius * 0.35, pupilGuide);

    // Corner brackets
    _drawCornerBrackets(canvas, center, guideRadius, color);

    // Quality arc (inner, shows quality score during burst)
    if (phase == _ScannerPhase.bursting && qualityScore > 0) {
      _drawQualityArc(canvas, center, guideRadius - 8);
    }

    // Burst progress ring (outer)
    if (phase == _ScannerPhase.bursting) {
      _drawProgressRing(canvas, center, guideRadius + 10);
    }

    // Status text above the guide
    final statusText = switch (phase) {
      _ScannerPhase.idle => 'Align eye, then press Start',
      _ScannerPhase.bursting => 'Capturing...',
      _ScannerPhase.processing => 'Processing...',
      _ScannerPhase.liveDetection => switch (status) {
          IrisDetectionStatus.notFound => 'Align the eye inside the circle',
          IrisDetectionStatus.tooFar => 'Move closer',
          IrisDetectionStatus.tooClose => 'Move back',
          IrisDetectionStatus.notCentered => 'Center the eye',
          IrisDetectionStatus.tooBlurry => 'Hold steady',
          IrisDetectionStatus.ready => 'Hold steady...',
        },
    };

    final textPainter = TextPainter(
      text: TextSpan(
        text: statusText,
        style: TextStyle(
          color: color,
          fontSize: 16,
          fontWeight: FontWeight.w600,
          shadows: const [
            Shadow(blurRadius: 4, color: Colors.black87),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(center.dx - textPainter.width / 2, center.dy - guideRadius - 36),
    );

    // Quality score number below the guide during burst
    if (phase == _ScannerPhase.bursting && qualityScore > 0) {
      final scorePainter = TextPainter(
        text: TextSpan(
          text: qualityScore.toStringAsFixed(0),
          style: TextStyle(
            color: _qualityColor(qualityScore),
            fontSize: 24,
            fontWeight: FontWeight.bold,
            shadows: const [
              Shadow(blurRadius: 4, color: Colors.black87),
            ],
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      scorePainter.layout();
      scorePainter.paint(
        canvas,
        Offset(
            center.dx - scorePainter.width / 2, center.dy + guideRadius + 16),
      );
    }
  }

  /// Draws a quality arc inside the guide circle.
  /// Color: red <40, orange 40-70, green >70.
  void _drawQualityArc(Canvas canvas, Offset center, double radius) {
    final arcPaint = Paint()
      ..color = _qualityColor(qualityScore)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4.0
      ..strokeCap = StrokeCap.round;

    final sweepAngle = (qualityScore / 100.0) * 2 * pi;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -pi / 2, // start from top
      sweepAngle,
      false,
      arcPaint,
    );
  }

  /// Draws a progress ring outside the guide circle.
  void _drawProgressRing(Canvas canvas, Offset center, double radius) {
    // Background track
    final trackPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.2)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;
    canvas.drawCircle(center, radius, trackPaint);

    // Progress arc
    final progressPaint = Paint()
      ..color = Colors.cyan
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round;

    final sweepAngle = burstProgress * 2 * pi;
    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      -pi / 2,
      sweepAngle,
      false,
      progressPaint,
    );
  }

  Color _qualityColor(double score) {
    if (score < 40) return Colors.red;
    if (score < 70) return Colors.orange;
    return Colors.greenAccent;
  }

  void _drawCornerBrackets(
    Canvas canvas, Offset center, double radius, Color color,
  ) {
    final paint = Paint()
      ..color = color.withValues(alpha: 0.7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..strokeCap = StrokeCap.round;

    final bracketLen = radius * 0.25;
    final offset = radius + 12;

    // Top-left
    canvas.drawLine(
      Offset(center.dx - offset, center.dy - offset),
      Offset(center.dx - offset + bracketLen, center.dy - offset),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx - offset, center.dy - offset),
      Offset(center.dx - offset, center.dy - offset + bracketLen),
      paint,
    );

    // Top-right
    canvas.drawLine(
      Offset(center.dx + offset, center.dy - offset),
      Offset(center.dx + offset - bracketLen, center.dy - offset),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx + offset, center.dy - offset),
      Offset(center.dx + offset, center.dy - offset + bracketLen),
      paint,
    );

    // Bottom-left
    canvas.drawLine(
      Offset(center.dx - offset, center.dy + offset),
      Offset(center.dx - offset + bracketLen, center.dy + offset),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx - offset, center.dy + offset),
      Offset(center.dx - offset, center.dy + offset - bracketLen),
      paint,
    );

    // Bottom-right
    canvas.drawLine(
      Offset(center.dx + offset, center.dy + offset),
      Offset(center.dx + offset - bracketLen, center.dy + offset),
      paint,
    );
    canvas.drawLine(
      Offset(center.dx + offset, center.dy + offset),
      Offset(center.dx + offset, center.dy + offset - bracketLen),
      paint,
    );
  }

  @override
  bool shouldRepaint(covariant _EyeGuidePainter oldDelegate) {
    return oldDelegate.status != status ||
        oldDelegate.phase != phase ||
        oldDelegate.qualityScore != qualityScore ||
        oldDelegate.burstProgress != burstProgress;
  }
}
