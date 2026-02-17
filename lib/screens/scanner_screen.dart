import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../services/iris_service.dart';
import '../services/database_service.dart';
import 'registration_screen.dart';
import 'person_detail_screen.dart';

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isProcessing = false;
  String _statusMessage = 'Initializing camera...';
  IrisSegmentation? _lastSegmentation;

  final DatabaseService _dbService = DatabaseService();
  late final IrisService _irisService;

  @override
  void initState() {
    super.initState();
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
        ResolutionPreset.max,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      await _cameraController!.initialize();

      if (mounted) {
        setState(() {
          _isInitialized = true;
          _statusMessage = 'Position the eye within the guide and capture';
        });
      }
    } catch (e) {
      setState(() => _statusMessage = 'Camera error: $e');
    }
  }

  Future<void> _captureAndProcess() async {
    if (_cameraController == null || _isProcessing) return;

    setState(() {
      _isProcessing = true;
      _lastSegmentation = null;
      _statusMessage = 'Capturing...';
    });

    try {
      final xFile = await _cameraController!.takePicture();
      final imageFile = File(xFile.path);

      setState(() => _statusMessage = 'Saving image...');
      final savedPath = await _irisService.saveEyeImage(imageFile);

      setState(() => _statusMessage = 'Processing iris (segmentation)...');

      // Run the full OpenCV pipeline
      final result = await _irisService.processIrisImage(savedPath);

      if (result == null) {
        if (mounted) {
          setState(() => _statusMessage = 'Could not detect iris. Try again with better framing.');
          await _showRetryDialog(
            'Iris not detected',
            'The iris boundaries could not be found in the image.\n\n'
            'Tips:\n'
            '• Get closer to the eye (fill 60%+ of the frame)\n'
            '• Ensure good, even lighting\n'
            '• Keep the eye fully open\n'
            '• Avoid reflections on the eye surface',
          );
        }
        return;
      }

      // Show segmentation result
      setState(() {
        _lastSegmentation = result.segmentation;
        _statusMessage = 'Iris detected! Searching for match...';
      });

      await Future.delayed(const Duration(milliseconds: 800));

      // Try to find a match
      final match = await _irisService.findMatch(result.template);

      if (!mounted) return;

      if (match != null) {
        setState(() => _statusMessage =
            'Match: ${match.person.fullName} (${(match.confidence * 100).toStringAsFixed(1)}%)');

        await Future.delayed(const Duration(milliseconds: 500));

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PersonDetailScreen(person: match.person),
            ),
          );
        }
      } else {
        final shouldRegister = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('No match found'),
            content: const Text(
              'This iris is not registered. Would you like to register a new person?',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Cancel'),
              ),
              FilledButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('Register'),
              ),
            ],
          ),
        );

        if (shouldRegister == true && mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => RegistrationScreen(
                irisImagePath: savedPath,
                irisTemplate: result.template,
              ),
            ),
          );
        }
      }
    } catch (e) {
      setState(() => _statusMessage = 'Error: $e');
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  Future<void> _showRetryDialog(String title, String message) async {
    await showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          FilledButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('OK, try again'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Iris Scanner'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
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
                          segmentation: _lastSegmentation,
                        ),
                      ),
                      if (_isProcessing)
                        Container(
                          color: Colors.black45,
                          child: const Center(
                            child: CircularProgressIndicator(
                              color: Colors.white,
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
            padding: const EdgeInsets.all(20),
            color: Theme.of(context).colorScheme.surface,
            child: Column(
              children: [
                Text(
                  _statusMessage,
                  style: Theme.of(context).textTheme.bodyLarge,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: FilledButton.icon(
                    onPressed: _isInitialized && !_isProcessing
                        ? _captureAndProcess
                        : null,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Capture Iris'),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Custom painter that draws the eye alignment guide.
/// If segmentation data is available, it also draws the detected circles.
class _EyeGuidePainter extends CustomPainter {
  final IrisSegmentation? segmentation;

  _EyeGuidePainter({this.segmentation});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.2;

    // Guide circles (always shown)
    final guidePaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawCircle(center, radius, guidePaint);
    canvas.drawCircle(center, radius * 0.4, guidePaint);

    // Crosshair
    final crossPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.3)
      ..strokeWidth = 1.0;

    canvas.drawLine(
      Offset(center.dx - radius * 1.3, center.dy),
      Offset(center.dx + radius * 1.3, center.dy),
      crossPaint,
    );
    canvas.drawLine(
      Offset(center.dx, center.dy - radius * 1.3),
      Offset(center.dx, center.dy + radius * 1.3),
      crossPaint,
    );

    // If we have segmentation results, draw detected circles in green
    if (segmentation != null) {
      final seg = segmentation!;

      final detectedPaint = Paint()
        ..color = Colors.greenAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;

      // Pupil
      canvas.drawCircle(
        Offset(seg.pupilCenter.x.toDouble(), seg.pupilCenter.y.toDouble()),
        seg.pupilRadius.toDouble(),
        detectedPaint,
      );

      // Iris
      final irisPaint = Paint()
        ..color = Colors.cyanAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;

      canvas.drawCircle(
        Offset(seg.irisCenter.x.toDouble(), seg.irisCenter.y.toDouble()),
        seg.irisRadius.toDouble(),
        irisPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _EyeGuidePainter oldDelegate) {
    return oldDelegate.segmentation != segmentation;
  }
}
