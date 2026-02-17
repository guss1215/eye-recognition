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

      // Prefer highest resolution back camera
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
      _statusMessage = 'Capturing...';
    });

    try {
      final xFile = await _cameraController!.takePicture();
      final imageFile = File(xFile.path);

      setState(() => _statusMessage = 'Processing iris...');

      // Save the image
      final savedPath = await _irisService.saveEyeImage(imageFile);

      // Generate iris template
      final template = await _irisService.generateIrisTemplate(savedPath);

      // Try to find a match
      setState(() => _statusMessage = 'Searching for match...');
      final match = await _irisService.findMatch(template);

      if (!mounted) return;

      if (match != null) {
        // Person found
        setState(() => _statusMessage = 'Match found!');
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
        // No match → offer registration
        final shouldRegister = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('No match found'),
            content: const Text(
              'This iris is not registered yet. Would you like to register a new person?',
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
                irisTemplate: template,
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
                      // Eye guide overlay
                      CustomPaint(
                        size: Size.infinite,
                        painter: _EyeGuidePainter(),
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

class _EyeGuidePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width * 0.2;

    final paint = Paint()
      ..color = Colors.white.withValues(alpha: 0.6)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // Outer circle (iris boundary guide)
    canvas.drawCircle(center, radius, paint);

    // Inner circle (pupil guide)
    canvas.drawCircle(center, radius * 0.4, paint);

    // Crosshair lines
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
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
