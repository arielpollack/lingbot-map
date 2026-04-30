// LingBot Capture — minimal Flutter POC.
//
// One screen, three actions:
//   1. Tap REC to start video. Tap STOP to end.
//   2. Tap UPLOAD to send to the backend.
//   3. When the run completes, tap VIEW to open the textured 3D mesh in the
//      built-in WebView (or copy the URL out to a desktop browser).
//
// Backend is the public Modal URL — no LAN setup needed.

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:url_launcher/url_launcher.dart';
import 'package:webview_flutter/webview_flutter.dart';

const String kBackendBase =
    'https://arielpollack--lingbot-map-web-fastapi-app.modal.run';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    _cameras = await availableCameras();
  } catch (_) {
    _cameras = [];
  }
  runApp(const CaptureApp());
}

class CaptureApp extends StatelessWidget {
  const CaptureApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LingBot Capture',
      theme: ThemeData(colorSchemeSeed: Colors.deepPurple, useMaterial3: true),
      home: const CaptureScreen(),
    );
  }
}

enum Phase {
  idle,
  recording,
  recorded,
  uploading,
  processing,
  completed,
  failed,
}

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  CameraController? _controller;
  bool _ready = false;
  Phase _phase = Phase.idle;
  String _status = 'Initializing camera…';
  String? _videoPath;
  String? _runId;
  String? _viewUrl;
  Timer? _pollTimer;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) {
      setState(() {
        _status = 'No camera available on this device.';
      });
      return;
    }
    final back = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras.first,
    );
    final controller = CameraController(
      back,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    try {
      await controller.initialize();
    } catch (e) {
      setState(() {
        _status = 'Camera init failed: $e';
      });
      return;
    }
    if (!mounted) {
      controller.dispose();
      return;
    }
    setState(() {
      _controller = controller;
      _ready = true;
      _status = 'Ready. Tap REC to start.';
    });
  }

  Future<void> _toggleRecording() async {
    final c = _controller;
    if (c == null || !_ready) return;
    if (_phase == Phase.recording) {
      try {
        final XFile file = await c.stopVideoRecording();
        setState(() {
          _videoPath = file.path;
          _phase = Phase.recorded;
          _status = 'Recorded ${file.path.split('/').last}. Tap UPLOAD.';
        });
      } catch (e) {
        setState(() {
          _phase = Phase.idle;
          _status = 'Stop failed: $e';
        });
      }
    } else {
      try {
        await c.startVideoRecording();
        setState(() {
          _phase = Phase.recording;
          _status = 'Recording… tap STOP when done.';
        });
      } catch (e) {
        setState(() {
          _status = 'Start failed: $e';
        });
      }
    }
  }

  Future<void> _upload() async {
    final path = _videoPath;
    if (path == null) return;
    setState(() {
      _phase = Phase.uploading;
      _status = 'Uploading…';
    });
    try {
      final uri = Uri.parse('$kBackendBase/api/runs');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('video', path));
      request.fields['fps'] = '10';
      request.fields['mode'] = 'streaming';
      final streamed =
          await request.send().timeout(const Duration(minutes: 15));
      final body = await streamed.stream.bytesToString();
      if (streamed.statusCode != 200) {
        throw 'HTTP ${streamed.statusCode}: $body';
      }
      final data = jsonDecode(body) as Map<String, dynamic>;
      final id = data['id'] as String;
      setState(() {
        _runId = id;
        _phase = Phase.processing;
        _status = 'Submitted. Waiting for completion…';
      });
      _startPolling(id);
    } catch (e) {
      setState(() {
        _phase = Phase.failed;
        _status = 'Upload failed: $e';
      });
    }
  }

  void _startPolling(String runId) {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (timer) async {
      try {
        final resp = await http
            .get(Uri.parse('$kBackendBase/api/runs/$runId'))
            .timeout(const Duration(seconds: 30));
        if (resp.statusCode != 200) return;
        final data = jsonDecode(resp.body) as Map<String, dynamic>;
        final status = data['status'] as String?;
        if (status == 'completed') {
          timer.cancel();
          final output = data['output'] as Map<String, dynamic>?;
          final hasMesh = output != null && output['mesh_key'] != null;
          final url = hasMesh
              ? '$kBackendBase/mesh?run=$runId'
              : '$kBackendBase/viewer?run=$runId';
          setState(() {
            _phase = Phase.completed;
            _viewUrl = url;
            _status = 'Done. Tap VIEW.';
          });
        } else if (status == 'failed') {
          timer.cancel();
          setState(() {
            _phase = Phase.failed;
            _status = 'Job failed: ${data['error'] ?? 'unknown'}';
          });
        } else {
          setState(() {
            _status = 'Processing… status=$status';
          });
        }
      } catch (_) {
        // transient; keep polling
      }
    });
  }

  Future<void> _openWebView() async {
    final url = _viewUrl;
    if (url == null) return;
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => MeshViewerScreen(url: url)),
    );
  }

  Future<void> _openExternal() async {
    final url = _viewUrl;
    if (url == null) return;
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  Future<void> _copyUrl() async {
    final url = _viewUrl;
    if (url == null) return;
    await Clipboard.setData(ClipboardData(text: url));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('URL copied')),
    );
  }

  void _reset() {
    _pollTimer?.cancel();
    setState(() {
      _phase = Phase.idle;
      _videoPath = null;
      _runId = null;
      _viewUrl = null;
      _status = 'Ready. Tap REC to start.';
    });
  }

  Widget _preview() {
    if (!_ready || _controller == null) {
      return const SizedBox(
        height: 240,
        child: Center(child: Text('Camera initializing…')),
      );
    }
    return AspectRatio(
      aspectRatio: _controller!.value.aspectRatio,
      child: CameraPreview(_controller!),
    );
  }

  Widget _bottomActions() {
    final children = <Widget>[];
    if (_phase == Phase.idle || _phase == Phase.recording) {
      children.add(
        FilledButton.icon(
          icon: Icon(
            _phase == Phase.recording
                ? Icons.stop_circle
                : Icons.fiber_manual_record,
          ),
          label: Text(_phase == Phase.recording ? 'STOP' : 'REC'),
          style: FilledButton.styleFrom(
            backgroundColor:
                _phase == Phase.recording ? Colors.red : null,
            minimumSize: const Size.fromHeight(56),
          ),
          onPressed: _ready ? _toggleRecording : null,
        ),
      );
    }
    if (_phase == Phase.recorded) {
      children.add(
        FilledButton.icon(
          icon: const Icon(Icons.cloud_upload),
          label: const Text('UPLOAD'),
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56)),
          onPressed: _upload,
        ),
      );
      children.add(const SizedBox(height: 8));
      children.add(
        OutlinedButton(
          onPressed: _reset,
          child: const Text('RECORD AGAIN'),
        ),
      );
    }
    if (_phase == Phase.uploading || _phase == Phase.processing) {
      children.add(
        const Padding(
          padding: EdgeInsets.symmetric(vertical: 16),
          child: LinearProgressIndicator(),
        ),
      );
    }
    if (_phase == Phase.completed) {
      children.add(
        FilledButton.icon(
          icon: const Icon(Icons.view_in_ar),
          label: const Text('VIEW IN APP'),
          style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(56)),
          onPressed: _openWebView,
        ),
      );
      children.add(const SizedBox(height: 8));
      children.add(
        OutlinedButton.icon(
          icon: const Icon(Icons.open_in_browser),
          label: const Text('OPEN IN BROWSER'),
          onPressed: _openExternal,
        ),
      );
      children.add(const SizedBox(height: 8));
      children.add(
        OutlinedButton.icon(
          icon: const Icon(Icons.content_copy),
          label: const Text('COPY URL'),
          onPressed: _copyUrl,
        ),
      );
      children.add(const SizedBox(height: 8));
      children.add(
        SelectableText(
          _viewUrl ?? '',
          style: const TextStyle(fontFamily: 'monospace', fontSize: 11),
          textAlign: TextAlign.center,
        ),
      );
      children.add(const SizedBox(height: 8));
      children.add(
        OutlinedButton(
          onPressed: _reset,
          child: const Text('NEW CAPTURE'),
        ),
      );
    }
    if (_phase == Phase.failed) {
      children.add(
        OutlinedButton(
          onPressed: _reset,
          child: const Text('RESET'),
        ),
      );
    }
    return Column(children: children);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('LingBot Capture'),
        actions: [
          if (_runId != null)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: Center(
                child: Text(
                  'run: ${_runId!.substring(0, 8)}…',
                  style: const TextStyle(fontSize: 11),
                ),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _preview(),
              const SizedBox(height: 16),
              Text(_status, textAlign: TextAlign.center),
              const SizedBox(height: 16),
              _bottomActions(),
              const SizedBox(height: 24),
              Text(
                'Backend: $kBackendBase',
                textAlign: TextAlign.center,
                style: const TextStyle(fontSize: 10, color: Colors.grey),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class MeshViewerScreen extends StatefulWidget {
  const MeshViewerScreen({super.key, required this.url});
  final String url;

  @override
  State<MeshViewerScreen> createState() => _MeshViewerScreenState();
}

class _MeshViewerScreenState extends State<MeshViewerScreen> {
  late final WebViewController _controller;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFF111418))
      ..loadRequest(Uri.parse(widget.url));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.url, style: const TextStyle(fontSize: 11)),
      ),
      body: WebViewWidget(controller: _controller),
    );
  }
}
