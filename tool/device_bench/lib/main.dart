/// LFM3 on a phone, through dianoia. The benchmark starts by itself at launch
/// (so it can run without touching the phone) and again on demand; it runs
/// in a worker isolate and this page shows its progress lines.
library;

import 'dart:io';
import 'dart:isolate';

import 'package:flutter/material.dart';

import 'bench.dart';

void main() => runApp(const BenchApp());

class BenchApp extends StatelessWidget {
  const BenchApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'LFM3 bench',
    theme: ThemeData(colorSchemeSeed: Colors.teal),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.teal,
      brightness: Brightness.dark,
    ),
    home: const BenchPage(),
  );
}

class BenchPage extends StatefulWidget {
  const BenchPage({super.key});

  @override
  State<BenchPage> createState() => _BenchPageState();
}

class _BenchPageState extends State<BenchPage> {
  final _lines = <String>[];
  var _running = false;

  @override
  void initState() {
    super.initState();
    _run();
  }

  Future<void> _run() async {
    setState(() {
      _running = true;
      _lines.clear();
    });
    final port = ReceivePort();
    await Isolate.spawn(_worker, port.sendPort);
    await for (final message in port) {
      if (message == null) break;
      debugPrint('lfm3_bench: $message');
      setState(() => _lines.add(message as String));
    }
    port.close();
    setState(() => _running = false);
  }

  // Metal buffers live for the life of the process, so each run leaks its
  // activations and KV caches (~150 MB); restart the app between many runs.
  static Future<void> _worker(SendPort out) async {
    try {
      await runBench(out.send);
    } catch (e, stack) {
      // Release builds print nowhere a Mac can read, so errors also go to
      // a file that devicectl can copy back.
      out.send('ERROR: $e\n$stack');
      try {
        File('${documentsPath()}/bench/error.txt')
          ..createSync(recursive: true)
          ..writeAsStringSync('$e\n$stack');
      } on FileSystemException {
        // Nothing more to do; the message is on screen.
      }
    }
    out.send(null);
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('LFM3 on this device')),
    body: SafeArea(
      child: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          for (final line in _lines)
            Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: SelectableText(
                line,
                style: const TextStyle(fontFamily: 'Menlo', fontSize: 12),
              ),
            ),
          if (_running) const LinearProgressIndicator(),
        ],
      ),
    ),
    floatingActionButton: _running
        ? null
        : FloatingActionButton.extended(
            onPressed: _run,
            label: const Text('Run again'),
          ),
  );
}
