/// A simple chat with LFM3, run on this device by dianoia.
///
/// The model bundle (`LFM3-1.7B-QAD-Q4`, written by `dianoia import-gguf`)
/// goes into the app's documents; see README.md.
library;

import 'package:flutter/material.dart';

import 'src/chat_controller.dart';
import 'src/chat_page.dart';

void main() => runApp(const DianoiaExampleApp());

class DianoiaExampleApp extends StatefulWidget {
  const DianoiaExampleApp({super.key});

  @override
  State<DianoiaExampleApp> createState() => _DianoiaExampleAppState();
}

class _DianoiaExampleAppState extends State<DianoiaExampleApp> {
  final _chat = ChatController();

  @override
  void initState() {
    super.initState();
    _chat.start();
  }

  @override
  void dispose() {
    _chat.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Dianoia',
    theme: ThemeData(colorSchemeSeed: Colors.indigo),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.indigo,
      brightness: Brightness.dark,
    ),
    home: ChatPage(controller: _chat),
  );
}
