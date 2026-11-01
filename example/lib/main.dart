/// A simple chat with an on-device model selected at launch.
///
/// Model bundles can be downloaded in the app or copied into the app's
/// documents; see README.md.
library;

import 'package:flutter/material.dart';

import 'src/model_selection_page.dart';

void main() => runApp(const DianoiaExampleApp());

class DianoiaExampleApp extends StatelessWidget {
  const DianoiaExampleApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Dianoia',
    theme: ThemeData(colorSchemeSeed: Colors.indigo),
    darkTheme: ThemeData(
      colorSchemeSeed: Colors.indigo,
      brightness: Brightness.dark,
    ),
    home: const ModelSelectionPage(),
  );
}
