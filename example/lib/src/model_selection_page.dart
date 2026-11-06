import 'package:flutter/material.dart';

import 'chat_page.dart';
import 'example_model.dart';
import 'audio_page.dart';
import 'speech_model.dart';

/// The models supported by the example, before any worker is started.
class ModelSelectionPage extends StatelessWidget {
  const ModelSelectionPage({super.key});

  /// LFM3-Omni's chat opens from its own screen, not from this list.
  static final models = [
    for (final model in ExampleModel.values)
      if (model != ExampleModel.omni) model,
  ];

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Dianoia')),
    body: SafeArea(
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 640),
          child: ListView(
            padding: const EdgeInsets.all(24),
            children: [
              Text(
                'Choose a model',
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 8),
              const Text(
                'Models run on this device. Select one to open a chat or '
                'set it up if needed.',
              ),
              const SizedBox(height: 24),
              for (final model in models)
                Card(
                  clipBehavior: Clip.antiAlias,
                  child: ListTile(
                    contentPadding: const EdgeInsets.all(16),
                    title: Text(model.name),
                    subtitle: Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        '${model.sizeLabel}\n'
                        'License: ${model.license}',
                      ),
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => ModelChatPage(model: model),
                      ),
                    ),
                  ),
                ),
              for (final speech in SpeechModel.values)
                Card(
                  child: ListTile(
                    contentPadding: const EdgeInsets.all(16),
                    title: Text(speech.name),
                    subtitle: Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text(
                        '${speech.summary}\n${speech.sizeLabel}\n'
                        'License: ${speech.spec.license}',
                      ),
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => AudioTranscriptionPage(model: speech),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    ),
  );
}
