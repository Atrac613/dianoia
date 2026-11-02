import 'package:flutter/material.dart';

import 'chat_controller.dart';
import 'chat_page.dart';
import 'example_model.dart';
import 'audio_page.dart';

/// The models supported by the example, before any worker is started.
class ModelSelectionPage extends StatelessWidget {
  const ModelSelectionPage({super.key});

  static const models = ExampleModel.values;

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
              Card(
                child: ListTile(
                  contentPadding: const EdgeInsets.all(16),
                  title: const Text('LFM3-Audio-2B Q4_0'),
                  subtitle: const Text(
                    'Experimental audio transcription · Local model setup',
                  ),
                  trailing: const Icon(Icons.mic),
                  onTap: () => Navigator.of(context).push(
                    MaterialPageRoute<void>(
                      builder: (_) => const AudioTranscriptionPage(),
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

/// Owns the worker for one visit to a model's chat.
class ModelChatPage extends StatefulWidget {
  const ModelChatPage({super.key, required this.model});

  final ExampleModel model;

  @override
  State<ModelChatPage> createState() => _ModelChatPageState();
}

class _ModelChatPageState extends State<ModelChatPage> {
  late final _chat = ChatController(model: widget.model);

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
  Widget build(BuildContext context) => ChatPage(controller: _chat);
}
