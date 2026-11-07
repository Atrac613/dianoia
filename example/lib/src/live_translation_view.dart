import 'package:flutter/material.dart';

import 'audio_controller.dart';
import 'live_translation.dart';

/// The controls and the running translation of [AudioTask.liveTranslation].
class LiveTranslationSettings extends StatelessWidget {
  const LiveTranslationSettings({
    super.key,
    required this.controller,
    required this.enabled,
  });
  final AudioController controller;
  final bool enabled;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: controller.translator,
    builder: (context, _) {
      final translator = controller.translator;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Speak English. Each finished sentence of the live transcript is '
            'translated as text by the same model, on the CPU beside the '
            'speech recognition, so translations follow a sentence behind. '
            'The last sentence is translated when you stop.',
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<String>(
            initialValue: translator.target,
            decoration: const InputDecoration(labelText: 'Translate into'),
            items: [
              for (final target in translationTargets)
                DropdownMenuItem(value: target, child: Text(target)),
            ],
            onChanged: enabled && !translator.busy
                ? (value) => translator.setTarget(value!)
                : null,
          ),
          if (translator.backendName case final name?) ...[
            const SizedBox(height: 8),
            Text(
              'Translator ready · $name',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ],
      );
    },
  );
}

/// Sentences with their translations, then the sentence being spoken.
class LiveTranslationTranscript extends StatelessWidget {
  const LiveTranslationTranscript({super.key, required this.controller});
  final AudioController controller;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: controller.translator,
    builder: (context, _) {
      final translator = controller.translator;
      final theme = Theme.of(context);
      final muted = theme.textTheme.bodyMedium?.copyWith(
        color: theme.colorScheme.onSurfaceVariant,
      );
      final speaking = translator.unfinished(controller.transcript);
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (translator.error case final error?) ...[
            Text(error, style: TextStyle(color: theme.colorScheme.error)),
            const SizedBox(height: 12),
          ],
          for (final segment in translator.segments)
            _Segment(segment: segment, muted: muted),
          if (speaking.isNotEmpty)
            Text(speaking, style: muted?.copyWith(fontStyle: FontStyle.italic)),
          if (translator.pending > 0) ...[
            const SizedBox(height: 8),
            Text(
              'Translating · ${translator.pending} sentence'
              '${translator.pending == 1 ? '' : 's'} waiting',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ],
      );
    },
  );
}

class _Segment extends StatelessWidget {
  const _Segment({required this.segment, required this.muted});
  final TranslatedSegment segment;
  final TextStyle? muted;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SelectableText(segment.source, style: muted),
          const SizedBox(height: 4),
          if (segment.failed)
            Text(
              'Not translated',
              style: TextStyle(color: theme.colorScheme.error),
            )
          else if (segment.translation.isEmpty)
            const Align(
              alignment: Alignment.centerLeft,
              child: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            )
          else
            SelectableText(
              segment.translation,
              style: theme.textTheme.bodyLarge,
            ),
        ],
      ),
    );
  }
}
