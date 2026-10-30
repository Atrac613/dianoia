import 'package:flutter/material.dart';

import 'chat_controller.dart';

/// One message: the user's on the right, the model's on the left, with the
/// reply's speed (or the reason it failed) underneath.
class MessageBubble extends StatelessWidget {
  const MessageBubble({super.key, required this.message});

  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final mine = message.role == ChatRole.user;
    final stats = message.stats;
    return Align(
      alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.sizeOf(context).width * 0.8,
        ),
        child: Column(
          crossAxisAlignment: mine
              ? CrossAxisAlignment.end
              : CrossAxisAlignment.start,
          children: [
            Container(
              margin: const EdgeInsets.symmetric(vertical: 4),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
              decoration: BoxDecoration(
                color: mine
                    ? scheme.primaryContainer
                    : scheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(16),
              ),
              child: message.text.isEmpty
                  ? const _Typing()
                  : SelectableText(
                      message.text,
                      style: TextStyle(
                        color: mine
                            ? scheme.onPrimaryContainer
                            : scheme.onSurface,
                      ),
                    ),
            ),
            if (stats != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Text(
                  stats,
                  style: Theme.of(context).textTheme.labelSmall?.copyWith(
                    color: message.failed
                        ? scheme.error
                        : scheme.onSurfaceVariant,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// Shown until the first token of a reply arrives.
class _Typing extends StatelessWidget {
  const _Typing();

  @override
  Widget build(BuildContext context) => const SizedBox(
    width: 24,
    height: 16,
    child: Center(
      child: SizedBox.square(
        dimension: 14,
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
    ),
  );
}
