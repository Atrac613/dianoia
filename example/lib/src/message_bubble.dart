import 'package:flutter/material.dart';

import 'chat_controller.dart';
import 'image_attachment.dart';

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
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (message.image case final image?)
                    ThumbnailPreview(image: image),
                  if (message.text.isNotEmpty)
                    SelectableText(
                      message.text,
                      style: TextStyle(
                        color: mine
                            ? scheme.onPrimaryContainer
                            : scheme.onSurface,
                      ),
                    )
                  else if (!mine && message.pending)
                    _Typing(status: message.progress),
                  if (message.text.isEmpty && !mine && !message.pending)
                    const Text('No reply text.'),
                ],
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
  const _Typing({required this.status});
  final String status;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      const SizedBox.square(
        dimension: 14,
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
      const SizedBox(width: 8),
      Flexible(child: Text(status)),
    ],
  );
}
