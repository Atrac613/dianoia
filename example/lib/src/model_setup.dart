import 'package:dianoia/dianoia.dart';
import 'package:flutter/material.dart';

import 'chat_controller.dart';

/// Offers to download the model, naming its size and license, and shows
/// the download and the bundle build.
class ModelSetup extends StatelessWidget {
  const ModelSetup({super.key, required this.chat});

  final ChatController chat;

  @override
  Widget build(BuildContext context) {
    final spec = chat.spec;
    if (spec == null) return const SizedBox.shrink();
    final theme = Theme.of(context);
    final installing = chat.setup == Setup.installing;
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 480),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(spec.name, style: theme.textTheme.titleLarge),
              const SizedBox(height: 8),
              Text(
                '${_gb(spec.downloadBytes)} GB to download from Hugging Face, '
                'once. The model then runs on this device: what you type '
                'stays here, along with any images you send.',
              ),
              if (spec is VisionGgufSpec) ...[
                const SizedBox(height: 12),
                const Text(
                  'Allow at least 5 GB of free storage during download and setup. '
                  'The installed model uses about 1.54 GB.',
                ),
              ],
              if (spec is VisionCheckpointSpec) ...[
                const SizedBox(height: 12),
                Text(
                  'Allow at least 13 GB of free storage during the first '
                  'download and conversion. The installed '
                  '${spec.quantization.name.toUpperCase()} model uses about '
                  '${spec.quantization == VisionQuantization.q4 ? '2.06' : '2.89'} GB.',
                ),
              ],
              const SizedBox(height: 12),
              Text('License: ${spec.license}'),
              SelectableText(
                spec.licenseUrl,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.primary,
                ),
              ),
              const SizedBox(height: 24),
              if (installing) ..._progress(context) else _download(context),
            ],
          ),
        ),
      ),
    );
  }

  Widget _download(BuildContext context) {
    final error = chat.installError;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (error != null)
          Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: SelectableText(
              error,
              style: TextStyle(color: Theme.of(context).colorScheme.error),
            ),
          ),
        FilledButton.icon(
          onPressed: chat.installModel,
          icon: const Icon(Icons.download),
          label: Text(error == null ? 'Download' : 'Resume download'),
        ),
      ],
    );
  }

  List<Widget> _progress(BuildContext context) {
    final building = chat.installStage != InstallStage.downloading;
    final total = chat.installTotal;
    return [
      LinearProgressIndicator(
        value: building || total == 0 ? null : chat.installBytes / total,
      ),
      const SizedBox(height: 8),
      Text(
        building
            ? (chat.spec is VisionGgufSpec
                  ? 'Preparing the model on this device…'
                  : 'Building the model on this device…')
            : '${_gb(chat.installBytes)} of ${_gb(total)} GB',
      ),
      if (!building)
        Align(
          alignment: Alignment.centerRight,
          child: TextButton(
            onPressed: chat.cancelInstall,
            child: const Text('Cancel'),
          ),
        ),
    ];
  }

  static String _gb(int bytes) => (bytes / 1e9).toStringAsFixed(2);
}
