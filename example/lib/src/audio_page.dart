import 'dart:async';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';

import 'audio_controller.dart';

/// A local-model preview, separate from text/image chat and its download UI.
class AudioTranscriptionPage extends StatefulWidget {
  const AudioTranscriptionPage({super.key, this.controller});

  /// An already configured controller; the route closes it when leaving.
  final AudioController? controller;
  @override
  State<AudioTranscriptionPage> createState() => _AudioTranscriptionPageState();
}

class _AudioTranscriptionPageState extends State<AudioTranscriptionPage>
    with WidgetsBindingObserver {
  late final _audio = widget.controller ?? AudioController();
  String? _defaultDirectory, _pathError;
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    if (widget.controller == null) unawaited(_findModel());
  }

  Future<void> _findModel() async {
    try {
      final parent = Platform.isAndroid
          ? await getExternalStorageDirectory()
          : Platform.isWindows
          ? await getApplicationCacheDirectory()
          : await getApplicationDocumentsDirectory();
      if (!mounted || parent == null) return;
      final path = '${parent.path}/LFM3-Audio-2B-Q4';
      setState(() => _defaultDirectory = path);
      if (await File('$path/language/audio_source.json').exists() && mounted) {
        await _audio.load(path);
      }
    } catch (e) {
      if (mounted) setState(() => _pathError = '$e');
    }
  }

  Future<void> _choose() async {
    try {
      final path = await getDirectoryPath(confirmButtonText: 'Use audio model');
      if (path != null && mounted) await _audio.load(path);
    } catch (e) {
      if (mounted) setState(() => _pathError = '$e');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final leaving =
        state == AppLifecycleState.paused ||
        state == AppLifecycleState.hidden ||
        state == AppLifecycleState.detached ||
        (state == AppLifecycleState.inactive &&
            (_audio.phase == AudioPhase.recording ||
                _audio.phase == AudioPhase.finishing));
    if (leaving) unawaited(_audio.cancel());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    if (widget.controller == null) {
      _audio.dispose();
    } else {
      unawaited(_audio.close());
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Audio transcription')),
    body: AudioTranscriptionView(
      controller: _audio,
      directory: _defaultDirectory,
      setupError: _pathError,
      onChoose: Platform.isMacOS || Platform.isWindows || Platform.isLinux
          ? _choose
          : null,
      onRetry: _findModel,
    ),
  );
}

class AudioTranscriptionView extends StatelessWidget {
  const AudioTranscriptionView({
    super.key,
    required this.controller,
    this.directory,
    this.setupError,
    this.onChoose,
    this.onRetry,
  });
  final AudioController controller;
  final String? directory, setupError;
  final VoidCallback? onChoose, onRetry;

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: controller,
    builder: (context, _) {
      final a = controller;
      final ready = a.phase == AudioPhase.ready;
      final needsModel =
          a.phase == AudioPhase.needsModel || a.phase == AudioPhase.failed;
      final status = switch (a.phase) {
        AudioPhase.needsModel =>
          'Choose a prepared LFM3-Audio-2B Q4_0 model to begin.',
        AudioPhase.loading => 'Checking and loading the audio model…',
        AudioPhase.ready => 'Ready · ${a.backendName ?? ''}',
        AudioPhase.requestingPermission => 'Requesting microphone access…',
        AudioPhase.recording =>
          'Recording · ${a.recordedSeconds.toStringAsFixed(1)} / 30 seconds',
        AudioPhase.finishing => 'Finishing transcription…',
        AudioPhase.cancelling => 'Stopping…',
        AudioPhase.failed => 'The audio model could not be loaded.',
        AudioPhase.closed => 'Closed',
      };
      return SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 640),
            child: ListView(
              padding: const EdgeInsets.all(24),
              children: [
                Text(
                  'LFM3-Audio-2B Q4_0',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                const Text(
                  'Experimental · Record up to 30 seconds, then stop to see the transcript. '
                  'Audio stays on this device and is not saved.',
                ),
                const SizedBox(height: 24),
                Text(status),
                if (a.phase == AudioPhase.loading ||
                    a.phase == AudioPhase.finishing ||
                    a.phase == AudioPhase.cancelling) ...[
                  const SizedBox(height: 12),
                  const LinearProgressIndicator(),
                ],
                if (needsModel) ...[
                  const SizedBox(height: 16),
                  const Text(
                    'This preview uses a locally prepared model. See the example guide for setup.',
                  ),
                  if (directory != null) ...[
                    const SizedBox(height: 8),
                    SelectableText('Model folder: $directory'),
                  ],
                  const SizedBox(height: 16),
                  Wrap(
                    spacing: 8,
                    children: [
                      if (onChoose != null)
                        FilledButton.icon(
                          onPressed: onChoose,
                          icon: const Icon(Icons.folder_open),
                          label: const Text('Choose model folder'),
                        ),
                      if (onRetry != null)
                        TextButton(
                          onPressed: onRetry,
                          child: const Text('Check again'),
                        ),
                    ],
                  ),
                ],
                if (a.error ?? setupError case final String error) ...[
                  const SizedBox(height: 16),
                  SelectableText(
                    error,
                    style: TextStyle(
                      color: Theme.of(context).colorScheme.error,
                    ),
                  ),
                ],
                const SizedBox(height: 24),
                DropdownButtonFormField<String>(
                  initialValue: a.language ?? 'auto',
                  decoration: const InputDecoration(
                    labelText: 'Speech language',
                  ),
                  items: const [
                    DropdownMenuItem(
                      value: 'auto',
                      child: Text('Detect language'),
                    ),
                    DropdownMenuItem(
                      value: 'Japanese',
                      child: Text('Japanese'),
                    ),
                    DropdownMenuItem(value: 'English', child: Text('English')),
                  ],
                  onChanged: ready
                      ? (value) {
                          a.language = value == 'auto' ? null : value;
                        }
                      : null,
                ),
                const SizedBox(height: 16),
                Wrap(
                  spacing: 12,
                  runSpacing: 8,
                  children: [
                    if (a.phase == AudioPhase.recording)
                      FilledButton.icon(
                        onPressed: a.stop,
                        icon: const Icon(Icons.stop),
                        label: const Text('Stop and transcribe'),
                      )
                    else
                      FilledButton.icon(
                        onPressed: ready ? a.record : null,
                        icon: const Icon(Icons.mic),
                        label: const Text('Record'),
                      ),
                    if (a.active && a.phase != AudioPhase.cancelling)
                      TextButton(
                        onPressed: a.cancel,
                        child: const Text('Cancel recording'),
                      ),
                  ],
                ),
                const SizedBox(height: 24),
                if (a.transcript.isNotEmpty)
                  SelectableText(
                    a.transcript,
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
              ],
            ),
          ),
        ),
      );
    },
  );
}
