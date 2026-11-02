import 'dart:async';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dianoia/dianoia.dart';

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
      if (!mounted) return;
      final storeParent = Platform.isWindows
          ? await getApplicationCacheDirectory()
          : await getApplicationSupportDirectory();
      if (!mounted) return;
      final store = '${storeParent.path}/models';
      _audio.storeDirectory = store;
      final path = parent == null ? null : '${parent.path}/LFM3-Audio-2B-Q4';
      setState(() => _defaultDirectory = path);
      if (path != null &&
          await File('$path/language/audio_source.json').exists() &&
          mounted) {
        await _audio.load(path);
      } else if (mounted) {
        await _audio.discover(store);
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
                _audio.phase == AudioPhase.finishing ||
                _audio.phase == AudioPhase.synthesizing ||
                _audio.phase == AudioPhase.playing));
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
    appBar: AppBar(title: const Text('Audio preview')),
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
          'Download LFM3-Audio-2B Q4_0 or choose a prepared model.',
        AudioPhase.loading => 'Checking and loading the audio model…',
        AudioPhase.installing =>
          a.installProgress?.stage == InstallStage.building
              ? 'Preparing the model on this device…'
              : 'Downloading the audio model…',
        AudioPhase.ready => 'Ready · ${a.backendName ?? ''}',
        AudioPhase.requestingPermission => 'Requesting microphone access…',
        AudioPhase.recording =>
          'Recording · ${a.recordedSeconds.toStringAsFixed(1)} / 30 seconds',
        AudioPhase.finishing => 'Finishing transcription…',
        AudioPhase.synthesizing => 'Generating speech…',
        AudioPhase.playing => 'Playing speech…',
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
                  'Or enter text to generate speech. Audio stays on this device. '
                  'Recordings are not saved; generated speech uses a temporary playback file.',
                ),
                const SizedBox(height: 24),
                Text(status),
                if (a.phase == AudioPhase.loading ||
                    a.phase == AudioPhase.finishing ||
                    a.phase == AudioPhase.synthesizing ||
                    a.phase == AudioPhase.cancelling) ...[
                  const SizedBox(height: 12),
                  const LinearProgressIndicator(),
                ],
                if (needsModel && a.storeDirectory != null) ...[
                  const SizedBox(height: 16),
                  const Text(
                    '1.23 GB to download from Hugging Face. Allow at least 4 GB of free storage during setup; about 1.30 GB remains installed.',
                  ),
                  const SizedBox(height: 8),
                  const Text('License: LFM Open License v1.0'),
                  SelectableText(AudioGgufSpec.lfm3AudioQ4.licenseUrl),
                  const SizedBox(height: 16),
                  FilledButton.icon(
                    onPressed: a.downloadModel,
                    icon: const Icon(Icons.download),
                    label: Text(
                      a.error == null ? 'Download' : 'Resume download',
                    ),
                  ),
                ],
                if (a.phase == AudioPhase.installing) ...[
                  const SizedBox(height: 16),
                  LinearProgressIndicator(
                    value: a.installProgress?.stage == InstallStage.downloading
                        ? a.installProgress!.fraction
                        : null,
                  ),
                  const SizedBox(height: 8),
                  if (a.installProgress?.stage == InstallStage.downloading)
                    Text(
                      '${(a.installProgress!.bytes / 1e9).toStringAsFixed(2)} of ${(a.installProgress!.totalBytes / 1e9).toStringAsFixed(2)} GB',
                    ),
                  TextButton(
                    onPressed: a.cancel,
                    child: const Text('Cancel download'),
                  ),
                ],
                if (needsModel) ...[
                  const SizedBox(height: 16),
                  const Text(
                    'You can also use a model prepared on another computer.',
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
                    if (a.active &&
                        a.phase != AudioPhase.cancelling &&
                        a.phase != AudioPhase.installing &&
                        a.phase != AudioPhase.synthesizing &&
                        a.phase != AudioPhase.playing)
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
                const SizedBox(height: 24),
                const Divider(),
                const SizedBox(height: 16),
                _SpeechComposer(controller: a, enabled: ready),
              ],
            ),
          ),
        ),
      );
    },
  );
}

class _SpeechComposer extends StatefulWidget {
  const _SpeechComposer({required this.controller, required this.enabled});
  final AudioController controller;
  final bool enabled;
  @override
  State<_SpeechComposer> createState() => _SpeechComposerState();
}

class _SpeechComposerState extends State<_SpeechComposer> {
  final _text = TextEditingController();
  String _language = 'ja', _voice = 'female';
  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      Text('Text to speech', style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      const Text(
        'Use a short sentence. Speech is generated before playback, up to 30 seconds.',
      ),
      const SizedBox(height: 16),
      TextField(
        controller: _text,
        enabled: widget.enabled,
        minLines: 2,
        maxLines: 4,
        maxLength: 1000,
        decoration: const InputDecoration(
          labelText: 'Text to speak',
          border: OutlineInputBorder(),
        ),
      ),
      const SizedBox(height: 8),
      Row(
        children: [
          Expanded(
            child: DropdownButtonFormField<String>(
              initialValue: _language,
              decoration: const InputDecoration(labelText: 'Output language'),
              items: const [
                DropdownMenuItem(value: 'ja', child: Text('Japanese')),
                DropdownMenuItem(value: 'en', child: Text('English')),
              ],
              onChanged: widget.enabled
                  ? (value) => setState(() => _language = value!)
                  : null,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: DropdownButtonFormField<String>(
              initialValue: _voice,
              decoration: const InputDecoration(labelText: 'Voice'),
              items: const [
                DropdownMenuItem(value: 'female', child: Text('Female')),
                DropdownMenuItem(value: 'male', child: Text('Male')),
              ],
              onChanged: widget.enabled
                  ? (value) => setState(() => _voice = value!)
                  : null,
            ),
          ),
        ],
      ),
      const SizedBox(height: 16),
      Align(
        alignment: Alignment.centerLeft,
        child: FilledButton.icon(
          onPressed:
              widget.controller.phase == AudioPhase.synthesizing ||
                  widget.controller.phase == AudioPhase.playing
              ? widget.controller.cancel
              : widget.enabled
              ? () => widget.controller.speak(
                  _text.text,
                  language: _language,
                  voice: _voice,
                )
              : null,
          icon: Icon(
            widget.controller.phase == AudioPhase.synthesizing ||
                    widget.controller.phase == AudioPhase.playing
                ? Icons.stop
                : Icons.volume_up,
          ),
          label: Text(
            widget.controller.phase == AudioPhase.synthesizing ||
                    widget.controller.phase == AudioPhase.playing
                ? 'Stop speech'
                : 'Generate and play',
          ),
        ),
      ),
    ],
  );
}
