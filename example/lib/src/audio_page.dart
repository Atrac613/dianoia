import 'dart:async';
import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:dianoia/dianoia.dart';

import 'audio_controller.dart';
import 'chat_page.dart';
import 'image_attachment.dart';
import 'example_model.dart';
import 'speech_model.dart';

/// A local-model preview, separate from text/image chat and its download UI.
class AudioTranscriptionPage extends StatefulWidget {
  const AudioTranscriptionPage({
    super.key,
    this.model = SpeechModel.audio,
    this.controller,
  });

  final SpeechModel model;

  /// An already configured controller; the route closes it when leaving.
  final AudioController? controller;
  @override
  State<AudioTranscriptionPage> createState() => _AudioTranscriptionPageState();
}

class _AudioTranscriptionPageState extends State<AudioTranscriptionPage>
    with WidgetsBindingObserver {
  late AudioController _audio =
      widget.controller ?? AudioController(model: widget.model);
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
      final model = _audio.model;
      final path = parent == null
          ? null
          : '${parent.path}/${model.directoryName}';
      setState(() => _defaultDirectory = path);
      if (path != null &&
          await File('$path/language/${model.receiptName}').exists() &&
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
      final path = await getDirectoryPath(confirmButtonText: 'Use this model');
      if (path != null && mounted) await _audio.load(path);
    } catch (e) {
      if (mounted) setState(() => _pathError = '$e');
    }
  }

  /// The chat runs its own worker, and model workers run one at a time: this
  /// screen's worker closes before the chat's starts, and a new one loads
  /// the model again on return.
  Future<void> _openChat(ExampleModel chat) async {
    final closed = _audio.close();
    await Navigator.of(
      context,
    ).push(MaterialPageRoute<void>(builder: (_) => ModelChatPage(model: chat)));
    await closed;
    // A supplied controller belongs to the caller and stays closed.
    if (!mounted || widget.controller != null) return;
    final previous = _audio;
    setState(() {
      _audio = AudioController(model: widget.model);
      _pathError = null;
    });
    previous.dispose();
    await _findModel();
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
                _audio.phase == AudioPhase.replying ||
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
    appBar: AppBar(
      title: Text(
        _audio.model == SpeechModel.omni ? 'Omni preview' : 'Audio preview',
      ),
      actions: [
        if (_audio.model.chatModel case final chat?)
          ListenableBuilder(
            listenable: _audio,
            builder: (context, _) => TextButton.icon(
              // Not while it records, replies or installs.
              onPressed: switch (_audio.phase) {
                AudioPhase.ready ||
                AudioPhase.needsModel ||
                AudioPhase.failed => () => _openChat(chat),
                _ => null,
              },
              icon: const Icon(Icons.chat_bubble_outline),
              label: const Text('Text chat'),
            ),
          ),
      ],
    ),
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
          'Download ${a.model.name} or choose a prepared model.',
        AudioPhase.loading => 'Checking and loading the model…',
        AudioPhase.installing =>
          a.installProgress?.stage == InstallStage.building
              ? 'Preparing the model on this device…'
              : 'Downloading the model…',
        AudioPhase.ready => 'Ready · ${a.backendName ?? ''}',
        AudioPhase.requestingPermission => 'Requesting microphone access…',
        AudioPhase.recording =>
          a.task == AudioTask.liveTranscription
              ? 'Transcribing live · ${a.recordedSeconds.toStringAsFixed(1)} seconds'
              : 'Recording · ${a.recordedSeconds.toStringAsFixed(1)} / 30 seconds',
        AudioPhase.finishing =>
          a.task == AudioTask.conversation
              ? 'Finishing the voice message…'
              : 'Finishing transcription…',
        AudioPhase.replying =>
          a.task == AudioTask.photoQuestion
              ? 'Looking at the photo and answering…'
              : 'Generating a spoken reply…',
        AudioPhase.resettingConversation => 'Starting a new conversation…',
        AudioPhase.synthesizing => 'Generating speech…',
        AudioPhase.playing => 'Playing speech…',
        AudioPhase.cancelling => 'Stopping…',
        AudioPhase.failed => 'The model could not be loaded.',
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
                  a.model.name,
                  style: Theme.of(context).textTheme.titleLarge,
                ),
                const SizedBox(height: 8),
                Text(
                  'Experimental · '
                  '${a.model.supportsLiveTranscription ? 'Live transcription continues until stopped. Other recording modes allow' : 'Recordings allow'} '
                  'up to 30 seconds. '
                  '${a.model.supportsPhotoQuestions ? 'Ask about a photo by voice or text, or enter' : 'Or enter'} '
                  'text to generate speech. '
                  '${a.model.supportsPhotoQuestions ? 'Audio and photos stay' : 'Audio stays'} on this device. '
                  'Recordings are not saved; generated speech uses a temporary playback file.',
                ),
                const SizedBox(height: 24),
                Text(status),
                if (a.phase == AudioPhase.loading ||
                    a.phase == AudioPhase.finishing ||
                    a.phase == AudioPhase.replying ||
                    a.phase == AudioPhase.synthesizing ||
                    a.phase == AudioPhase.cancelling) ...[
                  const SizedBox(height: 12),
                  const LinearProgressIndicator(),
                ],
                if (needsModel && a.storeDirectory != null) ...[
                  const SizedBox(height: 16),
                  Text(a.model.setupLabel),
                  const SizedBox(height: 8),
                  Text('License: ${a.model.spec.license}'),
                  SelectableText(a.model.spec.licenseUrl),
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
                DropdownButtonFormField<AudioTask>(
                  initialValue: a.task,
                  decoration: const InputDecoration(labelText: 'Audio mode'),
                  items: [
                    if (a.model.supportsPhotoQuestions)
                      const DropdownMenuItem(
                        value: AudioTask.photoQuestion,
                        child: Text('Ask about a photo'),
                      ),
                    const DropdownMenuItem(
                      value: AudioTask.transcription,
                      child: Text('Transcription and TTS'),
                    ),
                    const DropdownMenuItem(
                      value: AudioTask.conversation,
                      child: Text('Voice conversation (English)'),
                    ),
                    if (a.model.supportsLiveTranscription)
                      const DropdownMenuItem(
                        value: AudioTask.liveTranscription,
                        child: Text('Live transcription (English)'),
                      ),
                  ],
                  onChanged: ready ? (value) => a.setTask(value!) : null,
                ),
                const SizedBox(height: 16),
                if (a.task == AudioTask.conversation) ...[
                  const Text(
                    'Ask a question in English, then stop to receive text and speech. '
                    'Record again to continue the conversation. Replies play after generation. '
                    'Voice conversation currently supports English only.',
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      onPressed: ready ? a.newConversation : null,
                      icon: const Icon(Icons.add_comment_outlined),
                      label: const Text('New conversation'),
                    ),
                  ),
                ] else if (a.task == AudioTask.photoQuestion) ...[
                  _PhotoQuestion(controller: a, enabled: ready),
                ] else if (a.task == AudioTask.liveTranscription) ...[
                  const Text(
                    'English LiveASR · Continues until you stop. Words appear about once per second of audio, plus processing time. '
                    'Stop to finish the transcript. Cancel discards it. Japanese is available after recording in Transcription and TTS mode.',
                  ),
                ] else
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
                      DropdownMenuItem(
                        value: 'English',
                        child: Text('English'),
                      ),
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
                        label: Text(
                          a.task == AudioTask.conversation ||
                                  a.task == AudioTask.photoQuestion
                              ? 'Stop and reply'
                              : a.task == AudioTask.liveTranscription
                              ? 'Stop live transcription'
                              : 'Stop and transcribe',
                        ),
                      )
                    else
                      FilledButton.icon(
                        onPressed: ready ? a.record : null,
                        icon: const Icon(Icons.mic),
                        label: Text(
                          a.task == AudioTask.conversation ||
                                  a.task == AudioTask.photoQuestion
                              ? 'Record question'
                              : a.task == AudioTask.liveTranscription
                              ? 'Start live transcription'
                              : 'Record',
                        ),
                      ),
                    if (a.active &&
                        a.phase != AudioPhase.cancelling &&
                        a.phase != AudioPhase.installing &&
                        a.phase != AudioPhase.replying &&
                        a.phase != AudioPhase.synthesizing &&
                        a.phase != AudioPhase.playing)
                      TextButton(
                        onPressed: a.cancel,
                        child: Text(
                          a.task == AudioTask.conversation
                              ? 'Cancel voice message'
                              : 'Cancel recording',
                        ),
                      ),
                    if (a.phase == AudioPhase.replying ||
                        (a.phase == AudioPhase.playing &&
                            (a.task == AudioTask.conversation ||
                                a.task == AudioTask.photoQuestion)))
                      TextButton(
                        onPressed: a.cancel,
                        child: const Text('Stop reply'),
                      ),
                  ],
                ),
                const SizedBox(height: 24),
                if (a.task == AudioTask.photoQuestion) ...[
                  for (final exchange in a.photoExchanges) ...[
                    Text(
                      [
                        if (exchange.spoken) 'Spoken question',
                        if (exchange.question.isNotEmpty) exchange.question,
                      ].join(' · '),
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    SelectableText(
                      exchange.answer,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                    const SizedBox(height: 24),
                  ],
                  if (a.partialReply.isNotEmpty)
                    SelectableText(
                      a.partialReply,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                ] else if (a.task != AudioTask.conversation &&
                    a.transcript.isNotEmpty)
                  SelectableText(
                    a.transcript,
                    style: Theme.of(context).textTheme.bodyLarge,
                  ),
                if (a.task == AudioTask.conversation) ...[
                  for (var i = 0; i < a.voiceTurns.length; i++) ...[
                    Text(
                      'Voice message ${i + 1} · ${a.voiceTurns[i].seconds.toStringAsFixed(1)} s',
                      style: Theme.of(context).textTheme.labelLarge,
                    ),
                    const SizedBox(height: 8),
                    SelectableText(a.voiceTurns[i].reply),
                    const SizedBox(height: 24),
                  ],
                  if (a.partialReply.isNotEmpty) SelectableText(a.partialReply),
                ] else if (a.task == AudioTask.transcription) ...[
                  const SizedBox(height: 24),
                  const Divider(),
                  const SizedBox(height: 16),
                  _SpeechComposer(controller: a, enabled: ready),
                ],
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

class _PhotoQuestion extends StatefulWidget {
  const _PhotoQuestion({required this.controller, required this.enabled});
  final AudioController controller;
  final bool enabled;
  @override
  State<_PhotoQuestion> createState() => _PhotoQuestionState();
}

class _PhotoQuestionState extends State<_PhotoQuestion> {
  late final _text = TextEditingController(
    text: widget.controller.photoQuestion,
  );

  @override
  void dispose() {
    _text.dispose();
    super.dispose();
  }

  Future<void> _pick() async {
    try {
      final image = await pickChatImage();
      if (image != null) widget.controller.setPhoto(image);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('$e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final a = widget.controller;
    // The controller clears an answered question; keep the field in step.
    if (a.photoQuestion.isEmpty && _text.text.isNotEmpty) _text.clear();
    final thumbnail = a.photo?.thumbnail?.bytes;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text(
          'Choose a photo, then type a question or record one. Ask again to '
          'follow up; earlier answers stay in context until the photo changes. '
          'The photo is reduced to at most 256 image tokens and encoded once, '
          'which takes several seconds on the CPU.',
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            if (thumbnail != null)
              Padding(
                padding: const EdgeInsets.only(right: 12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.memory(thumbnail, height: 96),
                ),
              ),
            OutlinedButton.icon(
              onPressed: widget.enabled ? _pick : null,
              icon: const Icon(Icons.photo_outlined),
              label: Text(a.photo == null ? 'Choose photo' : 'Change photo'),
            ),
          ],
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _text,
          enabled: widget.enabled,
          minLines: 1,
          maxLines: 3,
          maxLength: 500,
          onChanged: (value) => a.photoQuestion = value,
          decoration: const InputDecoration(
            labelText: 'Question (optional with a recording)',
            border: OutlineInputBorder(),
          ),
        ),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Spoken reply'),
          subtitle: const Text('Answer with interleaved text and speech'),
          value: a.speakReply,
          onChanged: widget.enabled ? a.setSpeakReply : null,
        ),
        Align(
          alignment: Alignment.centerLeft,
          child: FilledButton.tonalIcon(
            onPressed: widget.enabled && a.photo != null ? a.askTyped : null,
            icon: const Icon(Icons.send),
            label: Text(
              a.photoExchanges.isEmpty ? 'Ask in text' : 'Follow up in text',
            ),
          ),
        ),
        if (a.photoExchanges.isNotEmpty)
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: widget.enabled ? a.newPhotoConversation : null,
              icon: const Icon(Icons.add_comment_outlined),
              label: const Text('New conversation'),
            ),
          ),
      ],
    );
  }
}
