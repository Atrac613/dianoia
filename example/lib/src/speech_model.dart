import 'package:dianoia/dianoia.dart';

import 'model_size.dart';

/// The speech-capable models the audio preview can run. Both share the
/// transcription, speech and voice conversation paths; Omni adds questions
/// about photos, and LiveASR exists for LFM3-Audio only.
enum SpeechModel {
  audio,
  omni;

  InstallableModel get spec => switch (this) {
    audio => AudioGgufSpec.lfm3AudioQ4,
    omni => OmniGgufSpec.lfm3OmniQ4,
  };

  String get name => spec.name;
  String get directoryName => switch (this) {
    audio => 'LFM3-Audio-2B-Q4',
    omni => 'LFM3-Omni-2B-Q4',
  };

  /// The receipt a prepared research folder carries in `language/`.
  String get receiptName => switch (this) {
    audio => 'audio_source.json',
    omni => 'omni_source.json',
  };

  bool get supportsLiveTranscription => this == audio;
  bool get supportsPhotoQuestions => this == omni;

  String get summary => switch (this) {
    audio => 'Voice conversation, transcription and TTS',
    omni =>
      'Questions about photos by voice or text, transcription, TTS '
          'and voice conversation',
  };

  String get sizeLabel {
    final installed = spec.installedBytes;
    return '${gigabytes(spec.downloadBytes)} GB download'
        '${installed == null ? '' : ' · ${gigabytes(installed)} GB after setup'}';
  }

  String get setupLabel =>
      '${gigabytes(spec.downloadBytes)} GB to download from Hugging Face. '
      'Allow at least ${gigabytes(spec.setupBytes ?? 0)} GB of free storage '
      'during setup; about ${gigabytes(spec.installedBytes ?? 0)} GB remains '
      'installed.';
}
