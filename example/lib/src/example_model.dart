import 'package:dianoia/dianoia.dart';
import 'model_size.dart';

/// Models offered by the example, downloaded or supplied as local bundles.
enum ExampleModel {
  lfm3,

  /// Qwen3-0.6B for text and tools, in Qwen's official Q8_0 build.
  qwen3,
  lfm3Vl,

  /// LFM3-Omni as a text chat with images, opened from its own screen,
  /// which shares the installed bundle.
  omni;

  /// A mobile CPU encoder budget; the library's default remains 1024.
  static const maxImageTokens = 256;

  /// Loaded through the public LFM3-VL API.
  bool get isVision => this == lfm3Vl;

  /// Offers an image button next to the message field.
  bool get acceptsImages => this != lfm3 && this != qwen3;

  /// How replies pick their tokens: Qwen recommends sampling for Qwen3,
  /// LFM3 runs greedily.
  SamplingParams get sampling =>
      this == qwen3 ? SamplingParams.qwen3NonThinking : SamplingParams.greedy;
  String get id => downloadSpec.id;
  String get name => downloadSpec.name;

  String get directoryName => switch (this) {
    lfm3 => 'LFM3-1.7B-QAD-Q4',
    qwen3 => 'Qwen3-0.6B-Q8',
    lfm3Vl => 'LFM3-VL-2B-QAD-Q4',
    omni => 'LFM3-Omni-2B-Q4',
  };
  String get license => downloadSpec.license;
  String get licenseUrl => downloadSpec.licenseUrl;

  InstallableModel get downloadSpec => switch (this) {
    lfm3 => ModelSpec.lfm3Q4,
    qwen3 => ModelSpec.qwen3_0_6bQ8,
    lfm3Vl => VisionGgufSpec.lfm3VlQadQ4,
    omni => OmniGgufSpec.lfm3OmniQ4,
  };
  String get sizeLabel {
    final spec = downloadSpec;
    final installed = spec.installedBytes;
    return '${gigabytes(spec.downloadBytes)} GB download'
        '${installed == null ? '' : ' · ${gigabytes(installed)} GB after setup'}';
  }
}
