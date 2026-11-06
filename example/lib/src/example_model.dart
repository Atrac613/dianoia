import 'package:dianoia/dianoia.dart';
import 'model_size.dart';

/// Models offered by the example, downloaded or supplied as local bundles.
enum ExampleModel {
  lfm3,
  lfm3Vl,

  /// LFM3-Omni as a text chat with images; its speech features live on the
  /// audio page, which shares the installed bundle.
  omni;

  /// A mobile CPU encoder budget; the library's default remains 1024.
  static const maxImageTokens = 256;

  /// Loaded through the public LFM3-VL API.
  bool get isVision => this == lfm3Vl;

  /// Offers an image button next to the message field.
  bool get acceptsImages => this != lfm3;
  String get id => downloadSpec.id;
  String get name => downloadSpec.name;

  /// The model list shows Omni twice: here, and as the audio page.
  String get title => this == omni ? '$name · Chat' : name;
  String get directoryName => switch (this) {
    lfm3 => 'LFM3-1.7B-QAD-Q4',
    lfm3Vl => 'LFM3-VL-2B-QAD-Q4',
    omni => 'LFM3-Omni-2B-Q4',
  };
  String get license => downloadSpec.license;
  String get licenseUrl => downloadSpec.licenseUrl;

  InstallableModel get downloadSpec => switch (this) {
    lfm3 => ModelSpec.lfm3Q4,
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
