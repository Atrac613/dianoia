import 'package:dianoia/dianoia.dart';
import 'model_size.dart';

/// Models offered by the example, downloaded or supplied as local bundles.
enum ExampleModel {
  lfm3,
  lfm3Vl;

  bool get isVision => this == lfm3Vl;
  String get id => downloadSpec.id;
  String get name => downloadSpec.name;
  String get directoryName =>
      isVision ? 'LFM3-VL-2B-QAD-Q4' : 'LFM3-1.7B-QAD-Q4';
  String get license => downloadSpec.license;
  String get licenseUrl => downloadSpec.licenseUrl;

  InstallableModel get downloadSpec =>
      isVision ? VisionGgufSpec.lfm3VlQadQ4 : ModelSpec.lfm3Q4;
  String get sizeLabel {
    final spec = downloadSpec;
    final installed = spec.installedBytes;
    return '${gigabytes(spec.downloadBytes)} GB download'
        '${installed == null ? '' : ' · ${gigabytes(installed)} GB after setup'}';
  }
}
