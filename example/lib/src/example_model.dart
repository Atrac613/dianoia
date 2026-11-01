import 'package:dianoia/dianoia.dart';

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
  String get sizeLabel => isVision
      ? '${(downloadSpec.downloadBytes / 1e9).toStringAsFixed(2)} GB download · 1.54 GB installed'
      : '${(ModelSpec.lfm3Q4.downloadBytes / 1e9).toStringAsFixed(2)} GB download';
}
