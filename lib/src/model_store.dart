/// Getting a model onto the device: its files from where the model's authors
/// publish them, pinned by revision and checked by SHA-256, and the bundle
/// dianoia runs built from them on the device, or a prebuilt VL bundle.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:crypto/crypto.dart';

import 'download.dart';
import 'gguf.dart';
import 'native.dart';
import 'model.dart';
import 'quant.dart';
import 'vision_backbone.dart';
import 'vision_bundle.dart';
import 'vision_gguf.dart';

export 'download.dart' show ModelDownloadException, RemoteFile;

/// A pinned checkpoint or prebuilt vision bundle installed by [ModelStore].
sealed class InstallableModel {
  const InstallableModel();

  String get id;
  String get name;
  String get license;
  String get licenseUrl;
  List<RemoteFile> get files;

  /// Total bytes downloaded, including metadata and license files.
  int get downloadBytes => files.fold(0, (sum, file) => sum + file.size);
}

/// Language matrix quantization; the tied output head stays Q8.
enum VisionQuantization { q8, q4 }

/// An upstream BF16 VL checkpoint converted into a standalone Q8/Q4 bundle.
/// Installation imports its language backbone, quantizes it, then packs it.
final class VisionCheckpointSpec extends InstallableModel {
  const VisionCheckpointSpec({
    required this.id,
    required this.name,
    required this.license,
    required this.licenseUrl,
    required this.files,
    this.quantization = VisionQuantization.q8,
  });

  @override
  final String id;
  @override
  final String name;
  @override
  final String license;
  @override
  final String licenseUrl;
  @override
  final List<RemoteFile> files;

  final VisionQuantization quantization;

  VisionCheckpointSpec fromMirror(String baseUrl) {
    final base = Uri.parse(baseUrl.endsWith('/') ? baseUrl : '$baseUrl/');
    if (!['http', 'https'].contains(base.scheme) ||
        !base.hasAuthority ||
        base.query.isNotEmpty ||
        base.fragment.isNotEmpty) {
      throw ArgumentError.value(
        baseUrl,
        'baseUrl',
        'expected HTTP(S) directory',
      );
    }
    return VisionCheckpointSpec(
      id: id,
      name: name,
      license: license,
      licenseUrl: licenseUrl,
      quantization: quantization,
      files: List.unmodifiable([
        for (final f in files)
          RemoteFile(
            f.name,
            base.resolve(f.name).toString(),
            size: f.size,
            sha256: f.sha256,
          ),
      ]),
    );
  }

  static const _repo =
      'https://huggingface.co/LiquidAI/LFM3-VL-2B/resolve/'
      'ba44bd51db20a2753f28c273a1571ccaa210d954';

  /// 3.85 GB downloaded, about 2.89 GB installed. Allow at least 13 GB free
  /// during conversion, including the temporary full language backbone.
  static const lfm3VlQ8 = VisionCheckpointSpec(
    id: 'lfm3-vl-2b-q8-ba44bd5',
    name: 'LFM3-VL-2B (Q8)',
    license: 'LFM Open License v1.0',
    licenseUrl:
        'https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/ba44bd51db20a2753f28c273a1571ccaa210d954/LICENSE',
    files: _lfm3VlFiles,
  );

  /// The same pinned checkpoint built as Q4: about 2.06 GB installed.
  /// Source download bytes are unchanged; the tied output head stays Q8.
  static const lfm3VlQ4 = VisionCheckpointSpec(
    id: 'lfm3-vl-2b-q4-ba44bd5',
    name: 'LFM3-VL-2B (Q4)',
    license: 'LFM Open License v1.0',
    licenseUrl:
        'https://huggingface.co/LiquidAI/LFM3-VL-2B/blob/ba44bd51db20a2753f28c273a1571ccaa210d954/LICENSE',
    quantization: VisionQuantization.q4,
    files: _lfm3VlFiles,
  );

  static const _lfm3VlFiles = [
    RemoteFile(
      'config.json',
      '$_repo/config.json',
      size: 2905,
      sha256:
          '18a02e8b5963e38c98683e2c948705ea45220d75f017e9b4bdd9cf54692e6dc3',
    ),
    RemoteFile(
      'processor_config.json',
      '$_repo/processor_config.json',
      size: 1535,
      sha256:
          'd3d5c3f9dc3d3f25c3177ba8bb05938b53c51799244af1ca6406cde78924b84e',
    ),
    RemoteFile(
      'tokenizer.json',
      '$_repo/tokenizer.json',
      size: 17908258,
      sha256:
          'a39eed58008cbcbc09e5b629b81338a654269c64cd13348aaa6090f4db75c7e7',
    ),
    RemoteFile(
      'LICENSE',
      '$_repo/LICENSE',
      size: 10574,
      sha256:
          '30adf9d6478191fb87f2424f63ba0728598335aaf99cd2848ef17e8e545fe94b',
    ),
    RemoteFile(
      'model.safetensors',
      '$_repo/model.safetensors',
      size: 3836417920,
      sha256:
          'a9bb8b1c159c9d6c3007bc1c7ad8018b599f759d97da2701714e37426283e131',
    ),
  ];
}

/// Paired official language QAD Q4_0 and BF16 vision GGUFs, imported on device.
final class VisionGgufSpec extends InstallableModel {
  const VisionGgufSpec({
    required this.id,
    required this.name,
    required this.license,
    required this.licenseUrl,
    required this.language,
    required this.vision,
    required this.config,
    required this.processor,
    required this.tokenizer,
    required this.licenseFile,
  });

  @override
  final String id;
  @override
  final String name;
  @override
  final String license;
  @override
  final String licenseUrl;
  final RemoteFile language, vision, config, processor, tokenizer, licenseFile;
  @override
  List<RemoteFile> get files => [
    config,
    processor,
    tokenizer,
    licenseFile,
    language,
    vision,
  ];

  VisionGgufSpec fromMirror(String baseUrl) {
    final base = Uri.parse(baseUrl.endsWith('/') ? baseUrl : '$baseUrl/');
    if (!['http', 'https'].contains(base.scheme) ||
        !base.hasAuthority ||
        base.query.isNotEmpty ||
        base.fragment.isNotEmpty) {
      throw ArgumentError.value(
        baseUrl,
        'baseUrl',
        'expected HTTP(S) directory',
      );
    }
    RemoteFile at(RemoteFile f) => RemoteFile(
      f.name,
      base.resolve(f.name).toString(),
      size: f.size,
      sha256: f.sha256,
    );
    return VisionGgufSpec(
      id: id,
      name: name,
      license: license,
      licenseUrl: licenseUrl,
      language: at(language),
      vision: at(vision),
      config: at(config),
      processor: at(processor),
      tokenizer: at(tokenizer),
      licenseFile: at(licenseFile),
    );
  }

  static const _repo =
      'https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF/resolve/'
      'b36554cfd35854b06429900131730cd80f092cdb';

  /// 1.45 GB downloaded, about 1.54 GB installed; allow 5 GB during setup.
  static final lfm3VlQadQ4 = VisionGgufSpec(
    id: 'lfm3-vl-2b-qad-q4-b36554c',
    name: 'LFM3-VL-2B (QAD Q4_0)',
    license: VisionCheckpointSpec.lfm3VlQ8.license,
    licenseUrl: VisionCheckpointSpec.lfm3VlQ8.licenseUrl,
    language: const RemoteFile(
      'LFM3-VL-2B-QAD-Q4_0.gguf',
      '$_repo/LFM3-VL-2B-QAD-Q4_0.gguf',
      size: 1049615648,
      sha256:
          '621c455376cb96a20891697deb9975c6fb47dd5ea98ed3ad1f3b7e8f24f0b1e6',
    ),
    vision: const RemoteFile(
      'mmproj-LFM3-VL-2B-BF16.gguf',
      '$_repo/mmproj-LFM3-VL-2B-BF16.gguf',
      size: 378996032,
      sha256:
          '8e75a88afc0e6a3bcd637becbe720447ea0794934b7b417a1f42d5cfe0d9c288',
    ),
    config: VisionCheckpointSpec._lfm3VlFiles.firstWhere(
      (f) => f.name == 'config.json',
    ),
    processor: VisionCheckpointSpec._lfm3VlFiles.firstWhere(
      (f) => f.name == 'processor_config.json',
    ),
    tokenizer: VisionCheckpointSpec._lfm3VlFiles.firstWhere(
      (f) => f.name == 'tokenizer.json',
    ),
    licenseFile: VisionCheckpointSpec._lfm3VlFiles.firstWhere(
      (f) => f.name == 'LICENSE',
    ),
  );
}

/// A prebuilt `pack-vl` bundle hosted by an app or its model distributor.
///
/// The application supplies a trusted, pinned manifest, for example from an
/// asset reviewed with its release. Never trust an unverified manifest fetched
/// from the same remote server: it defines the expected payload hashes.
final class VisionModelSpec extends InstallableModel {
  VisionModelSpec._({
    required this.id,
    required this.name,
    required this.license,
    required this.licenseUrl,
    required this.files,
  });

  /// [manifestJson] is the exact `vl_bundle.json` produced by `pack-vl`.
  /// All files, including that manifest, are fetched beneath [baseUrl] and
  /// pinned by their sizes and SHA-256. Whitespace in the manifest is significant.
  factory VisionModelSpec.fromManifest({
    required String id,
    required String name,
    required String license,
    required String licenseUrl,
    required String baseUrl,
    required String manifestJson,
  }) {
    final manifest = VlBundleManifest.parse(manifestJson);
    final bytes = utf8.encode(manifestJson);
    final files = [
      RemoteFile(
        'vl_bundle.json',
        '',
        size: bytes.length,
        sha256: sha256.convert(bytes).toString(),
      ),
      for (final entry in manifest.files.entries)
        RemoteFile(
          entry.key,
          '',
          size: entry.value.bytes,
          sha256: entry.value.sha256,
        ),
    ];
    return VisionModelSpec._(
      id: id,
      name: name,
      license: license,
      licenseUrl: licenseUrl,
      files: List.unmodifiable(files),
    ).fromMirror(baseUrl);
  }

  @override
  final String id;
  @override
  final String name;
  @override
  final String license;
  @override
  final String licenseUrl;
  @override
  final List<RemoteFile> files;

  /// Changes only the download host/path, preserving all pinned bytes.
  VisionModelSpec fromMirror(String baseUrl) {
    final base = Uri.parse(baseUrl.endsWith('/') ? baseUrl : '$baseUrl/');
    if (!['http', 'https'].contains(base.scheme) ||
        !base.hasAuthority ||
        base.query.isNotEmpty ||
        base.fragment.isNotEmpty) {
      throw ArgumentError.value(
        baseUrl,
        'baseUrl',
        'expected HTTP(S) directory',
      );
    }
    return VisionModelSpec._(
      id: id,
      name: name,
      license: license,
      licenseUrl: licenseUrl,
      files: List.unmodifiable([
        for (final file in files)
          RemoteFile(
            file.name,
            base.resolve(file.name).toString(),
            size: file.size,
            sha256: file.sha256,
          ),
      ]),
    );
  }
}

/// A model [ModelStore] can install: a GGUF checkpoint and the matching
/// Hugging Face model's config, tokenizer and license.
final class ModelSpec extends InstallableModel {
  const ModelSpec({
    required this.id,
    required this.name,
    required this.license,
    required this.licenseUrl,
    required this.gguf,
    required this.config,
    required this.tokenizer,
    required this.licenseFile,
  });

  /// The model's directory name in a store.
  @override
  final String id;
  @override
  final String name;

  /// The license the weights come under, which an app should show before
  /// downloading them, and where to read it.
  @override
  final String license;
  @override
  final String licenseUrl;

  final RemoteFile gguf, config, tokenizer, licenseFile;

  @override
  List<RemoteFile> get files => [config, tokenizer, licenseFile, gguf];

  /// This model with every file fetched from [baseUrl] instead, as
  /// `[baseUrl]/[RemoteFile.name]`: a mirror an app hosts, or a test
  /// server. The sizes and SHA-256 stay pinned, so a mirror can only serve
  /// the published bytes.
  ModelSpec fromMirror(String baseUrl) {
    RemoteFile at(RemoteFile f) => RemoteFile(
      f.name,
      '$baseUrl/${f.name}',
      size: f.size,
      sha256: f.sha256,
    );
    return ModelSpec(
      id: id,
      name: name,
      license: license,
      licenseUrl: licenseUrl,
      gguf: at(gguf),
      config: at(config),
      tokenizer: at(tokenizer),
      licenseFile: at(licenseFile),
    );
  }

  /// LFM3-1.7B in Liquid's quantization-aware Q4_0 build, from
  /// LiquidAI/LFM3-1.7B-GGUF and LiquidAI/LFM3-1.7B on Hugging Face: 1.07 GB
  /// to download, 1.14 GB installed.
  static const lfm3Q4 = ModelSpec(
    id: 'lfm3-1.7b-qad-q4',
    name: 'LFM3-1.7B (QAD Q4_0)',
    license: 'LFM Open License v1.0',
    licenseUrl: 'https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE',
    gguf: RemoteFile(
      'LFM3-1.7B-QAD-Q4_0.gguf',
      '$_ggufRepo/LFM3-1.7B-QAD-Q4_0.gguf',
      size: 1049615680,
      sha256:
          '43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452',
    ),
    config: RemoteFile(
      'config.json',
      '$_baseRepo/config.json',
      size: 1454,
      sha256:
          '8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d',
    ),
    tokenizer: RemoteFile(
      'tokenizer.json',
      '$_baseRepo/tokenizer.json',
      size: 17908635,
      sha256:
          '7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701',
    ),
    licenseFile: RemoteFile(
      'LICENSE',
      '$_baseRepo/LICENSE',
      size: 10574,
      sha256:
          '4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c',
    ),
  );

  static const _ggufRepo =
      'https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/'
      'bad5729bba61d076fff20d2548e62526665af215';
  static const _baseRepo =
      'https://huggingface.co/LiquidAI/LFM3-1.7B/resolve/'
      'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7';
}

enum InstallStage { downloading, building, done }

/// Where a [ModelStore.install] stands.
final class InstallProgress {
  const InstallProgress(this.stage, {this.bytes = 0, this.totalBytes = 0});

  final InstallStage stage;

  /// Bytes downloaded so far, over all files.
  final int bytes;
  final int totalBytes;

  double get fraction => totalBytes == 0 ? 0 : bytes / totalBytes;
}

/// The models on this device, one directory each under [directory] (an
/// app's support directory, say).
final class ModelStore {
  ModelStore(this.directory);

  final String directory;

  /// The installed path for [LocalModel.load] or [LocalVisionModel.loadBundle].
  String bundlePath(InstallableModel model) {
    _checkId(model.id);
    return '$directory/${model.id}/bundle';
  }

  /// Checks installation presence and, for VL, pinned manifest identity.
  /// Payload integrity is checked by [LocalVisionModel.loadBundle] when loading.
  bool isInstalled(InstallableModel model) {
    final bundle = bundlePath(model);
    if (!Directory(bundle).existsSync()) return false;
    if (model is VisionModelSpec) {
      final manifest = File('$bundle/vl_bundle.json');
      final pinned = model.files.first;
      return manifest.existsSync() &&
          manifest.lengthSync() == pinned.size &&
          sha256.convert(manifest.readAsBytesSync()).toString() ==
              pinned.sha256;
    }
    if (model is VisionGgufSpec) {
      try {
        final m = VlBundleManifest.parse(
          File('$bundle/vl_bundle.json').readAsStringSync(),
        );
        return m.weights == 'q4' && m.inputEmbedding == 'q8-cache';
      } on Object {
        return false;
      }
    }
    return true;
  }

  /// Downloads [model]'s files and builds its bundle, reporting progress.
  /// What a previous install left half done is resumed; a lost connection
  /// is retried [retries] times. Cancelling the subscription stops the
  /// download and keeps what arrived; the cancel completes once the files
  /// are closed, so another install may start right after. One install of a
  /// model runs at a time: a second throws [StateError]. The stream ends
  /// with [InstallStage.done], or with an error.
  ///
  /// Text models need about the download size again in free space while
  /// building; downloads are deleted afterwards. Vision bundles are validated
  /// in an isolate during [InstallStage.building], then renamed into place,
  /// without a second copy or quantization. [VisionCheckpointSpec] instead
  /// builds Q8/Q4 from upstream BF16 files, allowing about 13 GB during conversion
  /// for the built-in LFM3-VL model. Cancellation during conversion or verification
  /// waits for it to finish and leaves the downloads unpublished. [VisionGgufSpec]
  /// imports prequantized QAD files with about 5 GB peak setup storage.
  Stream<InstallProgress> install(InstallableModel model, {int retries = 3}) {
    _checkId(model.id);
    if (model is VisionCheckpointSpec) {
      const required = {
        'config.json',
        'processor_config.json',
        'tokenizer.json',
        'LICENSE',
        'model.safetensors',
      };
      final names = model.files.map((f) => f.name).toSet();
      if (model.files.length != required.length ||
          names.length != required.length ||
          !names.containsAll(required)) {
        throw ArgumentError(
          'VL checkpoint must contain exactly the five required files',
        );
      }
    }
    if (model is VisionGgufSpec) {
      final names = model.files.map((f) => f.name).toSet();
      if (model.config.name != 'config.json' ||
          model.processor.name != 'processor_config.json' ||
          model.tokenizer.name != 'tokenizer.json' ||
          model.licenseFile.name != 'LICENSE' ||
          names.length != 6 ||
          names.any(
            (n) => !RegExp(r'^[A-Za-z0-9][A-Za-z0-9._-]*$').hasMatch(n),
          )) {
        throw ArgumentError(
          'GGUF model files must have six distinct safe names and canonical metadata',
        );
      }
    }
    final key = '$directory/${model.id}';
    if (!_installing.add(key)) {
      throw StateError('${model.id} is already being installed in $directory');
    }
    var cancelled = false;
    final client = HttpClient()
      ..connectionTimeout = const Duration(seconds: 30)
      ..userAgent = 'dianoia';
    late final StreamController<InstallProgress> out;
    void report(InstallProgress progress) {
      if (!cancelled) out.add(progress);
    }

    Future<void> run() async {
      final total = model.downloadBytes;
      try {
        if (!isInstalled(model)) {
          if ((model is VisionModelSpec || model is VisionGgufSpec) &&
              Directory(bundlePath(model)).existsSync()) {
            throw StateError(
              'installed VL manifest differs; remove ${model.id} before replacing it',
            );
          }
          final downloads = Directory('$directory/${model.id}/downloads')
            ..createSync(recursive: true);
          var done = 0;
          for (final file in model.files) {
            File(
              '${downloads.path}/${file.name}',
            ).parent.createSync(recursive: true);
            for (var attempt = 0; ; attempt++) {
              try {
                await downloadFile(
                  client,
                  file,
                  '${downloads.path}/${file.name}',
                  onBytes: (bytes) => report(
                    InstallProgress(
                      InstallStage.downloading,
                      bytes: done + bytes,
                      totalBytes: total,
                    ),
                  ),
                  cancelled: () => cancelled,
                );
                break;
              } on IOException {
                if (cancelled || attempt >= retries) rethrow;
                await Future<void>.delayed(Duration(seconds: 1 << attempt));
              }
            }
            if (cancelled) return;
            done += file.size;
          }
          report(
            InstallProgress(
              InstallStage.building,
              bytes: total,
              totalBytes: total,
            ),
          );
          if (model is VisionModelSpec) {
            await _verifyVision(downloads.path);
            if (cancelled) return;
            downloads.renameSync(bundlePath(model));
          } else if (model is VisionGgufSpec) {
            _removeStaging('$directory/${model.id}');
            final partial = Directory('${bundlePath(model)}.partial');
            try {
              if (partial.existsSync()) partial.deleteSync(recursive: true);
              await _buildQad(model, downloads.path, partial.path);
              if (cancelled) return;
              partial.renameSync(bundlePath(model));
              await downloads.delete(recursive: true);
            } finally {
              if (partial.existsSync()) partial.deleteSync(recursive: true);
            }
          } else if (model is VisionCheckpointSpec) {
            _removeStaging('$directory/${model.id}');
            final partial = Directory('${bundlePath(model)}.partial');
            try {
              await _buildVision(
                downloads.path,
                partial.path,
                model.quantization,
              );
              if (cancelled) return;
              partial.renameSync(bundlePath(model));
              await downloads.delete(recursive: true);
            } finally {
              if (partial.existsSync()) partial.deleteSync(recursive: true);
            }
          } else {
            await _build(model as ModelSpec, downloads.path, bundlePath(model));
            await downloads.delete(recursive: true);
          }
        }
        report(
          InstallProgress(InstallStage.done, bytes: total, totalBytes: total),
        );
      } catch (e, stack) {
        if (!cancelled) out.addError(e, stack);
      } finally {
        client.close(force: true);
        _installing.remove(key);
        // Not awaited: after a cancel, closing waits for onCancel, which
        // waits for this.
        unawaited(out.close());
      }
    }

    Future<void>? running;
    out = StreamController(
      onListen: () => running = run(),
      onCancel: () {
        cancelled = true;
        client.close(force: true);
        // A resumed install must not append to a part still being written.
        return running;
      },
    );
    return out.stream;
  }

  /// Deletes the randomly named staging directories that a VL build of a
  /// killed process left in [modelDir]; its finally blocks never ran.
  /// Only one install of a model runs at a time, so none is in use.
  static void _removeStaging(String modelDir) {
    for (final entry in Directory(modelDir).listSync(followLinks: false)) {
      final name = entry.uri.pathSegments.where((s) => s.isNotEmpty).last;
      if (entry is Directory && name.startsWith('.dianoia-vl-')) {
        entry.deleteSync(recursive: true);
      }
    }
  }

  static Future<void> _buildQad(
    VisionGgufSpec model,
    String downloads,
    String bundle,
  ) => Isolate.run(
    () => importVlQadBundle(
      '$downloads/${model.language.name}',
      '$downloads/${model.vision.name}',
      downloads,
      bundle,
    ),
  );

  // Use separate isolates for quantization and packing, so the first one's
  // temporary quantized matrices are released before the packed cache is mapped.
  static Future<void> _buildVision(
    String downloads,
    String bundle,
    VisionQuantization quantization,
  ) async {
    final weights = quantization == VisionQuantization.q4
        ? WeightFormat.q4
        : WeightFormat.q8;
    final backbone = '$bundle.backbone';
    try {
      for (final path in [backbone, bundle]) {
        final directory = Directory(path);
        if (directory.existsSync()) directory.deleteSync(recursive: true);
      }
      await _prepareVision(downloads, backbone, weights);
      await _packVision(downloads, backbone, bundle, weights);
    } finally {
      final directory = Directory(backbone);
      if (directory.existsSync()) directory.deleteSync(recursive: true);
    }
  }

  static Future<void> _prepareVision(
    String downloads,
    String backbone,
    WeightFormat weights,
  ) => Isolate.run(() async {
    importVlBackbone(downloads, backbone);
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    final model = await Lfm3Model.load(
      backbone,
      maxSeq: 1,
      maxChunk: 1,
      weights: weights,
      q8CachePath: '$backbone/lfm3-${weights.name}.cache',
    );
    await model.close();
  });

  static Future<void> _packVision(
    String downloads,
    String backbone,
    String bundle,
    WeightFormat weights,
  ) => Isolate.run(
    () => packVlBundle(downloads, backbone, bundle, weights: weights),
  );

  /// The installs running in this isolate, by model directory.
  static final _installing = <String>{};

  /// Deletes [model]'s bundle and any downloads it left.
  Future<void> remove(InstallableModel model) async {
    _checkId(model.id);
    if (_installing.contains('$directory/${model.id}')) {
      throw StateError('cannot remove ${model.id} during installation');
    }
    final dir = Directory('$directory/${model.id}');
    if (dir.existsSync()) await dir.delete(recursive: true);
  }

  // Keep the isolate closure outside install's context, which holds unsendable
  // stream controllers, clients and futures.
  static Future<void> _verifyVision(String directory) =>
      Isolate.run(() => validateVlBundle(directory));

  static void _checkId(String id) {
    if (!RegExp(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$').hasMatch(id)) {
      throw ArgumentError.value(id, 'model.id', 'expected a directory name');
    }
  }

  /// Builds the bundle from the downloads in another isolate, so that an
  /// app's UI keeps running, into a directory renamed into place at the end.
  static Future<void> _build(
    ModelSpec model,
    String downloads,
    String bundle,
  ) => Isolate.run(() {
    final partial = Directory('$bundle.partial');
    if (partial.existsSync()) partial.deleteSync(recursive: true);
    // The kernels quantize the embedding to Q8 for the LM head.
    if (!NativeKernels.isLoaded) NativeKernels.load(null);
    importGguf('$downloads/${model.gguf.name}', downloads, partial.path);
    File(
      '$downloads/${model.licenseFile.name}',
    ).copySync('${partial.path}/LICENSE');
    partial.renameSync(bundle);
  });
}
