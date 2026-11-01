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
import 'vision_bundle.dart';

export 'download.dart' show ModelDownloadException, RemoteFile;

/// A pinned text checkpoint or prebuilt vision bundle installed by [ModelStore].
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
  /// without a second copy or quantization. Cancellation during verification
  /// waits for it to finish and leaves the downloads unpublished.
  Stream<InstallProgress> install(InstallableModel model, {int retries = 3}) {
    _checkId(model.id);
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
          if (model is VisionModelSpec &&
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
