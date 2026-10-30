/// Getting a model onto the device: its files from where the model's authors
/// publish them, pinned by revision and checked by SHA-256, and the bundle
/// dianoia runs built from them on the device. Nothing is redistributed.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'download.dart';
import 'gguf.dart';
import 'native.dart';

export 'download.dart' show ModelDownloadException, RemoteFile;

/// A model [ModelStore] can install: a GGUF checkpoint and the matching
/// Hugging Face model's config, tokenizer and license.
final class ModelSpec {
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
  final String id;
  final String name;

  /// The license the weights come under, which an app should show before
  /// downloading them, and where to read it.
  final String license;
  final String licenseUrl;

  final RemoteFile gguf, config, tokenizer, licenseFile;

  List<RemoteFile> get files => [config, tokenizer, licenseFile, gguf];

  /// Bytes to download.
  int get downloadBytes => files.fold(0, (sum, f) => sum + f.size);

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

/// Where an [ModelStore.install] stands.
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

  /// The bundle [LocalModel.load] takes, once [model] is installed.
  String bundlePath(ModelSpec model) => '$directory/${model.id}/bundle';

  bool isInstalled(ModelSpec model) =>
      Directory(bundlePath(model)).existsSync();

  /// Downloads [model]'s files and builds its bundle, reporting progress.
  /// What a previous install left half done is resumed; a lost connection
  /// is retried [retries] times. Cancelling the subscription stops the
  /// download and keeps what arrived; the cancel completes once the files
  /// are closed, so another install may start right after. One install of a
  /// model runs at a time: a second throws [StateError]. The stream ends
  /// with [InstallStage.done], or with an error.
  ///
  /// Needs about the download size again in free space while building; the
  /// downloads are deleted afterwards.
  Stream<InstallProgress> install(ModelSpec model, {int retries = 3}) {
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
          final downloads = Directory('$directory/${model.id}/downloads')
            ..createSync(recursive: true);
          var done = 0;
          for (final file in model.files) {
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
          await _build(model, downloads.path, bundlePath(model));
          await downloads.delete(recursive: true);
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
  Future<void> remove(ModelSpec model) async {
    final dir = Directory('$directory/${model.id}');
    if (dir.existsSync()) await dir.delete(recursive: true);
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
