/// Text model bundles: a model directory's quantized weights plus only what
/// the cache lacks, as `LocalModel.load` opens them.
///
/// A bundle holds config.json, tokenizer.json, generation_config.json when
/// the model has one, the quantized cache `<architecture>-<weights>.cache`,
/// and a model.safetensors with the 1-D tensors, the conv kernels and the
/// bf16 embedding for token lookup.
library;

import 'dart:io';

import 'config.dart';
import 'hf_import.dart';
import 'model.dart';
import 'native.dart';
import 'q8_cache.dart';
import 'quant.dart';
import 'safetensors.dart';

/// The quantized cache file of [config]'s architecture under [weights]:
/// `lfm3-q8.cache`, `qwen3-q4.cache` and so on.
String quantizedCacheName(TextModelConfig config, WeightFormat weights) =>
    '${config.architecture.name}-${weights.name}.cache';

/// Writes the bundle of the model directory [modelDir], whose [weights]
/// cache is already built (by loading it with those weights), to [outDir].
void packTextBundle(String modelDir, String outDir, WeightFormat weights) {
  if (!weights.isNative) {
    throw ArgumentError.value(weights, 'weights', 'expected q8 or q4');
  }
  final name = quantizedCacheName(TextModelConfig.load(modelDir), weights);
  final cache =
      Q8Cache.open(
        '$modelDir/$name',
        Q8Cache.keyFor(modelDir, weights: weights.name),
      ) ??
      (throw StateError(
        'no ${weights.name} cache for $modelDir: load it with '
        '${weights.name} weights first',
      ));
  Directory(outDir).createSync(recursive: true);
  for (final file in ['config.json', 'tokenizer.json']) {
    File('$modelDir/$file').copySync('$outDir/$file');
  }
  final generation = File('$modelDir/generation_config.json');
  if (generation.existsSync()) {
    generation.copySync('$outDir/generation_config.json');
  }
  final st = SafetensorsFile.open('$modelDir/model.safetensors');
  try {
    // Matrices live in the cache, except that the embedding is also read as
    // bf16 rows for token lookup.
    st.writeSubset(
      '$outDir/model.safetensors',
      (name, info) =>
          info.shape.length != 2 || name == 'model.embed_tokens.weight',
    );
  } finally {
    st.close();
  }
  // The cache key fingerprints the safetensors header, so it changes.
  Q8Cache.write(
    '$outDir/$name',
    Q8Cache.keyFor(outDir, weights: weights.name),
    {
      for (final (matrix, rows, cols) in cache.shapes)
        matrix: cache.matrix(matrix, rows, cols)!,
    },
  );
}

/// Builds the [weights] bundle of the Hugging Face checkpoint in [hfDir]
/// into [outDir], with [stagingDir] for the full-precision model directory
/// in between (deleted afterwards). Loads the native kernels, which
/// quantize, unless they are loaded. A checkpoint of another family is first renamed into the
/// engine's layout (hf_import.dart).
Future<void> buildTextBundle(
  String hfDir,
  String outDir, {
  required String stagingDir,
  required WeightFormat weights,
}) async {
  if (!NativeKernels.isLoaded) NativeKernels.load(null);
  final config = TextModelConfig.load(hfDir);
  final staging = Directory(stagingDir);
  if (staging.existsSync()) staging.deleteSync(recursive: true);
  try {
    final String modelDir;
    if (config.architecture == ModelArchitecture.lfm3) {
      modelDir = hfDir;
    } else {
      importHfCheckpoint(hfDir, stagingDir);
      modelDir = stagingDir;
    }
    // Loading quantizes every matrix into the cache, which packing copies.
    final cache = '$modelDir/${quantizedCacheName(config, weights)}';
    final model = await TextModel.load(
      modelDir,
      maxSeq: 16,
      weights: weights,
      q8CachePath: cache,
    );
    await model.close();
    packTextBundle(modelDir, outDir, weights);
    if (modelDir == hfDir) File(cache).deleteSync();
  } finally {
    if (staging.existsSync()) staging.deleteSync(recursive: true);
  }
}
