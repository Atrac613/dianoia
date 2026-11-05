/// Pinned official LFM3-Omni Q4_0 sources and their vision configuration.
library;

import 'dart:convert';
import 'dart:io';

import 'audio_gguf.dart';
import 'download.dart';
import 'tokenizer.dart';
import 'vision_config.dart';

/// The official Q4_0 language GGUF and its combined vision/audio mmproj.
/// The language tensors share LFM3-Audio's layout; the mmproj adds the
/// LFM3-VL vision tower and projector to the audio encoder and generator.
abstract final class Lfm3OmniQ4Source {
  static const repository = 'LiquidAI/LFM3-Omni-2B-GGUF';
  static const revision = 'd508b88f08a5b4286f2d2805a6a8d69a0ff44c72';
  static const _base = 'https://huggingface.co/$repository/resolve/$revision';
  static const language = RemoteFile(
    'LFM3-Omni-2B-Q4_0.gguf',
    '$_base/LFM3-Omni-2B-Q4_0.gguf',
    size: 1049617920,
    sha256: '6d2273a383b92dd7c5db0eb4c8dfeae13e805b485a6621c4b391ae179c2b5f28',
  );
  static const mmproj = RemoteFile(
    'mmproj-LFM3-Omni-2B-Q4_0.gguf',
    '$_base/mmproj-LFM3-Omni-2B-Q4_0.gguf',
    size: 270576544,
    sha256: 'c7341273f1757b33ccd9e6f8b17007eb81307d26c90d8c36c22efb45fd3d82eb',
  );
  static const files = [language, mmproj];
  static const downloadBytes = 1320194464;
  static const metadataRevision = '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67';
  static const _metadataBase =
      'https://huggingface.co/LiquidAI/LFM3-Omni-2B/resolve/$metadataRevision';
  static const config = RemoteFile(
    'omni_config.json',
    '$_metadataBase/config.json',
    size: 5912,
    sha256: '17e3792551ef56b37e85bbd1e6e968f70dc5b4b14c96f1339d0be85e0ba46875',
  );
  static const processorConfig = RemoteFile(
    'processor_config.json',
    '$_metadataBase/processor_config.json',
    size: 2374,
    sha256: 'dd756829f85a10d343c4488c7d3479d3dc782e6f35721b7dae2fb588f781fc4a',
  );
  static const tokenizer = RemoteFile(
    'tokenizer.json',
    '$_metadataBase/tokenizer.json',
    size: 17908634,
    sha256: '41927bd314d17d7d7fadb44e10cd0919f7b71528d7741596d8cbd15c6945ec44',
  );
  static const license = RemoteFile(
    'LICENSE',
    '$_metadataBase/LICENSE',
    size: 10574,
    sha256: '4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c',
  );
  static const metadataFiles = [config, processorConfig, tokenizer, license];
}

/// The vision part of an Omni config.json. Omni names no span marker IDs in
/// its config; they are the tokenizer's `<|vision_start|>`/`<|vision_end|>`.
Lfm3VlConfig loadOmniVisionConfig(String configPath, Lfm3Tokenizer tokenizer) {
  int marker(String text) =>
      tokenizer.addedTokenId(text) ??
      (throw FormatException('Omni tokenizer lacks $text'));
  return Lfm3VlConfig.fromOmniJson(
    jsonDecode(File(configPath).readAsStringSync()) as Map<String, Object?>,
    visionStartTokenId: marker('<|vision_start|>'),
    visionEndTokenId: marker('<|vision_end|>'),
  );
}

/// The speech mmproj of a prepared research directory: LFM3-Omni's when
/// present, otherwise LFM3-Audio's.
String speechMmprojIn(String directory) {
  for (final name in [
    Lfm3OmniQ4Source.mmproj.name,
    Lfm3AudioQ4Source.audio.name,
  ]) {
    if (File('$directory/$name').existsSync()) return '$directory/$name';
  }
  throw StateError('$directory holds no prepared LFM3-Omni or Audio mmproj');
}
