// BpeTokenizer on Qwen3's tokenizer.json against Hugging Face tokenizers
// (test/fixtures/qwen3_text_oracle.json, from tool/qwen3_oracle.py). The
// tokenizer is not committed: point DIANOIA_QWEN3_DIR at a Qwen3 model
// directory (such as ../dart-llm-lab/models/Qwen3-0.6B) to run these.
import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final dir = Platform.environment['DIANOIA_QWEN3_DIR'];
  final skip = dir == null ? 'set DIANOIA_QWEN3_DIR to a Qwen3 model' : null;
  final oracle =
      jsonDecode(
            File('test/fixtures/qwen3_text_oracle.json').readAsStringSync(),
          )
          as Map<String, Object?>;
  late final BpeTokenizer tokenizer = BpeTokenizer.load(dir!);

  group('encode', () {
    for (final c in (oracle['encode'] as List).cast<Map<String, Object?>>()) {
      final text = c['text'] as String;
      // Qwen3 normalizes to NFC first, which Dart cannot (see tokenizer.dart).
      final nfd = text.contains('́');
      test(
        jsonEncode(text),
        () => expect(tokenizer.encode(text), c['ids']),
        skip: skip ?? (nfd ? 'NFC normalization is a known gap' : null),
      );
    }
  });

  group('decode', () {
    for (final c in (oracle['decode'] as List).cast<Map<String, Object?>>()) {
      final ids = (c['ids'] as List).cast<int>();
      final skipSpecial = c['skip_special_tokens'] as bool;
      test(
        '${ids.take(6).toList()} skip_special=$skipSpecial',
        () => expect(
          tokenizer.decode(ids, skipSpecialTokens: skipSpecial),
          c['text'],
        ),
        skip: skip,
      );
    }
  });
}
