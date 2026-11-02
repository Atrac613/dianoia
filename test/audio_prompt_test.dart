import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

import 'support/synthetic_tokenizer.dart';

const audioSpecials = [
  '<|audio|>',
  '<|audio_out|>',
  '<|audio_start|>',
  '<|mixed_start|>',
  '<|text_end|>',
];

void main() {
  late Directory temp;
  late Lfm3Tokenizer tokenizer;
  setUp(() {
    temp = Directory.systemTemp.createTempSync('dianoia-audio-prompt-');
    writeSyntheticTokenizer(temp.path, additionalSpecials: audioSpecials);
    tokenizer = Lfm3Tokenizer.load(temp.path);
  });
  tearDown(() => temp.deleteSync(recursive: true));
  final oracle =
      jsonDecode(
            File('test/fixtures/audio/prompt_oracle.json').readAsStringSync(),
          )
          as Map;
  for (final c in oracle['cases'] as List) {
    test('ASR prompt matches upstream template: ${c['language']}', () {
      final prompt = prepareAudioTranscriptionPrompt(
        tokenizer: tokenizer,
        audioTokens: 17,
        hiddenSize: 32,
        language: c['language'] as String?,
      );
      expect(prompt.text, c['text']);
      final audioId = tokenizer.addedTokenId('<|audio|>')!;
      expect(
        prompt.ids.sublist(prompt.audioStart, prompt.audioStart + 17),
        List.filled(17, audioId),
      );
      expect(
        tokenizer.decode(prompt.ids),
        (c['text'] as String).replaceFirst(
          '<|audio|>',
          List.filled(17, '<|audio|>').join(),
        ),
      );
      expect(() => prompt.ids.add(1), throwsUnsupportedError);
      expect(() => prompt.suppressedTokenIds.add(1), throwsUnsupportedError);
    });
  }
  test(
    'text generation suppresses speech modes and stops on both turn markers',
    () {
      final prompt = prepareAudioTranscriptionPrompt(
        tokenizer: tokenizer,
        audioTokens: 1,
        hiddenSize: 32,
      );
      final logits = Float32List(280)..fillRange(0, 280, -10);
      logits[42] = 3;
      for (final id in prompt.suppressedTokenIds) {
        logits[id] = 100;
      }
      expect(prompt.greedyToken(logits), 42);
      expect(prompt.stopTokenIds, {
        tokenizer.addedTokenId('<|im_end|>'),
        tokenizer.addedTokenId('<|text_end|>'),
      });
      logits[43] = double.nan;
      expect(() => prompt.greedyToken(logits), throwsStateError);
      expect(() => prompt.greedyToken(Float32List(0)), throwsStateError);
    },
  );
  test(
    'rejects malformed language names, dimensions and missing audio markers',
    () {
      for (final language in [
        '',
        'English\n',
        '<|audio|>',
        'Japanese<|im_end|>',
        'x' * 65,
      ]) {
        expect(
          () => prepareAudioTranscriptionPrompt(
            tokenizer: tokenizer,
            audioTokens: 1,
            hiddenSize: 32,
            language: language,
          ),
          throwsArgumentError,
        );
      }
      for (final dims in [(0, 32), (-1, 32), (1, 0)]) {
        expect(
          () => prepareAudioTranscriptionPrompt(
            tokenizer: tokenizer,
            audioTokens: dims.$1,
            hiddenSize: dims.$2,
          ),
          throwsArgumentError,
        );
      }
      writeSyntheticTokenizer(temp.path);
      final missing = Lfm3Tokenizer.load(temp.path);
      expect(
        () => prepareAudioTranscriptionPrompt(
          tokenizer: missing,
          audioTokens: 1,
          hiddenSize: 32,
        ),
        throwsFormatException,
      );
    },
  );
}
