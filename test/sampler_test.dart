import 'dart:math' as math;
import 'dart:typed_data';

import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

/// Logits whose softmax at temperature 1 is [probs].
Float32List _logits(List<double> probs) =>
    Float32List.fromList([for (final p in probs) math.log(p)]);

/// How often each token is drawn in [n] draws.
List<double> _frequencies(Sampler sampler, Float32List logits, int n) {
  final counts = List.filled(logits.length, 0);
  for (var i = 0; i < n; i++) {
    counts[sampler.next(logits)]++;
  }
  return [for (final c in counts) c / n];
}

void main() {
  final logits = _logits([0.6, 0.3, 0.1]);

  test('greedy picks the best token outside the suppressed set', () {
    final sampler = Sampler(SamplingParams.greedy);
    expect(sampler.next(logits), 0);
    expect(sampler.next(logits, suppressed: {0}), 1);
    expect(() => sampler.next(logits, suppressed: {0, 1, 2}), throwsStateError);
  });

  test('draws follow the softmax at temperature 1', () {
    final f = _frequencies(
      Sampler(const SamplingParams(temperature: 1, seed: 1)),
      logits,
      20000,
    );
    expect(f[0], closeTo(0.6, 0.02));
    expect(f[1], closeTo(0.3, 0.02));
    expect(f[2], closeTo(0.1, 0.02));
  });

  test('a low temperature sharpens the distribution', () {
    final f = _frequencies(
      Sampler(const SamplingParams(temperature: 0.25, seed: 2)),
      logits,
      5000,
    );
    // 0.6^4 : 0.3^4 : 0.1^4, normalized.
    expect(f[0], closeTo(0.94, 0.02));
  });

  test('a seed makes replies reproducible', () {
    List<int> draws(int seed) {
      final sampler = Sampler(SamplingParams(temperature: 1, seed: seed));
      return [for (var i = 0; i < 50; i++) sampler.next(logits)];
    }

    expect(draws(7), draws(7));
    expect(draws(7), isNot(draws(8)));
  });

  test('top-k keeps the k most likely tokens', () {
    final f = _frequencies(
      Sampler(const SamplingParams(temperature: 1, topK: 2, seed: 3)),
      logits,
      5000,
    );
    expect(f[2], 0);
    expect(f[0], closeTo(2 / 3, 0.03));
  });

  test('top-p keeps tokens while the mass before them is under p', () {
    final f = _frequencies(
      Sampler(const SamplingParams(temperature: 1, topP: 0.75, seed: 4)),
      logits,
      5000,
    );
    expect(f[2], 0);
    expect(f[1], greaterThan(0));
    final first = _frequencies(
      Sampler(const SamplingParams(temperature: 1, topP: 0.1, seed: 4)),
      logits,
      200,
    );
    expect(first, [1, 0, 0]);
  });

  test('min-p drops tokens less likely than p times the best', () {
    final f = _frequencies(
      Sampler(const SamplingParams(temperature: 1, minP: 0.4, seed: 5)),
      logits,
      5000,
    );
    expect(f[2], 0);
    expect(f[1], greaterThan(0));
  });

  test('suppressed tokens are never drawn', () {
    final sampler = Sampler(const SamplingParams(temperature: 1, seed: 6));
    for (var i = 0; i < 500; i++) {
      expect(sampler.next(logits, suppressed: {0}), isNot(0));
    }
  });

  group('presence penalty', () {
    test('lowers each seen token once and restores the logits', () {
      final before = Float32List.fromList(logits);
      final sampler = Sampler(const SamplingParams(presencePenalty: 2));
      // log 0.6 - 2 < log 0.3: token 0 loses to token 1.
      expect(sampler.next(logits, seen: {0}), 1);
      expect(logits, orderedEquals(before));
      // log 0.6 - 2 and log 0.3 - 2 are both under log 0.1.
      expect(sampler.next(logits, seen: {0, 1}), 2);
      expect(logits, orderedEquals(before));
    });

    test('changes nothing at zero or with nothing seen', () {
      expect(Sampler(SamplingParams.greedy).next(logits, seen: {0}), 0);
      expect(Sampler(const SamplingParams(presencePenalty: 2)).next(logits), 0);
    });

    test('shifts sampled draws away from seen tokens', () {
      final sampler = Sampler(
        const SamplingParams(temperature: 1, presencePenalty: 1, seed: 9),
      );
      final counts = List.filled(3, 0);
      for (var i = 0; i < 5000; i++) {
        counts[sampler.next(logits, seen: {0})]++;
      }
      // Weights 0.6/e, 0.3, 0.1 renormalized: about 0.36, 0.48, 0.16.
      expect(counts[0] / 5000, closeTo(0.36, 0.03));
      expect(counts[1] / 5000, closeTo(0.48, 0.03));
    });

    test('restores the logits when every token is suppressed', () {
      final before = Float32List.fromList(logits);
      final sampler = Sampler(const SamplingParams(presencePenalty: 1));
      expect(
        () => sampler.next(logits, suppressed: {0, 1, 2}, seen: {0}),
        throwsStateError,
      );
      expect(logits, orderedEquals(before));
    });
  });
}
