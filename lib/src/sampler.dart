/// Choosing the next token from the logits: greedily, or by sampling.
///
/// Sampling follows the order of Hugging Face's logits processors:
/// temperature, then top-k, then top-p, then min-p, and a draw from what
/// is left, renormalized. A presence penalty, applied first, lowers the
/// logit of every token the reply has already generated, as vLLM's
/// `presence_penalty` does. Tokens whose probability is below 1e-9 of the
/// most likely one are never drawn: leaving them out spares sorting the
/// whole vocabulary on every step and changes no draw a caller could tell.
library;

import 'dart:math' as math;
import 'dart:typed_data';

/// How replies pick their tokens. [temperature] 0 is greedy decoding: the
/// most likely token, the other fields unused.
final class SamplingParams {
  const SamplingParams({
    this.temperature = 0,
    this.topK = 0,
    this.topP = 1,
    this.minP = 0,
    this.presencePenalty = 0,
    this.seed,
  }) : assert(temperature >= 0),
       assert(presencePenalty >= 0),
       assert(topK >= 0),
       assert(topP > 0 && topP <= 1),
       assert(minP >= 0 && minP <= 1);

  /// Greedy decoding.
  static const greedy = SamplingParams();

  /// Qwen3's recommendation with thinking enabled.
  static const qwen3Thinking = SamplingParams(
    temperature: 0.6,
    topK: 20,
    topP: 0.95,
  );

  /// Qwen3's recommendation with thinking disabled.
  static const qwen3NonThinking = SamplingParams(
    temperature: 0.7,
    topK: 20,
    topP: 0.8,
  );

  /// Divides the logits; 0 means greedy decoding.
  final double temperature;

  /// Keeps the [topK] most likely tokens; 0 keeps them all.
  final int topK;

  /// Keeps the most likely tokens whose probabilities sum to [topP].
  final double topP;

  /// Drops tokens less likely than [minP] times the most likely one.
  final double minP;

  /// Subtracted once from the logit of each token already in the reply,
  /// however often it appeared: 0 leaves the logits as they are. Qwen
  /// suggests values up to 2 against endless repetition; high values can
  /// make a reply switch languages. It applies to greedy decoding too.
  final double presencePenalty;

  /// Seeds the random draws, for reproducible replies; null seeds them
  /// from the clock.
  final int? seed;

  bool get isGreedy => temperature == 0;
}

/// Picks tokens under [params], drawing from one random sequence across
/// calls.
final class Sampler {
  Sampler(this.params) : _random = math.Random(params.seed);

  final SamplingParams params;
  final math.Random _random;

  /// log(1e-9): candidates must be at least this likely, relative to the
  /// most likely token.
  static final double _floor = math.log(1e-9);

  /// The next token from [logits], never one in [suppressed]. [seen] are
  /// the tokens the reply has generated so far, which
  /// [SamplingParams.presencePenalty] counts against. [logits] are
  /// penalized in place while the token is chosen and restored before
  /// this returns.
  int next(
    Float32List logits, {
    Set<int> suppressed = const {},
    Set<int> seen = const {},
  }) {
    final penalty = params.presencePenalty;
    if (penalty == 0 || seen.isEmpty) return _next(logits, suppressed);
    final ids = seen.toList(growable: false);
    final saved = Float32List(ids.length);
    for (var i = 0; i < ids.length; i++) {
      saved[i] = logits[ids[i]];
      logits[ids[i]] = saved[i] - penalty;
    }
    try {
      return _next(logits, suppressed);
    } finally {
      for (var i = 0; i < ids.length; i++) {
        logits[ids[i]] = saved[i];
      }
    }
  }

  int _next(Float32List logits, Set<int> suppressed) {
    if (params.isGreedy) return argmaxToken(logits, suppressed: suppressed);
    final t = params.temperature;
    // The vocabulary is scanned twice per token, so the loops stay plain:
    // the suppressed set is consulted only for the few tokens near the top.
    final max = logits[argmaxToken(logits, suppressed: suppressed)];
    // Scaled logits relative to the best: (logit - max) / t <= 0.
    final cutoff = max + _floor * t;
    final candidates = <(int, double)>[];
    for (var i = 0; i < logits.length; i++) {
      final logit = logits[i];
      if (logit < cutoff || suppressed.contains(i)) continue;
      candidates.add((i, (logit - max) / t));
    }
    // Ties keep the lower id first, as a stable sort by score would.
    candidates.sort((a, b) {
      final byScore = b.$2.compareTo(a.$2);
      return byScore != 0 ? byScore : a.$1.compareTo(b.$1);
    });
    var kept = params.topK > 0 && params.topK < candidates.length
        ? candidates.sublist(0, params.topK)
        : candidates;

    var probs = [for (final (_, s) in kept) math.exp(s)];
    var total = probs.fold(0.0, (a, b) => a + b);
    if (params.topP < 1) {
      // Keep each token while the probability before it is under topP; the
      // most likely always stays.
      var cumulative = 0.0;
      var n = 0;
      while (n < probs.length && (n == 0 || cumulative < params.topP)) {
        cumulative += probs[n] / total;
        n++;
      }
      kept = kept.sublist(0, n);
      probs = probs.sublist(0, n);
      total = probs.fold(0.0, (a, b) => a + b);
    }
    if (params.minP > 0) {
      // probs[0] is the most likely token's weight.
      final threshold = params.minP * probs[0];
      var n = 1;
      while (n < probs.length && probs[n] >= threshold) {
        n++;
      }
      kept = kept.sublist(0, n);
      probs = probs.sublist(0, n);
      total = probs.fold(0.0, (a, b) => a + b);
    }

    var draw = _random.nextDouble() * total;
    for (var i = 0; i < kept.length; i++) {
      draw -= probs[i];
      if (draw < 0) return kept[i].$1;
    }
    return kept.last.$1;
  }
}

/// The index of the largest of [logits] outside [suppressed]; the first
/// one on a tie.
int argmaxToken(Float32List logits, {Set<int> suppressed = const {}}) {
  if (suppressed.isEmpty) {
    var best = 0;
    for (var i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) best = i;
    }
    return best;
  }
  var best = -1;
  for (var i = 0; i < logits.length; i++) {
    if (suppressed.contains(i)) continue;
    if (best < 0 || logits[i] > logits[best]) best = i;
  }
  if (best < 0) throw StateError('every token is suppressed');
  return best;
}
