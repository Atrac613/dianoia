/// Noticing a reply that has started to repeat itself.
///
/// Small models, above all under greedy decoding or after quantization,
/// can fall into a loop: the same sentence or paragraph again and again
/// until the token limit. Once a block of tokens has come back several
/// times in a row, the rest of the reply is almost always more of it, so a
/// session can stop there or move on to the answer instead.
library;

/// Recognizes a run of tokens that ends in the same block, of at most
/// [maxPeriod] tokens, repeated at least [minRepeats] times in a row and
/// spanning at least [minTokens] tokens.
///
/// [minTokens] keeps short legitimate runs, such as a row of dashes or a
/// few identical list markers, from counting as a loop.
final class RepetitionGuard {
  const RepetitionGuard({
    this.maxPeriod = 64,
    this.minRepeats = 3,
    this.minTokens = 32,
  }) : assert(maxPeriod >= 1),
       assert(minRepeats >= 2),
       assert(minTokens >= 1);

  /// The longest repeated block looked for, in tokens.
  final int maxPeriod;

  /// How many times in a row the block must appear.
  final int minRepeats;

  /// The fewest tokens the repetition must cover.
  final int minTokens;

  /// Whether [tokens] end in a loop. Each call looks back at most
  /// `maxPeriod * max(minRepeats * maxPeriod, minTokens)` tokens, so it
  /// can run after every generated token.
  bool loops(List<int> tokens) {
    final n = tokens.length;
    for (var period = 1; period <= maxPeriod; period++) {
      var span = period * minRepeats;
      if (span < minTokens) span = minTokens;
      if (span > n) return false; // longer periods need even more tokens
      // The last `span` tokens repeat with this period when each equals
      // the one `period` before it.
      var repeats = true;
      for (var i = n - 1; i >= n - span + period; i--) {
        if (tokens[i] != tokens[i - period]) {
          repeats = false;
          break;
        }
      }
      if (repeats) return true;
    }
    return false;
  }
}
