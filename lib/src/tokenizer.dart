/// The byte-level BPE tokenizers of LFM3 and Qwen3, reimplemented from
/// tokenizer.json.
///
/// Encoding runs the same four stages as Hugging Face tokenizers:
///
/// 1. Added tokens (`<|im_start|>`, `<tool_call>`, ...) are cut out of the
///    text first, leftmost-longest. None of them strip neighbouring spaces.
/// 2. The rest is split by the pre-tokenizer regex (contractions, letter
///    runs, digit runs, symbol runs, newlines, other whitespace). LFM3 keeps
///    up to three digits together, Qwen3 one.
/// 3. Each piece's UTF-8 bytes are mapped to printable characters with
///    GPT-2's byte-to-unicode table, so every byte has a vocabulary entry.
/// 4. BPE: with `ignore_merges` (LFM3), a piece that is itself in the
///    vocabulary becomes one token. Otherwise the lowest-ranked adjacent
///    pair is merged until no ranked pair is left.
///
/// Known gaps: Dart's `\s` (JavaScript semantics) and the Oniguruma `\s`
/// used by tokenizers disagree on U+0085 and U+FEFF, neither of which
/// appears in the oracle corpora. Qwen3's NFC normalizer is not applied:
/// Dart has no Unicode normalization, so text in another normalization form
/// (decomposed accents, such as 'e' followed by U+0301) encodes to
/// different, still decodable, tokens. Keyboards and most sources produce
/// NFC already.
library;

import 'dart:convert';
import 'dart:io';

/// Older name of [BpeTokenizer], from when LFM3 was the only family.
typedef Lfm3Tokenizer = BpeTokenizer;

final class BpeTokenizer {
  BpeTokenizer._(
    this._vocab,
    this._ranks,
    this._added,
    this._special,
    this._pieceOf,
    this._pretokenizer,
    this._ignoreMerges,
  );

  factory BpeTokenizer.load(String modelDir) {
    final json =
        jsonDecode(File('$modelDir/tokenizer.json').readAsStringSync())
            as Map<String, Object?>;
    final model = json['model'] as Map<String, Object?>;
    if (model['type'] != 'BPE' || model['byte_fallback'] == true) {
      throw const FormatException('tokenizer.json: expected byte-level BPE');
    }
    final normalizer = json['normalizer'] as Map<String, Object?>?;
    if (normalizer != null && normalizer['type'] != 'NFC') {
      throw FormatException(
        'tokenizer.json: unsupported normalizer ${normalizer['type']}',
      );
    }
    final vocab = (model['vocab'] as Map<String, Object?>).cast<String, int>();
    final ranks = <String, int>{};
    final merges = model['merges'] as List;
    for (var i = 0; i < merges.length; i++) {
      final merge = merges[i];
      // Newer files store [left, right]; older ones "left right".
      final key = merge is List ? '${merge[0]} ${merge[1]}' : merge as String;
      ranks.putIfAbsent(key, () => i);
    }
    final addedTokens = (json['added_tokens'] as List)
        .cast<Map<String, Object?>>();
    final added = <String, int>{
      for (final token in addedTokens)
        token['content'] as String: token['id'] as int,
    };
    final special = <int>{
      for (final token in addedTokens)
        if (token['special'] == true) token['id'] as int,
    };
    final pieceOf = <int, String>{
      for (final MapEntry(:key, :value) in vocab.entries) value: key,
      for (final MapEntry(:key, :value) in added.entries) value: key,
    };
    return BpeTokenizer._(
      vocab,
      ranks,
      added,
      special,
      pieceOf,
      _pretokenizerOf(json['pre_tokenizer']),
      model['ignore_merges'] == true,
    );
  }

  /// The regex of the pre-tokenizer's Split step: one of the two this
  /// engine knows, whose `(?i:...)` group it spells out with character
  /// classes because Dart's RegExp has no inline flags.
  static RegExp _pretokenizerOf(Object? preTokenizer) {
    String? pattern;
    if (preTokenizer case {
      'type': 'Sequence',
      'pretokenizers': [
        {
          'type': 'Split',
          'pattern': {'Regex': final String regex},
          'behavior': 'Isolated',
          'invert': false,
        },
        {'type': 'ByteLevel', 'add_prefix_space': false, 'use_regex': false},
      ],
    }) {
      pattern = regex;
    }
    final digits = switch (pattern) {
      _lfm3Pattern => r'\p{N}{1,3}',
      _qwen2Pattern => r'\p{N}',
      _ => throw const FormatException(
        'tokenizer.json: unsupported pre-tokenizer',
      ),
    };
    return RegExp(
      r"'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"
      r'|[^\r\n\p{L}\p{N}]?\p{L}+'
      '|$digits'
      r'| ?[^\s\p{L}\p{N}]+[\r\n]*'
      r'|\s*[\r\n]+'
      r'|\s+(?!\S)'
      r'|\s+',
      unicode: true,
    );
  }

  static const _lfm3Pattern =
      r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| "
      r'?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';
  static const _qwen2Pattern =
      r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| "
      r'?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+';

  final Map<String, int> _vocab;

  /// Merge rank keyed by "left right". Byte-level symbols never contain a
  /// raw space (0x20 maps to 'Ġ'), so the space is an unambiguous separator.
  final Map<String, int> _ranks;
  final Map<String, int> _added;

  /// Ids of added tokens marked special (skipped by [decode] on request).
  final Set<int> _special;
  final Map<int, String> _pieceOf;
  final RegExp _pretokenizer;

  /// Whether a piece found whole in the vocabulary skips the merges.
  final bool _ignoreMerges;
  final Map<String, List<int>> _cache = {};

  late final Map<int, List<String>> _addedByFirstUnit = () {
    final byFirst = <int, List<String>>{};
    for (final content in _added.keys) {
      byFirst.putIfAbsent(content.codeUnitAt(0), () => []).add(content);
    }
    for (final list in byFirst.values) {
      list.sort((a, b) => b.length.compareTo(a.length));
    }
    return byFirst;
  }();

  static final List<String> _byteToChar = _buildByteToChar();
  static final Map<int, int> _charToByte = {
    for (var b = 0; b < 256; b++) _byteToChar[b].codeUnitAt(0): b,
  };

  /// Id of an added token such as `<|im_end|>`, or null.
  int? addedTokenId(String content) => _added[content];

  /// The first added token spelled out in [text], or null. [encode] turns
  /// such spellings into control tokens, so text that must stay plain (a
  /// typed question, a replayed answer) is checked with this first.
  String? addedTokenIn(String text) {
    for (var i = 0; i < text.length; i++) {
      if (_addedAt(text, i) case final match?) return match;
    }
    return null;
  }

  List<int> encode(String text) {
    final ids = <int>[];
    var plainStart = 0;
    var i = 0;
    while (i < text.length) {
      final match = _addedAt(text, i);
      if (match == null) {
        i++;
        continue;
      }
      _encodePlain(text.substring(plainStart, i), ids);
      ids.add(_added[match]!);
      i += match.length;
      plainStart = i;
    }
    _encodePlain(text.substring(plainStart), ids);
    return ids;
  }

  /// The text of [ids]; with [skipSpecialTokens], without the added tokens
  /// marked special (`<|im_end|>`, but not Qwen3's `<think>`).
  String decode(List<int> ids, {bool skipSpecialTokens = false}) {
    final bytes = <int>[];
    for (final id in ids) {
      if (skipSpecialTokens && _special.contains(id)) continue;
      final piece = _pieceOf[id];
      if (piece == null) continue;
      if (_added.containsKey(piece)) {
        bytes.addAll(utf8.encode(piece));
        continue;
      }
      for (final unit in piece.codeUnits) {
        bytes.add(_charToByte[unit] ?? 0x3f);
      }
    }
    return utf8.decode(bytes, allowMalformed: true);
  }

  String? _addedAt(String text, int i) {
    final candidates = _addedByFirstUnit[text.codeUnitAt(i)];
    if (candidates == null) return null;
    for (final content in candidates) {
      if (text.startsWith(content, i)) return content;
    }
    return null;
  }

  void _encodePlain(String text, List<int> ids) {
    if (text.isEmpty) return;
    var last = 0;
    for (final match in _pretokenizer.allMatches(text)) {
      // The pattern covers every character, but keep any gap as a piece
      // rather than dropping text silently.
      if (match.start > last) {
        ids.addAll(_bpe(text.substring(last, match.start)));
      }
      ids.addAll(_bpe(match[0]!));
      last = match.end;
    }
    if (last < text.length) ids.addAll(_bpe(text.substring(last)));
  }

  List<int> _bpe(String piece) {
    final cached = _cache[piece];
    if (cached != null) return cached;
    final mapped = StringBuffer();
    for (final b in utf8.encode(piece)) {
      mapped.write(_byteToChar[b]);
    }
    final word = mapped.toString();
    if (_ignoreMerges) {
      final whole = _vocab[word];
      if (whole != null) return _cache[piece] = [whole];
    }

    var symbols = [
      for (final unit in word.codeUnits) String.fromCharCode(unit),
    ];
    while (symbols.length > 1) {
      var bestRank = -1;
      var bestLeft = '';
      var bestRight = '';
      for (var i = 0; i + 1 < symbols.length; i++) {
        final rank = _ranks['${symbols[i]} ${symbols[i + 1]}'];
        if (rank != null && (bestRank < 0 || rank < bestRank)) {
          bestRank = rank;
          bestLeft = symbols[i];
          bestRight = symbols[i + 1];
        }
      }
      if (bestRank < 0) break;
      // Merge every occurrence of the best pair, left to right.
      final merged = <String>[];
      for (var i = 0; i < symbols.length; i++) {
        if (i + 1 < symbols.length &&
            symbols[i] == bestLeft &&
            symbols[i + 1] == bestRight) {
          merged.add(bestLeft + bestRight);
          i++;
        } else {
          merged.add(symbols[i]);
        }
      }
      symbols = merged;
    }
    final ids = [
      for (final symbol in symbols)
        _vocab[symbol] ??
            (throw StateError(
              'symbol ${jsonEncode(symbol)} has no vocabulary id',
            )),
    ];
    return _cache[piece] = ids;
  }

  /// GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest
  /// to code points from 256 upward, in byte order.
  static List<String> _buildByteToChar() {
    bool printable(int b) =>
        (b >= 0x21 && b <= 0x7e) ||
        (b >= 0xa1 && b <= 0xac) ||
        (b >= 0xae && b <= 0xff);
    var next = 256;
    return [
      for (var b = 0; b < 256; b++)
        String.fromCharCode(printable(b) ? b : next++),
    ];
  }
}
