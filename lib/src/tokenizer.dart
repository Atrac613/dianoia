/// LFM3's tokenizer, reimplemented from tokenizer.json.
///
/// Encoding runs the same four stages as Hugging Face tokenizers:
///
/// 1. Added tokens (`<|im_start|>`, `<tool_call>`, ...) are cut out of the
///    text first, leftmost-longest. None of them strip neighbouring spaces.
/// 2. The rest is split by the pre-tokenizer regex (contractions, letter
///    runs, 1-3 digit runs, symbol runs, newlines, other whitespace).
/// 3. Each piece's UTF-8 bytes are mapped to printable characters with
///    GPT-2's byte-to-unicode table, so every byte has a vocabulary entry.
/// 4. BPE: a piece that is itself in the vocabulary becomes one token
///    (`ignore_merges`). Otherwise the lowest-ranked adjacent pair is merged
///    until no ranked pair is left.
///
/// Known gap: Dart's `\s` (JavaScript semantics) and the Oniguruma `\s` used
/// by tokenizers disagree on U+0085 and U+FEFF. Neither appears in the
/// oracle corpus.
library;

import 'dart:convert';
import 'dart:io';

final class Lfm3Tokenizer {
  Lfm3Tokenizer._(this._vocab, this._ranks, this._added, this._pieceOf);

  factory Lfm3Tokenizer.load(String modelDir) {
    final json =
        jsonDecode(File('$modelDir/tokenizer.json').readAsStringSync())
            as Map<String, Object?>;
    final model = json['model'] as Map<String, Object?>;
    if (model['type'] != 'BPE' || model['ignore_merges'] != true) {
      throw const FormatException(
        'tokenizer.json: expected BPE with ignore_merges',
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
    final added = <String, int>{
      for (final token in json['added_tokens'] as List)
        (token as Map<String, Object?>)['content'] as String:
            token['id'] as int,
    };
    final pieceOf = <int, String>{
      for (final MapEntry(:key, :value) in vocab.entries) value: key,
      for (final MapEntry(:key, :value) in added.entries) value: key,
    };
    return Lfm3Tokenizer._(vocab, ranks, added, pieceOf);
  }

  final Map<String, int> _vocab;

  /// Merge rank keyed by "left right". Byte-level symbols never contain a
  /// raw space (0x20 maps to 'Ġ'), so the space is an unambiguous separator.
  final Map<String, int> _ranks;
  final Map<String, int> _added;
  final Map<int, String> _pieceOf;
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

  /// The (?i:'s|'t|...) group of the original pattern is spelled out with
  /// character classes because Dart's RegExp has no inline flags.
  static final RegExp _pretokenizer = RegExp(
    r"'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])"
    r'|[^\r\n\p{L}\p{N}]?\p{L}+'
    r'|\p{N}{1,3}'
    r'| ?[^\s\p{L}\p{N}]+[\r\n]*'
    r'|\s*[\r\n]+'
    r'|\s+(?!\S)'
    r'|\s+',
    unicode: true,
  );

  static final List<String> _byteToChar = _buildByteToChar();
  static final Map<int, int> _charToByte = {
    for (var b = 0; b < 256; b++) _byteToChar[b].codeUnitAt(0): b,
  };

  /// Id of an added token such as `<|im_end|>`, or null.
  int? addedTokenId(String content) => _added[content];

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

  String decode(List<int> ids) {
    final bytes = <int>[];
    for (final id in ids) {
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
    final whole = _vocab[word];
    if (whole != null) return _cache[piece] = [whole];

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
