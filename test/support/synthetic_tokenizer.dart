import 'dart:convert';
import 'dart:io';

const _specials = [
  '<|startoftext|>',
  '<|im_start|>',
  '<|im_end|>',
  '<think>',
  '</think>',
  '<|vision_start|>',
  '<|vision_end|>',
  '<|image|>',
  '<|video|>',
];

void writeSyntheticTokenizer(
  String dir, {
  List<String> additionalSpecials = const [],
}) {
  final specials = [..._specials, ...additionalSpecials];
  final visible = [
    for (var i = 33; i <= 126; i++) i,
    for (var i = 161; i <= 172; i++) i,
    for (var i = 174; i <= 255; i++) i,
  ];
  var extra = 256;
  final vocab = <String, int>{};
  for (var byte = 0; byte < 256; byte++) {
    vocab[String.fromCharCode(visible.contains(byte) ? byte : extra++)] = byte;
  }
  File('$dir/tokenizer.json').writeAsStringSync(
    jsonEncode({
      'model': {
        'type': 'BPE',
        'ignore_merges': true,
        'vocab': vocab,
        'merges': [],
      },
      // LFM3's pre-tokenizer, as its tokenizer.json spells it.
      'pre_tokenizer': {
        'type': 'Sequence',
        'pretokenizers': [
          {
            'type': 'Split',
            'pattern': {
              'Regex':
                  r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+"
                  r'|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+'
                  r'|\s+(?!\S)|\s+',
            },
            'behavior': 'Isolated',
            'invert': false,
          },
          {
            'type': 'ByteLevel',
            'add_prefix_space': false,
            'trim_offsets': true,
            'use_regex': false,
          },
        ],
      },
      'added_tokens': [
        for (var i = 0; i < specials.length; i++)
          {'content': specials[i], 'id': 256 + i},
      ],
    }),
  );
}
