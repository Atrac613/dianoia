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

void writeSyntheticTokenizer(String dir) {
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
      'added_tokens': [
        for (var i = 0; i < _specials.length; i++)
          {'content': _specials[i], 'id': 256 + i},
      ],
    }),
  );
}
