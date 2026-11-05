import 'dart:typed_data';

import 'package:dianoia/lab.dart';

/// System prompts of the photo conversation, one per reply mode.
const photoTextSystem = 'Respond in text.';

/// One completed question about the photo: the typed [text] and/or the
/// projected rows of a spoken question ([audio]), and the model's [answer].
typedef PhotoTurn = ({String? text, Float32List? audio, String answer});

/// The state of a conversation about one photo, kept in the inference
/// worker. The photo is encoded once; every question re-renders the whole
/// conversation with the official template (earlier answers as text, the
/// photo and spoken questions as their cached rows), so the spoken-reply
/// setting may change between questions and other tasks may use the backbone
/// in between.
final class PhotoConversation {
  PhotoConversation(this.imageRows, this.imageTokens, this.hiddenSize) {
    if (imageTokens <= 0 || imageRows.length != imageTokens * hiddenSize) {
      throw ArgumentError('Photo rows do not match its token count');
    }
  }

  final Float32List imageRows;
  final int imageTokens, hiddenSize;
  final _turns = <PhotoTurn>[];
  List<PhotoTurn> get turns => List.unmodifiable(_turns);

  /// The messages for the next question, which carries [text] and/or a
  /// spoken question when [spoken]. The photo opens the first user turn.
  List<Map<String, Object?>> messages({
    required bool speak,
    String? text,
    required bool spoken,
  }) {
    List<Map<String, Object?>> user(String? text, bool audio, bool image) => [
      {
        'role': 'user',
        'content': [
          if (image) {'type': 'image'},
          if (audio) {'type': 'audio'},
          if (text != null && text.isNotEmpty) {'type': 'text', 'text': text},
        ],
      },
    ];
    return [
      {
        'role': 'system',
        'content': speak ? audioConversationSystem : photoTextSystem,
      },
      for (final (i, turn) in _turns.indexed) ...[
        ...user(turn.text, turn.audio != null, i == 0),
        {'role': 'assistant', 'content': turn.answer},
      ],
      ...user(text, spoken, _turns.isEmpty),
    ];
  }

  /// Projected rows of earlier spoken questions, then [current] if any, in
  /// prompt order.
  List<Float32List> audio([Float32List? current]) => [
    for (final turn in _turns) ?turn.audio,
    ?current,
  ];

  /// Token counts matching [audio].
  List<int> audioTokens([Float32List? current]) => [
    for (final rows in audio(current)) rows.length ~/ hiddenSize,
  ];

  /// Forgets the questions after the first [count], which the caller never
  /// received (a reply cancelled after it was generated).
  void keep(int count) {
    if (count < 0 || count > _turns.length) {
      throw StateError('The photo conversation was restarted');
    }
    _turns.removeRange(count, _turns.length);
  }

  void add({String? text, Float32List? audio, required String answer}) {
    if ((text == null || text.isEmpty) && audio == null) {
      throw ArgumentError('A question needs text or audio');
    }
    _turns.add((text: text, audio: audio, answer: answer));
  }
}
