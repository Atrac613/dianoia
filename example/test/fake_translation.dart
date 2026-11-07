import 'dart:async';

import 'package:dianoia_example/src/live_translation.dart';

/// Translates by tagging the text; [hold] keeps a translation in flight.
class FakeTranslation implements TranslationBackend {
  final requests = <(String, String)>[];
  final loads = <String>[];
  Completer<void>? hold;
  Object? failure;
  int cancels = 0, closes = 0;
  @override
  Future<String> load(String languageDirectory) async {
    loads.add(languageDirectory);
    if (failure case final error?) throw error;
    return 'CPU';
  }

  @override
  Future<String> translate(
    String text,
    String target, {
    void Function(String)? onText,
  }) async {
    requests.add((text, target));
    onText?.call('[$target]');
    await hold?.future;
    return '[$target] $text';
  }

  @override
  Future<void> cancel() async {
    cancels++;
    if (hold case final c? when !c.isCompleted) {
      c.completeError(StateError('cancelled'));
    }
  }

  @override
  Future<void> close() async => closes++;
}
