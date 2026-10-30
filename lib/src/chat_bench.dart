/// Time to first token in an app-like setting, with and without reusing the
/// model's state (chat_session.dart), shared by `lfm3 chatbench` and the
/// iPhone bench app.
///
/// Every case of a tool-case file (example/tool_cases.json) is asked as its
/// own conversation, all with one system prompt and the whole tool catalog,
/// as an app would. Reported: the cold first request, the warm-up of the
/// system and tools block, saving that state to a file and loading it back
/// (as a later app start would), each conversation started from that
/// checkpoint, and a tool round trip continued in place. Reusing state,
/// from memory or from the file, must not change a bit of the logits, and
/// that is checked against computing from scratch.
library;

import 'dart:io';
import 'dart:typed_data';

import 'chat_session.dart';
import 'model.dart';
import 'tokenizer.dart';
import 'tool_calls.dart';

const benchSystemPrompt =
    'You are an assistant on the user\'s phone. Answer briefly. When the '
    'user asks for an action or for live information, call a tool.';

const _toolResult =
    '{"city": "Paris", "temperature_c": 18, "condition": "light rain"}';

/// One call with the expected name whose every expected argument contains
/// one of its fragments, ignoring case; a null [expect] wants no call.
bool toolCallMatches(List<ToolCall> calls, Map<String, Object?>? expect) {
  if (expect == null) return calls.isEmpty;
  if (calls.length != 1 || calls.single.name != expect['name']) return false;
  final args = calls.single.arguments;
  for (final MapEntry(:key, :value)
      in (expect['arguments'] as Map<String, Object?>).entries) {
    final actual = (args[key] ?? '').toLowerCase();
    if (!(value as List).cast<String>().any(actual.contains)) return false;
  }
  return true;
}

/// The tool definitions of a tool-case file, in template form.
List<Map<String, Object?>> catalogTools(Map<String, Object?> spec) => [
  for (final MapEntry(:key, :value)
      in (spec['tools'] as Map).cast<String, Object?>().entries)
    <String, Object?>{
      'type': 'function',
      'function': <String, Object?>{
        'name': key,
        ...(value as Map).cast<String, Object?>(),
      },
    },
];

/// Runs the benchmark on [model] with the cases of [spec], reporting lines
/// to [log], and returns a record of the timings.
Future<Map<String, Object?>> chatBench(
  Lfm3Model model,
  Lfm3Tokenizer tokenizer,
  Map<String, Object?> spec,
  void Function(String) log, {
  required String statePath,
}) async {
  final tools = catalogTools(spec);
  final cases = (spec['cases'] as List).cast<Map<String, Object?>>();
  final session = ChatSession(
    model,
    tokenizer,
    system: benchSystemPrompt,
    tools: tools,
    maxNew: 96,
  );
  Map<String, Object?> user(int i) => {
    'role': 'user',
    'content': cases[i]['prompt'],
  };
  String ms(double s) => '${(s * 1000).toStringAsFixed(0)} ms';

  // Reuse must give the logits of a from-scratch run, bit for bit: a
  // conversation started from the warmed-up block, and a tool result
  // continued in place.
  session.keepPromptLogits = true;
  await model.reset();
  await session.warmUp();
  session.clear();
  final first = await session.send(user(0));
  final reusedLogits = session.promptLogits!;
  final afterTool = await session.send({
    'role': 'tool',
    'content': _toolResult,
  });
  final continuedLogits = session.promptLogits!;
  session.keepPromptLogits = false;
  final roundTrip = session.messages.sublist(0, 3);
  final scratch = await _fromScratch(model, session, [user(0)]);
  var identical =
      _same(scratch, reusedLogits) &&
      _same(await _fromScratch(model, session, roundTrip), continuedLogits);

  // The same block saved to a file and loaded back, as at a later start.
  await model.reset();
  await session.warmUp();
  var sw = Stopwatch()..start();
  await model.saveState(statePath);
  final saveSeconds = sw.elapsedMicroseconds / 1e6;
  final stateBytes = File(statePath).lengthSync();
  await model.reset();
  sw = Stopwatch()..start();
  final loaded = await session.warmUp(statePath: statePath);
  final loadSeconds = sw.elapsedMicroseconds / 1e6;
  session
    ..clear()
    ..keepPromptLogits = true;
  await session.send(user(0));
  session.keepPromptLogits = false;
  identical = identical && loaded && _same(scratch, session.promptLogits!);
  log(
    'Reused state, from memory and from a file, gives the from-scratch '
    'logits: ${identical ? 'bit for bit' : 'NO, THEY DIFFER'}',
  );
  log(
    'State file of the system and tools block: '
    '${(stateBytes / 1e6).toStringAsFixed(1)} MB, saved in '
    '${ms(saveSeconds)}, loaded in ${ms(loadSeconds)}'
    '${loaded ? '' : ' (NOT LOADED)'}',
  );

  // Cold: nothing to reuse.
  await model.reset();
  session.clear();
  final cold = await session.send(user(0));
  log(
    'Cold request: ${cold.promptTokens} prompt tokens '
    '(system and tools ${session.prefix.length}), first token after '
    '${ms(cold.prefillSeconds)}',
  );
  await model.reset();
  sw = Stopwatch()..start();
  await session.warmUp();
  final warmUpSeconds = sw.elapsedMicroseconds / 1e6;
  log('Warm-up of the system and tools block: ${ms(warmUpSeconds)}');

  final runs = <Map<String, Object?>>[];
  var passed = 0;
  for (var i = 0; i < cases.length; i++) {
    session.clear();
    final turn = await session.send(user(i));
    final ok = toolCallMatches(
      turn.reply.toolCalls,
      cases[i]['expect'] as Map<String, Object?>?,
    );
    if (ok) passed++;
    runs.add({
      'prompt_tokens': turn.promptTokens,
      'new_tokens': turn.newTokens,
      'ttft_s': turn.prefillSeconds,
      'decode_tok_per_s': turn.generatedTokens / turn.decodeSeconds,
      'passed': ok,
    });
  }
  final ttfts = [for (final r in runs) r['ttft_s'] as double]..sort();
  log(
    'Warm requests: ${cases.length} conversations, first token after '
    '${ms(ttfts[ttfts.length ~/ 2])} median (${ms(ttfts.first)}-'
    '${ms(ttfts.last)}); tool calls right $passed/${cases.length} with all '
    '${tools.length} tools offered',
  );
  log(
    'Tool round trip: the tool result (${afterTool.newTokens} new tokens) '
    'continued in place, first token after ${ms(afterTool.prefillSeconds)}; '
    'reply: ${afterTool.text.replaceAll('\n', ' ')}',
  );
  return {
    'identical_logits': identical,
    'prefix_tokens': session.prefix.length,
    'cold': {'prompt_tokens': cold.promptTokens, 'ttft_s': cold.prefillSeconds},
    'warm_up_s': warmUpSeconds,
    'state_bytes': stateBytes,
    'state_save_s': saveSeconds,
    'state_load_s': loadSeconds,
    'warm': runs,
    'warm_ttft_median_s': ttfts[ttfts.length ~/ 2],
    'passed': passed,
    'cases': cases.length,
    'first_reply': first.text,
    'tool_round_trip': {
      'new_tokens': afterTool.newTokens,
      'ttft_s': afterTool.prefillSeconds,
      'reply': afterTool.text,
    },
  };
}

/// The prompt logits for [conversation], computed from an empty state.
Future<Float32List> _fromScratch(
  Lfm3Model model,
  ChatSession session,
  List<Map<String, Object?>> conversation,
) async {
  await model.reset();
  await model.prefill(session.promptIds(conversation));
  return Float32List.fromList(model.logits);
}

bool _same(Float32List a, Float32List b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i] && !(a[i].isNaN && b[i].isNaN)) return false;
  }
  return true;
}
