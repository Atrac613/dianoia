/// A live canary for long reasoning: rounds of the running-sum task of
/// lib/src/reasoning_canary.dart, each longer than the last, graded line by
/// line as the reply streams in.
///
/// Usage: `dart run tool/reasoning_canary.dart <bundle>`
///        `[--backend=auto|cpu|gpu] [--context=4096]`
///        `[--steps=25,50,100,200,400] [--rounds=0] [--seed=1]`
///        `[--temperature=0.7] [--top-k=20] [--top-p=0.8]`
///        `[--presence-penalty=0] [--guard=off|on]`
///        `[--thinking=off|on] [--budget=N] [--stop-on-error=off|on]`
///        `[--verify=off|on] [--retries=4] [--indexed=off|on]`
///        `[--out=canary.jsonl] [--keep-text=off|on]`
///
/// Rounds walk the --steps ladder, skipping lengths whose prompt and
/// answer do not fit the context, and start over at the bottom with the
/// next seed; --rounds=0 runs until Ctrl-C. A status line shows the round
/// as it goes: steps written, steps right since the first, and when and
/// how the first mistake came. With --verify=on each line is checked as it
/// is written and a wrong one is rewound and sampled again, up to
/// --retries times (ChatSession.send's checkLine); the round then ends at
/// the last right line. --indexed=on labels each number in the prompt
/// with its step (RunningSumTask.indexed). Each finished round prints a line and,
/// with --out, appends a JSON record (with the reply's text under
/// --keep-text=on); Ctrl-C ends the round in progress
/// and prints a summary per length.
library;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:dianoia/lab.dart';

const _usage =
    'usage: dart run tool/reasoning_canary.dart <bundle> '
    '[--backend=auto|cpu|gpu] [--context=N] [--steps=25,50,...] '
    '[--rounds=0] [--seed=1] [--temperature=0.7] [--top-k=20] [--top-p=0.8] '
    '[--presence-penalty=0] [--guard=off|on] [--thinking=off|on] '
    '[--budget=N] [--stop-on-error=off|on] [--verify=off|on] [--retries=4] '
    '[--indexed=off|on] '
    '[--out=canary.jsonl] [--keep-text=off|on]';

Never _fail(String message) {
  stderr
    ..writeln(message)
    ..writeln(_usage);
  exit(64);
}

Future<void> main(List<String> args) async {
  final positional = <String>[];
  final options = <String, String>{};
  for (final arg in args) {
    if (arg.startsWith('--') && arg.contains('=')) {
      final at = arg.indexOf('=');
      options[arg.substring(2, at)] = arg.substring(at + 1);
    } else {
      positional.add(arg);
    }
  }
  const known = {
    'backend',
    'context',
    'steps',
    'rounds',
    'seed',
    'temperature',
    'top-k',
    'top-p',
    'presence-penalty',
    'guard',
    'thinking',
    'budget',
    'stop-on-error',
    'verify',
    'retries',
    'indexed',
    'out',
    'keep-text',
  };
  if (positional.length != 1) _fail('expected one bundle directory');
  final unknown = options.keys.toSet().difference(known);
  if (unknown.isNotEmpty) _fail('unknown options: ${unknown.join(', ')}');

  T parse<T>(String name, String fallback, T? Function(String) parser) =>
      parser(options[name] ?? fallback) ?? _fail('bad --$name');
  bool flag(String name) => switch (options[name] ?? 'off') {
    'on' => true,
    'off' => false,
    _ => _fail('--$name takes on or off'),
  };

  final backend = InferenceBackend.values
      .where((b) => b.name == (options['backend'] ?? 'auto'))
      .firstOrNull;
  if (backend == null) _fail('bad --backend');
  final context = options['context'] == null
      ? null
      : parse('context', '', int.tryParse);
  final ladder = parse('steps', '25,50,100,200,400', (s) {
    final steps = s.split(',').map(int.tryParse).toList();
    return steps.any((n) => n == null || n < 1) ? null : steps.cast<int>();
  });
  final rounds = parse('rounds', '0', int.tryParse);
  var seed = parse('seed', '1', int.tryParse);
  final sampling = SamplingParams(
    temperature: parse('temperature', '0.7', double.tryParse),
    topK: parse('top-k', '20', int.tryParse),
    topP: parse('top-p', '0.8', double.tryParse),
    presencePenalty: parse('presence-penalty', '0', double.tryParse),
    seed: seed,
  );
  final guard = flag('guard') ? const RepetitionGuard() : null;
  final thinking = flag('thinking');
  final budget = options['budget'] == null
      ? null
      : parse('budget', '', int.tryParse);
  final stopOnError = flag('stop-on-error');
  final verify = flag('verify');
  final indexed = flag('indexed');
  final retries = parse('retries', '4', int.tryParse);
  if (retries < 0) _fail('--retries must not be negative');
  final out = options['out'] == null ? null : File(options['out']!);
  final keepText = flag('keep-text');

  stdout.writeln('loading ${positional.single} ...');
  final model = await LocalModel.load(
    positional.single,
    backend: backend,
    contextLength: context,
  );
  final settings = {
    // ModelStore keeps bundles in <model>/bundle.
    'model': () {
      final parts = positional.single.split('/').where((p) => p.isNotEmpty);
      return parts.last == 'bundle' && parts.length > 1
          ? parts.elementAt(parts.length - 2)
          : parts.last;
    }(),
    'backend': model.backend.name,
    'context': model.contextLength,
    'temperature': sampling.temperature,
    'top_k': sampling.topK,
    'top_p': sampling.topP,
    'presence_penalty': sampling.presencePenalty,
    'guard': guard != null,
    'thinking': thinking,
    'budget': budget,
    'verify': verify,
    'indexed': indexed,
    if (verify) 'retries': retries,
  };
  stdout.writeln(
    'settings: ${settings.entries.map((e) => '${e.key}=${e.value}').join(' ')}',
  );

  var stopping = false;
  final sigint = ProcessSignal.sigint.watch().listen((_) {
    if (stopping) exit(130); // a second Ctrl-C does not wait
    stopping = true;
    stdout.writeln('\nstopping after this round ...');
  });

  final records = <Map<String, Object?>>[];
  var round = 0;
  while (!stopping && (rounds == 0 || round < rounds)) {
    var ran = false;
    for (final steps in ladder) {
      if (stopping || (rounds > 0 && round >= rounds)) break;
      final task = RunningSumTask(steps, seed: seed, indexed: indexed);
      final session = model.chat(
        maxNewTokens: model.contextLength,
        sampling: SamplingParams(
          temperature: sampling.temperature,
          topK: sampling.topK,
          topP: sampling.topP,
          presencePenalty: sampling.presencePenalty,
          seed: seed * 1000 + steps,
        ),
        enableThinking: thinking,
        thinkingBudget: budget,
        repetitionGuard: guard,
      );
      final message = {'role': 'user', 'content': task.prompt};
      // The answer with a tenth to spare, and the budget when thinking.
      final needed =
          session.promptIds([message]).length +
          (session.tokenizer.encode(task.answer).length * 1.1).ceil() +
          (thinking ? (budget ?? 0) : 0);
      if (needed > model.contextLength) continue;
      ran = true;
      round++;
      final record = await _round(
        session,
        task,
        message,
        label: 'r$round N=$steps seed=$seed',
        stopOnError: stopOnError,
        verify: verify,
        retries: retries,
        keepText: keepText,
        stopping: () => stopping,
      );
      final full = {
        'time': DateTime.now().toUtc().toIso8601String(),
        ...settings,
        'round': round,
        ...record,
      };
      records.add(full);
      out?.writeAsStringSync('${jsonEncode(full)}\n', mode: FileMode.append);
    }
    if (!ran) {
      stdout.writeln(
        'no --steps length fits a ${model.contextLength}-token context',
      );
      break;
    }
    seed++;
  }
  await sigint.cancel();
  _summary(records);
  await model.close();
}

/// Runs one round, keeping a status line current, and returns its record.
Future<Map<String, Object?>> _round(
  ChatSession session,
  RunningSumTask task,
  Map<String, Object?> message, {
  required String label,
  required bool stopOnError,
  required bool verify,
  required int retries,
  required bool keepText,
  required bool Function() stopping,
}) async {
  final sw = Stopwatch()..start();
  var tokens = 0;
  var grade = task.grade('');
  var answerFrom = 0; // tokens before the answer, when thinking
  int? errorTokens;
  double? errorSeconds;
  var correctTokens = 0;
  var correctSeconds = 0.0;
  var status = '';
  final shown = Stopwatch()..start();
  var rejected = 0;

  void show({bool force = false}) {
    if (!force && shown.elapsedMilliseconds < 200) return;
    shown.reset();
    final seconds = sw.elapsedMicroseconds / 1e6;
    final rate = seconds > 0 ? tokens / seconds : 0;
    // With --verify a wrong line shows until it is rewound, with no time:
    // only one that stays counts.
    final error = switch ((grade.firstError, errorSeconds)) {
      (null, _) => '',
      (final kind?, null) => '  ✗ step ${grade.firstErrorStep} (${kind.name})',
      (final kind?, final at?) =>
        '  ✗ step ${grade.firstErrorStep} (${kind.name}) '
            'at $errorTokens tok, ${at.toStringAsFixed(1)} s',
    };
    status =
        '[$label] ${grade.lines}/${task.steps} lines  ✓${grade.correct}  '
        '~${grade.consistent}  $tokens tok  ${seconds.toStringAsFixed(1)} s  '
        '${rate.toStringAsFixed(1)} tok/s'
        '${verify ? '  ↺$rejected' : ''}$error';
    stdout.write('\r\x1b[2K$status');
  }

  String answerOf(String text) {
    final end = text.lastIndexOf('</think>');
    return end < 0
        ? (session.enableThinking ? '' : text)
        : text.substring(end + '</think>'.length);
  }

  final turn = await session.send(
    message,
    // Every graded line so far right; lines before the first step pass.
    checkLine: verify
        ? (reply) {
            final g = task.grade(answerOf(reply));
            if (g.correct == g.lines) return true;
            rejected++;
            return false;
          }
        : null,
    lineRetries: retries,
    onText: (text) async {
      tokens++;
      if (session.enableThinking &&
          answerFrom == 0 &&
          text.contains('</think>')) {
        answerFrom = tokens;
      }
      if (text.endsWith('\n')) {
        final before = grade;
        grade = task.grade(answerOf(text));
        if (grade.correct > before.correct) {
          correctTokens = tokens;
          correctSeconds = sw.elapsedMicroseconds / 1e6;
        }
        if (!verify && before.firstError == null && grade.firstError != null) {
          errorTokens = tokens;
          errorSeconds = sw.elapsedMicroseconds / 1e6;
          show(force: true);
        }
      }
      show();
      // Decoding completes its futures without returning to the event
      // loop, which delivers Ctrl-C: let it run once a line.
      if (text.endsWith('\n')) await Future<void>.delayed(Duration.zero);
      if (stopping()) return false;
      if (grade.lines >= task.steps) return false;
      return !(stopOnError && grade.firstError != null);
    },
  );
  final seconds = sw.elapsedMicroseconds / 1e6;
  final before = grade;
  grade = task.grade(answerOf(turn.text), finished: true);
  if (before.firstError == null && grade.firstError != null) {
    errorTokens = tokens;
    errorSeconds = seconds;
  } else if (grade.correct > before.correct) {
    correctTokens = tokens;
    correctSeconds = seconds;
  }
  final end = turn.promptTokens + turn.generatedTokens;
  final stop = switch (null) {
    _ when grade.lines >= task.steps => 'done',
    _ when turn.stoppedOnCheck => 'check',
    _ when turn.stoppedOnRepetition => 'loop',
    _ when stopOnError && grade.firstError != null => 'error',
    _ when stopping() => 'interrupted',
    _ when end >= session.model.maxSeq => 'context',
    _ when turn.generatedTokens >= session.maxNew => 'max_new',
    _ => 'end',
  };
  show(force: true);
  stdout.writeln('  → $stop${turn.thinkingCut ? ' (thinking cut)' : ''}');
  return {
    'steps': task.steps,
    'seed': task.seed,
    ...grade.toJson(),
    'stop': stop,
    'prompt_tokens': turn.promptTokens,
    'generated_tokens': turn.generatedTokens,
    'thinking_tokens': answerFrom,
    'thinking_cut': turn.thinkingCut,
    'rejected_lines': turn.rejectedLines,
    // With --verify, every token fed, rejected lines included.
    'work_tokens': tokens,
    'seconds': seconds,
    'tokens_per_second': seconds > 0 ? turn.generatedTokens / seconds : 0,
    'correct_tokens': correctTokens,
    'correct_seconds': correctSeconds,
    'first_error_tokens': errorTokens,
    'first_error_seconds': errorSeconds,
    if (keepText) 'text': turn.text,
  };
}

/// How each length fared: rounds, fully right, and steps held on average.
void _summary(List<Map<String, Object?>> records) {
  if (records.isEmpty) return;
  stdout.writeln('\nsteps  rounds  all right  mean ✓  mean ~  longest ✓ (s)');
  final lengths = {for (final r in records) r['steps']! as int}.toList()
    ..sort();
  for (final steps in lengths) {
    final rs = [
      for (final r in records)
        if (r['steps'] == steps) r,
    ];
    double mean(String key) =>
        rs.map((r) => r[key]! as int).reduce((a, b) => a + b) / rs.length;
    final allRight = rs.where((r) => r['correct'] == steps).length;
    final longest = rs
        .map((r) => r['correct_seconds']! as double)
        .reduce((a, b) => a > b ? a : b);
    stdout.writeln(
      '${'$steps'.padLeft(5)}  ${'${rs.length}'.padLeft(6)}  '
      '${'$allRight'.padLeft(9)}  ${mean('correct').toStringAsFixed(1).padLeft(6)}  '
      '${mean('consistent').toStringAsFixed(1).padLeft(6)}  '
      '${longest.toStringAsFixed(1).padLeft(13)}',
    );
  }
}
