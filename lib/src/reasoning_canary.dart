/// A long task whose every step can be checked as it streams in, for
/// watching how long a model keeps reasoning correctly
/// (tool/reasoning_canary.dart).
///
/// The model adds a list of two-digit numbers one at a time and writes
/// each step as `i: previous + number = total`. Every line tests the four
/// things long reasoning wears down: keeping count ([CanaryError.step]),
/// finding the right number far back in the prompt
/// ([CanaryError.operand]), carrying its own last result forward
/// ([CanaryError.carry]) and adding ([CanaryError.arithmetic]). The
/// numbers come from a seed, so a run can be repeated exactly.
library;

import 'dart:math' as math;

/// What went wrong in a step, in the order a line is checked.
enum CanaryError {
  /// The line is not `i: a + b = c`.
  format,

  /// The step number is not the next one.
  step,

  /// The number added is not the one the prompt lists for this step.
  operand,

  /// The previous total is not the total of the line before.
  carry,

  /// The total is not the sum of the line's own two numbers.
  arithmetic,
}

/// One running sum to work through.
final class RunningSumTask {
  RunningSumTask(this.steps, {required this.seed, this.indexed = false})
    : assert(steps > 0),
      numbers = _numbers(steps, seed);

  final int steps;
  final int seed;

  /// Whether the prompt labels each number with its step (`31) 71`), so
  /// that finding the next one is a lookup of its label rather than
  /// counting along the list. The numbers and answer stay the same.
  final bool indexed;

  /// The numbers to add, one per step, each in 10..99.
  final List<int> numbers;

  static List<int> _numbers(int steps, int seed) {
    final random = math.Random(seed);
    return [for (var i = 0; i < steps; i++) 10 + random.nextInt(90)];
  }

  /// The correct total after each step.
  late final List<int> totals = () {
    var sum = 0;
    return [for (final n in numbers) sum += n];
  }();

  /// The user message that asks for the steps. Two worked lines show the
  /// form: small models copy a `<placeholder>` literally.
  String get prompt {
    final lines = answer.split('\n');
    return 'Add these $steps numbers one at a time, starting from 0. Write '
        'one line per number, numbered from 1, and nothing else. The first '
        'lines are:\n${lines.take(2).join('\n')}\n\n'
        'Numbers: ${indexed ? [for (var i = 0; i < steps; i++) '${i + 1}) ${numbers[i]}'].join(', ') : numbers.join(', ')}';
  }

  /// The full correct answer, for sizing a round against the context.
  String get answer => [
    for (var i = 0; i < steps; i++)
      '${i + 1}: ${i == 0 ? 0 : totals[i - 1]} + ${numbers[i]} = ${totals[i]}',
  ].join('\n');

  static final _line = RegExp(
    r'^\s*(\d+)\s*:\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)',
  );

  /// Grades the complete lines of [text], the reply so far: a last line
  /// without its newline counts only when [finished]. Wrapping that larger
  /// models add is skipped: anything before the first step line (`Here
  /// is the output:`), blank lines and code fences.
  CanaryGrade grade(String text, {bool finished = false}) {
    var lines = text.split('\n');
    if (!finished) lines = lines.sublist(0, lines.length - 1);
    final first = lines.indexWhere(_line.hasMatch);
    lines = [
      if (first >= 0)
        for (final line in lines.skip(first))
          if (line.trim().isNotEmpty && !line.trim().startsWith('```')) line,
    ];
    var correct = 0;
    var consistent = 0;
    CanaryError? firstError;
    int? firstErrorStep;
    var stillCorrect = true;
    int? ownTotal = 0;
    for (var i = 0; i < lines.length && i < steps; i++) {
      final error = _check(lines[i], i, ownTotal);
      final match = _line.firstMatch(lines[i]);
      ownTotal = match == null ? null : int.parse(match.group(4)!);
      if (error == null) consistent++;
      // While every line before was right, a line that holds on its own
      // terms carried the true total, so it is right too.
      if (!stillCorrect) continue;
      if (error == null) {
        correct++;
      } else {
        stillCorrect = false;
        firstErrorStep = i + 1;
        firstError = error;
      }
    }
    return CanaryGrade(
      lines: lines.length < steps ? lines.length : steps,
      correct: correct,
      consistent: consistent,
      firstError: firstError,
      firstErrorStep: firstErrorStep,
    );
  }

  /// The error in [line], step [i] (from 0), after a line whose own total
  /// was [previous] (null when it had none); null when the line holds.
  CanaryError? _check(String line, int i, int? previous) {
    final match = _line.firstMatch(line);
    if (match == null) return CanaryError.format;
    final [index, a, b, c] = [
      for (var g = 1; g <= 4; g++) int.parse(match.group(g)!),
    ];
    if (index != i + 1) return CanaryError.step;
    if (b != numbers[i]) return CanaryError.operand;
    if (previous != null && a != previous) return CanaryError.carry;
    if (a + b != c) return CanaryError.arithmetic;
    return null;
  }
}

/// How far a reply got through a [RunningSumTask].
final class CanaryGrade {
  const CanaryGrade({
    required this.lines,
    required this.correct,
    required this.consistent,
    required this.firstError,
    required this.firstErrorStep,
  });

  /// Complete step lines graded, at most the task's steps.
  final int lines;

  /// Steps right from the first one up to the first mistake: how long the
  /// reasoning held.
  final int correct;

  /// Steps that hold on their own terms (right number, carried total and
  /// sum), wherever they are: how well the model recovers after a mistake.
  final int consistent;

  /// The first mistake and its step (from 1); null while all is right.
  final CanaryError? firstError;
  final int? firstErrorStep;

  Map<String, Object?> toJson() => {
    'lines': lines,
    'correct': correct,
    'consistent': consistent,
    'first_error': firstError?.name,
    'first_error_step': firstErrorStep,
  };
}
