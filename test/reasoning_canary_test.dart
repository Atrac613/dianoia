import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

void main() {
  final task = RunningSumTask(5, seed: 1);

  test('numbers and totals come from the seed', () {
    expect(RunningSumTask(5, seed: 1).numbers, task.numbers);
    expect(RunningSumTask(5, seed: 2).numbers, isNot(task.numbers));
    expect(task.numbers.every((n) => n >= 10 && n <= 99), isTrue);
    expect(task.totals.last, task.numbers.reduce((a, b) => a + b));
    expect(task.prompt, contains(task.numbers.join(', ')));
  });

  test('the correct answer grades fully correct', () {
    final grade = task.grade(task.answer, finished: true);
    expect(grade.lines, 5);
    expect(grade.correct, 5);
    expect(grade.consistent, 5);
    expect(grade.firstError, isNull);
  });

  test('an unfinished last line waits for its newline', () {
    final lines = task.answer.split('\n');
    final partial = '${lines[0]}\n${lines[1].substring(0, 4)}';
    expect(task.grade(partial).lines, 1);
    expect(task.grade('${lines[0]}\n${lines[1]}').lines, 1);
    expect(task.grade('${lines[0]}\n${lines[1]}\n').lines, 2);
  });

  /// The answer with step [step] (from 1) replaced by [line].
  String withLine(int step, String line) {
    final lines = task.answer.split('\n');
    lines[step - 1] = line;
    return lines.join('\n');
  }

  String stepLine(int index, int a, int b, int c) => '$index: $a + $b = $c';

  test('each kind of mistake is named at its step', () {
    final n = task.numbers;
    final t = task.totals;
    final cases = {
      CanaryError.format: 'the total is ${t[2]}',
      CanaryError.step: stepLine(4, t[1], n[2], t[2]),
      CanaryError.operand: stepLine(3, t[1], n[2] + 1, t[1] + n[2] + 1),
      CanaryError.carry: stepLine(3, t[1] + 1, n[2], t[1] + 1 + n[2]),
      CanaryError.arithmetic: stepLine(3, t[1], n[2], t[2] + 10),
    };
    for (final MapEntry(key: kind, value: line) in cases.entries) {
      final grade = task.grade(withLine(3, line), finished: true);
      expect(grade.firstError, kind, reason: line);
      expect(grade.firstErrorStep, 3);
      expect(grade.correct, 2);
    }
  });

  test('a carried mistake breaks correctness but not consistency', () {
    final n = task.numbers;
    final t = task.totals;
    // Step 2 adds wrongly; later steps carry its total correctly.
    final lines = [
      stepLine(1, 0, n[0], t[0]),
      stepLine(2, t[0], n[1], t[1] + 1),
      for (var i = 2; i < 5; i++) stepLine(i + 1, t[i - 1] + 1, n[i], t[i] + 1),
    ];
    final grade = task.grade(lines.join('\n'), finished: true);
    expect(grade.correct, 1);
    expect(grade.firstError, CanaryError.arithmetic);
    expect(grade.consistent, 4);
  });

  test('blank lines are skipped and lines past the task ignored', () {
    final text = '\n${task.answer.replaceAll('\n', '\n\n')}\n6: 1 + 1 = 2\n';
    final grade = task.grade(text);
    expect(grade.lines, 5);
    expect(grade.correct, 5);
  });

  test('an introduction and code fences around the steps are skipped', () {
    final text = 'Here is the output:\n\n```\n${task.answer}\n```\n';
    final grade = task.grade(text);
    expect(grade.lines, 5);
    expect(grade.correct, 5);
    expect(task.grade('Here is the output:\n').lines, 0);
  });

  test('an indexed prompt labels each number and keeps the task', () {
    final indexed = RunningSumTask(5, seed: 1, indexed: true);
    expect(indexed.numbers, task.numbers);
    expect(indexed.answer, task.answer);
    final n = task.numbers;
    expect(indexed.prompt, contains('1) ${n[0]}, 2) ${n[1]}, 3) ${n[2]}'));
    expect(task.prompt, isNot(contains('1) ')));
  });
}
