import 'package:dianoia/dianoia.dart';
import 'package:test/test.dart';

void main() {
  const guard = RepetitionGuard(maxPeriod: 8, minRepeats: 3, minTokens: 6);

  List<int> repeat(List<int> block, int times) => [
    for (var i = 0; i < times; i++) ...block,
  ];

  test('a block repeated enough times in a row is a loop', () {
    expect(
      guard.loops([
        9,
        9,
        ...repeat([1, 2, 3], 3),
      ]),
      isTrue,
    );
    expect(guard.loops(repeat([5], 6)), isTrue);
    expect(guard.loops(repeat([1, 2, 3, 4, 5, 6, 7, 8], 3)), isTrue);
  });

  test('too few repeats or too few tokens are not a loop', () {
    expect(guard.loops(repeat([1, 2, 3], 2)), isFalse);
    // Three repeats, but only five tokens.
    expect(guard.loops(repeat([5], 5)), isFalse);
    expect(guard.loops(const []), isFalse);
  });

  test('only the end of the tokens counts', () {
    expect(
      guard.loops([
        ...repeat([1, 2, 3], 3),
        4,
      ]),
      isFalse,
    );
  });

  test('blocks longer than maxPeriod are not looked for', () {
    final block = [for (var i = 0; i < 9; i++) i];
    expect(guard.loops(repeat(block, 3)), isFalse);
    expect(const RepetitionGuard(maxPeriod: 9).loops(repeat(block, 4)), isTrue);
  });

  test('the default needs at least 32 repeated tokens', () {
    const standard = RepetitionGuard();
    expect(standard.loops(repeat([7], 31)), isFalse);
    expect(standard.loops(repeat([7], 32)), isTrue);
    expect(standard.loops(repeat([1, 2, 3, 4], 8)), isTrue);
    expect(standard.loops(repeat([1, 2, 3, 4], 7)), isFalse);
  });
}
