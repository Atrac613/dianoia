/// Which models a conversation is using between its forward passes (a
/// ChatSession reply or warm-up), so that a control-vector change, which
/// resets the model, refuses to pull the sequence from under it. Internal:
/// neither `dianoia.dart` nor `lab.dart` exports it.
library;

final _uses = Expando<int>('model uses');

/// Whether a conversation is using [model].
bool modelInUse(Object model) => (_uses[model] ?? 0) > 0;

/// Marks [model] in use until the matching [endModelUse].
void beginModelUse(Object model) => _uses[model] = (_uses[model] ?? 0) + 1;

/// Ends a [beginModelUse].
void endModelUse(Object model) {
  final n = _uses[model] ?? 0;
  if (n == 0) throw StateError('endModelUse without beginModelUse');
  _uses[model] = n == 1 ? null : n - 1;
}
