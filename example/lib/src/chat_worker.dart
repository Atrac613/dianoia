/// The chat worker: an isolate that owns the model, so that the forward
/// pass never blocks the UI.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';

import 'chat_protocol.dart';

const _system =
    'You are a helpful assistant running entirely on this device. '
    'Answer concisely, in the language the user writes in.';

/// GPU attention holds one score per key in threadgroup memory, so the
/// context stays well inside what phone GPUs offer.
const _maxSeq = 2048;

/// The worker isolate's entry point.
Future<void> runChatWorker((SendPort, ModelLocation) args) async {
  final (out, location) = args;
  final ChatSession session;
  final String backend;
  try {
    if (location.modelDir != location.dropDir) {
      out.send(const WorkerProgress('Copying the model to app storage…'));
      _copyBundle(location.dropDir, location.modelDir);
    }
    final dir = location.modelDir;
    if (!File('$dir/lfm3-q4.cache').existsSync()) {
      out.send(
        WorkerFailed(
          'No model found. Copy the ${ModelLocation.bundleName} bundle '
          '(from `dianoia import-gguf`) to ${location.dropDir} and restart '
          'the app.',
        ),
      );
      return;
    }
    out.send(const WorkerProgress('Loading the model…'));
    NativeKernels.load(null);
    (session, backend) = await _open(dir);
    await session.warmUp();
  } catch (e) {
    out.send(WorkerFailed('$e'));
    return;
  }

  var stopRequested = false;
  final commands = ReceivePort();
  commands.listen((command) {
    switch (command as ChatCommand) {
      case SendMessage(:final text):
        stopRequested = false;
        unawaited(_reply(session, text, out, () => stopRequested));
      case StopReply():
        stopRequested = true;
      case NewChat():
        session.clear();
    }
  });
  out.send(WorkerReady(commands.sendPort, backend));
}

/// The model on the GPU, or on the CPU kernels where there is no usable GPU.
Future<(ChatSession, String)> _open(String dir) async {
  Future<Lfm3Model> load({required bool gpu}) => Lfm3Model.load(
    dir,
    maxSeq: _maxSeq,
    weights: WeightFormat.q4,
    q8CachePath: '$dir/lfm3-q4.cache',
    gpu: gpu,
  );
  var backend = 'GPU';
  Lfm3Model model;
  try {
    model = await load(gpu: true);
  } on StateError {
    backend = 'CPU';
    model = await load(gpu: false);
  }
  final session = ChatSession(
    model,
    Lfm3Tokenizer.load(dir),
    system: _system,
    maxNew: 512,
  );
  return (session, backend);
}

Future<void> _reply(
  ChatSession session,
  String text,
  SendPort out,
  bool Function() stopRequested,
) async {
  final before = session.messages.length;
  try {
    final turn = await session.send(
      {'role': 'user', 'content': text},
      onText: (partial) async {
        out.send(ReplyText(partial));
        // A timer, not a microtask: it lets the stop command arrive.
        await Future<void>.delayed(Duration.zero);
        return !stopRequested();
      },
    );
    out.send(
      ReplyDone(
        text: turn.text,
        tokens: turn.generatedTokens,
        firstTokenMs: (turn.prefillSeconds * 1000).round(),
        tokensPerSecond: turn.decodeSeconds > 0
            ? turn.generatedTokens / turn.decodeSeconds
            : 0,
      ),
    );
  } catch (e) {
    // Leave the conversation as it was before this message.
    session.messages.removeRange(before, session.messages.length);
    out.send(
      ReplyFailed(
        e is StateError && '$e'.contains('maxSeq')
            ? 'The conversation is longer than the model\'s context '
                  '($_maxSeq tokens). Start a new chat.'
            : '$e',
      ),
    );
  }
}

/// Copies the files of [from] to [to] unless copies of the same sizes are
/// already there. Mapping the copy in app storage, rather than the pushed
/// files in external storage, follows the device benchmark.
void _copyBundle(String from, String to) {
  final source = Directory(from);
  if (!source.existsSync()) return;
  Directory(to).createSync(recursive: true);
  for (final file in source.listSync().whereType<File>()) {
    final copy = File('$to/${file.uri.pathSegments.last}');
    if (copy.existsSync() && copy.lengthSync() == file.lengthSync()) continue;
    file.copySync(copy.path);
  }
}
