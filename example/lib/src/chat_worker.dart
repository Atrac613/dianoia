/// The chat worker: an isolate that owns the model, so that the forward
/// pass, the download's hashing and the bundle build never block the UI.
library;

import 'dart:async';
import 'dart:io';
import 'dart:isolate';

import 'package:dianoia/dianoia.dart';

import 'chat_protocol.dart';

const _system =
    'You are a helpful assistant running entirely on this device. '
    'Answer concisely, in the language the user writes in.';

/// `--dart-define=DIANOIA_MODEL_MIRROR=URL` fetches the model from a mirror
/// (example/tool/serve_model.dart, say) instead of Hugging Face.
const _mirror = String.fromEnvironment('DIANOIA_MODEL_MIRROR');
final _spec = _mirror.isEmpty
    ? ModelSpec.lfm3Q4
    : ModelSpec.lfm3Q4.fromMirror(_mirror);

/// The worker isolate's entry point.
Future<void> runChatWorker((SendPort, ModelLocation) args) async {
  final (out, location) = args;
  final store = ModelStore(location.storeDir);
  final commands = ReceivePort();
  out.send(WorkerStarted(commands.sendPort));

  ChatSession? session;
  StreamSubscription<InstallProgress>? install;
  var stopRequested = false;

  Future<void> open(String dir) async {
    out.send(const WorkerProgress('Loading the model…'));
    try {
      final model = await LocalModel.load(dir);
      final chat = model.chat(system: _system);
      await chat.warmUp();
      session = chat;
      out.send(WorkerReady(model.backend.name.toUpperCase()));
    } catch (e) {
      out.send(WorkerFailed('$e'));
    }
  }

  commands.listen((command) {
    switch (command as ChatCommand) {
      case SendMessage(:final text):
        stopRequested = false;
        unawaited(_reply(session!, text, out, () => stopRequested));
      case StopReply():
        stopRequested = true;
      case NewChat():
        session?.clear();
      case InstallModel():
        install ??= store
            .install(_spec)
            .listen(
              (p) => out.send(InstallUpdate(p.stage, p.bytes, p.totalBytes)),
              onError: (Object e) {
                install = null;
                out.send(InstallFailed('$e'));
              },
              onDone: () {
                if (install == null) return; // failed
                install = null;
                unawaited(open(store.bundlePath(_spec)));
              },
            );
      case CancelInstall():
        unawaited(install?.cancel());
        install = null;
        out.send(ModelNeeded(_spec));
    }
  });

  final sideLoaded = _sideLoaded(location, out);
  if (sideLoaded != null) {
    await open(sideLoaded);
  } else if (store.isInstalled(_spec)) {
    await open(store.bundlePath(_spec));
  } else {
    out.send(ModelNeeded(_spec));
  }
}

/// A bundle a developer copied into [ModelLocation.dropDir], ready to map,
/// or null.
String? _sideLoaded(ModelLocation location, SendPort out) {
  final cache = 'lfm3-q4.cache';
  if (!File('${location.dropDir}/$cache').existsSync()) return null;
  if (location.copyDir != location.dropDir) {
    out.send(const WorkerProgress('Copying the model to app storage…'));
    _copyBundle(location.dropDir, location.copyDir);
  }
  return location.copyDir;
}

Future<void> _reply(
  ChatSession session,
  String text,
  SendPort out,
  bool Function() stopRequested,
) async {
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
  } on ContextLengthExceeded {
    out.send(
      const ReplyFailed(
        'The conversation no longer fits the model\'s context. '
        'Start a new chat.',
      ),
    );
  } catch (e) {
    out.send(ReplyFailed('$e'));
  }
}

/// Copies the files of [from] to [to] unless copies of the same sizes are
/// already there. Mapping the copy in app storage, rather than the pushed
/// files in external storage, follows the device benchmark.
void _copyBundle(String from, String to) {
  Directory(to).createSync(recursive: true);
  for (final file in Directory(from).listSync().whereType<File>()) {
    final copy = File('$to/${file.uri.pathSegments.last}');
    if (copy.existsSync() && copy.lengthSync() == file.lengthSync()) continue;
    file.copySync(copy.path);
  }
}
