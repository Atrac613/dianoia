// Serves a model's published files from local directories, honouring the
// open-ended Range requests (bytes=N-) that a resumed download sends, so the
// app can install from this machine instead of Hugging Face
// (DIANOIA_MODEL_MIRROR). For development only: plain HTTP.
//
//   dart run tool/serve_model.dart --port=8642 [--mbps=50] DIR...
//
// --mbps throttles each response to that many megabytes a second, slow
// enough to cancel and resume a download by hand.
// A request for /NAME gets the first file called NAME in the DIRs. On an
// Android phone, `adb reverse tcp:8642 tcp:8642` makes it
// http://127.0.0.1:8642 there.
import 'dart:io';

Future<void> main(List<String> args) async {
  String? flag(String name) => args
      .firstWhere((a) => a.startsWith('--$name='), orElse: () => '')
      .split('=')
      .elementAtOrNull(1);
  final port = int.parse(flag('port') ?? '8642');
  final mbps = double.tryParse(flag('mbps') ?? '');
  final dirs = args.where((a) => !a.startsWith('--')).toList();
  if (dirs.isEmpty) {
    stderr.writeln('usage: serve_model.dart [--port=8642] [--mbps=N] DIR...');
    exitCode = 2;
    return;
  }
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, port);
  stdout.writeln('serving ${dirs.join(', ')} at http://127.0.0.1:$port');
  // Requests are served concurrently: a resumed download must not wait
  // for the cancelled response to drain.
  server.listen((request) => _serve(request, dirs, mbps));
}

Future<void> _serve(
  HttpRequest request,
  List<String> dirs,
  double? mbps,
) async {
  final response = request.response;
  final name = request.uri.pathSegments.lastOrNull ?? '';
  final file = dirs
      .map((d) => File('$d/$name'))
      .where((f) => name.isNotEmpty && f.existsSync())
      .firstOrNull;
  if (file == null) {
    response.statusCode = HttpStatus.notFound;
    await response.close();
    return;
  }
  final length = file.lengthSync();
  final range = request.headers.value(HttpHeaders.rangeHeader);
  final match = range == null
      ? null
      : RegExp(r'bytes=(\d+)-').firstMatch(range);
  final from = match == null ? 0 : int.parse(match[1]!);
  if (match != null) {
    response
      ..statusCode = HttpStatus.partialContent
      ..headers.set(
        HttpHeaders.contentRangeHeader,
        'bytes $from-${length - 1}/$length',
      );
  }
  stdout.writeln('${request.method} /$name from $from');
  response.contentLength = length - from;
  try {
    if (mbps == null) {
      await response.addStream(file.openRead(from));
    } else {
      await for (final chunk in file.openRead(from)) {
        response.add(chunk);
        await response.flush();
        await Future<void>.delayed(
          Duration(microseconds: chunk.length ~/ mbps),
        );
      }
    }
    await response.close();
  } on IOException {
    stdout.writeln('/$name: the client went away');
  }
}
