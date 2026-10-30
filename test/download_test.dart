// downloadFile against a local server that redirects, honours or ignores
// Range requests, and can cut a response short.
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:dianoia/lab.dart';
import 'package:test/test.dart';

/// Serves [content] at /file behind a redirect from /start. With
/// [ranges] false it ignores Range headers; [cutAfter] ends the first
/// response after that many bytes.
final class _Server {
  _Server(this.content, {this.ranges = true, this.cutAfter});

  final Uint8List content;
  final bool ranges;
  int? cutAfter;
  final rangeHeaders = <String?>[];
  late final HttpServer _server;

  String get url => 'http://127.0.0.1:${_server.port}/start';

  Future<void> start() async {
    _server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _server.listen((request) async {
      final response = request.response;
      if (request.uri.path == '/start') {
        response
          ..statusCode = HttpStatus.temporaryRedirect
          ..headers.set(HttpHeaders.locationHeader, '/file');
        await response.close();
        return;
      }
      final range = request.headers.value(HttpHeaders.rangeHeader);
      rangeHeaders.add(range);
      var from = 0;
      if (ranges && range != null) {
        from = int.parse(RegExp(r'bytes=(\d+)-').firstMatch(range)![1]!);
        response
          ..statusCode = HttpStatus.partialContent
          ..headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $from-${content.length - 1}/${content.length}',
          );
      }
      final body = Uint8List.sublistView(content, from);
      final cut = cutAfter;
      if (cut != null) {
        cutAfter = null;
        response.contentLength = body.length;
        final socket = await response.detachSocket();
        socket.add(Uint8List.sublistView(body, 0, cut));
        await socket.flush();
        socket.destroy();
        return;
      }
      response.contentLength = body.length;
      response.add(body);
      await response.close();
    });
  }

  Future<void> stop() => _server.close(force: true);
}

void main() {
  final content = Uint8List.fromList(
    List.generate(3 << 20, (i) => Random(i ~/ 4096).nextInt(256)),
  );
  late Directory dir;
  late HttpClient client;

  setUp(() {
    dir = Directory.systemTemp.createTempSync('dianoia_download');
    client = HttpClient();
  });
  tearDown(() {
    client.close(force: true);
    dir.deleteSync(recursive: true);
  });

  RemoteFile remote(_Server server, {String? hash}) => RemoteFile(
    'file.bin',
    server.url,
    size: content.length,
    sha256: hash ?? sha256.convert(content).toString(),
  );

  test('follows the redirect and keeps a file that matches', () async {
    final server = _Server(content);
    await server.start();
    addTearDown(server.stop);
    final path = '${dir.path}/file.bin';
    await downloadFile(client, remote(server), path);
    expect(File(path).readAsBytesSync(), content);
    expect(File('$path.part').existsSync(), isFalse);
  });

  test('resumes a cut download with a Range request', () async {
    final server = _Server(content, cutAfter: 1 << 20);
    await server.start();
    addTearDown(server.stop);
    final path = '${dir.path}/file.bin';
    await expectLater(
      downloadFile(client, remote(server), path),
      throwsA(isA<IOException>()),
    );
    final kept = File('$path.part').lengthSync();
    expect(kept, greaterThan(0));
    await downloadFile(client, remote(server), path);
    expect(server.rangeHeaders.last, 'bytes=$kept-');
    expect(File(path).readAsBytesSync(), content);
  });

  test('starts over when the server ignores the range', () async {
    final server = _Server(content, ranges: false);
    await server.start();
    addTearDown(server.stop);
    final path = '${dir.path}/file.bin';
    File('$path.part').writeAsBytesSync(Uint8List.sublistView(content, 0, 99));
    await downloadFile(client, remote(server), path);
    expect(File(path).readAsBytesSync(), content);
  });

  test('refuses a file whose hash differs, and drops it', () async {
    final server = _Server(content);
    await server.start();
    addTearDown(server.stop);
    final path = '${dir.path}/file.bin';
    await expectLater(
      downloadFile(client, remote(server, hash: '0' * 64), path),
      throwsA(isA<ModelDownloadException>()),
    );
    expect(File(path).existsSync(), isFalse);
    expect(File('$path.part').existsSync(), isFalse);
  });

  test('stops when cancelled, and a later call finishes the part', () async {
    final server = _Server(content);
    await server.start();
    addTearDown(server.stop);
    final path = '${dir.path}/file.bin';
    await downloadFile(client, remote(server), path, cancelled: () => true);
    expect(File(path).existsSync(), isFalse);
    expect(File('$path.part').existsSync(), isTrue);
    await downloadFile(client, remote(server), path);
    expect(File(path).readAsBytesSync(), content);
  });
}
