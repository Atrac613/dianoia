/// Downloading one file that must arrive exactly as published: resumable,
/// with its size and SHA-256 checked before it takes its final name.
library;

import 'dart:async';
import 'dart:io';

import 'package:crypto/crypto.dart';

/// A file to download: where it is published, and the size and SHA-256 it
/// must have.
final class RemoteFile {
  const RemoteFile(
    this.name,
    this.url, {
    required this.size,
    required this.sha256,
  });

  /// The name it gets on disk.
  final String name;
  final String url;
  final int size;

  /// Lowercase hex.
  final String sha256;
}

/// A download that failed for good: the server answered with an error, or
/// the file arrived different from what was published.
final class ModelDownloadException implements Exception {
  const ModelDownloadException(this.message);

  final String message;

  @override
  String toString() => 'ModelDownloadException: $message';
}

/// Downloads [file] to [path] unless it is already there. Bytes arrive in
/// `[path].part`, which a later call resumes with a Range request; only a
/// file of the right size and SHA-256 is renamed to [path]. [onBytes] sees
/// how many bytes of the file are on disk; [cancelled] is polled between
/// chunks and stops the download, keeping the part.
///
/// Throws [ModelDownloadException] when the server refuses the file or it
/// does not match (the part is deleted then), and lets I/O errors through
/// for the caller to retry.
Future<void> downloadFile(
  HttpClient client,
  RemoteFile file,
  String path, {
  void Function(int bytes)? onBytes,
  bool Function()? cancelled,
}) async {
  if (File(path).existsSync()) {
    onBytes?.call(file.size);
    return;
  }
  final part = File('$path.part');
  var have = part.existsSync() ? part.lengthSync() : 0;
  if (have > file.size) {
    part.deleteSync();
    have = 0;
  }
  final response = have == file.size
      ? null
      : await _get(client, Uri.parse(file.url), from: have);
  final digest = _Digest();
  final hash = sha256.startChunkedConversion(digest);
  switch (response?.statusCode) {
    case null || HttpStatus.partialContent:
      // Resuming: the hash first covers what is already on disk.
      if (response != null && !_startsAt(response, have)) {
        await response.drain<void>();
        throw ModelDownloadException('${file.url}: unexpected content range');
      }
      await part.openRead().forEach(hash.add);
    case HttpStatus.ok:
      have = 0; // the server sent the whole file
    case final status:
      await response!.drain<void>();
      throw ModelDownloadException('${file.url}: HTTP $status');
  }

  var got = have;
  if (response != null) {
    final sink = part.openWrite(
      mode: have > 0 ? FileMode.writeOnlyAppend : FileMode.writeOnly,
    );
    var reported = got;
    try {
      await for (final chunk in response) {
        sink.add(chunk);
        hash.add(chunk);
        got += chunk.length;
        if (got - reported >= 1 << 20) {
          reported = got;
          onBytes?.call(got);
        }
        if (cancelled?.call() ?? false) break;
      }
    } finally {
      await sink.close();
    }
    if (cancelled?.call() ?? false) return;
  }
  hash.close();
  if (got != file.size || digest.value.toString() != file.sha256) {
    part.deleteSync();
    throw ModelDownloadException(
      '${file.name} arrived as $got bytes with SHA-256 ${digest.value}, '
      'expected ${file.size} bytes with ${file.sha256}',
    );
  }
  part.renameSync(path);
  onBytes?.call(got);
}

/// GET [url] from byte [from] on, following redirects by hand: the Range
/// header must reach the server that finally answers (Hugging Face sends
/// files from a CDN).
Future<HttpClientResponse> _get(
  HttpClient client,
  Uri url, {
  required int from,
}) async {
  for (var hops = 0; hops < 8; hops++) {
    final request = await client.getUrl(url);
    request.followRedirects = false;
    if (from > 0) request.headers.set(HttpHeaders.rangeHeader, 'bytes=$from-');
    final response = await request.close();
    if (!response.isRedirect) return response;
    final location = response.headers.value(HttpHeaders.locationHeader);
    await response.drain<void>();
    if (location == null) {
      throw ModelDownloadException('$url: redirect without a location');
    }
    url = url.resolve(location);
  }
  throw ModelDownloadException('$url: too many redirects');
}

bool _startsAt(HttpClientResponse response, int from) =>
    response.headers
        .value(HttpHeaders.contentRangeHeader)
        ?.startsWith('bytes $from-') ??
    false;

final class _Digest implements Sink<Digest> {
  late Digest value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}
