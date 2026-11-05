import 'dart:typed_data';
import 'dart:ui' as ui;
import 'dart:math' as math;

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import 'chat_protocol.dart';

/// HEIC and HEIF photos (the iPhone camera default) are decoded here by the
/// platform codec, which also applies their orientation, and handed on as a
/// PNG no larger than [maxConvertedSide]: the worker reads only PNG and JPEG,
/// and the models see at most a few hundred image tokens anyway.
const maxConvertedSide = 2048;

/// The platform codec downsamples while decoding, so converted photos may be
/// larger than the [ImageAttachment] pixel limit (24 and 48 MP are common).
const maxConvertedPixels = 50000000;

bool _isPng(Uint8List data) =>
    data.length >= 8 &&
    data[0] == 137 &&
    data[1] == 80 &&
    data[2] == 78 &&
    data[3] == 71 &&
    data[4] == 13 &&
    data[5] == 10 &&
    data[6] == 26 &&
    data[7] == 10;

bool _isJpeg(Uint8List data) =>
    data.length >= 3 && data[0] == 255 && data[1] == 216 && data[2] == 255;

const _heifBrands = {
  'heic',
  'heix',
  'heim',
  'heis',
  'hevc',
  'hevx',
  'mif1',
  'msf1',
};

/// An ISO base media file whose major brand is one of the HEIF brands.
bool isHeif(Uint8List data) =>
    data.length >= 12 &&
    String.fromCharCodes(data, 4, 8) == 'ftyp' &&
    _heifBrands.contains(String.fromCharCodes(data, 8, 12));

/// Decodes [descriptor] scaled to fit [maxSide] and encodes it as PNG.
Future<Uint8List> _renderPng(ui.ImageDescriptor descriptor, int maxSide) async {
  final width = descriptor.width, height = descriptor.height;
  final scale = math.min(1.0, maxSide / math.max(width, height));
  final codec = await descriptor.instantiateCodec(
    targetWidth: math.max(1, (width * scale).round()),
    targetHeight: math.max(1, (height * scale).round()),
  );
  try {
    final frame = await codec.getNextFrame();
    try {
      final png = (await frame.image.toByteData(
        format: ui.ImageByteFormat.png,
      ))!;
      return Uint8List.fromList(
        png.buffer.asUint8List(png.offsetInBytes, png.lengthInBytes),
      );
    } finally {
      frame.image.dispose();
    }
  } finally {
    codec.dispose();
  }
}

/// Read on the UI isolate while the OS picker grants access, retaining a small
/// thumbnail for history. Model preprocessing runs in the inference worker.
Future<ImageAttachment?> pickChatImage({
  Future<XFile?> Function()? chooseFile,
}) async {
  final file = chooseFile != null
      ? await chooseFile()
      : await openFile(
          acceptedTypeGroups: const [
            XTypeGroup(
              label: 'PNG, JPEG or HEIC',
              extensions: ['png', 'jpg', 'jpeg', 'heic', 'heif'],
              mimeTypes: [
                'image/png',
                'image/jpeg',
                'image/heic',
                'image/heif',
              ],
              uniformTypeIdentifiers: [
                'public.png',
                'public.jpeg',
                'public.heic',
                'public.heif',
              ],
            ),
          ],
        );
  if (file == null) return null;
  if (await file.length() > ImageAttachment.maxBytes) {
    throw const FormatException('Choose an image smaller than 32 MiB.');
  }
  final bytes = BytesBuilder(copy: false);
  await for (final chunk in file.openRead()) {
    if (bytes.length + chunk.length > ImageAttachment.maxBytes) {
      throw const FormatException('Choose an image smaller than 32 MiB.');
    }
    bytes.add(chunk);
  }
  var data = bytes.takeBytes();
  final heif = isHeif(data);
  if (!heif && !_isPng(data) && !_isJpeg(data)) {
    throw const FormatException('Choose a PNG, JPEG or HEIC image.');
  }
  final buffer = await ui.ImmutableBuffer.fromUint8List(data);
  try {
    final ui.ImageDescriptor descriptor;
    try {
      descriptor = await ui.ImageDescriptor.encoded(buffer);
    } on Object {
      if (!heif) rethrow;
      throw const FormatException(
        'This device cannot read HEIC photos. Choose a PNG or JPEG image.',
      );
    }
    try {
      final width = descriptor.width, height = descriptor.height;
      final limit = heif ? maxConvertedPixels : 16000000;
      if (width * height > limit) {
        throw FormatException('Choose an image with at most $limit pixels.');
      }
      if (heif) data = await _renderPng(descriptor, maxConvertedSide);
      return ImageAttachment(
        file.name,
        data,
        thumbnail: ImageThumbnail(file.name, await _renderPng(descriptor, 240)),
      );
    } finally {
      descriptor.dispose();
    }
  } finally {
    buffer.dispose();
  }
}

class AttachmentPreview extends StatelessWidget {
  const AttachmentPreview({super.key, required this.image, this.onRemove});
  final ImageAttachment image;
  final VoidCallback? onRemove;
  @override
  Widget build(BuildContext context) => ThumbnailPreview(
    image: image.thumbnail ?? ImageThumbnail(image.name, image.bytes),
    onRemove: onRemove,
  );
}

class ThumbnailPreview extends StatelessWidget {
  const ThumbnailPreview({super.key, required this.image, this.onRemove});
  final ImageThumbnail image;
  final VoidCallback? onRemove;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Flexible(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (image.bytes != null)
              Image.memory(
                image.bytes!,
                width: 120,
                height: 90,
                // One dimension only: both would decode at exactly 240x180
                // and distort images of any other aspect ratio.
                cacheWidth: 240,
                fit: BoxFit.contain,
                errorBuilder: (_, _, _) => const SizedBox(
                  width: 120,
                  height: 90,
                  child: Icon(Icons.broken_image_outlined),
                ),
              ),
            Text(image.name, maxLines: 1, overflow: TextOverflow.ellipsis),
          ],
        ),
      ),
      if (onRemove != null)
        IconButton(
          tooltip: 'Remove image',
          onPressed: onRemove,
          icon: const Icon(Icons.close),
        ),
    ],
  );
}
