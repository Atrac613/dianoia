import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';

import 'chat_protocol.dart';

/// Read on the UI isolate while the OS picker grants access. Decode and vision
/// preprocessing happen in the inference worker, never through a picker path.
Future<ImageAttachment?> pickChatImage({
  Future<XFile?> Function()? chooseFile,
}) async {
  final file = chooseFile != null
      ? await chooseFile()
      : await openFile(
          acceptedTypeGroups: const [
            XTypeGroup(
              label: 'PNG or JPEG',
              extensions: ['png', 'jpg', 'jpeg'],
              mimeTypes: ['image/png', 'image/jpeg'],
              uniformTypeIdentifiers: ['public.png', 'public.jpeg'],
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
  final data = bytes.takeBytes();
  final png =
      data.length >= 8 &&
      data[0] == 137 &&
      data[1] == 80 &&
      data[2] == 78 &&
      data[3] == 71 &&
      data[4] == 13 &&
      data[5] == 10 &&
      data[6] == 26 &&
      data[7] == 10;
  final jpeg =
      data.length >= 3 && data[0] == 255 && data[1] == 216 && data[2] == 255;
  if (!png && !jpeg) throw const FormatException('Choose a PNG or JPEG image.');
  return ImageAttachment(file.name, data);
}

class AttachmentPreview extends StatelessWidget {
  const AttachmentPreview({super.key, required this.image, this.onRemove});
  final ImageAttachment image;
  final VoidCallback? onRemove;

  @override
  Widget build(BuildContext context) => Row(
    mainAxisSize: MainAxisSize.min,
    children: [
      Flexible(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Image.memory(
              image.bytes,
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
