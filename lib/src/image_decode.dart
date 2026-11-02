/// PNG/JPEG input adapters with explicit byte and pixel budgets.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as image;

const defaultMaxEncodedImageBytes = 32 * 1024 * 1024;
const defaultMaxDecodedImagePixels = 16000000;

typedef DecodedRgbImage = ({Uint8List pixels, int width, int height});

void _checkLimits(int maxEncodedBytes, int maxPixels) {
  if (maxEncodedBytes <= 0 || maxPixels <= 0) {
    throw ArgumentError('image byte and pixel limits must be positive');
  }
}

void _checkSize(int bytes, int limit) {
  if (bytes <= 0 || bytes > limit) {
    throw FormatException('encoded image must contain 1..$limit bytes');
  }
}

/// Decodes one still PNG or JPEG to RGB8 without resizing. JPEG EXIF orientation
/// is applied by the decoder. Alpha is composited over white in channel space.
/// PNG samples use stored orientation; ICC color conversion is not performed.
/// Pixel limits are supplied to both decoders before raster allocation.
DecodedRgbImage decodeImageRgb(
  Uint8List encoded, {
  int maxEncodedBytes = defaultMaxEncodedImageBytes,
  int maxPixels = defaultMaxDecodedImagePixels,
}) {
  _checkLimits(maxEncodedBytes, maxPixels);
  _checkSize(encoded.length, maxEncodedBytes);
  final image.Decoder decoder;
  if (encoded.length >= 8 && image.PngDecoder().isValidFile(encoded)) {
    decoder = image.PngDecoder(maxPixels: maxPixels);
  } else if (encoded.length >= 2 && encoded[0] == 0xff && encoded[1] == 0xd8) {
    decoder = image.JpegDecoder(maxPixels: maxPixels);
  } else {
    throw const FormatException('expected a still PNG or JPEG image');
  }
  try {
    final info = decoder.startDecode(encoded);
    // Division keeps malicious PNG dimensions from overflowing an int product.
    if (info == null ||
        info.width <= 0 ||
        info.height <= 0 ||
        info.width > maxPixels ~/ info.height) {
      throw FormatException('invalid image or more than $maxPixels pixels');
    }
    if (info.numFrames != 1 || (info is image.PngInfo && info.isAnimated)) {
      throw const FormatException('animated images are not supported');
    }
    final decoded = decoder.decodeFrame(0);
    if (decoded == null ||
        decoded.width <= 0 ||
        decoded.height <= 0 ||
        decoded.width > maxPixels ~/ decoded.height) {
      throw const FormatException('invalid decoded image dimensions');
    }
    final rgb = Uint8List(decoded.width * decoded.height * 3);
    var offset = 0;
    final grayscale = !decoded.hasPalette && decoded.numChannels < 3;
    for (final pixel in decoded) {
      final alpha = pixel.aNormalized;
      int channel(num value) =>
          ((value * alpha + 1 - alpha) * 255).round().clamp(0, 255);
      rgb[offset++] = channel(pixel.rNormalized);
      rgb[offset++] = channel(
        grayscale ? pixel.rNormalized : pixel.gNormalized,
      );
      rgb[offset++] = channel(
        grayscale ? pixel.rNormalized : pixel.bNormalized,
      );
    }
    return (pixels: rgb, width: decoded.width, height: decoded.height);
  } on image.ImageException catch (error) {
    throw FormatException('invalid PNG/JPEG image: $error');
  }
}

/// Reads a local file with a bounded stream, then performs synchronous decoding.
/// No URL fetching or filename-extension detection. A file that grows between
/// the length check and the read cannot bypass the encoded-byte limit.
Future<DecodedRgbImage> readImageRgb(
  String path, {
  int maxEncodedBytes = defaultMaxEncodedImageBytes,
  int maxPixels = defaultMaxDecodedImagePixels,
}) async {
  _checkLimits(maxEncodedBytes, maxPixels);
  final file = File(path);
  _checkSize(await file.length(), maxEncodedBytes);
  final bytes = BytesBuilder(copy: false);
  await for (final chunk in file.openRead(0, maxEncodedBytes + 1)) {
    if (bytes.length + chunk.length > maxEncodedBytes) {
      throw FormatException('encoded image exceeds $maxEncodedBytes bytes');
    }
    bytes.add(chunk);
  }
  return decodeImageRgb(
    bytes.takeBytes(),
    maxEncodedBytes: maxEncodedBytes,
    maxPixels: maxPixels,
  );
}
