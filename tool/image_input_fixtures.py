#!/usr/bin/env python3
"""Generate tiny image-input fixtures with Pillow, independently of Dart codecs.

Run with Pillow 13.0.0. Files contain synthetic pixels only, not model weights.
"""
from __future__ import annotations

import struct
import zlib
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / 'test/fixtures/image_input'
ROOT.mkdir(parents=True, exist_ok=True)


def save(mode: str, size: tuple[int, int], pixels: Sequence, name: str) -> Image.Image:
    """Writes a Pillow image of mode and size filled with pixels to name.

    Args:
        mode: Pillow image mode, such as 'RGB' or 'I;16'.
        size: Width and height.
        pixels: Row-major pixel values in that mode.
        name: File name under the fixture directory; its extension picks the format.

    Returns:
        The image as saved.
    """
    image = Image.new(mode, size)
    image.putdata(pixels)
    image.save(ROOT / name)
    return image


save('RGB', (3, 2), [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                     (10, 20, 30), (128, 64, 32), (255, 255, 255)], 'rgb.png')
save('RGBA', (3, 2), [(255, 0, 0, 128), (0, 255, 0, 0), (0, 0, 255, 255),
                      (0, 0, 0, 64), (255, 255, 255, 128), (10, 20, 30, 0)], 'rgba.png')
save('L', (3, 2), [0, 17, 64, 128, 200, 255], 'gray.png')
save('I;16', (3, 2), [0, 257, 32768, 65535, 43690, 514], 'gray16.png')
save('LA', (3, 2), [(0, 0), (32, 128), (80, 255), (160, 64), (200, 255), (255, 1)],
     'gray-alpha.png')
palette = Image.new('P', (3, 2))
palette.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (768 - 9))
palette.putdata([0, 1, 2, 2, 1, 0])
palette.save(ROOT / 'indexed.png', transparency=bytes([0, 128, 255]))
save('RGB', (8, 8), [(x * 31, y * 29, (x + y) * 17)
                     for y in range(8) for x in range(8)], 'gradient.png')

# Six asymmetric gray blocks, each an exact JPEG DCT block. Orientations 1..8
# include horizontal/vertical reflections, transpose, and quarter rotations.
values = [[20, 50, 80], [110, 140, 170]]
photo = Image.new('RGB', (24, 16))
photo.putdata([(values[y // 8][x // 8],) * 3 for y in range(16) for x in range(24)])
for orientation in range(1, 9):
    exif = Image.Exif()
    exif[274] = orientation
    photo.save(ROOT / f'orientation-{orientation}.jpg', quality=100, subsampling=0, exif=exif)

first = Image.new('RGB', (8, 8), (255, 0, 0))
first.save(ROOT / 'animated.png', save_all=True,
           append_images=[Image.new('RGB', (8, 8), (0, 0, 255))], duration=100, loop=0)


# Explicit one-frame APNG: Pillow optimizes a one-frame save to plain PNG.
def chunk(kind: bytes, data: bytes) -> bytes:
    """A PNG chunk of kind holding data, with its length and CRC."""
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))


first.save(ROOT / 'one-frame-apng.png')
png = (ROOT / 'one-frame-apng.png').read_bytes()
control = chunk(b'acTL', struct.pack('>II', 1, 0))
frame = chunk(b'fcTL', struct.pack('>IIIIIHHBB', 0, 8, 8, 0, 0, 1, 10, 0, 0))
(ROOT / 'one-frame-apng.png').write_bytes(png[:33] + control + frame + png[33:])
first.save(ROOT / 'unsupported.gif')

# Valid image headers declaring excessive dimensions, backed by tiny bodies.
# These must fail on dimensions, before raster allocation or decompression.
png = bytearray((ROOT / 'rgb.png').read_bytes())
png[16:24] = struct.pack('>II', 100000, 100000)
png[29:33] = struct.pack('>I', zlib.crc32(png[12:29]))
(ROOT / 'large-header.png').write_bytes(png)
jpeg = bytearray((ROOT / 'orientation-1.jpg').read_bytes())
sof = jpeg.index(b'\xff\xc0')
jpeg[sof + 5:sof + 9] = struct.pack('>HH', 65535, 65535)
(ROOT / 'large-header.jpg').write_bytes(jpeg)

# Also exercise dimension multiplication overflow in hostile PNG headers.
png[16:24] = struct.pack('>II', 0xffffffff, 0xffffffff)
png[29:33] = struct.pack('>I', zlib.crc32(png[12:29]))
(ROOT / 'overflow-header.png').write_bytes(png)
