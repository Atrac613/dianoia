# Image input fixtures

Synthetic pixels only; no model weights. Regenerate with Pillow 13.0.0:

```sh
python tool/image_input_fixtures.py
```

PNG files cover RGB, RGBA, indexed transparency, grayscale, grayscale/alpha and
16-bit grayscale. The 8x8 gradient has RGB `(31*x, 29*y, 17*(x+y))` and provides
an independent exact input for the public inference comparison.

JPEGs contain six 8x8 grayscale blocks in a 24x16 raster, with values
`[[20, 50, 80], [110, 140, 170]]`. Each file has a different EXIF orientation
(1–8). Tests check the independently specified oriented block layout, allowing
one RGB level of JPEG decoding error. They do not obtain expected orientations
from the codec under test.

`animated.png` is a two-frame APNG; `one-frame-apng.png` has explicit animation
control and frame control chunks for one frame. Both are rejected by the still
image API. The GIF is rejected as an unsupported format.

The two `large-header` files reuse tiny image bodies with dimensions changed to
100000x100000 (PNG, valid updated IHDR CRC) and 65535x65535 (JPEG SOF). They test
rejection before raster allocation, not successful image decoding.

`overflow-header.png` declares 4294967295x4294967295 pixels with a valid IHDR
CRC, exercising overflow-safe dimension rejection before decoding.
