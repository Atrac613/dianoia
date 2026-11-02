"""Unit checks for the validation-only inverse mmproj conversion."""
from __future__ import annotations
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from prepare_vl_qad import canonical, read_gguf


def string(value: str) -> bytes:
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


class PrepareQadTests(unittest.TestCase):
    def test_canonical_names(self) -> None:
        self.assertEqual(canonical('mm.input_norm.bias'), 'model.multi_modal_projector.layer_norm.bias')
        self.assertEqual(canonical('v.blk.2.attn_q_norm.weight'), 'model.vision_tower.layers.2.self_attn.q_layernorm.weight')
        self.assertEqual(canonical('v.blk.3.conv2d_spatial.weight'), 'model.vision_tower.layers.3.conv.spatial_conv.weight')
        with self.assertRaises(ValueError):
            canonical('unexpected.weight')

    def test_raw_norm_and_temporal_patch_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root/'config'
            config.mkdir()
            for name in ['config.json', 'processor_config.json', 'tokenizer.json', 'LICENSE']:
                (config/name).write_text('{}')
            norm = struct.pack('<1024f', *([1.25]*1024))
            count = 1024*3*16*16
            a, b = struct.pack('<e', 1.5)*count, struct.pack('<e', -2.5)*count
            bf16 = struct.pack('<4H', 0x3f80, 0xbf80, 0x4000, 0xc000)
            tensors = [('v.embedding_norm.weight', [1024], 0, norm),
                       ('mm.0.weight', [2,2], 30, bf16),
                       ('v.patch_embd.weight', [16,16,3,1024], 1, a),
                       ('v.patch_embd.weight.1', [16,16,3,1024], 1, b)]
            header = b'GGUF'+struct.pack('<IQQ',3,len(tensors),1)
            header += string('clip.vision.projector_type')+struct.pack('<I',8)+string('lfm3vl')
            offset = 0
            payload = bytearray()
            for name, dims, typ, data in tensors:
                header += string(name)+struct.pack('<I',len(dims))
                header += struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',typ,offset)
                payload.extend(data)
                payload.extend(b' '*(-len(payload)%32))
                offset = len(payload)
            header += b' '*(-len(header)%32)
            gguf = root/'mmproj.gguf'
            gguf.write_bytes(header+payload)
            self.assertEqual(len(read_gguf(gguf)['tensors']),4)
            subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_vl_qad.py')),
                f'--mmproj={gguf}',f'--config={config}',f'--out={root}/out'],check=True,capture_output=True)
            with (root/'out/model.safetensors').open('rb') as f:
                n = struct.unpack('<Q',f.read(8))[0]
                info = json.loads(f.read(n))
                start = f.tell()
                entry = info['model.vision_tower.embedding_norm.weight']
                f.seek(start+entry['data_offsets'][0])
                self.assertEqual(f.read(len(norm)),norm)
                matrix = info['model.multi_modal_projector.linear_1.weight']
                self.assertEqual(matrix['dtype'], 'BF16')
                f.seek(start+matrix['data_offsets'][0])
                self.assertEqual(f.read(len(bf16)),bf16)
                patch = info['model.vision_tower.patchifier.weight']
                self.assertEqual(patch['shape'],[1024,1536])
                for row in [0,1,1023]:
                    for channel in range(3):
                        for frame, expected in enumerate([1.5,-2.5]):
                            index = ((row*3+channel)*2+frame)*256
                            f.seek(start+patch['data_offsets'][0]+index*4)
                            self.assertEqual(struct.unpack('<256f',f.read(1024)),(expected,)*256)

    def test_rejects_invalid_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'invalid.gguf'
            path.write_bytes(b'bad!')
            with self.assertRaises(ValueError):
                read_gguf(path)


if __name__ == '__main__':
    unittest.main()
