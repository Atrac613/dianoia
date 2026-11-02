"""Check that the real-model evidence comparison rejects false matches."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vision_compare import compare


class VisionCompareTest(unittest.TestCase):
    """compare() on two identical synthetic dumps, one then altered."""

    def setUp(self) -> None:
        """Writes matching reference and actual dumps to a temporary directory."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.reference = Path(self.temp.name)/'reference'
        self.actual = Path(self.temp.name)/'actual'
        for path in [self.reference, self.actual]:
            path.mkdir()
            (path/'meta.json').write_text(json.dumps(dict(complete=True, prompt='Color?',
                thinking=False, grid_thw=[1, 2, 2], prompt_tokens=3,
                stages=['embedding', 'projected'], generated_ids=[4])))
            np.array([1, 2, 3], dtype='<i4').tofile(path/'ids.i32')
            np.array([4], dtype='<i4').tofile(path/'gen.i32')
            for name in ['pixels.f32', 'vision_embedding.f32', 'vision_projected.f32',
                         'logits.f32', 'gen_logits.f32']:
                np.array([0.0, 1.0, -1.0, 0.5], dtype='<f4').tofile(path/name)

    def check(self, **kwargs: float) -> dict:
        """compare() with the default tolerances and any logit overrides."""
        return compare(self.reference, self.actual, 0.002, 0.0002, **kwargs)

    def test_equal_complete_evidence_passes(self) -> None:
        self.assertTrue(self.check()['checks']['passed'])

    def test_small_preprocessing_difference_is_always_rejected(self) -> None:
        data = np.fromfile(self.actual/'pixels.f32', dtype='<f4')
        data[0] = 1e-6
        data.tofile(self.actual/'pixels.f32')
        self.assertFalse(self.check()['checks']['passed'])

    def test_nonfinite_and_truncated_features_are_rejected(self) -> None:
        np.array([float('nan')]*4, dtype='<f4').tofile(self.actual/'vision_projected.f32')
        self.assertFalse(self.check()['checks']['passed'])
        json.dumps(self.check(), allow_nan=False)
        np.array([0.0], dtype='<f4').tofile(self.actual/'vision_projected.f32')
        self.assertFalse(self.check()['checks']['passed'])

    def test_changed_tokens_and_incomplete_run_are_rejected(self) -> None:
        np.array([5], dtype='<i4').tofile(self.actual/'gen.i32')
        self.assertFalse(self.check()['checks']['passed'])
        meta = json.loads((self.actual/'meta.json').read_text())
        meta['complete'] = False
        (self.actual/'meta.json').write_text(json.dumps(meta))
        self.assertFalse(self.check()['checks']['passed'])

    def test_logit_tolerance_does_not_relax_encoder_tolerance(self) -> None:
        (np.array([0.0, 1.0, -1.0, 0.5], dtype='<f4')+0.1).tofile(self.actual/'logits.f32')
        self.assertFalse(self.check()['checks']['passed'])
        self.assertTrue(self.check(logits_atol=0.2)['checks']['passed'])
        (np.array([0.0, 1.0, -1.0, 0.5], dtype='<f4')+0.1).tofile(self.actual/'vision_projected.f32')
        self.assertFalse(self.check(logits_atol=0.2)['checks']['passed'])


if __name__ == '__main__':
    unittest.main()
