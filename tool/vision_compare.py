#!/usr/bin/env python3
"""Compare completed upstream and dianoia visual inference dumps.

Always requires exact preprocessing, prompt IDs, grid and generated token IDs.
--atol/--rtol bound each encoder/projector/logit element; quantized backends
need separately chosen tolerances. A successful report is not a broad quality
benchmark. Returns nonzero for any failed check and saves an inspectable report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference: Path, actual: Path, atol: float, rtol: float,
            logits_atol: float | None = None, logits_rtol: float | None = None) -> dict:
    """Checks a vision-run dump against an upstream reference dump.

    Args:
        reference: Directory written by vision_reference.py.
        actual: Directory written by `dianoia vision-run`.
        atol: Absolute tolerance for encoder, projector and logit elements.
        rtol: Relative tolerance, scaled by the reference magnitude.
        logits_atol: Absolute tolerance for logits only; atol when None.
        logits_rtol: Relative tolerance for logits only; rtol when None.

    Returns:
        A JSON-ready report: per-check booleans, with checks['passed'] true
        only if all hold, and error metrics per compared file.
    """
    a = json.loads((reference/'meta.json').read_text())
    b = json.loads((actual/'meta.json').read_text())
    report = {'checks': {}, 'metrics': {}}
    checks, metrics = report['checks'], report['metrics']
    checks['complete'] = a.get('complete') is True and b.get('complete') is True
    for key in ['prompt', 'thinking', 'grid_thw', 'prompt_tokens', 'stages']:
        checks[key] = a.get(key) == b.get(key)
    for name in ['ids.i32', 'gen.i32']:
        checks[name] = (reference/name).read_bytes() == (actual/name).read_bytes()
    checks['vision_stages'] = all(stage in a['stages'] for stage in ['embedding', 'projected'])
    for label, path, meta in [('reference', reference, a), ('actual', actual, b)]:
        checks[label+'_generated_meta'] = np.fromfile(path/'gen.i32', dtype='<i4').tolist() == meta.get('generated_ids')
    names = ['pixels.f32', 'logits.f32', 'gen_logits.f32'] + [f'vision_{stage}.f32' for stage in a['stages']]
    for name in names:
        x = np.fromfile(reference/name, dtype='<f4')
        y = np.fromfile(actual/name, dtype='<f4')
        shape_ok = x.shape == y.shape and (x.size > 0 or name == 'gen_logits.f32')
        if not shape_ok:
            checks[name] = False
            metrics[name] = {'reference_elements': int(x.size), 'actual_elements': int(y.size)}
            continue
        error = np.abs(x.astype('float64')-y.astype('float64'))
        finite = np.isfinite(x).all() and np.isfinite(y).all()
        limit_a = logits_atol if name in ['logits.f32', 'gen_logits.f32'] and logits_atol is not None else atol
        limit_r = logits_rtol if name in ['logits.f32', 'gen_logits.f32'] and logits_rtol is not None else rtol
        checks[name] = bool(finite and (np.array_equal(x, y) if name == 'pixels.f32' else
            np.all(error <= limit_a + limit_r*np.abs(x))))
        metrics[name] = dict(max_abs=(float(error.max()) if x.size else 0.0) if finite else None,
                            mean_abs=(float(error.mean()) if x.size else 0.0) if finite else None,
                            rmse=(float(np.sqrt(np.mean(error**2))) if x.size else 0.0) if finite else None)
        if name == 'logits.f32':
            metrics[name]['reference_argmax'] = int(x.argmax()) if finite else None
            metrics[name]['actual_argmax'] = int(y.argmax()) if finite else None
    checks['passed'] = all(checks.values())
    report.update(atol=atol, rtol=rtol, logits_atol=logits_atol, logits_rtol=logits_rtol, reference_text=a.get('text'), actual_text=b.get('text'))
    return report


def main() -> None:
    """Compares the dumps named on the command line; exits 1 on a failure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--actual', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--atol', type=float, default=0.002)
    p.add_argument('--rtol', type=float, default=0.0002)
    p.add_argument('--logits-atol', type=float)
    p.add_argument('--logits-rtol', type=float)
    a = p.parse_args()
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in [a.atol, a.rtol, a.logits_atol, a.logits_rtol]):
        p.error('tolerances must be nonnegative')
    result = compare(a.reference, a.actual, a.atol, a.rtol, a.logits_atol, a.logits_rtol)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['checks']['passed'] else 1)


if __name__ == '__main__':
    main()
