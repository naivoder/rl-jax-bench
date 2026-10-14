"""Shrink oracle_iter.npz into the committed test fixture (lossless for these values): one-hot/binary S, O -> uint8;
permutations / actions / masks -> small ints; rewards (integers -2/18/-102 ...) -> int16; weights stay float64."""
import sys
import numpy as np
z = np.load(sys.argv[1])
out = {}
for k in z.files:
    v = z[k]
    leaf = k.split('/')[-1]
    if leaf in ('S', 'O'):
        assert np.all((v == 0) | (v == 1)), k
        v = v.astype(np.uint8)
    elif leaf in ('A', 'ND', 'vq_tag'):
        v = v.astype(np.int8)
    elif leaf == 'R':
        assert np.all(v == np.round(v)), k
        v = v.astype(np.int16)
    elif leaf in ('vq', 'sel') or leaf.startswith('rb') and leaf != 'rb_n':
        v = v.astype(np.int16)
    out[k] = v
np.savez_compressed(sys.argv[2], **out)
print('wrote', len(out), 'arrays')
