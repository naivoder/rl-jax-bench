"""Differential test 1 (guide port): numpy guide logits vs JAX reference logits on 4,096 demo states; then the guide's true
mean on the original-MinAtar wrapper (1,000 episodes, eps 0.01) vs our JAX measurement 6.784 +- 0.029 (data/guides/../data/demos/Breakout_guide.json)."""
import sys, json, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle_common import NumpyGuide, Breakout
ref = np.load('guide_ref_logits.npz')
g = NumpyGuide()
L = g.logits(ref['obs'])
err = np.abs(L - ref['logits']).max()
srt = np.sort(ref['logits'], -1); margin = srt[:, -1] - srt[:, -2]
agree = (L.argmax(-1) == ref['logits'].argmax(-1)).mean()
dem = (L.argmax(-1) == ref['demo_act']).mean()
out = dict(max_abs_logit_err=float(err), min_top2_margin=float(margin.min()), argmax_agree=float(agree), agree_with_demo_labels=float(dem))
t0 = time.time(); rets, lens = [], []
for k in range(1000):
    e = Breakout(seed=10_000 + k); o, _ = e.reset(); done = False; R = 0.0
    while not done:
        a, _ = g.predict(o); o, r, te, tr, _ = e.step(a[0]); R += r; done = te or tr
    rets.append(R); lens.append(e.t)
out.update(guide_mean=float(np.mean(rets)), guide_se=float(np.std(rets, ddof=1) / np.sqrt(1000)), mean_len=float(np.mean(lens)), cap_frac=float(np.mean(np.array(lens) >= 1000)), secs=time.time() - t0)
z = (out['guide_mean'] - 6.784) / np.hypot(out['guide_se'], 0.02926)
out['z_vs_jax_6.784'] = float(z)
print(json.dumps(out, indent=1)); json.dump(out, open('test_guide_port.json', 'w'), indent=1)
