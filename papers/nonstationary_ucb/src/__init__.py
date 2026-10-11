"""nonstationary_ucb: JAX re-implementation of Garivier & Moulines, D-UCB and SW-UCB for switching bandits.

Importing the package runs the GPU bootstrap before anything can import jax, so every entry point
(``python -m src.train``, ``python -m src.eval``, pytest) gets the host CUDA libs, the shared compile
cache and the determinism flags. float64 is enabled in ``src.common``.
"""
from rl_jax_bench import gpu_bootstrap as _gb

_gb.setup()
