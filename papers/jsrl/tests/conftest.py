import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# tests are CPU-only and compile-bound: skip expensive XLA optimisation passes (smoke test < 30 s); numerics are not compared bitwise anywhere
os.environ.setdefault("XLA_FLAGS", "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
