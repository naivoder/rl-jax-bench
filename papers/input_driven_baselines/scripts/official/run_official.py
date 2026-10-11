"""Run an UNMODIFIED official script under TF 2.13 with tf.compat.v1 aliased as `tensorflow` (labeled shim:
TF 1.2.1 / py3.6 cannot be installed here). Usage: run_official.py <script.py> [args...]"""
import os, sys, runpy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
OFF = os.path.join(os.environ.get('OFFICIAL_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external', 'official')), 'input_driven_rl_example')  # the authors' clone
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import tensorflow.python  # keep the real submodules importable (nn_ops: from tensorflow.python.framework import ops)
sys.modules['tensorflow'] = tf1
sys.path.insert(0, OFF)
script = os.path.join(OFF, sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
