"""Traceability: every field of the training Config has a tagged entry in src/configs/sources.yaml."""
import dataclasses
import os
import yaml

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from src.train import Config  # noqa: E402


def test_every_config_key_is_sourced():
    y = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'src', 'configs', 'sources.yaml')))
    keys = y['config_keys']
    missing = [f.name for f in dataclasses.fields(Config) if f.name not in keys or not keys[f.name].get('source')]
    assert not missing, missing
