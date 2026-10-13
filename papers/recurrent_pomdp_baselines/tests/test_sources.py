"""Every config key used by any src/configs/*.yaml appears in sources.yaml with a source tag."""
import glob
import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_all_keys_tagged():
    src = yaml.safe_load(open(os.path.join(HERE, "src", "configs", "sources.yaml")))
    tagged = set()
    for sec, d in src.items():
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, dict) and v.get("source"):
                    tagged.add(k)
    used = set()
    for f in glob.glob(os.path.join(HERE, "src", "configs", "*.yaml")):
        if f.endswith("sources.yaml"):
            continue
        used |= set(yaml.safe_load(open(f)).keys()) - {"base"}
    missing = used - tagged
    assert not missing, missing

