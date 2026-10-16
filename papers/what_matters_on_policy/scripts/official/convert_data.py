# Convert the released study dump into per-experiment parquet/pickle with flat columns for later stages.
import json, io, pickle, numpy as np, pandas as pd
import os  # noqa: E402
OFFICIAL_ROOT = os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "external", "official"))  # authors' checkout, see README
p=os.path.join(OFFICIAL_ROOT, "data", "mujoco123_data.json")
tmp=json.load(open(p))
out={}
for k,v in tmp.items():
    df=pd.read_json(io.StringIO(v[0]))
    meta=dict(params=v[1], title=v[2], a=v[3], b=v[4])
    for col in ('_eval_','_eval_original_avg','_eval_det'):
        if col in df: df[col]=df[col].map(lambda d: {int(a):b for a,b in d.items()} if isinstance(d,dict) else d)
    out[k]=(df,meta)
    print(k, meta['title'], len(df), sorted(set(df.env_name)), 'params', len(meta['params']), flush=True)
pickle.dump(out, open(os.path.join(OFFICIAL_ROOT, "data", "mujoco123.pkl"),'wb'))
