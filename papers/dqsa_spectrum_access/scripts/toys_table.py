import json, sys, glob, os
R = os.path.join(os.path.dirname(__file__), "..", "runs", "toys")
rows = []
for f in ["toys.json", "toys_t1_lr1e-3.json", "toys_t1.json"]:
    p = os.path.join(R, f)
    if not os.path.exists(p): continue
    d = json.load(open(p))
    tag = "" if f == "toys.json" else ("[T1 lr 1e-3, 6000 its]" if "lr1e-3" in f else "[T1 lr 3e-4, 10000 its]")
    for c in d["checks"]:
        if f == "toys.json" and c["name"].startswith("T1 Q"): continue
        rows.append((c["name"] + tag, c["ours"], c["truth"], c["tol"], c["ok"], c.get("note", "")))
print("| check | ours | truth | tolerance | PASS | note |\n|---|---|---|---|---|---|")
for n, o, t, tol, ok, note in rows:
    print(f"| {n} | {o:.6g} | {t:.6g} | {tol:.3g} | {'PASS' if ok else 'FAIL'} | {note} |")
print(f"\n{sum(r[4] for r in rows)}/{len(rows)} PASS")
