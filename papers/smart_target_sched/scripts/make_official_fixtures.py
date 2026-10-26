"""Build the 'authors' output' fixtures used by tests/test_official_parity.py from the paper's own arXiv TeX source
($OFFICIAL_ROOT/arxiv_src, fetched by scripts/official/setup.sh; sha256 in the fixture meta).  There is NO released code for this paper
(see the README); the authors' computed numbers that DO exist are (i) Tables I-IV as typeset in the TeX and (ii) the
vector figures figs/*.pdf, whose polyline vertices are the authors' plotted data to ~1e-4 relative.  Not part of the reproduction pipeline."""
import hashlib, json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import extract_fig_vectors as X

SRC = os.path.join(os.environ.get("OFFICIAL_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "official")), "arxiv_src")
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
if "--gz" in sys.argv:        # the authors' G(P,z) vs z figure (4 curves x 14901 points per type), a SEPARATE fixture so paper_fig_curves.npz stays byte-stable
    _sha = hashlib.sha256(open(os.path.join(SRC, "2312.07858v1-src.tgz"), "rb").read()).hexdigest()
    _arr = {}
    for _f in ["Gzbeta0.9q4_rec", "Gzbeta0.9q4_cau"]:
        _r = X.extract(os.path.join(SRC, "figs", _f + ".pdf"))
        for _i, _c in enumerate(_r["curves"]):
            _arr[f"{_f}__{_i}__x"] = np.array(_c["x"]); _arr[f"{_f}__{_i}__y"] = np.array(_c["y"])
    _arr["_src_sha256"] = np.array(_sha)
    np.savez_compressed(os.path.join(OUT, "paper_fig_Gz.npz"), **_arr)
    print("wrote", len(_arr), "arrays to paper_fig_Gz.npz"); sys.exit(0)

sha = hashlib.sha256(open(os.path.join(SRC, "2312.07858v1-src.tgz"), "rb").read()).hexdigest()

# ---- tables
tex = open(os.path.join(SRC, "main_doublecolumn.tex")).read()
tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
labels = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
tables = {}
for lab, body in zip(labels, tabs):
    rows = [r for r in re.split(r"\\\\|\\hline", body) if re.search(r"\d+\.\d\d", r)]
    tables[lab] = [[float(x) for x in re.findall(r"(?<![\d.{,])(\d+\.\d\d)(?!\d)", r)] for r in rows]
json.dump({"source": "arXiv:2312.07858v1 TeX main_doublecolumn.tex", "src_sha256": sha, "tables": tables}, open(os.path.join(OUT, "paper_tex_tables.json"), "w"), indent=1)

# ---- figures (vector polylines)
arr = {}
for f in ["gfbeta09z4q4_rec", "gfbeta09z10q4_rec", "gfbeta09z4q10_rec", "gfbeta09z10q10_rec",
          "gfbeta09z4q4_cau", "gfbeta09z10q4_cau", "gfbeta09z4q10_cau", "gfbeta09z10q10_cau",
          "type-q4", "type-q10", "Q-x1", "Q-x10",
          "Bound_gap_rec", "Bound_gap_cau", "Bound_gap_rec_cau", "Bound_gap_rec_cau_ratio",
          "multi-dimensional/Bound_performance_rec_cau", "multi-dimensional/Bound_performance_rec_cau_ratio"]:
    r = X.extract(os.path.join(SRC, "figs", f + ".pdf"))
    name = f.split("/")[-1]
    for i, c in enumerate(r["curves"]):
        arr[f"{name}__{i}__x"] = np.array(c["x"]); arr[f"{name}__{i}__y"] = np.array(c["y"])
    arr[f"{name}__axis_resid"] = np.array([r["axis_fit_resid"][0] or 0.0, r["axis_fit_resid"][1] or 0.0])
arr["_src_sha256"] = np.array(sha)
np.savez_compressed(os.path.join(OUT, "paper_fig_curves.npz"), **arr)
print("wrote", len(arr), "arrays; src sha", sha)
