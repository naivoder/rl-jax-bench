"""Parse Tables I-IV out of the arXiv TeX source of 2312.07858v1 (main_doublecolumn.tex) into the cell -> K -> (index, myopic, TEV/TEC) layout of
src/eval.py::PAPER, as an INDEPENDENT transcription check of the PDF reading used for the MATCH judgments.  usage: parse_tex_tables.py <tex> <out.json>"""
import json, re, sys

tex = open(sys.argv[1]).read()
tabs = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex, flags=re.S)
labels = re.findall(r"\\label\{(tab:[^}]+)\}", tex)
out = {}
for lab, body in zip(labels, tabs):
    rows = [r for r in re.split(r"\\\\|\\hline", body) if re.search(r"\d+\.\d\d", r)]
    vals = []
    for r in rows:
        nums = re.findall(r"(?<![\d.{,])(\d+\.\d\d)(?!\d)", r)
        vals.append([float(x) for x in nums])
    out[lab] = vals
json.dump(out, open(sys.argv[2], "w"), indent=1)
for k, v in out.items():
    print(k, len(v), v[:3])
