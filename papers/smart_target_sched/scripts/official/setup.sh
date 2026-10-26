#!/bin/sh
# Fetch the authors' published output: the arXiv TeX source of 2312.07858v1 (Tables I-IV as typeset, vector figures
# figs/*.pdf). No code was released for this paper. After this, `uv run python scripts/make_official_fixtures.py` and
# `... --gz` rebuild tests/fixtures/paper_tex_tables.json, paper_fig_curves.npz and paper_fig_Gz.npz (needs pymupdf:
# `uv sync --extra pdf`). The shipped fixtures were built from the tarball with the sha256 below.
set -e
cd "$(dirname "$0")/../.."
OFFICIAL_ROOT=${OFFICIAL_ROOT:-external/official}
SHA=b453b186d7e52d3db42c550c8a4c3c58a21bcbd5a930012848879aaa08ebb1bc
mkdir -p "$OFFICIAL_ROOT/arxiv_src"
cd "$OFFICIAL_ROOT/arxiv_src"
[ -f 2312.07858v1-src.tgz ] || curl -fL -o 2312.07858v1-src.tgz https://arxiv.org/src/2312.07858v1
echo "$SHA  2312.07858v1-src.tgz" | sha256sum -c - || echo "WARNING: tarball differs from the one the shipped fixtures were built from"
tar xzf 2312.07858v1-src.tgz
echo "arXiv source in $OFFICIAL_ROOT/arxiv_src"
