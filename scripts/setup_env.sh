#!/bin/bash
# The evaluation environment, into the active Python 3.11 (a venv or a container), plus the
# TotalSegmentator weights the organisers' ts_utils.py loads (Dataset297, total, 3 mm):
#   scripts/setup_env.sh <weights dir> [pip install options]
#   export SYNTHRAD_EVAL_TS_RESULTS=<weights dir>
# e.g. '--target <dir> --upgrade' for a package dir next to a read-only image (then PYTHONPATH=<dir>).
set -euo pipefail
cd "$(dirname "$0")/.."
TS=${1:?usage: scripts/setup_env.sh <weights dir> [pip install options]}
shift
PYTHON=${PYTHON:-python3}
$PYTHON -c 'import sys; assert sys.version_info[:2] == (3, 11), f"Python 3.11 expected (the container), not {sys.version}"'
$PYTHON -m pip install "$@" -r requirements.txt
$PYTHON -m pip install "$@" --no-deps -r requirements-nodeps.txt
$PYTHON - "$TS" <<'PY'
import io, sys, urllib.request, zipfile
from pathlib import Path
dst = Path(sys.argv[1])
if not (dst / "Dataset297_TotalSegmentator_total_3mm_1559subj").is_dir():
    url = ("https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/"
           "Dataset297_TotalSegmentator_total_3mm_1559subj.zip")
    dst.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(dst)
print("weights:", dst)
PY
