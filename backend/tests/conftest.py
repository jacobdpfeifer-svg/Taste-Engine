"""Test environment: isolate the sqlite db and kernels dir, and make sure the generated
stimulus bank exists (scripts/generate_stimuli.py is deterministic, so this is stable).

Env must be set before app.config / app.main are imported by any test module.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent

if not (BACKEND / "data" / "stimuli.json").exists():
    subprocess.run(
        [sys.executable, "scripts/generate_stimuli.py"], cwd=BACKEND, check=True
    )

os.environ["STIMULI_PATH"] = str(BACKEND / "data" / "stimuli.json")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test_taste.sqlite3")
os.environ["KERNELS_DIR"] = tempfile.mkdtemp()
