"""Put the evals package root on sys.path for its tests.

Library code lives under evalkit/ (a unique top-level name — the flat
`db`/`seed`/`run` names stay as uninstalled scripts, never as importable
modules, after one editable-install shadowing incident). Run on its own:

    pytest evals
"""

import sys
from pathlib import Path

_pkg_root = str(Path(__file__).resolve().parents[1])
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
