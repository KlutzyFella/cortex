"""Root conftest for the ingestion service.

Inserts the services/ingestion directory into sys.path so that pytest
discovers the local modules (config, db, processor, embedder) regardless
of the working directory from which pytest is invoked.
"""

import sys
from pathlib import Path

# Ensure the package root (services/ingestion/) is on sys.path so that
# flat imports like `from config import ...` resolve correctly in tests.
_pkg_root = str(Path(__file__).parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
