"""Put the generator service root on sys.path for its tests.

The services are installed as editable packages and keep flat top-level module
names (config, db, embedder, llm, main, server). llm is unique to this service,
but config is shared by all three, so this directory is only safe to put on
sys.path when no other service's tests are being collected.

Run these tests on their own:

    pytest services/generator
"""

import sys
from pathlib import Path

_service_root = str(Path(__file__).resolve().parents[1])
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)