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

# The root pyproject pins pythonpath=["services/ingestion"], which pytest
# applies *after* this conftest runs — so ingestion's root would otherwise sit
# ahead of ours and `import config` would resolve to the ingestion service
# (whose GeneratorConfig does not exist). Evict the other service roots so the
# suite is independent of ini application order. This bit us the moment a test
# imported the shared `config` module name; the llm-only tests never noticed.
_repo_root = Path(__file__).resolve().parents[2]
for _entry in list(sys.path):
    try:
        _p = Path(_entry).resolve()
    except OSError:
        continue
    if (
        _p != Path(_service_root).resolve()
        and _repo_root in _p.parents
        and _p.parent.name == "services"
    ):
        sys.path.remove(_entry)