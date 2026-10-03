"""Put the retriever service root on sys.path for its tests.

The services are installed as editable packages and keep flat top-level module
names (config, db, embedder, llm, main, server). config/db/embedder are shared
by name with the ingestion service, so this directory is only safe to put on
sys.path when no other service's tests are being collected.

The root pyproject pins pythonpath=["services/ingestion"], which pytest
applies around conftest loading — without the eviction below, `import config`
would resolve to the ingestion service. (Same lesson as the generator suite:
its llm-only tests never noticed until a test imported the shared name.)

Run these tests on their own:

    pytest services/retriever
"""

import sys
from pathlib import Path

_service_root = str(Path(__file__).resolve().parents[1])
if _service_root not in sys.path:
    sys.path.insert(0, _service_root)

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
