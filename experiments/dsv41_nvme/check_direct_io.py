"""Small Linux direct-I/O check; no model download or GPU allocation."""
import hashlib
import json
import sys

from async_store import AsyncExpertStore
from expert_store import ExpertStore
from posix_support import DIRECT_IO
from test_async_store import AsyncStoreTests


def main():
    # Unsupported direct I/O is a failure, never a silent pass.
    if not DIRECT_IO:
        print(json.dumps({"passed": False, "direct": False,
                          "reason": "os.O_DIRECT/preadv unavailable on this platform"}, indent=2))
        return 1
    # Reuse the public synthetic fixture, never the private checkpoint.
    AsyncStoreTests.setUpClass()
    results = []
    try:
        for cls in (ExpertStore, AsyncExpertStore):
            store = cls(AsyncStoreTests.bank, AsyncStoreTests.digest, direct=True)
            try:
                if isinstance(store, AsyncExpertStore):
                    store.plan(store.records)
                for key, record in store.records.items():
                    with store.read(key) as data:
                        assert hashlib.sha256(data).hexdigest() == record["sha256"]
                results.append({"store": cls.__name__, "reads": store.reads,
                                "bytes_read": store.bytes_read,
                                "peak_staging_bytes": store.peak_staging_bytes})
            finally:
                store.close()
            assert store.fd is None and store.active_reads == 0
    finally:
        AsyncStoreTests.tearDownClass()
    print(json.dumps({"passed": True, "direct": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
