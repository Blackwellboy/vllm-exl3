"""CPU contract test for the ExpertCache lease stream ownership rule.

Scope and honesty: this is a CONTRACT test driven by a documented test double for
`torch.cuda`, not GPU evidence. It makes no CUDA call, allocates no device
memory, loads no model and builds no arena, so it runs on a CPU-only host and
verifies only the ownership rule itself:

    the stream that owns a lease is the stream current at ACQUISITION, and the
    release event is recorded on that stream - not on whatever stream happens to
    be current when the caller leaves the lease.

The double implements exactly the three torch entry points the lease path uses
(`torch.cuda.device`, `torch.cuda.current_stream`, `torch.cuda.stream`,
`torch.cuda.Event`) and records the stream object each event was recorded on.
The double is scoped to these tests and every touchpoint goes through it, so a
CUDA-capable host runs this test without touching the device.

The test is RED against the pre-fix release, which read the current stream at
release time (`event.record(torch.cuda.current_stream(self.device))`), and GREEN
against the fix, which captures the owning stream at acquisition. Nothing here
is waived or skipped: a regression fails.
"""
import contextlib
import importlib.util
import sys
import threading
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parent
_LOADS = [0]


def load_expert_cache(directory):
    """Import `expert_cache.py` from *directory* without exllamav3 installed.

    `LinearEXL3` is imported at module import time but the lease path never
    constructs one, so an import shim keeps this check independent of the
    quantisation backend. The shim raises if the lease path ever does try to
    build a projection, which would mean this test is no longer testing the
    lease.
    """
    directory = str(directory)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    if "exllamav3" not in sys.modules:
        package = types.ModuleType("exllamav3")
        package.__path__ = []
        modules = types.ModuleType("exllamav3.modules")
        modules.__path__ = []
        quant = types.ModuleType("exllamav3.modules.quant")
        quant.__path__ = []
        exl3 = types.ModuleType("exllamav3.modules.quant.exl3")

        class LinearEXL3:
            def __init__(self, *args, **kwargs):
                raise AssertionError("the lease path must not construct LinearEXL3")

        exl3.LinearEXL3 = LinearEXL3
        package.modules = modules
        modules.quant = quant
        quant.exl3 = exl3
        sys.modules["exllamav3"] = package
        sys.modules["exllamav3.modules"] = modules
        sys.modules["exllamav3.modules.quant"] = quant
        sys.modules["exllamav3.modules.quant.exl3"] = exl3
    _LOADS[0] += 1
    name = f"expert_cache_under_test_{_LOADS[0]}"
    spec = importlib.util.spec_from_file_location(name, Path(directory) / "expert_cache.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStream:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"FakeStream({self.name})"


class FakeEvent:
    def __init__(self, double):
        self.double = double
        self.stream = None
        self.synchronized = False

    def record(self, stream=None):
        if self.stream is not None:
            raise AssertionError("a CUDA event can only be recorded once")
        self.stream = stream if stream is not None else self.double.current_stream(None)
        self.double.events.append(self)

    def query(self):
        return True

    def synchronize(self):
        self.synchronized = True


class CudaDouble:
    """Records which stream each event was recorded on; never touches a device."""

    def __init__(self):
        self._local = threading.local()
        self.events = []

    # -- the four torch.cuda entry points the lease path uses -----------------
    def current_stream(self, device=None):
        return getattr(self._local, "stream", None) or FakeStream("default")

    @contextlib.contextmanager
    def stream(self, stream):
        previous = getattr(self._local, "stream", None)
        self._local.stream = stream
        try:
            yield stream
        finally:
            self._local.stream = previous

    @contextlib.contextmanager
    def device(self, device):
        yield device

    def Event(self):
        return FakeEvent(self)

    # ------------------------------------------------------------------------
    @contextlib.contextmanager
    def active(self):
        with patch.object(torch.cuda, "current_stream", self.current_stream), \
             patch.object(torch.cuda, "stream", self.stream), \
             patch.object(torch.cuda, "device", self.device), \
             patch.object(torch.cuda, "Event", self.Event):
            yield self


def build_cache(module):
    """A cache shaped like a post-`_load` hit, without arena, model or CUDA.

    `__init__` is bypassed deliberately: it checks host RAM, allocates the
    device arena and builds `Regions`, none of which the lease path needs.
    """
    cache = module.ExpertCache.__new__(module.ExpertCache)
    cache.device = torch.device("cuda:0")
    cache.lock = threading.RLock()
    cache.closed = False
    cache.store = SimpleNamespace(records={"0:1": {"bytes": 4096}})
    cache.entries = OrderedDict()
    cache.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0)
    cache.generation = 1
    cache.regions = {}
    cache.resident_bytes = 4096
    entry = module.Entry("0:1", 0, 4096, 1, {})
    cache.entries["0:1"] = entry
    return cache, entry


class LeaseStreamContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_expert_cache(ROOT)

    def test_release_after_the_stream_context_exits_records_on_the_acquisition_stream(self):
        cache, entry = build_cache(self.module)
        cuda = CudaDouble()
        owning = FakeStream("s")
        with cuda.active():
            lease = cache.lease("0:1")
            with torch.cuda.stream(owning):          # acquisition stream
                leased = lease.__enter__()
                self.assertIs(leased, entry)
                self.assertEqual(entry.users, 1)
            lease.__exit__(None, None, None)         # release, outside the stream block
            self.assertEqual(entry.users, 0)
            recorded = [event.stream for event in entry.events]
        self.assertEqual(len(recorded), 1)
        self.assertIs(
            recorded[0], owning,
            "the release event must be recorded on the stream captured at acquisition, "
            "not on the stream current at release",
        )

    def test_concurrent_leases_on_different_streams_each_keep_their_own_stream(self):
        cache, entry = build_cache(self.module)
        cuda = CudaDouble()
        first_stream, second_stream = FakeStream("a"), FakeStream("b")
        with cuda.active():
            first, second = cache.lease("0:1"), cache.lease("0:1")
            with torch.cuda.stream(first_stream):
                first.__enter__()
            with torch.cuda.stream(second_stream):
                second.__enter__()
            self.assertEqual(entry.users, 2)
            # Released outside both stream blocks, newest lease first.
            second.__exit__(None, None, None)
            first.__exit__(None, None, None)
            self.assertEqual(entry.users, 0)
            recorded = [event.stream for event in entry.events]
        self.assertEqual(recorded, [second_stream, first_stream])
        self.assertEqual(cache.stats["hits"] + cache.stats["misses"], 2)

    def test_release_still_records_once_on_the_owning_stream_when_the_body_raises(self):
        cache, entry = build_cache(self.module)
        cuda = CudaDouble()
        owning = FakeStream("s")
        try:
            raise ValueError("consumer failure")
        except ValueError:
            exc = sys.exc_info()
        with cuda.active():
            lease = cache.lease("0:1")
            with torch.cuda.stream(owning):
                lease.__enter__()
            suppressed = lease.__exit__(*exc)
            self.assertFalse(suppressed, "a lease must not swallow the consumer's exception")
            self.assertEqual(entry.users, 0)
            recorded = [event.stream for event in entry.events]
        self.assertEqual(recorded, [owning])

    def test_completed_event_dropping_and_lease_accounting_are_unchanged(self):
        cache, entry = build_cache(self.module)
        cuda = CudaDouble()
        with cuda.active():
            for _ in range(3):
                with cache.lease("0:1"):
                    pass
            # FakeEvent.query() is always True, so every hit drops the previous lot.
            self.assertEqual(len(entry.events), 1)
            with cache.lease("0:1"):
                self.assertEqual(len(entry.events), 0, "hit path must drop completed events")
            self.assertEqual(entry.users, 0)
        self.assertEqual(cache.stats["hits"], 4)
        self.assertEqual(cache.stats["misses"], 0)
        self.assertEqual(len(cuda.events), 4)
        self.assertEqual({event.stream.name for event in cuda.events}, {"default"})


if __name__ == "__main__":
    unittest.main()
