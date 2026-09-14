"""Opt-in CUDA fixture for the ExpertCache lease stream contract. NOT RUN HERE.

This is the tiny real-hardware counterpart of `test_lease_stream_contract.py`,
prepared so the parent can run it on a GPU host without a model, weights,
exllamav3 or a large cache: it allocates one 1 MiB arena and two scratch
tensors, uses real CUDA streams and real `torch.cuda.Event` objects, and
constructs no `LinearEXL3` (the leased `Entry` carries an empty projection dict,
which is all `_evict()`'s `projections.clear()` needs).

It is skipped unless BOTH are true:
  * `torch.cuda.is_available()`, and
  * `DSV41_NVME_CUDA_CONTRACT=1` (explicit opt-in; never runs by accident).

Run it with:
    DSV41_NVME_CUDA_CONTRACT=1 python -m pytest experiments/dsv41_nvme/test_lease_stream_cuda.py -v

Discrimination for the L1 defect uses stream occupancy, not luck: the correct
code records the release event on the stream that owns the lease, so the event
completes as soon as that stream's work drains; the pre-fix code records on
whatever stream is current at release (here: the default stream, deliberately
kept busy), so the event sits behind unrelated queued work. Each check first
confirms the discriminator is actually in place (the busy stream's own probe
event must still be pending) and fails loudly instead of passing silently if the
device drains too fast for the margin.

No CUDA receipt is claimed by this repository for these checks: they are
prepared, gated and unrun on the review host.
"""
import os
import sys
import threading
import time
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lease_stream_contract import load_expert_cache  # noqa: E402  (import shim + loader)

# The environment gate is evaluated FIRST: without the explicit opt-in no CUDA
# query of any kind is made, so a review host with a device present is untouched.
OPT_IN = os.environ.get("DSV41_NVME_CUDA_CONTRACT") == "1"
ENABLED = OPT_IN and torch.cuda.is_available()
SKIP_REASON = (
    "requires a CUDA device and DSV41_NVME_CUDA_CONTRACT=1 (opt-in); "
    "prepared but NOT RUN on the review host - no CUDA receipt is claimed"
)
ARENA_BYTES = 1 << 20  # 1 MiB, deliberately tiny
SPIN = 1.0000017


def _spin(iterations, tensor):
    """Enqueue a deterministic amount of elementwise work on the current stream."""
    for _ in range(iterations):
        tensor.mul_(SPIN)
    return tensor


def calibrate(device):
    """A 1 MiB scratch tensor plus the _spin iteration count costing ~1 second."""
    tensor = torch.ones(1 << 18, dtype=torch.float32, device=device)
    timer_stream = torch.cuda.Stream()
    start = time.monotonic()
    with torch.cuda.stream(timer_stream):
        _spin(64, tensor)
    timer_stream.synchronize()
    per_iteration = (time.monotonic() - start) / 64
    iterations = max(256, min(2_000_000, int(1.0 / max(per_iteration, 5e-6))))
    return iterations, tensor


def build_cache(module, key="0:1", size=4096):
    """A real arena, real Regions accounting, no model and no projection handles."""
    cache = module.ExpertCache.__new__(module.ExpertCache)
    cache.device = torch.device("cuda:0")
    cache.lock = threading.RLock()
    cache.closed = False
    cache.store = SimpleNamespace(records={key: {"bytes": size}})
    cache.entries = OrderedDict()
    cache.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0)
    cache.generation = 1
    cache.regions = {int(key.split(":")[0]): module.Regions(ARENA_BYTES, 0)}
    cache.resident_bytes = size
    cache.arena = torch.empty(ARENA_BYTES, dtype=torch.uint8, device=cache.device)
    offset = cache.regions[int(key.split(":")[0])].allocate(size)
    assert offset is not None
    cache.entries[key] = module.Entry(key, offset, size, 1, {})
    return cache, cache.entries[key]


def wait_pending(event, timeout=10.0):
    """True if *event* completes within *timeout*, reported with its latency."""
    start = time.monotonic()
    while not event.query():
        if time.monotonic() - start > timeout:
            return False, time.monotonic() - start
        time.sleep(0.002)
    return True, time.monotonic() - start


@unittest.skipUnless(ENABLED, SKIP_REASON)
class LeaseStreamCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_expert_cache(Path(__file__).resolve().parent)
        cls.iterations, cls.scratch = calibrate(torch.device("cuda:0"))

    def setUp(self):
        torch.cuda.synchronize()

    def tearDown(self):
        torch.cuda.synchronize()

    def busy_default(self):
        """Keep the default stream busy and confirm the discriminator is live."""
        default = torch.cuda.current_stream(torch.device("cuda:0"))
        probe = torch.cuda.Event()
        with torch.cuda.stream(default):
            _spin(self.iterations, self.scratch)
            probe.record()
        self.assertFalse(
            probe.query(),
            "calibration failed to keep the default stream busy; re-run with a larger "
            "calibrate() target rather than trusting a silent pass",
        )
        return probe

    def test_release_after_the_stream_context_exits_records_on_the_acquisition_stream(self):
        cache, entry = build_cache(self.module)
        owning = torch.cuda.Stream()
        self.busy_default()
        lease = cache.lease("0:1")
        with torch.cuda.stream(owning):
            lease.__enter__()
            _spin(32, self.scratch)  # the consumer's work, on the acquisition stream
        lease.__exit__(None, None, None)  # release happens outside the stream block
        self.assertEqual(entry.users, 0)
        completed, latency = wait_pending(entry.events[-1])
        self.assertTrue(completed, "release event never completed")
        self.assertLess(
            latency, 0.25,
            "the release event waited behind work on the stream current at release "
            "instead of the stream captured at acquisition",
        )

    def test_concurrent_leases_on_different_streams_keep_their_own_event_stream(self):
        cache, entry = build_cache(self.module)
        busy_stream, quiet_stream = torch.cuda.Stream(), torch.cuda.Stream()
        default = torch.cuda.current_stream(torch.device("cuda:0"))
        probe = torch.cuda.Event()
        with torch.cuda.stream(default):
            _spin(self.iterations, self.scratch)
            probe.record()
        self.assertFalse(probe.query(), "discriminator not in place; refusing a silent pass")
        first, second = cache.lease("0:1"), cache.lease("0:1")
        with torch.cuda.stream(busy_stream):
            first.__enter__()
        with torch.cuda.stream(quiet_stream):
            second.__enter__()
        self.assertEqual(entry.users, 2)
        second.__exit__(None, None, None)   # quiet stream, released outside its block
        first.__exit__(None, None, None)    # busy stream, released outside its block
        self.assertEqual(entry.users, 0)
        self.assertEqual(len(entry.events), 2)
        quiet_done, _ = wait_pending(entry.events[0], timeout=5.0)
        self.assertTrue(quiet_done, "the quiet lease's event must complete on its own stream")
        self.assertFalse(
            entry.events[1].query(),
            "the busy lease's event must still be pending on ITS acquisition stream; "
            "pending nowhere means both events landed on the idle default stream",
        )

    def test_eviction_waits_for_work_queued_on_the_captured_stream(self):
        cache, entry = build_cache(self.module)
        owning = torch.cuda.Stream()
        lease = cache.lease("0:1")
        with torch.cuda.stream(owning):
            lease.__enter__()
            _spin(self.iterations, self.scratch)  # queued before the release event
        lease.__exit__(None, None, None)
        start = time.monotonic()
        cache._evict("0:1")
        elapsed = time.monotonic() - start
        self.assertGreaterEqual(
            elapsed, 0.5,
            "_evict() returned before the work queued on the lease's captured stream "
            "drained; the gating event was not recorded on that stream",
        )
        self.assertNotIn("0:1", cache.entries)
        self.assertEqual(cache.resident_bytes, 0)
        self.assertEqual(cache.stats["evictions"], 1)
        self.assertTrue(entry.events[0].query())


if __name__ == "__main__":
    unittest.main()
