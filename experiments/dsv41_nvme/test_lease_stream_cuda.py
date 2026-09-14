"""Opt-in real-CUDA lease/event contract; no model or expert projection load."""
import os
import sys
import threading
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_lease_stream_contract import load_expert_cache

ENABLED = os.environ.get('DSV41_NVME_CUDA_CONTRACT') == '1' and torch.cuda.is_available()

@unittest.skipUnless(ENABLED, 'requires explicit CUDA contract opt-in and real CUDA')
class LeaseStreamCudaTests(unittest.TestCase):
    def setUp(self):
        self.module = load_expert_cache(Path(__file__).resolve().parent)
        real_event = torch.cuda.Event
        class ObservedEvent:
            # Instrument real events; never emulate CUDA completion.
            def __init__(self):
                self.native = real_event()
                self.stream = None
                self.synchronized = False
            def record(self, stream=None):
                self.stream = stream if stream is not None else torch.cuda.current_stream()
                self.native.record(self.stream)
            def query(self):
                return self.native.query()
            def synchronize(self):
                self.native.synchronize()
                self.synchronized = True
        self.observer = patch.object(torch.cuda, 'Event', ObservedEvent)
        self.observer.start()
        self.addCleanup(self.observer.stop)
        cache = self.module.ExpertCache.__new__(self.module.ExpertCache)
        cache.device = torch.device('cuda:0')
        cache.lock = threading.RLock()
        cache.closed = False
        cache.store = SimpleNamespace(records={'0:1': {'bytes':4096}})
        cache.entries = OrderedDict()
        cache.stats = dict(hits=0, misses=0, evictions=0, failures=0, peak_resident_bytes=0)
        cache.generation = 1
        cache.regions = {0: self.module.Regions(1 << 20, 0)}
        cache.resident_bytes = 4096
        cache.arena = torch.zeros(1 << 20, dtype=torch.uint8, device=cache.device)
        offset = cache.regions[0].allocate(4096)
        cache.entries['0:1'] = self.module.Entry('0:1',offset,4096,1,{})
        torch.cuda.synchronize()
        self.cache, self.entry = cache, cache.entries['0:1']
        self.addCleanup(torch.cuda.synchronize)

    def test_release_after_stream_context_uses_acquisition_stream(self):
        owning = torch.cuda.Stream()
        lease = self.cache.lease('0:1')
        with torch.cuda.stream(owning):
            lease.__enter__()
            self.cache.arena[:4096].fill_(7)
        lease.__exit__(None,None,None)
        event = self.entry.events[-1]
        self.assertEqual(event.stream.cuda_stream, owning.cuda_stream)
        event.synchronize()
        self.assertTrue(event.query())
        self.assertTrue(torch.equal(self.cache.arena[:4096].cpu(), torch.full((4096,),7,dtype=torch.uint8)))
        self.assertEqual(self.entry.users,0)

    def test_concurrent_leases_keep_distinct_streams(self):
        a,b = torch.cuda.Stream(),torch.cuda.Stream()
        first,second = self.cache.lease('0:1'),self.cache.lease('0:1')
        with torch.cuda.stream(a):
            first.__enter__();self.cache.arena[:2048].fill_(3)
        with torch.cuda.stream(b):
            second.__enter__();self.cache.arena[2048:4096].fill_(5)
        second.__exit__(None,None,None);first.__exit__(None,None,None)
        events = list(self.entry.events)
        self.assertEqual([e.stream.cuda_stream for e in events],[b.cuda_stream,a.cuda_stream])
        for event in events:event.synchronize()
        self.assertTrue(torch.equal(self.cache.arena[:4096].cpu(),torch.cat([torch.full((2048,),3,dtype=torch.uint8),torch.full((2048,),5,dtype=torch.uint8)])))
        self.assertEqual(self.entry.users,0)

    def test_eviction_waits_for_real_events_before_region_release(self):
        owning = torch.cuda.Stream();lease=self.cache.lease('0:1')
        with torch.cuda.stream(owning):
            lease.__enter__();self.cache.arena[:4096].fill_(11)
        lease.__exit__(None,None,None)
        events=list(self.entry.events)
        self.assertEqual(events[0].stream.cuda_stream,owning.cuda_stream)
        region=self.cache.regions[0];release=region.release
        def checked_release(offset):
            self.assertTrue(all(e.synchronized and e.query() for e in events))
            return release(offset)
        with patch.object(region,'release',checked_release):self.cache._evict('0:1')
        self.assertNotIn('0:1',self.cache.entries)
        self.assertEqual(self.cache.resident_bytes,0)
        self.assertTrue(torch.equal(self.cache.arena[:4096].cpu(),torch.full((4096,),11,dtype=torch.uint8)))

if __name__ == '__main__':unittest.main()
