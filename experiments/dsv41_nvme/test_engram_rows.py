"""Synthetic raw-row caching tests; real Engram numerical gates are separate."""
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from engram_rows import EngramRows
from posix_support import requires_posix_reads


@requires_posix_reads
class EngramRowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=Path(self.temp.name)
        self.w=root/'weights';self.s=root/'scales'
        self.w.write_bytes(b'HEADER'+b''.join(bytes([i])*256 for i in range(256)))
        self.s.write_bytes(b'SCALES'+b''.join(bytes([255-i])*8 for i in range(256)))
        self.wfd=os.open(self.w,os.O_RDONLY);self.sfd=os.open(self.s,os.O_RDONLY)
        self.cache=EngramRows(self.wfd,6,self.sfd,6,256,256,8,capacity_rows=16)

    def tearDown(self):
        self.cache.close();os.close(self.wfd);os.close(self.sfd);self.temp.cleanup()

    def gather(self,ids,owned=None):
        if owned is None:owned=[True]*len(ids)
        weight=bytearray(len(ids)*256);scale=bytearray(len(ids)*8)
        self.cache.gather_into(ids,owned,memoryview(weight),memoryview(scale))
        for i,(key,keep) in enumerate(zip(ids,owned)):
            self.assertEqual(weight[i*256:(i+1)*256],bytes([key if keep else 0])*256)
            self.assertEqual(scale[i*8:(i+1)*8],bytes([255-key if keep else 0])*8)

    def test_duplicates_and_ownership_avoid_redundant_reads(self):
        self.gather([3,3,4,5],[True,True,False,True])
        self.assertEqual(self.cache.stats['unique_misses'],2)
        self.assertEqual(self.cache.stats['disk_weight_bytes'],512)
        self.assertNotIn(4,self.cache.cache)
        self.gather([5,3,3])
        self.assertEqual(self.cache.stats['unique_misses'],2)
        self.assertEqual(self.cache.stats['unique_hits'],2)

    def test_request_larger_than_cache_and_churn_keep_all_output_bytes(self):
        self.gather(list(range(64)))
        self.assertEqual(len(self.cache.cache),16)
        self.assertEqual(self.cache.stats['evictions'],48)
        self.gather([0,63,32,0])
        self.assertLessEqual(self.cache.stats['peak_cached_rows'],16)

    def test_prefetch_returns_while_workers_read_and_only_one_is_pending(self):
        real=os.pread;entered=threading.Event();release=threading.Event()
        def held(*args):
            entered.set()
            if not release.wait(timeout=3):raise TimeoutError('Test did not release reader')
            return real(*args)
        try:
            with patch('engram_rows.os.pread',held):
                self.cache.prefetch(list(range(64)),[True]*64)
                self.assertTrue(entered.wait(timeout=2))
                self.assertFalse(self.cache.pending[1].done())
                with self.assertRaises(RuntimeError):self.cache.prefetch([1],[True])
                release.set()
                self.gather(list(range(64)))
            self.assertEqual(self.cache.stats['prefetch_consumed'],1)
        finally:release.set()

    def test_unmatched_prefetch_does_not_supply_wrong_rows(self):
        self.cache.prefetch([1,2],[True,True])
        self.gather([5,6])
        self.assertEqual(self.cache.stats['prefetch_unused'],1)

    def _mutate_one_payload_byte(self):
        """Same-size in-place payload mutation with a deterministic time stamp.

        `EngramRows` holds the identity tuple (st_dev, st_ino, st_size,
        st_mtime_ns, st_ctime_ns) taken with fstat at construction. Linux stamps
        inode times from a coarse (timer-tick) clock, so a write that lands in the
        same tick as file creation leaves the whole tuple unchanged and the guard
        cannot see it; relying on a later tick is a race, not a check. Pinning
        st_mtime_ns explicitly to a value one second away makes the tuple change
        on any filesystem, including second-granularity ones, without sleeping.
        """
        recorded=EngramRows._identity(self.cache.w_fd)
        with self.w.open('r+b') as source:
            source.seek(6+3*256);source.write(b'x');source.flush();os.fsync(source.fileno())
        os.utime(self.w,ns=(recorded[3]-10**9,recorded[3]-10**9))
        observed=EngramRows._identity(self.cache.w_fd)
        # If this ever fires the write stopped being size-preserving and the test
        # would be proving the wrong thing.
        self.assertEqual(observed[2],recorded[2])
        if observed[3]==recorded[3]:
            self.skipTest('filesystem does not persist an explicit nanosecond mtime')

    def test_changed_source_is_rejected_even_for_cache_hit(self):
        self.gather([3])
        self._mutate_one_payload_byte()
        with self.assertRaises(ValueError):self.gather([3])

    def test_source_size_change_is_rejected_even_for_cache_hit(self):
        # Stat-only identity catches a size change with no timestamp dependency
        # at all, so this path stays deterministic even where st_mtime_ns is
        # quantized to seconds and the same-tick case above is undetectable.
        self.gather([3])
        with self.w.open('ab') as source:source.write(b'z')
        with self.assertRaises(ValueError):self.gather([3])

    def test_short_read_and_invalid_request_are_rejected(self):
        with patch('engram_rows.os.pread',return_value=b''):
            with self.assertRaises(OSError):self.gather([4])
        self.assertEqual(len(self.cache.cache),0)
        for ids,owned in (([-1],[False]),([256],[True]),([1],[1]),([1]*8193,[True]*8193)):
            with self.assertRaises(ValueError):self.gather(ids,owned)

    def test_close_releases_cache_and_owned_descriptors(self):
        self.cache.prefetch([3],[True]);self.cache.close()
        self.assertEqual(len(self.cache.cache),0)
        with self.assertRaises(OSError):os.fstat(self.cache.w_fd)
        self.assertGreater(os.fstat(self.wfd).st_size,0)
        with self.assertRaises(RuntimeError):self.cache.prefetch([3],[True])


if __name__=='__main__':unittest.main()
