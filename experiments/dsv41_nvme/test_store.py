import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from expert_store import ALIGN, ExpertStore, Regions, build_bank
from posix_support import requires_posix_reads


class RegionsTests(unittest.TestCase):
    def test_fragmentation_coalescing_and_double_free(self):
        r=Regions(4*ALIGN)
        a=r.allocate(ALIGN); b=r.allocate(2*ALIGN); c=r.allocate(ALIGN)
        self.assertIsNone(r.allocate(1))
        r.release(b);self.assertIsNone(r.allocate(3*ALIGN))
        r.release(c);self.assertEqual(r.allocate(3*ALIGN),b)
        r.release(a)
        with self.assertRaises(KeyError):r.release(a)


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.bank=Path(cls.tmp.name)/'bank'
        from test_async_store import AsyncStoreTests
        AsyncStoreTests.setUpClass()
        cls.bank = AsyncStoreTests.bank
        cls.digest = AsyncStoreTests.digest


    @classmethod
    def tearDownClass(cls):
        from test_async_store import AsyncStoreTests
        AsyncStoreTests.tearDownClass()
        cls.tmp.cleanup()

    @requires_posix_reads
    def test_every_tensor_byte_hash_matches(self):
        s=ExpertStore(self.bank,self.digest,direct=False)
        try:
            for key,r in s.records.items():
                with s.read(key) as b:
                    for t in r['tensors'].values():
                        self.assertEqual(hashlib.sha256(b[t['offset']:t['offset']+t['bytes']]).hexdigest(),t['sha256'])
            self.assertEqual(s.active_reads,0)
            self.assertLessEqual(s.peak_staging_bytes,64*2**20)
        finally:s.close()

    def test_bad_identity_and_cancel(self):
        with self.assertRaises(ValueError):ExpertStore(self.bank,'0'*64,direct=False)
        s=ExpertStore(self.bank,self.digest,direct=False);cancel=threading.Event();cancel.set()
        try:
            with self.assertRaises(InterruptedError):
                with s.read('0:1',cancel=cancel):pass
            self.assertEqual(s.reads,0)
        finally:s.close()

    @requires_posix_reads
    def test_interrupted_syscall_retries_and_eof_fails(self):
        import os
        s=ExpertStore(self.bank,self.digest,direct=False);real=os.pread;calls=[0]
        def interrupted(*a):
            calls[0]+=1
            if calls[0]==1:raise InterruptedError()
            return real(*a)
        try:
            with patch('expert_store.os.pread',interrupted):
                with s.read('0:1'):pass
            with patch('expert_store.os.pread',return_value=b''):
                with self.assertRaises(OSError):
                    with s.read('0:1'):pass
            self.assertEqual(s.active_reads,0)
        finally:s.close()

    @requires_posix_reads
    def test_corrupt_record_rejected(self):
        import os
        s=ExpertStore(self.bank,self.digest,direct=False);real=os.pread
        def corrupt(*args):
            b=bytearray(real(*args))
            if b:b[0]^=1
            return bytes(b)
        try:
            with patch('expert_store.os.pread',corrupt):
                with self.assertRaises(ValueError):
                    with s.read('0:1'):pass
        finally:s.close()


if __name__=='__main__':unittest.main()
