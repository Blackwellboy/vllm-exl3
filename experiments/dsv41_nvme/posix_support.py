"""Platform capabilities shared by the NVMe component checks.

The extracted transports call POSIX `os.pread`/`os.preadv` directly and never
silently substitute a different syscall, so a platform without them cannot
exercise these components. Those checks skip explicitly here instead of
failing on an AttributeError, which keeps a non-POSIX run honest: skips are
visible, not counted as passes.
"""
import os
import unittest

POSIX_READS = hasattr(os, "pread") and hasattr(os, "preadv")
DIRECT_IO = POSIX_READS and hasattr(os, "O_DIRECT")

requires_posix_reads = unittest.skipUnless(
    POSIX_READS, "requires POSIX os.pread/os.preadv (unsupported on this platform)")
