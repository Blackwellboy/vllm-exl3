"""Regression: MADV after H2D must use tensor VIEW range, not base storage.

P1 on vcruz305/vllm-exl3#14: ``untyped_storage().data_ptr()/nbytes()`` covers
later unconsumed views of a fused shard and can zero them on heap (or force
re-fault / negate reclaim on safetensors MAP_PRIVATE).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vllm_exl3.exl3 import (  # noqa: E402
    _find_containing_vma,
    _madv_dontneed_cpu_tensor,
    direct_fill_stats,
)


PAGE = os.sysconf("SC_PAGESIZE")
MADV_DONTNEED = 4


def _sha(t: torch.Tensor) -> str:
    u8 = t.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(u8.numpy().tobytes()).hexdigest()


def _old_madv_storage_range(src: torch.Tensor) -> bool:
    """Pre-fix helper: advise base storage span (the P1 bug)."""
    ptr = int(src.untyped_storage().data_ptr())
    nbytes = int(src.untyped_storage().nbytes())
    start = ptr + ((PAGE - (ptr % PAGE)) % PAGE)
    end = (ptr + nbytes) - ((ptr + nbytes) % PAGE)
    if end <= start:
        return False
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    return libc.madvise(ctypes.c_void_p(start), ctypes.c_size_t(end - start), MADV_DONTNEED) == 0


def test_old_helper_corrupts_later_heap_view():
    """Prove the P1: storage-range MADV zeros a later contiguous view."""
    n = PAGE * 8
    base = torch.empty(n, dtype=torch.uint8)
    base[:] = (torch.arange(n) % 251).to(torch.uint8)
    piece0 = base.narrow(0, 0, PAGE * 2)
    piece1 = base.narrow(0, PAGE * 2, PAGE * 2)
    h1 = _sha(piece1)
    assert piece0.storage_offset() == 0
    assert piece1.storage_offset() == PAGE * 2
    assert piece0.untyped_storage().nbytes() == n
    assert _old_madv_storage_range(piece0)
    # Heap MADV_DONTNEED → subsequent reads are zero-filled.
    assert int((piece1 == 0).sum()) == piece1.numel()
    assert _sha(piece1) != h1


def test_new_helper_skips_heap_and_preserves_later_view():
    """New helper refuses non-safetensors VMAs; later views stay intact."""
    n = PAGE * 8
    base = torch.empty(n, dtype=torch.uint8)
    base[:] = (torch.arange(n) % 251).to(torch.uint8)
    piece0 = base.narrow(0, PAGE, PAGE * 2)  # nonzero storage offset
    piece1 = base.narrow(0, PAGE * 3, PAGE * 2)
    h0, h1 = _sha(piece0), _sha(piece1)
    os.environ["VLLM_EXL3_MADV_AFTER_H2D"] = "1"
    asserted = _madv_dontneed_cpu_tensor(piece0)
    assert asserted is False  # heap VMA → skip
    assert _sha(piece0) == h0
    assert _sha(piece1) == h1


def test_new_helper_skips_noncontiguous():
    n = PAGE * 4
    base = torch.arange(n, dtype=torch.int32)
    view = base[::2]
    assert not view.is_contiguous()
    assert _madv_dontneed_cpu_tensor(view) is False


def test_new_helper_zero_offset_and_subpage_safe():
    """Sub-page and zero-offset views must not SEGV or widen the range."""
    # Sub-page: fewer than one full page of view bytes → no advise (inward align).
    small = torch.arange(16, dtype=torch.uint8)
    assert _madv_dontneed_cpu_tensor(small) is False
    # Zero-offset multi-page heap still skipped (not safetensors).
    big = torch.arange(PAGE * 2, dtype=torch.uint8)
    assert big.storage_offset() == 0
    assert _madv_dontneed_cpu_tensor(big) is False


def test_find_containing_vma_resolves_heap():
    t = torch.arange(PAGE, dtype=torch.uint8)
    vma = _find_containing_vma(int(t.data_ptr()))
    assert vma is not None
    lo, hi, path = vma
    assert lo <= int(t.data_ptr()) < hi
    # Anonymous heap mappings have empty pathname.
    assert path == "" or not path.endswith(".safetensors")


@pytest.mark.skipif(
    not Path("/var/tmp/models/DSV4.1-Flash-EXL3-4.75bpw").is_dir(),
    reason="real EXL3 checkpoint not mounted on this host",
)
def test_safetensors_view_madv_preserves_later_piece():
    """File-backed: advise piece0 view only; piece1 bytes unchanged."""
    pytest.importorskip("safetensors")
    from safetensors.torch import safe_open

    shard = Path(
        "/var/tmp/models/DSV4.1-Flash-EXL3-4.75bpw/model-00001-of-00032.safetensors"
    )
    assert shard.is_file()
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # Prefer a multi-page tensor we can split into contiguous views.
        key = None
        for k in keys:
            t = f.get_tensor(k)
            if t.is_contiguous() and t.numel() * t.element_size() >= PAGE * 4:
                key = k
                break
        assert key is not None
        tensor = f.get_tensor(key)

    flat = tensor.view(-1).contiguous()
    # Re-open via byte views on the same storage when possible.
    # Build two adjacent views covering distinct page ranges of `flat`.
    elems_per_page = max(1, PAGE // flat.element_size())
    n0 = elems_per_page * 2
    n1 = elems_per_page * 2
    assert flat.numel() >= n0 + n1
    piece0 = flat.narrow(0, 0, n0)
    piece1 = flat.narrow(0, n0, n1)
    assert piece0.is_contiguous() and piece1.is_contiguous()

    vma = _find_containing_vma(int(piece0.data_ptr()))
    # safetensors may copy to heap depending on version; if not file-backed, skip.
    if vma is None or not vma[2].endswith(".safetensors"):
        pytest.skip(f"tensor not file-backed safetensors VMA: {vma}")

    h1 = _sha(piece1)
    # Simulate H2D of piece0 then new MADV.
    if torch.cuda.is_available():
        dest = torch.empty_like(piece0, device="cuda")
        dest.copy_(piece0)
        torch.cuda.current_stream().synchronize()
    else:
        dest = piece0.clone()
    _ = dest
    before_calls = direct_fill_stats().get("MADV_AFTER_H2D_CALLS", 0)
    ok = _madv_dontneed_cpu_tensor(piece0)
    assert ok is True
    assert direct_fill_stats()["MADV_AFTER_H2D_CALLS"] >= before_calls + 1
    assert _sha(piece1) == h1
