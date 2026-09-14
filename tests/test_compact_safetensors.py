"""No model download or CUDA needed; readback uses the real safetensors parser."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
from unittest.mock import patch

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/exl3_pack_tools/compact_safetensors.py"
SPEC = importlib.util.spec_from_file_location("compact_safetensors", PATH)
compact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compact)


def fixture(tmp_path):
    header = {
        "marker": {"dtype": "I32", "shape": [], "data_offsets": [4, 8]},
        "empty": {"dtype": "I16", "shape": [0], "data_offsets": [8, 8]},
        "values": {"dtype": "I16", "shape": [3], "data_offsets": [12, 18]},
        "__metadata__": {"format": "pt", "description": "mixed-K MUL1"},
    }
    raw = compact.encode_header(header)
    data = b"gap!" + struct.pack("<i", -2082680531) + b"gap!" + struct.pack("<hhh", -32768, 7, 32767) + b"tail"
    source = tmp_path / "source.safetensors"
    source.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    return source, tmp_path / "repaired.safetensors"


def test_real_parser_readback_preserves_bytes_metadata_and_source(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("numpy")
    source, destination = fixture(tmp_path)
    original = source.read_bytes()
    with pytest.raises(safetensors.SafetensorError):
        with safetensors.safe_open(source, framework="np"):
            pass
    receipts = compact.compact_file(source, destination)
    assert source.read_bytes() == original
    with safetensors.safe_open(destination, framework="np") as f:
        assert set(f.keys()) == {"marker", "empty", "values"}
        assert f.metadata() == {"format": "pt", "description": "mixed-K MUL1"}
        assert f.get_tensor("marker").item() == -2082680531
        assert f.get_tensor("empty").shape == (0,)
        assert f.get_tensor("values").tolist() == [-32768, 7, 32767]
        for receipt in receipts:
            assert hashlib.sha256(f.get_tensor(receipt["name"]).tobytes()).hexdigest() == receipt["sha256"]


@pytest.mark.parametrize("descriptor", [
    {"dtype": "I32", "shape": [], "data_offsets": [2, 6]},
    {"dtype": "I32", "shape": [], "data_offsets": [8, 12]},
    {"dtype": "I32", "shape": [2], "data_offsets": [4, 8]},
    {"dtype": "I32", "shape": [True], "data_offsets": [4, 8]},
    {"dtype": "I32", "shape": [], "data_offsets": [4.0, 8]},
    {"dtype": "unknown", "shape": [], "data_offsets": [4, 8]},
])
def test_invalid_extents_and_types_are_rejected(descriptor):
    with pytest.raises(ValueError):
        compact.layout({"a": {"dtype": "I32", "shape": [], "data_offsets": [0, 4]}, "b": descriptor}, 8)


@pytest.mark.parametrize("payload", [b"", b"abc", struct.pack("<Q", 100) + b"{}"])
def test_truncated_header_never_creates_destination(tmp_path, payload):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(payload)
    with pytest.raises(ValueError):
        compact.compact_file(source, destination)
    assert not destination.exists()


def test_duplicate_keys_and_invalid_metadata_are_rejected(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    raw = b'{"a":{},"a":{}}'
    source.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ValueError, match="Duplicate"):
        compact.compact_file(source, destination)
    with pytest.raises(ValueError, match="Metadata"):
        compact.layout({"__metadata__": {"a": 3}}, 0)


def test_existing_source_destination_and_symlink_are_preserved(tmp_path):
    source, destination = fixture(tmp_path)
    original = source.read_bytes()
    for target in (source, destination):
        if target == destination:
            target.symlink_to(tmp_path / "absent")
        with pytest.raises(ValueError, match="new file"):
            compact.compact_file(source, target)
    assert source.read_bytes() == original
    assert destination.is_symlink()


def test_publish_race_does_not_overwrite_or_leave_temporary_file(tmp_path):
    source, destination = fixture(tmp_path)
    link = compact.os.link

    def concurrent_publish(temporary, target):
        destination.write_bytes(b"other writer")
        link(temporary, target)

    with patch.object(compact.os, "link", concurrent_publish):
        with pytest.raises(FileExistsError):
            compact.compact_file(source, destination)
    assert destination.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".compact-*"))


def test_cli_emits_receipts_and_refuses_second_copy(tmp_path):
    source, destination = fixture(tmp_path)
    command = [sys.executable, str(PATH), str(source), str(destination)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert len(json.loads(run.stdout)["tensors"]) == 3
    assert subprocess.run(command, capture_output=True).returncode == 2


def gapped_fixture(tmp_path, layout, payload, metadata=None):
    """Build a source whose payload contains gaps/trailing bytes around the tensors."""
    header = {
        name: {"dtype": dtype, "shape": shape, "data_offsets": list(offsets)}
        for name, dtype, shape, offsets in layout
    }
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = compact.encode_header(header)
    source = tmp_path / "gapped.safetensors"
    source.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return source, tmp_path / "gapped.out"


def read_header_bytes(path):
    body = path.read_bytes()
    n = struct.unpack("<Q", body[:8])[0]
    return json.loads(body[8:8 + n]), len(body) - 8 - n


@pytest.mark.parametrize("layout", [
    [("a", "I32", [2], (0, 8)), ("b", "I32", [2], (4, 12))],
    [("a", "I32", [2], (0, 8)), ("z", "I8", [0], (4, 4))],
])
def test_overlapping_source_is_rejected_by_tool_and_parser(tmp_path, layout):
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("numpy")
    source, destination = gapped_fixture(
        tmp_path, layout, struct.pack("<ii", 1, 2) + struct.pack("<ii", 3, 4))
    with pytest.raises(safetensors.SafetensorError):
        with safetensors.safe_open(source, framework="np"):
            pass
    with pytest.raises(ValueError, match="Overlapping"):
        compact.compact_file(source, destination)
    assert not destination.exists()


def test_copy_discards_only_unreferenced_padding(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("numpy")
    a, b = struct.pack("<ii", -2082680531, 7), struct.pack("<hhh", -32768, 0, 32767)
    payload = a + b"\x00" * 8 + b + b"trailing"
    source, destination = gapped_fixture(tmp_path, [
        ("a", "I32", [2], (0, 8)),
        ("empty", "I16", [0], (8, 8)),
        ("b", "I16", [3], (16, 22)),
    ], payload, metadata={"format": "pt"})
    original = source.read_bytes()
    receipts = compact.compact_file(source, destination)
    assert source.read_bytes() == original
    header, data_bytes = read_header_bytes(destination)
    assert header == {
        "a": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]},
        "empty": {"dtype": "I16", "shape": [0], "data_offsets": [8, 8]},
        "b": {"dtype": "I16", "shape": [3], "data_offsets": [8, 14]},
        "__metadata__": {"format": "pt"},
    }
    assert data_bytes == len(a) + len(b)
    assert [receipt["name"] for receipt in receipts] == ["a", "empty", "b"]
    with safetensors.safe_open(destination, framework="np") as handle:
        assert handle.get_tensor("a").tobytes() == a
        assert handle.get_tensor("b").tobytes() == b
        assert handle.get_tensor("empty").shape == (0,)
        assert handle.metadata() == {"format": "pt"}


def test_valid_shard_is_copied_without_layout_churn(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    numpy = pytest.importorskip("numpy")
    from safetensors.numpy import save_file
    tensors = {
        "w1_MUL1": numpy.array([-2082680531, 7], dtype=numpy.int32),
        "scalar": numpy.array(1.25, dtype=numpy.float32),
        "empty": numpy.zeros((0,), dtype=numpy.int16),
        "mo\u00e9l.q_a": numpy.arange(6, dtype=numpy.float16).reshape(2, 3),
    }
    source = tmp_path / "valid.safetensors"
    save_file(tensors, str(source), metadata={"format": "pt", "note": "mixed-K"})
    original = source.read_bytes()
    destination = tmp_path / "valid.out"
    compact.compact_file(source, destination)
    assert source.read_bytes() == original
    assert destination.stat().st_size == source.stat().st_size
    with safetensors.safe_open(destination, framework="np") as handle:
        assert set(handle.keys()) == set(tensors)
        assert handle.metadata() == {"format": "pt", "note": "mixed-K"}
        for name, tensor in tensors.items():
            copied = handle.get_tensor(name)
            assert copied.dtype == tensor.dtype
            assert copied.shape == tensor.shape
            assert copied.tobytes() == tensor.tobytes()


# safetensors exposes these through NumPy; BF16/F8 have no NumPy dtype.
NUMPY_READABLE = {"U8", "I8", "I16", "U16", "I32", "U32", "I64", "U64", "F16", "F32", "F64"}


def test_every_supported_dtype_round_trips_through_the_real_parser(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    pytest.importorskip("numpy")
    layout, payload, slices, offsets = [], b"", {}, 0
    for dtype, width in sorted(compact.WIDTH.items()):
        layout.append((dtype, dtype, [2], (offsets, offsets + 2 * width)))
        payload += bytes(range(1, 2 * width + 1))
        slices[dtype] = payload[offsets:offsets + 2 * width]
        offsets += 2 * width
    source, destination = gapped_fixture(tmp_path, layout, payload)
    compact.compact_file(source, destination)
    with safetensors.safe_open(destination, framework="np") as handle:
        assert set(handle.keys()) == set(compact.WIDTH)
        for dtype in sorted(NUMPY_READABLE):
            copied = handle.get_tensor(dtype)
            assert copied.shape == (2,)
            assert copied.tobytes() == slices[dtype]


def test_cli_reports_compaction_accounting(tmp_path):
    source, destination = fixture(tmp_path)
    command = [sys.executable, str(PATH), str(source), str(destination)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["source_bytes"] == source.stat().st_size
    assert report["destination_bytes"] == destination.stat().st_size
    assert report["discarded_bytes"] == report["source_bytes"] - report["destination_bytes"]
    assert report["discarded_bytes"] > 0
