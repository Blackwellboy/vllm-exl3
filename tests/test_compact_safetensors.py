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
