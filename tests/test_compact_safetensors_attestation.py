"""Publication is fail-closed: the real parser and a destination readback must
both clear *before* the finished file is linked into place.

No CUDA, no model payloads, CPU-only interpreter.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/exl3_pack_tools/compact_safetensors.py"
SPEC = importlib.util.spec_from_file_location("compact_safetensors_attest", PATH)
compact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compact)


def fixture(tmp_path):
    header = {
        "values": {"dtype": "I16", "shape": [3], "data_offsets": [4, 10]},
        "__metadata__": {"format": "pt"},
    }
    raw = compact.encode_header(header)
    source = tmp_path / "source.safetensors"
    source.write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"gap!" + struct.pack("<hhh", -32768, 7, 32767)
    )
    return source, tmp_path / "repaired.safetensors"


def assert_nothing_published(tmp_path, destination):
    assert not destination.exists()
    assert not list(tmp_path.glob(".compact-*"))


def test_unavailable_parser_blocks_publication(tmp_path, monkeypatch):
    """Fail closed: no parser means no publish, not an unverified copy."""
    source, destination = fixture(tmp_path)

    def unavailable(name):
        raise ImportError("safetensors is not installed")

    monkeypatch.setattr(compact.importlib, "import_module", unavailable)
    with pytest.raises(ValueError, match="refusing to publish"):
        compact.compact_file(source, destination)
    assert_nothing_published(tmp_path, destination)


def test_parser_rejection_blocks_publication(tmp_path, monkeypatch):
    source, destination = fixture(tmp_path)

    def rejected(parser, path, header):
        raise ValueError("Parser rejected the copied file: boom")

    monkeypatch.setattr(compact, "parser_layout", rejected)
    with pytest.raises(ValueError, match="rejected the copied file"):
        compact.compact_file(source, destination)
    assert_nothing_published(tmp_path, destination)


def test_corrupted_destination_is_caught_by_readback_before_publish(tmp_path, monkeypatch):
    """The digest that authorizes publication comes from the written file."""
    source, destination = fixture(tmp_path)
    fsync = os.fsync

    def corrupt_after_write(fd):
        os.lseek(fd, -1, os.SEEK_END)
        os.write(fd, b"\xff")
        fsync(fd)

    monkeypatch.setattr(compact.os, "fsync", corrupt_after_write)
    with pytest.raises(ValueError, match="disagree"):
        compact.compact_file(source, destination)
    assert_nothing_published(tmp_path, destination)


def test_verification_runs_on_the_unpublished_file_before_link(tmp_path, monkeypatch):
    source, destination = fixture(tmp_path)
    steps = []
    verify = compact.verify_before_publish
    link = compact.os.link

    def recording_verify(path, header_length, header, receipts):
        steps.append(("verify", os.path.lexists(destination)))
        return verify(path, header_length, header, receipts)

    def recording_link(temporary, target):
        steps.append(("link", Path(temporary).exists()))
        return link(temporary, target)

    monkeypatch.setattr(compact, "verify_before_publish", recording_verify)
    monkeypatch.setattr(compact.os, "link", recording_link)
    receipts = compact.compact_file(source, destination)
    assert [step for step, _ in steps] == ["verify", "link"]
    # At verification time the destination did not exist; the link target did.
    assert steps[0][1] is False
    assert steps[1][1] is True
    assert destination.exists()
    assert [receipt["destination_sha256"] for receipt in receipts] == [
        receipt["sha256"] for receipt in receipts
    ]


def test_receipt_digests_are_confirmed_by_the_parser(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    numpy = pytest.importorskip("numpy")
    from safetensors.numpy import save_file

    tensors = {
        "w1_MUL1": numpy.array([-2082680531, 7], dtype=numpy.int32),
        "empty": numpy.zeros((0,), dtype=numpy.int16),
        "mo\u00e9l.q_a": numpy.arange(6, dtype=numpy.float16).reshape(2, 3),
    }
    source = tmp_path / "valid.safetensors"
    save_file(tensors, str(source), metadata={"format": "pt"})
    destination = tmp_path / "valid.out"
    receipts = compact.compact_file(source, destination)
    assert len(receipts) == len(tensors)
    with safetensors.safe_open(destination, framework="np") as handle:
        for receipt in receipts:
            digest = hashlib.sha256(handle.get_tensor(receipt["name"]).tobytes()).hexdigest()
            assert receipt["sha256"] == digest
            assert receipt["destination_sha256"] == digest


def test_cli_report_attests_parser_and_readback(tmp_path):
    source, destination = fixture(tmp_path)
    run = subprocess.run(
        [sys.executable, str(PATH), str(source), str(destination)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report["parser"]["module"] == "safetensors"
    assert report["parser"]["verified_tensors"] == len(report["tensors"]) == 1
    for receipt in report["tensors"]:
        assert receipt["destination_sha256"] == receipt["sha256"]
        assert len(receipt["sha256"]) == 64
