import importlib.util
import json
from pathlib import Path
import struct
import pytest


def test_source_change_during_readback_refuses_publication(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/exl3_pack_tools/compact_safetensors.py'
    spec = importlib.util.spec_from_file_location('compact_verify_window', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source, destination = tmp_path/'source.safetensors', tmp_path/'new.safetensors'
    raw = json.dumps({'x': {'dtype':'U8', 'shape':[1], 'data_offsets':[0,1]}}).encode()
    raw += b' ' * (-len(raw) % 8)
    source.write_bytes(struct.pack('<Q',len(raw))+raw+b'x')
    verify = mod.verify_before_publish
    def change_after_verification(*args):
        result = verify(*args)
        with source.open('ab') as f:
            f.write(b'y')
        return result
    monkeypatch.setattr(mod, 'verify_before_publish', change_after_verification)
    with pytest.raises(ValueError, match='Source changed'):
        mod.compact_file(source,destination)
    assert not destination.exists()
