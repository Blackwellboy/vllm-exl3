"""CPU tests for the two n-gram table layouts (``shard_<i>.trellis`` and one unsharded
``trellis``) and the two table homes (resident tensor, memory-mapped checkpoint views)."""

import importlib.util
import json
import os
import struct
import subprocess
import sys

import pytest
import torch

pytest.importorskip("vllm")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
_EXL3_PATH = os.path.join(_ROOT, "src", "vllm_exl3", "exl3.py")
_SCAN = os.path.join(_ROOT, "tools", "exl3_pack_tools", "qwen_pack_scan.py")

_spec = importlib.util.spec_from_file_location("_exl3_ngram_layouts", _EXL3_PATH)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

ROOT = "model.layers.1.ple.ple_embedding.ngram_embedding"
BITS, HEADS, ROWS = 3, 2, 64
WORDS = X.ngram_words_per_row(BITS)


def _write_safetensors(path: str, tensors: dict[str, torch.Tensor]) -> None:
    header, blobs, off = {}, [], 0
    dtypes = {torch.int16: "I16", torch.float16: "F16", torch.int64: "I64"}
    for name, t in tensors.items():
        b = t.contiguous().numpy().tobytes()
        header[name] = {"dtype": dtypes[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def _table(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-32768, 32767, (ROWS, WORDS), generator=g, dtype=torch.int16)


def _aux() -> dict[str, torch.Tensor]:
    return {
        f"{ROOT}.head_bias": torch.zeros(HEADS, X.NGRAM_ROW_DIM, dtype=torch.float16),
        f"{ROOT}.head_offsets": torch.tensor([0, ROWS // 2], dtype=torch.int64),
        f"{ROOT}.head_vocab_sizes": torch.tensor([ROWS // 2, ROWS // 2], dtype=torch.int64),
        f"{ROOT}.layer_multipliers": torch.tensor([3, 5, 7], dtype=torch.int64),
    }


def _run_scan(pack: str) -> dict:
    subprocess.run([sys.executable, _SCAN, pack], check=True, capture_output=True)
    return json.load(open(os.path.join(pack, "pack_scan.json")))


def test_scan_reports_sharded_layout(tmp_path):
    t = _table()
    tensors = {f"{ROOT}.shard_0.trellis": t[: ROWS // 2], f"{ROOT}.shard_1.trellis": t[ROWS // 2 :]}
    tensors.update(_aux())
    _write_safetensors(str(tmp_path / "ngram_embedding.safetensors"), tensors)
    scan = _run_scan(str(tmp_path))
    tab = scan["ngram_tables"][ROOT]
    assert (tab["num_shards"], tab["rows_per_shard"], tab["bits"], tab["sharded"]) == (2, ROWS // 2, BITS, True)
    assert scan["ngram_problems"] == []


def test_scan_reports_unsharded_layout_and_keeps_it_out_of_dense_map(tmp_path):
    tensors = {f"{ROOT}.trellis": _table()}
    tensors.update(_aux())
    # a dense linear trellis without aux siblings must not be mistaken for a table
    tensors["model.layers.1.linear_attn.in_proj_qkv.trellis"] = torch.zeros(160, 48, dtype=torch.int16)
    _write_safetensors(str(tmp_path / "ngram_embedding.safetensors"), tensors)
    scan = _run_scan(str(tmp_path))
    tab = scan["ngram_tables"][ROOT]
    assert (tab["num_shards"], tab["rows_per_shard"], tab["bits"], tab["sharded"]) == (1, ROWS, BITS, False)
    assert "head_offsets" in tab["aux"] and "trellis" not in tab["aux"]
    assert scan["ngram_problems"] == []
    assert list(scan["ngram_tables"]) == [ROOT]


def _method(spec_extra: dict, table_mode: str, monkeypatch):
    monkeypatch.setenv(X.NGRAM_TABLE_ENV, table_mode)
    monkeypatch.setenv("VLLM_EXL3_NGRAM_KERNEL", "torch")
    monkeypatch.setattr(X, "_resolve_tp_geometry", lambda layer: (0, 1))
    monkeypatch.setattr(X, "_check_ngram_disk_graph_mode", lambda: None)
    cfg = X.Exl3Config(bits=3, codebook="mul1")
    spec = {"bits": BITS, "num_shards": 2, "rows_per_shard": ROWS // 2, "num_heads": HEADS}
    spec.update(spec_extra)
    return X.Exl3EmbeddingMethod(cfg, spec)


def _build(method, table: torch.Tensor, sharded: bool):
    layer = torch.nn.Module()
    method.create_weights(layer, X.NGRAM_ROW_DIM, [ROWS], X.NGRAM_ROW_DIM, ROWS, torch.float16)
    if sharded:
        half = ROWS // 2
        layer.shard_0.trellis.weight_loader(layer.shard_0.trellis, table[:half])
        layer.shard_1.trellis.weight_loader(layer.shard_1.trellis, table[half:])
    else:
        layer.trellis.weight_loader(layer.trellis, table)
    aux = _aux()
    for name in ("head_bias", "head_offsets", "head_vocab_sizes", "layer_multipliers"):
        p = getattr(layer, name)
        p.weight_loader(p, aux[f"{ROOT}.{name}"])
    method.process_weights_after_loading(layer)
    return layer


def _lookup(method, layer, ids):
    return method._embedding_impl(layer, ids)


def test_unsharded_and_disk_lookups_match_the_resident_sharded_table(monkeypatch):
    table = _table(1)
    ids = torch.tensor([[0, 5, 31, 32, 63, 5]], dtype=torch.long)

    ref = _method({}, "resident", monkeypatch)
    ref_out = _lookup(ref, _build(ref, table, sharded=True), ids)
    assert ref_out.shape == (1, 6, X.NGRAM_ROW_DIM)
    assert torch.equal(ref_out[0, 1], ref_out[0, 5])

    uns = _method({"num_shards": 1, "rows_per_shard": ROWS, "sharded": False}, "resident", monkeypatch)
    assert torch.equal(_lookup(uns, _build(uns, table, sharded=False), ids), ref_out)

    disk = _method({}, "disk", monkeypatch)
    layer = _build(disk, table, sharded=True)
    assert getattr(layer, "_exl3_ngram_table", None) is None
    assert layer._exl3_ngram_rows is None
    assert torch.equal(_lookup(disk, layer, ids), ref_out)

    disk_uns = _method({"num_shards": 1, "rows_per_shard": ROWS, "sharded": False}, "disk", monkeypatch)
    assert torch.equal(_lookup(disk_uns, _build(disk_uns, table, sharded=False), ids), ref_out)


def test_disk_table_routes_rows_across_shards():
    t = _table(2)
    # every shard but the last holds rows_per_shard rows; the last may be short
    d = X._NgramDiskTable([t[:20], t[20:40], t[40:60], t[60:]], rows_per_shard=20)
    uids = torch.tensor([0, 19, 20, 39, 40, 63], dtype=torch.long)
    assert torch.equal(d.gather(uids), t.index_select(0, uids))
    assert d.num_rows == ROWS


def test_only_the_disk_table_asks_for_the_caller_allocated_output(monkeypatch):
    """The opt-in disk mode needs the out-variant lookup; the default resident table
    keeps the returning lookup op, so the mainline serving path is untouched."""
    table = _table(3)
    resident = _method({}, "resident", monkeypatch)
    assert resident._ngram_lookup_uses_out_variant(_build(resident, table, sharded=True)) is False
    resident_uns = _method({"num_shards": 1, "rows_per_shard": ROWS, "sharded": False}, "resident", monkeypatch)
    assert resident_uns._ngram_lookup_uses_out_variant(
        _build(resident_uns, table, sharded=False)
    ) is False
    disk = _method({}, "disk", monkeypatch)
    assert disk._ngram_lookup_uses_out_variant(_build(disk, table, sharded=True)) is True


def test_unsharded_spec_requires_one_shard(monkeypatch):
    with pytest.raises(ValueError):
        _method({"sharded": False}, "resident", monkeypatch)


def test_table_env_is_validated(monkeypatch):
    monkeypatch.setenv(X.NGRAM_TABLE_ENV, "nvme")
    with pytest.raises(ValueError):
        X.Exl3EmbeddingMethod(X.Exl3Config(bits=3, codebook="mul1"),
                              {"bits": BITS, "num_shards": 1, "rows_per_shard": ROWS, "num_heads": HEADS})
