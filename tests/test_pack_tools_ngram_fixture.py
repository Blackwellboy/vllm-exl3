"""CPU tests for the pack tools' n-gram hand-off, driven by a synthetic MoE pack.

``qwen_pack_scan.py`` reads safetensors *headers only*, and ``qwen_pack_config.py``
consumes the scan JSON, so a header-only pack plus a small ``config.json`` exercises
the whole path with no weights and no GPU. That is the gap this file closes: the
PR changed what the config tool emits for ``ngram_embedding`` (``sharded`` /
``num_shards: 1`` for the exllamav3-1.5.0 unsharded layout) and nothing covered it.

Deliberately does not import vllm or torch, so it runs in the host CPU venv.
"""

import json
import os
import re
import struct
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.join(_HERE, "..", "tools", "exl3_pack_tools")
_SCAN = os.path.join(_TOOLS, "qwen_pack_scan.py")
_CONFIG = os.path.join(_TOOLS, "qwen_pack_config.py")

TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
BITS, HEADS, ROWS = 3, 2, 64
WORDS = 1 + 160 * BITS // 16
ROWS_PER_SHARD = 20


def _write_pack(pack: str, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    """One safetensors file, header exact and payload zero-filled to the declared size."""
    dtypes = {"I16": 2, "F16": 2, "I64": 8}
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = dtypes[dtype]
        for d in shape:
            n *= d
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(os.path.join(pack, "ngram_embedding.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        f.write(b"\0" * off)


def _aux() -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        f"{TABLE}.head_bias": ("F16", (HEADS, 160)),
        f"{TABLE}.head_offsets": ("I64", (HEADS,)),
        f"{TABLE}.head_vocab_sizes": ("I64", (HEADS,)),
        f"{TABLE}.layer_multipliers": ("I64", (3,)),
    }


def _pack(tmp_path, *, unsharded: bool, config: dict | None = None) -> str:
    """A pack with one n-gram table, three MoE layers and two dense linears."""
    pack = str(tmp_path / "pack")
    os.makedirs(pack, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    if unsharded:
        tensors[f"{TABLE}.trellis"] = ("I16", (ROWS, WORDS))
    else:
        for i in range(2):
            tensors[f"{TABLE}.shard_{i}.trellis"] = ("I16", (ROWS // 2, WORDS))
    tensors.update(_aux())
    # MoE: layers 3 and 4 at K=3 (base), layer 5 at K=2 (a layer_bits override)
    for layer, words in ((3, 48), (4, 48), (5, 32)):
        for expert in range(4):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensors[
                    f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.trellis"
                ] = ("I16", (64, words))
    tensors["model.language_model.layers.0.self_attn.q_proj.trellis"] = ("I16", (160, 32))
    tensors["model.language_model.layers.0.self_attn.o_proj.trellis"] = ("I16", (160, 32))
    tensors["mtp.layers.0.self_attn.q_proj.trellis"] = ("I16", (160, 32))
    _write_pack(pack, tensors)
    cfg = {
        "text_config": {
            "num_hidden_layers": 6,
            "quantization_config": {"quant_method": "exl3", "bits": 4, "codebook": "mcg"},
        }
    }
    cfg.update(config or {})
    json.dump(cfg, open(os.path.join(pack, "config.json"), "w"), indent=2)
    return pack


def _run(tool: str, pack: str, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, tool, pack, *extra], capture_output=True, text=True, check=False
    )


def _scan(pack: str) -> dict:
    r = _run(_SCAN, pack)
    assert r.returncode == 0, r.stderr
    return json.load(open(os.path.join(pack, "pack_scan.json")))


def test_synthetic_moe_pack_scan_reports_moe_dense_and_unsharded_ngram(tmp_path):
    scan = _scan(_pack(tmp_path, unsharded=True))
    assert sorted(scan["expert_k_per_layer"]) == ["3", "4", "5"]
    assert scan["expert_k_per_layer"]["3"] == {"down_proj": [3], "gate_proj": [3], "up_proj": [3]}
    assert scan["expert_k_per_layer"]["5"]["gate_proj"] == [2]
    assert scan["expert_k_nonuniform"] == []
    assert scan["ngram_problems"] == []
    tab = scan["ngram_tables"][TABLE]
    assert (tab["num_shards"], tab["rows_per_shard"], tab["bits"], tab["sharded"]) == (1, ROWS, BITS, False)
    assert "head_offsets" in tab["aux"] and "trellis" not in tab["aux"]


def test_config_tool_emits_unsharded_ngram_spec_from_that_scan(tmp_path):
    pack = _pack(tmp_path, unsharded=True)
    _scan(pack)
    r = _run(_CONFIG, pack, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "'sharded': False" in r.stdout and "'num_shards': 1" in r.stdout

    assert _run(_CONFIG, pack).returncode == 0
    cfg = json.load(open(os.path.join(pack, "config.json")))
    q = cfg["text_config"]["quantization_config"]
    assert q["quant_method"] == "exl3" and q["bits"] == 3  # base K = the mode across MoE layers
    assert q["layer_bits"] == {"5": 2}
    assert q["ngram_embedding"] == {
        "bits": BITS,
        "num_shards": 1,
        "rows_per_shard": ROWS,
        "num_heads": HEADS,
        "modules": ["ngram_embedding"],
        "sharded": False,
    }
    assert cfg["quantization_config"]["ngram_embedding"]["sharded"] is False
    assert os.path.exists(os.path.join(pack, "config.json.native"))
    # dense map carries the checkpoint prefix, the fused vLLM module, the vLLM-side root
    # and the MTP renumbering (checkpoint mtp.layers.0 -> module mtp.layers.<num_layers>)
    layers = q["non_routed_exl3"]["layers"]
    assert {k: v["bits"] for k, v in layers.items()} == {
        f"{root}{leaf}": 2
        for root in ("model.language_model.", "language_model.model.", "model.")
        for leaf in ("layers.0.self_attn.q_proj", "layers.0.self_attn.qkv_proj",
                     "layers.0.self_attn.o_proj")
    } | {"mtp.layers.0.self_attn.q_proj": 2, "mtp.layers.0.self_attn.qkv_proj": 2,
         "mtp.layers.6.self_attn.q_proj": 2, "mtp.layers.6.self_attn.qkv_proj": 2}
    assert q["non_routed_exl3"]["modules"] == [
        "self_attn.o_proj", "self_attn.q_proj", "self_attn.qkv_proj"
    ]

    # re-running starts from the preserved native block instead of nesting it
    assert _run(_CONFIG, pack).returncode == 0
    again = json.load(open(os.path.join(pack, "config.json")))["text_config"]["quantization_config"]
    assert again["ngram_embedding"]["sharded"] is False
    assert again["native_quantization_config"]["quant_method"] == "exl3"


def test_config_tool_keeps_sharded_tables_sharded(tmp_path):
    pack = _pack(tmp_path, unsharded=False)
    assert _scan(pack)["ngram_tables"][TABLE]["sharded"] is True
    assert _run(_CONFIG, pack).returncode == 0
    q = json.load(open(os.path.join(pack, "config.json")))["text_config"]["quantization_config"]
    assert q["ngram_embedding"]["sharded"] is True
    assert q["ngram_embedding"]["num_shards"] == 2


@pytest.mark.parametrize(
    "tables,message",
    [
        (
            {
                "a": {"bits": 3, "num_shards": 1, "rows_per_shard": 64, "sharded": False,
                      "aux": {"head_bias": {"shape": [2, 160]}}},
                "b": {"bits": 3, "num_shards": 1, "rows_per_shard": 64, "sharded": True,
                      "aux": {"head_bias": {"shape": [2, 160]}}},
            },
            "differ in geometry",
        ),
        (
            {
                "a": {"bits": 3, "num_shards": 1, "rows_per_shard": 64, "sharded": False,
                      "aux": {"head_vocab_sizes": {"shape": [2]}}},
            },
            "head_bias",
        ),
        (
            {
                "a": {"bits": 3, "num_shards": 1, "rows_per_shard": 64, "sharded": False,
                      "aux": {"head_bias": {"shape": [2, 160]}}},
                "b": {"bits": 3, "num_shards": 1, "rows_per_shard": 64, "sharded": False,
                      "aux": {"head_bias": {"shape": [3, 160]}}},
            },
            "head count",
        ),
    ],
)
def test_config_tool_refuses_incompatible_ngram_tables(tmp_path, tables, message):
    pack = _pack(tmp_path, unsharded=True)
    scan = _scan(pack)
    scan["ngram_tables"] = tables
    json.dump(scan, open(os.path.join(pack, "pack_scan.json"), "w"))
    r = _run(_CONFIG, pack, "--dry-run")
    assert r.returncode == 2
    assert message in r.stdout
    assert json.load(open(os.path.join(pack, "config.json")))["text_config"][
        "quantization_config"
    ]["quant_method"] == "exl3"  # refused before any rewrite


def test_config_tool_refuses_scan_problems(tmp_path):
    pack = _pack(tmp_path, unsharded=True)
    scan = _scan(pack)
    scan["ngram_problems"] = [f"{TABLE}: shard indices not contiguous"]
    json.dump(scan, open(os.path.join(pack, "pack_scan.json"), "w"))
    r = _run(_CONFIG, pack, "--dry-run")
    assert r.returncode == 2 and "REFUSE" in r.stdout
    assert re.search(r"shard indices not contiguous", r.stdout)
