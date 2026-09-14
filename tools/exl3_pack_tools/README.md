# tools/exl3_pack_tools

Tools for pointing this plugin at a pack quantized directly by turboderp's own
ExLlamaV3 tooling ("native" packs), rather than one produced by this
project's own conversion path. A native pack can carry a fractional average
bit width (e.g. 3.05 bpw) by assigning bits per tensor, so it needs its
`config.json` rewritten into the vocabulary the plugin already understands
before vLLM can load it.

- `qwen_pack_scan.py` inventories every tensor in the pack by reading each
  `*.safetensors` header only (no tensor data). It reports, per MoE layer,
  the set of K values across experts for gate/up/down (non-uniform expert K
  is supported via ragged trellis storage + python_loop); per dense linear
  (attention, dense MLP, shared expert, `lm_head`), its K; and for each
  row-wise n-gram embedding table, its shard count, row count, K and
  auxiliary tensors. Writes `pack_scan.json` next to the pack.

  `python3 qwen_pack_scan.py <pack_dir> [--out scan.json]`

- `qwen_pack_config.py` reads that scan and rewrites `config.json`'s
  `quantization_config` into the plugin's own fields: `layer_bits` (per-layer
  expert K overrides), `non_routed_exl3.layers` (a per-prefix bit map for
  dense linears), and `ngram_embedding` (the row-wise table spec). Non-uniform
  expert K is allowed and noted; it still refuses (exit 2) if the pack's
  n-gram tables have inconsistent geometry. The original block is kept under
  `native_quantization_config` and `config.json` is backed up to
  `config.json.native` first.

  `python3 qwen_pack_config.py <pack_dir> [--scan pack_scan.json] [--dry-run]`

- `regenerate_safetensors_index.py` rebuilds
  `model.safetensors.index.json` from the safetensors files actually present
  in the pack directory. **This is a hard requirement**: vLLM's loader trusts
  the index's `weight_map` over the files on disk, so if the shipped index
  still names files that were renamed, split, or dropped while assembling a
  native pack, vLLM silently skips or mis-sources tensors instead of
  failing loudly. Run this any time the pack's file set changes. It sums
  each tensor's `data_offsets` span into `metadata.total_size`, backs the
  original index up to `model.safetensors.index.json.native` (only if that
  backup does not already exist), and refuses (exit 2) if the same tensor
  name appears in more than one shard.

  `python3 regenerate_safetensors_index.py <pack_dir> [--dry-run]`

Typical order: scan, rewrite the config, regenerate the index, then run the
gates in `tools/verify_native_pack/` before booting a server.

## Repair a shard with gaps between tensors

`compact_safetensors.py` copies an existing shard into a new, contiguous
container when a safetensors reader rejects gaps in its payload layout.
Names, shapes, dtypes, metadata and tensor bytes are preserved, including
per-projection packed K and signed MUL1 markers. It does not requantize.

```sh
python3 tools/exl3_pack_tools/compact_safetensors.py source.safetensors repaired.safetensors > receipts.json
python3 -m pytest -q tests/test_compact_safetensors.py
```

The destination must not exist. The tool rejects overlaps, truncation,
duplicate JSON keys and inconsistent shapes, and streams with a 1 MiB copy
buffer plus the header and per-tensor receipts. Unreferenced padding is
discarded; unsupported dtypes fail rather than being guessed. The source
must remain idle during the copy. Receipts contain SHA256 hashes of copied
tensor bytes, not proof of a trusted model revision. The CLI report also
carries `source_bytes`, `destination_bytes` and the signed `discarded_bytes`
delta, so a run can attest that only unreferenced bytes were dropped. Keep
the original checkpoint and revision checks. If shard filenames change in
the serving copy, regenerate its index before loading it.
