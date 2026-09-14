# Experimental SAGE 3.30 NVMe components

These are reusable parts of a two-GB10 deployment that generated responses
with the original SAGE 3.30 checkpoint. They are **not an installed plugin
backend or a complete launch recipe**. They preserve existing packed weights;
they do not quantize, provide a CPU MoE executor, or enable CUDA graphs.

The source extraction is for integration review. Mainline has advanced since
the measured deployment. In particular, its mixed-K loader and disk-Engram
overlay now have different integration contracts. No serving defaults, source
pins, entry points or model architecture are changed by these files.

## What can be reused

| Component | Contract |
| --- | --- |
| `expert_store.py` | Copies existing MUL1 K2–K8 tensors into 4096-byte-aligned records; checks manifest identity and whole-record SHA256 before exposing bytes. Direct I/O is required by default and never silently falls back. |
| `async_store.py` | Overlaps read/hash work using at most two staging records, including completed and leased records. Reader threads make no CUDA calls. Unexpected demand displaces only unleased work. |
| `expert_cache.py` | Fixed CUDA arena with per-layer quotas, per-projection `LinearEXL3` handles, explicit leases and CUDA events before eviction. CPU-controlled, demand-only, eager execution; multi-stream serving and graphs are not qualified. |
| `engram_rows.py` | Bounded cache of original 256-byte FP8 rows plus 8-byte scale rows; one pending prefetch batch, up to four readers, 8192 rows/request, and 131072 cached rows/table. Unowned rows cause no disk reads. No hash-ID generation or dequantization is replaced. |

The expert reader accepts only
`vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` at revision
`e831e9e4d6bfeafa6d630848296417b1393404a3`. The single-shard bank builder is
for existing source tensors or fixtures; full-checkpoint packaging needs a
separate index-aware builder. A model/revision label is not source trust:
verify the source snapshot before building a bank. The manifest records the
source file digest, and each tensor retains its original bytes and hash.

The cache must receive a **complete manifest for its rank**. `apply()` treats
absent expert IDs as belonging to another EP rank. It cannot distinguish a
missing owned expert from an unowned one; the vLLM adapter must validate the
complete ownership set before admission. The tested topology was TP2+EP2,
192 whole experts/rank, C1, DP1, PP1, no EPLB or speculation.

## Run the portable checks

From the repository root, with Python 3.12 or later on a POSIX platform:

```sh
python3 -m unittest discover -s experiments/dsv41_nvme -p 'test_*.py' -v
```

The tests create synthetic fixtures in temporary directories. No model
download, Torch, CUDA, server, credentials or private host configuration is
needed. They cover corruption, short reads, retry, cancellation, retained
views, concurrent read bounds, eviction accounting, duplicate rows, ownership,
changed files and shutdown. Buffered I/O is explicitly selected for these
tests; they do not prove Linux direct I/O or GPU numerical correctness.

These components require POSIX `os.pread`/`os.preadv`, in buffered mode too:
neither transport falls back to another syscall on a platform that lacks them.
`posix_support.py` is the single capability check, and a run on a platform
without those primitives reports explicit **skips** (not passes, not errors).
Windows cannot exercise these components as written; only the pure-validation
and arena-accounting checks still run there.

For the Linux O_DIRECT check on a compatible local filesystem:

```sh
python3 experiments/dsv41_nvme/check_direct_io.py
```

This creates a small synthetic bank, reads and hashes it synchronously and
asynchronously, then removes only its temporary directory. Unsupported direct
I/O fails; it is not counted as a pass. The check is Linux-only by
construction: `os.O_DIRECT` and aligned `preadv` do not exist elsewhere.

## Installation status

Nothing here is installed. These files are not a plugin backend, not a
registered entry point and not a default production backend. `pyproject.toml`
does not reference them, no production module imports them, and
`test_provenance.py` asserts both facts. Any serving path that would consume
them requires an explicit, reviewed adapter that does not exist in this PR.

## Integration and qualification boundary

The pins below are **historical**: they describe the deployment that was
measured, not current mainline. vLLM `0.1.dev20904+g179dd0fa9`, Torch
`2.13.0+cu130` and plugin `8f4517e80416466fa4a3ad2eb28685021d39e95f` have no
in-tree equivalent on current mainline and are not verified against it; treat
them as provenance of the measurement, never as a build recipe. ExLlamaV3
`be57335b087e4f001c5caae061544df3c06ba01e` is still the ref CI pins in
`.github/workflows/ci.yml` (`EXLLAMAV3_SPARK_REF`), so that one pin is current.
Source hashes in `source-provenance.json` identify the extracted
implementation. Local changes to tests replace private weight fixtures with
synthetic data. The GPU cache calls ExLlamaV3's existing API; no ExLlamaV3
kernels are copied here.

### C3 boundary

These components move bytes; they do not define the model. The authoritative
configuration of `vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw` at revision
`e831e9e4d6bfeafa6d630848296417b1393404a3` (C3) is unchanged by this PR:

- No precision, config, checkpoint or architecture change is made or implied.
- Original packed K2-K8 expert bytes are passed through verbatim; the bank
  builder copies existing tensors and hashes them, it never re-quantizes or
  reinterprets a row width.
- EXL3 is never treated as FP8. Runtime FP8 `[32, 32]` delegation is the only
  FP8 relationship, and it stays with the existing mainline runtime path.
- No layer-global-K assumption is introduced: geometry is validated per
  projection against that tensor's own recorded shape.
- `EngramRows` caches original FP8 rows plus E8M0 scale rows and defers all
  dequantization, hashing and gating to its caller.

Merging these files alone cannot reproduce the deployment, and reusing them
must not be read as authority to change C3.

The complete model additionally needed a loader that skips full expert and
Engram allocation, verifies every owned expert tensor before finalization,
streams native weights with page release, and preserves vLLM routing,
collectives and attention. These integration changes are not installed by
this PR. `EngramRows` expects local row IDs plus the native ownership mask;
its caller remains responsible for native hashing, FP8/E8M0 dequantization,
TP collectives and gating. The historical Engram adapter is deliberately not
copied over the newer mainline overlay.

Before promoting a backend, qualify the current loader/overlay contracts,
fault handling across both ranks, GPU stream ownership, full-model quality,
and the memory budget. The reference GPU cache synchronizes before releasing
staging storage or recycling slots; CUDA-context failure requires worker
termination. Its host RAM/swap check is a guard for Spark unified memory, not
a replacement for cgroup limits and an external watchdog.

Full-model throughput remains slow. Isolated cache hits or overlapping reads
are not evidence of a serving speedup. CUDA graphs, DSpark and larger-context
memory improvements remain separate work.
