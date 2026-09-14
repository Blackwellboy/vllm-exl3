"""Compatibility shim for mixed EXL3/BF16 source-page reclamation on UMA."""

from __future__ import annotations

from functools import wraps
from typing import Any


def _loaded_shard_ids(
    loaded_weight: Any,
    loaded_shard_id: str | int | tuple | list | None,
    n_shards: int,
    output_partition_sizes: list[int],
) -> set[int]:
    if isinstance(loaded_shard_id, (tuple, list)):
        return {int(i) for i in loaded_shard_id}
    if isinstance(loaded_shard_id, int):
        return {loaded_shard_id}
    if isinstance(loaded_shard_id, str):
        qkv = {"q": 0, "k": 1, "v": 2}
        return {qkv[loaded_shard_id]} if loaded_shard_id in qkv else set()
    if n_shards <= 1:
        return {0}
    try:
        loaded_out = int(loaded_weight.shape[0])
    except Exception:
        return {0}
    if loaded_out == sum(int(x) for x in output_partition_sizes):
        return set(range(n_shards))
    return {0}


def install_mixed_bf16_madv_compat(module: Any) -> None:
    """Reclaim CPU safetensors pages after the mixed-BF16 direct H2D branch.

    ``Exl3LinearMethod._make_weight_loader`` has a special ``weight`` branch for
    BF16 shards.  The branch copies directly into a CUDA staging parameter and
    returns before the ordinary post-H2D MADV path.  Wrap the generated loader so
    the consumed source (or fully consumed fused source) is synchronized and
    reclaimed using the same view/VMA-bounded helper as the rest of the loader.
    """
    cls = getattr(module, "Exl3LinearMethod", None)
    if cls is None:
        return
    original = getattr(cls, "_make_weight_loader", None)
    if original is None or getattr(original, "_vllm_exl3_bf16_madv_wrapped", False):
        return

    @wraps(original)
    def wrapped_make(
        self,
        suffix,
        n_shards,
        output_partition_sizes,
        is_row_parallel,
        bf16_shards,
        layer=None,
        is_qkv_parallel=False,
    ):
        loader = original(
            self,
            suffix,
            n_shards,
            output_partition_sizes,
            is_row_parallel,
            bf16_shards,
            layer,
            is_qkv_parallel,
        )
        if suffix != "weight" or not bf16_shards:
            return loader

        @wraps(loader)
        def reclaiming_loader(param, loaded_weight, loaded_shard_id=None):
            result = loader(param, loaded_weight, loaded_shard_id)
            consumed = _loaded_shard_ids(
                loaded_weight,
                loaded_shard_id,
                int(n_shards),
                list(output_partition_sizes),
            )
            if consumed.intersection(int(i) for i in bf16_shards):
                try:
                    torch = module.torch
                    if (
                        torch is not None
                        and getattr(param, "device", None) is not None
                        and param.device.type == "cuda"
                        and getattr(loaded_weight, "device", None) is not None
                        and loaded_weight.device.type == "cpu"
                    ):
                        torch.cuda.current_stream().synchronize()
                        module._madv_dontneed_cpu_tensor(loaded_weight)
                except Exception:
                    # Reclamation is an optimization/safety aid, never a reason
                    # to make weight loading fail after a successful copy.
                    pass
            return result

        return reclaiming_loader

    wrapped_make._vllm_exl3_bf16_madv_wrapped = True
    cls._make_weight_loader = wrapped_make
    module._vllm_exl3_bf16_madv_compat_installed = True
