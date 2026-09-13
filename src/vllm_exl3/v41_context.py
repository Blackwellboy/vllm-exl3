"""DeepSeek V4.1 serving-memory and MoE-topology planning helpers.

The older ``compute_mla_kv_cache_bytes`` helper models V4-era MLA storage and
must not be reused as V4.1 admission math. V4.1 shares four persistent global
KV/index caches across layers; three are pooled 2:1 and one remains 1:1.

The 890 bytes/token figure is the logical persistent global-cache footprint
reported by the SGLang V4.1 kernel-optimization write-up. Runtime backends may
allocate more because of cache layout, padding, local SWA, metadata, and other
workspace. Accordingly, this module never treats the logical number alone as a
hardware-qualified capacity result.
"""
from __future__ import annotations

from typing import Any

V41_GLOBAL_CACHE_SOURCE_LAYERS = (2, 8, 14, 20)
V41_POOLED_2_TO_1_LAYERS = (2, 8, 14)
V41_MAIN_KV_BYTES_PER_ENTRY = 288
V41_INDEXER_K_BYTES_PER_ENTRY = 68
V41_ENTRY_BYTES = V41_MAIN_KV_BYTES_PER_ENTRY + V41_INDEXER_K_BYTES_PER_ENTRY
V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN = 890
V41_LOCAL_REPLAY_WINDOW = 128


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed != value:
        raise ValueError(f"{name} must be an integer")
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def compute_v41_logical_global_kv_bytes(context_len: int) -> int:
    """Return V4.1's logical persistent global KV+index footprint.

    Three global-cache producers pool 2:1 and the fourth is 1:1, so each
    original token accounts for ``(288 + 68) * (3/2 + 1) = 890`` bytes.
    This excludes local-window replay/cache storage and backend padding.
    """
    context = _positive_int("context_len", context_len)
    return context * V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN


def estimate_v41_kv_cache(
    context_len: int,
    *,
    backend_bytes_per_token: int | None = None,
) -> dict[str, Any]:
    """Describe logical and, when measured, backend-allocated V4.1 KV.

    ``backend_bytes_per_token`` should come from the actual serving backend's
    cache allocation receipt. It must not be guessed from the logical 890 B/tok
    value. When omitted, the returned estimate is explicitly a lower bound.
    """
    context = _positive_int("context_len", context_len)
    logical = compute_v41_logical_global_kv_bytes(context)
    measured_bpt: int | None = None
    allocated: int | None = None
    if backend_bytes_per_token is not None:
        measured_bpt = _positive_int("backend_bytes_per_token", backend_bytes_per_token)
        if measured_bpt < V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN:
            raise ValueError(
                "backend_bytes_per_token cannot be smaller than the V4.1 logical "
                f"global-cache floor ({V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN})"
            )
        allocated = context * measured_bpt
    return {
        "context_len": context,
        "global_cache_source_layers": list(V41_GLOBAL_CACHE_SOURCE_LAYERS),
        "pooled_2_to_1_layers": list(V41_POOLED_2_TO_1_LAYERS),
        "logical_bytes_per_token": V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN,
        "logical_global_kv_bytes": logical,
        "logical_global_kv_gib": logical / (1024**3),
        "backend_bytes_per_token": measured_bpt,
        "backend_allocated_kv_bytes": allocated,
        "backend_allocated_kv_gib": None if allocated is None else allocated / (1024**3),
        "capacity_qualified": allocated is not None,
        "basis": "measured_backend_allocation" if allocated is not None else "logical_lower_bound_only",
        "local_replay_window": V41_LOCAL_REPLAY_WINDOW,
        "note": (
            "Logical global KV excludes local-window/cache storage, backend padding, "
            "metadata and runtime workspaces. Use measured backend allocation for "
            "capacity claims."
        ),
    }


def validate_v41_context_scaling(
    max_model_len: int,
    *,
    model_resident_gib: float,
    total_mem_gib: float = 128.0,
    mem_util: float = 0.75,
    other_runtime_gib: float = 0.0,
    backend_bytes_per_token: int | None = None,
) -> dict[str, Any]:
    """Return a V4.1 memory-budget receipt without overstating qualification."""
    estimate = estimate_v41_kv_cache(
        max_model_len, backend_bytes_per_token=backend_bytes_per_token
    )
    for name, value in (
        ("model_resident_gib", model_resident_gib),
        ("total_mem_gib", total_mem_gib),
        ("mem_util", mem_util),
        ("other_runtime_gib", other_runtime_gib),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be numeric") from exc
        if name == "total_mem_gib" and parsed <= 0:
            raise ValueError("total_mem_gib must be positive")
        if name == "mem_util" and not (0.0 < parsed <= 1.0):
            raise ValueError("mem_util must be in (0, 1]")
        if name in {"model_resident_gib", "other_runtime_gib"} and parsed < 0:
            raise ValueError(f"{name} must be non-negative")

    kv_gib = (
        estimate["backend_allocated_kv_gib"]
        if estimate["backend_allocated_kv_gib"] is not None
        else estimate["logical_global_kv_gib"]
    )
    usable = float(total_mem_gib) * float(mem_util)
    remaining = usable - float(model_resident_gib) - float(other_runtime_gib) - float(kv_gib)
    return {
        **estimate,
        "model_resident_gib": float(model_resident_gib),
        "total_mem_gib": float(total_mem_gib),
        "mem_util": float(mem_util),
        "other_runtime_gib": float(other_runtime_gib),
        "usable_mem_gib": usable,
        "remaining_gib": remaining,
        "fits_budget": remaining > 0,
        "capacity_claim_allowed": bool(estimate["capacity_qualified"]),
    }


def plan_v41_moe_topologies(tp_size: int) -> dict[str, Any]:
    """Compare expert-parallel and pure-MoE-TP geometry for V4.1.

    This is geometry only, not a performance recommendation. In particular,
    pure TP4 produces a 576-wide local expert slice, while pure TP2 produces
    1152 which is already exactly 128-aligned.
    """
    tp = _positive_int("tp_size", tp_size)
    total_experts = 384
    hidden = 5120
    intermediate = 2304
    if total_experts % tp:
        raise ValueError(f"384 experts are not divisible by tp_size={tp}")
    if intermediate % tp:
        raise ValueError(f"2304 intermediate width is not divisible by tp_size={tp}")
    local_tp_width = intermediate // tp
    padded = ((local_tp_width + 127) // 128) * 128
    return {
        "tp_size": tp,
        "ep": {
            "experts_per_rank": total_experts // tp,
            "expert_hidden": hidden,
            "expert_intermediate_local": intermediate,
            "intermediate_128_aligned": intermediate % 128 == 0,
        },
        "pure_moe_tp": {
            "experts_per_rank": total_experts,
            "expert_hidden": hidden,
            "expert_intermediate_local": local_tp_width,
            "padded_intermediate_128": padded,
            "padding_fraction": (padded - local_tp_width) / local_tp_width,
            "intermediate_128_aligned": local_tp_width % 128 == 0,
        },
        "qualification": "experimental_ab_only",
    }
