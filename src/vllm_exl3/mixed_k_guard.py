"""Safety guards for mixed-K trellis prescan.

The low-memory arena prescan maps local expert ids onto checkpoint expert ids.
That mapping is only valid when vLLM uses linear expert placement. Other
placement/EPLB strategies must fall back to the normal loader, which already
uses vLLM's authoritative expert map.

For pure MoE tensor parallelism the safetensors header describes the unsharded
checkpoint tensor while ``_load_exl3`` narrows gate/up and down trellises before
copying them into local storage. Prescan must mirror that geometry or it will
allocate full-width arenas and later reject the correctly sharded tensor.
"""
from __future__ import annotations

from typing import Any


def _placement_strategy(layer: Any) -> str | None:
    value = getattr(layer, "expert_placement_strategy", None)
    if value is None:
        manager = getattr(layer, "expert_map_manager", None)
        value = getattr(manager, "placement_strategy", None) if manager is not None else None
    return str(value).lower() if value is not None else None


def _eplb_active(layer: Any) -> bool:
    for owner in (layer, getattr(layer, "expert_map_manager", None)):
        if owner is None:
            continue
        for name in ("enable_eplb", "eplb_enabled"):
            if bool(getattr(owner, name, False)):
                return True
    return False


def shard_prescanned_trellis_shapes(
    shapes: dict[str, dict[int, tuple[int, ...]]],
    tp_size: int,
) -> dict[str, dict[int, tuple[int, ...]]]:
    """Mirror EXL3 gate/up column sharding and down row sharding in metadata."""
    tp = int(tp_size)
    if tp <= 1:
        return {
            proj: {int(eid): tuple(shape) for eid, shape in per_expert.items()}
            for proj, per_expert in shapes.items()
        }
    out: dict[str, dict[int, tuple[int, ...]]] = {"gate": {}, "up": {}, "down": {}}
    for proj in ("gate", "up", "down"):
        for eid, shape in (shapes.get(proj) or {}).items():
            dims = list(int(x) for x in shape)
            if len(dims) != 3:
                raise RuntimeError(
                    f"EXL3 prescan expected 3-D trellis for {proj} expert={eid}, got {shape}"
                )
            shard_dim = 1 if proj in ("gate", "up") else 0
            if dims[shard_dim] % tp:
                raise RuntimeError(
                    f"EXL3 prescan cannot TP-shard {proj} expert={eid} shape={shape} "
                    f"by tp_size={tp}"
                )
            dims[shard_dim] //= tp
            out[proj][int(eid)] = tuple(dims)
    return out


def install_mixed_k_prescan_guard(exl3_module: Any) -> None:
    current = getattr(exl3_module, "_try_prescan_trellis_shapes", None)
    if not callable(current):
        return
    if bool(getattr(current, "_vllm_exl3_mixed_k_prescan_guard", False)):
        return

    def guarded(layer: Any, num_experts: int):
        placement = _placement_strategy(layer)
        if _eplb_active(layer) or placement not in (None, "linear"):
            logger = getattr(exl3_module, "logger", None)
            if logger is not None:
                getattr(logger, "info_once", logger.info)(
                    "EXL3 mixed-K arena prescan disabled for expert placement=%s "
                    "eplb=%s; falling back to authoritative loader mapping",
                    placement or "unknown",
                    _eplb_active(layer),
                )
            return None
        shapes = current(layer, num_experts)
        if shapes is None:
            return None
        resolver = getattr(exl3_module, "_resolve_tp_geometry", None)
        tp_size = 1
        if callable(resolver):
            try:
                _rank, tp_size = resolver(layer)
            except ModuleNotFoundError:
                # CPU-only unit tests intentionally run without vLLM. If the
                # synthetic layer carries no TP metadata, TP1 is the only safe
                # interpretation and matches the legacy prescan contract.
                tp_size = 1
        return shard_prescanned_trellis_shapes(shapes, int(tp_size))

    guarded._vllm_exl3_mixed_k_prescan_guard = True
    guarded._vllm_exl3_original = current
    exl3_module._try_prescan_trellis_shapes = guarded
    exl3_module._vllm_exl3_mixed_k_prescan_guard_installed = True
    exl3_module._vllm_exl3_mixed_k_prescan_tp_aware = True
