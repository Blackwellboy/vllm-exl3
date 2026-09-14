"""Worker extension: sample live CUDA EXL3 weights for parity diagnostics."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha16_cpu(t) -> str:
    import torch

    x = t.detach().to("cpu").contiguous()
    u8 = x.view(torch.uint8)
    return hashlib.sha256(u8.numpy().tobytes()).hexdigest()[:16]


def _global_expert_id(mod: Any, local_i: int, global_map: Any = None) -> int | None:
    """Resolve a local EP slot to its global expert id when placement is known."""
    if global_map is not None:
        try:
            return int(global_map[local_i])
        except Exception:
            pass
    starting_offset = getattr(mod, "starting_expert_offset", None)
    if starting_offset is not None:
        try:
            return int(starting_offset) + int(local_i)
        except Exception:
            pass
    return None


class ParityWorkerExtension:
    """Methods are injected onto the vLLM worker via --worker-extension-cls."""

    def parity_sample_weights(self, spec_json: str = "{}") -> dict[str, Any]:
        """Return compact hashes of selected live parameters.

        ``spec_json`` is a JSON object:
          layers: list[int]
          expert_ids: list[int]  (global expert ids)
          include_embed: bool
          include_attn: bool
        """
        import torch

        spec = json.loads(spec_json) if spec_json else {}
        layers = set(spec.get("layers", [0, 10, 20, 30, 39]))
        expert_ids = set(spec.get("expert_ids", [0, 2, 16, 32, 64, 80, 95]))
        include_embed = bool(spec.get("include_embed", True))
        include_attn = bool(spec.get("include_attn", True))

        # Worker -> model runner -> model
        model = None
        for path in (
            ("model_runner", "model"),
            ("worker", "model_runner", "model"),
            ("model_runner", "model", "model"),
        ):
            obj: Any = self
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if obj is not None and hasattr(obj, "named_parameters"):
                    model = obj
                    break
            except Exception:
                continue
        if model is None:
            # Fallback: dig for nn.Module with many params
            import gc
            import torch.nn as nn

            best = None
            best_n = 0
            for o in gc.get_objects():
                try:
                    if isinstance(o, nn.Module):
                        n = sum(1 for _ in o.parameters())
                        if n > best_n:
                            best_n = n
                            best = o
                except Exception:
                    continue
            model = best
        if model is None:
            return {"ok": False, "error": "model_not_found"}

        # Rank geometry
        tp_rank = getattr(self, "tp_rank", None)
        tp_size = getattr(self, "tp_size", None)
        if tp_rank is None:
            try:
                from vllm.distributed import get_tp_group

                tp_rank = get_tp_group().rank_in_group
                tp_size = get_tp_group().world_size
            except Exception:
                tp_rank, tp_size = -1, -1

        samples: list[dict[str, Any]] = []

        def add(name: str, tensor: Any, extra: dict | None = None) -> None:
            if tensor is None or not torch.is_tensor(tensor):
                return
            try:
                cpu = tensor.detach().to("cpu")
                item = {
                    "PARAM_NAME": name,
                    "TP_RANK": int(tp_rank) if tp_rank is not None else -1,
                    "TP_SIZE": int(tp_size) if tp_size is not None else -1,
                    "LIVE_SHAPE": list(cpu.shape),
                    "LIVE_DTYPE": str(cpu.dtype),
                    "LIVE_DEVICE": str(tensor.device),
                    "LIVE_HASH": _sha16_cpu(cpu),
                }
                if extra:
                    item.update(extra)
                samples.append(item)
            except Exception as e:
                samples.append({"PARAM_NAME": name, "error": str(e), "TP_RANK": tp_rank})

        # Embed / lm_head
        if include_embed:
            for n, p in model.named_parameters():
                if any(s in n for s in ("embed.weight", "embed_tokens.weight", "lm_head.weight")):
                    add(n, p, {"FAMILY": "embed"})

        # Attention dense weights
        if include_attn:
            for n, p in model.named_parameters():
                for L in layers:
                    if f"layers.{L}." not in n:
                        continue
                    if any(
                        x in n
                        for x in (
                            "q_a_proj",
                            "q_b_proj",
                            "kv_a_proj",
                            "kv_b_proj",
                            "o_proj",
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "wq_b",
                            "wkv_b",
                            "wo",
                        )
                    ):
                        add(n, p, {"FAMILY": "attn", "LAYER": L})

        # EXL3 modules: trellis/suh/svh
        for n, mod in model.named_modules():
            if not (
                hasattr(mod, "trellis")
                or type(mod).__name__ in ("LinearEXL3", "Exl3Linear", "LinearBase")
            ):
                # still check for expert lists
                pass
            layer_id = None
            for L in layers:
                if f"layers.{L}." in n:
                    layer_id = L
                    break
            if layer_id is None and "layers." in n:
                continue

            # Per-expert ParameterLists on MoE modules
            for list_name in (
                "gate_trellis",
                "up_trellis",
                "down_trellis",
                "gate_suh",
                "up_suh",
                "down_suh",
                "gate_svh",
                "up_svh",
                "down_svh",
            ):
                plist = getattr(mod, list_name, None)
                if plist is None:
                    continue
                try:
                    length = len(plist)
                except Exception:
                    continue
                # Map local index -> global expert id using explicit placement
                # first, then the EP starting offset used by current RoutedExperts.
                global_map = getattr(mod, "expert_ids", None)
                if global_map is None:
                    global_map = getattr(mod, "global_expert_ids", None)
                for local_i in range(length):
                    try:
                        t = plist[local_i]
                    except Exception:
                        continue
                    if not torch.is_tensor(t):
                        t = getattr(t, "data", t)
                    geid = _global_expert_id(mod, local_i, global_map)
                    if geid is not None and geid not in expert_ids and expert_ids:
                        continue
                    if geid is None and local_i not in expert_ids and expert_ids:
                        # Last-resort local-index sampling only when placement
                        # metadata is genuinely unavailable.
                        continue
                    K = None
                    if "trellis" in list_name and torch.is_tensor(t) and t.ndim == 3:
                        K = int(t.shape[-1]) // 16
                    add(
                        f"{n}.{list_name}[{local_i}]",
                        t,
                        {
                            "FAMILY": "exl3_moe",
                            "LAYER": layer_id,
                            "LOCAL_EXPERT": local_i,
                            "GLOBAL_EXPERT": geid,
                            "LIST": list_name,
                            "K": K,
                            "MODULE": type(mod).__name__,
                        },
                    )

            # Single LinearEXL3 (shared experts / dense)
            if hasattr(mod, "trellis") and torch.is_tensor(getattr(mod, "trellis")):
                if "shared" in n or "shared_experts" in n or layer_id is not None:
                    for attr in ("trellis", "suh", "svh"):
                        t = getattr(mod, attr, None)
                        if torch.is_tensor(t):
                            K = int(t.shape[-1]) // 16 if attr == "trellis" and t.ndim == 3 else None
                            add(
                                f"{n}.{attr}",
                                t,
                                {
                                    "FAMILY": "exl3_linear",
                                    "LAYER": layer_id,
                                    "K": K,
                                    "MODULE": type(mod).__name__,
                                },
                            )

        return {
            "ok": True,
            "model_type": type(model).__name__,
            "tp_rank": tp_rank,
            "tp_size": tp_size,
            "n_samples": len(samples),
            "samples": samples,
        }
