import pytest

from vllm_exl3.v41_context import (
    V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN,
    compute_v41_logical_global_kv_bytes,
    estimate_v41_kv_cache,
    plan_v41_moe_topologies,
    validate_v41_context_scaling,
)


def test_v41_logical_global_kv_matches_890_bytes_per_token():
    context = 131072
    expected = context * 890
    assert V41_LOGICAL_GLOBAL_KV_BYTES_PER_TOKEN == 890
    assert compute_v41_logical_global_kv_bytes(context) == expected
    assert expected / (1024**3) == pytest.approx(0.108642578125)


def test_unmeasured_v41_cache_is_lower_bound_not_capacity_claim():
    receipt = estimate_v41_kv_cache(8192)
    assert receipt["capacity_qualified"] is False
    assert receipt["backend_allocated_kv_bytes"] is None
    assert receipt["basis"] == "logical_lower_bound_only"


def test_measured_backend_cache_enables_capacity_receipt():
    receipt = validate_v41_context_scaling(
        8192,
        model_resident_gib=86.0,
        other_runtime_gib=8.0,
        backend_bytes_per_token=2048,
    )
    assert receipt["capacity_claim_allowed"] is True
    assert receipt["backend_allocated_kv_bytes"] == 8192 * 2048


def test_backend_bytes_per_token_cannot_undercut_logical_floor():
    with pytest.raises(ValueError):
        estimate_v41_kv_cache(8192, backend_bytes_per_token=889)


def test_tp2_pure_moe_tp_is_128_aligned_without_padding():
    plan = plan_v41_moe_topologies(2)
    assert plan["ep"]["experts_per_rank"] == 192
    assert plan["ep"]["expert_intermediate_local"] == 2304
    assert plan["pure_moe_tp"]["experts_per_rank"] == 384
    assert plan["pure_moe_tp"]["expert_intermediate_local"] == 1152
    assert plan["pure_moe_tp"]["padded_intermediate_128"] == 1152
    assert plan["pure_moe_tp"]["padding_fraction"] == 0


def test_tp4_pure_moe_tp_exposes_576_to_640_padding_candidate():
    plan = plan_v41_moe_topologies(4)
    assert plan["pure_moe_tp"]["expert_intermediate_local"] == 576
    assert plan["pure_moe_tp"]["padded_intermediate_128"] == 640
    assert plan["pure_moe_tp"]["padding_fraction"] == pytest.approx(64 / 576)
