import pytest

from vllm_exl3.mixed_k_guard import shard_prescanned_trellis_shapes


def _shapes():
    return {
        "gate": {0: (320, 144, 48), 1: (320, 144, 64)},
        "up": {0: (320, 144, 80), 1: (320, 144, 96)},
        "down": {0: (144, 320, 48), 1: (144, 320, 64)},
    }


def test_tp2_prescan_matches_loader_sharding_geometry():
    out = shard_prescanned_trellis_shapes(_shapes(), 2)
    assert out["gate"][0] == (320, 72, 48)
    assert out["up"][1] == (320, 72, 96)
    assert out["down"][0] == (72, 320, 48)


def test_tp1_prescan_is_unchanged():
    assert shard_prescanned_trellis_shapes(_shapes(), 1) == _shapes()


def test_prescan_rejects_nondivisible_tp_geometry():
    with pytest.raises(RuntimeError, match="cannot TP-shard"):
        shard_prescanned_trellis_shapes(
            {"gate": {0: (320, 145, 64)}, "up": {}, "down": {}}, 2
        )
