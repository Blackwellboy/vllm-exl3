from types import SimpleNamespace

from vllm_exl3.worker_parity import _global_expert_id


def test_global_expert_id_prefers_explicit_map() -> None:
    mod = SimpleNamespace(starting_expert_offset=24)
    assert _global_expert_id(mod, 0, [40, 41]) == 40
    assert _global_expert_id(mod, 1, [40, 41]) == 41


def test_global_expert_id_uses_starting_expert_offset() -> None:
    mod = SimpleNamespace(starting_expert_offset=24)
    assert _global_expert_id(mod, 0) == 24
    assert _global_expert_id(mod, 8) == 32
    assert _global_expert_id(mod, 95) == 119


def test_global_expert_id_unknown_without_placement() -> None:
    mod = SimpleNamespace()
    assert _global_expert_id(mod, 3) is None
