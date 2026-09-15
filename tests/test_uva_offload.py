from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_exl3.uva_offload import (
    EXL3_MOE_UVA_PARAMETER_SEGMENTS,
    EXL3_TRELLIS_ARENA_ATTRS,
    inspect_exl3_moe_uva_layer,
    install_uva_expert_validation,
    validate_exl3_moe_uva_arenas,
    validate_exl3_moe_uva_layer,
)


class _Param:
    def __init__(self, device_type: str = "cuda", uva: bool = False, numel: int = 1):
        self.device = SimpleNamespace(type=device_type)
        self._numel = numel
        if uva:
            self._vllm_is_uva_offloaded = True

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return 2


class _Layer:
    def __init__(
        self,
        *,
        uva_names: set[str] | None = None,
        cpu_names: set[str] | None = None,
    ):
        uva_names = uva_names or set()
        cpu_names = cpu_names or set()
        for name in EXL3_MOE_UVA_PARAMETER_SEGMENTS:
            setattr(
                self,
                name,
                _Param(
                    device_type="cpu" if name in cpu_names else "cuda",
                    uva=name in uva_names,
                ),
            )


def test_resident_layer_is_not_uva():
    status = inspect_exl3_moe_uva_layer(_Layer(), required=False)
    assert status.applicable
    assert not status.fully_uva_offloaded
    assert not status.partially_uva_offloaded
    assert set(status.resident_parameters) == set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)


def test_complete_mapped_payload_is_accepted():
    names = set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    status = validate_exl3_moe_uva_layer(_Layer(uva_names=names))
    assert status.fully_uva_offloaded
    assert not status.partially_uva_offloaded
    assert not status.cpu_fallback_parameters


def test_partial_uva_is_rejected():
    names = {EXL3_MOE_UVA_PARAMETER_SEGMENTS[0]}
    with pytest.raises(RuntimeError, match="partial"):
        validate_exl3_moe_uva_layer(_Layer(uva_names=names))


def test_non_uva_cpu_fallback_is_rejected():
    cpu = {EXL3_MOE_UVA_PARAMETER_SEGMENTS[0]}
    with pytest.raises(RuntimeError, match="ordinary CPU tensors"):
        validate_exl3_moe_uva_layer(_Layer(cpu_names=cpu))


def test_install_guard_runs_before_original_post_load(monkeypatch):
    names = set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    layer = _Layer(uva_names=names)
    events: list[str] = []

    class FakeMethod:
        def process_weights_after_loading(self, target):
            events.append("original")
            assert hasattr(target, "_exl3_uva_expert_status")
            return "ok"

    fake_module = SimpleNamespace(Exl3MoEMethod=FakeMethod)
    monkeypatch.setenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", "1")
    install_uva_expert_validation(fake_module)

    result = FakeMethod().process_weights_after_loading(layer)
    assert result == "ok"
    assert events == ["original"]
    assert layer._exl3_uva_expert_status["fully_uva_offloaded"] is True


def test_guard_is_noop_when_not_requested(monkeypatch):
    layer = _Layer()
    events: list[str] = []

    class FakeMethod:
        def process_weights_after_loading(self, target):
            events.append("original")
            return "ok"

    fake_module = SimpleNamespace(Exl3MoEMethod=FakeMethod)
    monkeypatch.delenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", raising=False)
    install_uva_expert_validation(fake_module)

    assert FakeMethod().process_weights_after_loading(layer) == "ok"
    assert events == ["original"]
    assert not hasattr(layer, "_exl3_uva_expert_status")


def _layer_with_arenas(*, marked: bool, numel: int = 64):
    layer = _Layer(uva_names=set(EXL3_MOE_UVA_PARAMETER_SEGMENTS))
    layer.gate_trellis = []
    for attr in EXL3_TRELLIS_ARENA_ATTRS:
        setattr(layer, attr, [_Param(uva=marked, numel=numel)])
    return layer


def test_zero_element_placeholders_are_not_counted_as_placed():
    names = set(EXL3_MOE_UVA_PARAMETER_SEGMENTS)
    layer = _Layer(uva_names=names)
    for name in ("w13_trellis", "w2_trellis"):
        setattr(layer, name, _Param(uva=True, numel=0))
    status = inspect_exl3_moe_uva_layer(layer, required=True)
    assert set(status.placeholder_parameters) == {"w13_trellis", "w2_trellis"}
    assert set(status.uva_parameters) == names - {"w13_trellis", "w2_trellis"}
    # Placeholders are tolerated before packing; the arena check decides.
    assert status.fully_uva_offloaded


def test_resident_arenas_are_rejected_after_packing():
    with pytest.raises(RuntimeError, match="allocated on the accelerator"):
        validate_exl3_moe_uva_arenas(_layer_with_arenas(marked=False))


def test_missing_arenas_are_rejected_after_packing():
    layer = _layer_with_arenas(marked=True)
    for attr in EXL3_TRELLIS_ARENA_ATTRS:
        setattr(layer, attr, [])
    with pytest.raises(RuntimeError, match="no trellis arenas"):
        validate_exl3_moe_uva_arenas(layer)


def test_uva_arenas_are_accepted_after_packing():
    summary = validate_exl3_moe_uva_arenas(_layer_with_arenas(marked=True))
    assert summary["uva_arenas"] == 3 and summary["resident_arenas"] == 0


def test_guard_checks_arenas_after_original_post_load(monkeypatch):
    layer = _layer_with_arenas(marked=False)
    for name in ("w13_trellis", "w2_trellis"):
        setattr(layer, name, _Param(uva=True, numel=0))

    class FakeMethod:
        def process_weights_after_loading(self, target):
            return "ok"

    fake_module = SimpleNamespace(Exl3MoEMethod=FakeMethod)
    monkeypatch.setenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", "1")
    install_uva_expert_validation(fake_module)
    with pytest.raises(RuntimeError, match="allocated on the accelerator"):
        FakeMethod().process_weights_after_loading(layer)


def test_alloc_trellis_arena_uses_pinned_host_view_when_required(monkeypatch):
    torch = pytest.importorskip("torch")
    from vllm_exl3 import exl3 as exl3_mod

    calls: list[tuple[int, ...]] = []

    def fake_pinned(shape, dtype):
        calls.append(tuple(shape))
        return torch.empty(shape, dtype=dtype)

    def fake_view(host):
        return host  # CPU stand-in for the mapped accelerator view

    monkeypatch.setattr(exl3_mod, "_pinned_host_empty", fake_pinned)
    fake_utils = SimpleNamespace(get_accelerator_view_from_cpu_tensor=fake_view)
    monkeypatch.setitem(__import__("sys").modules, "vllm.utils.torch_utils", fake_utils)
    monkeypatch.setenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", "1")
    layer = SimpleNamespace()
    arena = exl3_mod._alloc_trellis_arena(layer, (2, 4, 4, 48), torch.device("cpu"))
    assert calls == [(2, 4, 4, 48)]
    assert arena._vllm_is_uva_offloaded is True
    assert layer._exl3_uva_host_arenas[0] is arena
    param = exl3_mod._arena_parameter(arena)
    assert param._vllm_is_uva_offloaded is True

    monkeypatch.delenv("VLLM_EXL3_REQUIRE_UVA_EXPERTS", raising=False)
    plain = exl3_mod._alloc_trellis_arena(SimpleNamespace(), (1, 4, 4, 48), torch.device("cpu"))
    assert not getattr(plain, "_vllm_is_uva_offloaded", False)

