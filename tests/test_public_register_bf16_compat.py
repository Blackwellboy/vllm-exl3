from __future__ import annotations

import sys
from types import ModuleType

import vllm_exl3


def _stub_installer_module(
    monkeypatch,
    module_name: str,
    function_name: str,
    calls: list[str],
    label: str,
) -> ModuleType:
    mod = ModuleType(module_name)

    def installer(exl3_module):
        calls.append(label)

    setattr(mod, function_name, installer)
    monkeypatch.setitem(sys.modules, module_name, mod)
    return mod


def test_public_register_installs_mixed_bf16_madv_compat(monkeypatch) -> None:
    calls: list[str] = []

    fake_exl3 = ModuleType("vllm_exl3.exl3")
    monkeypatch.setitem(sys.modules, "vllm_exl3.exl3", fake_exl3)
    monkeypatch.setattr(vllm_exl3, "exl3", fake_exl3, raising=False)

    installers = [
        ("vllm_exl3.bf16_madv_compat", "install_mixed_bf16_madv_compat", "bf16_madv"),
        ("vllm_exl3.deepseek_v41", "install_deepseek_v41_compat", "deepseek_v41"),
        ("vllm_exl3.k78_compat", "install_k78_config_compat", "k78"),
        ("vllm_exl3.mixed_k_guard", "install_mixed_k_prescan_guard", "mixed_k"),
        ("vllm_exl3.physical_k_compat", "install_physical_fused_k_compat", "physical_k"),
        ("vllm_exl3.runtime_policy", "install_native_row_policy", "runtime_policy"),
        ("vllm_exl3.tp_geometry_compat", "install_tp_geometry_compat", "tp_geometry"),
        ("vllm_exl3.uva_offload", "install_uva_expert_validation", "uva"),
    ]
    for module_name, function_name, label in installers:
        _stub_installer_module(
            monkeypatch, module_name, function_name, calls, label
        )

    vllm_exl3.register()

    assert calls.count("bf16_madv") == 1
    assert set(calls) == {
        "bf16_madv",
        "deepseek_v41",
        "k78",
        "mixed_k",
        "physical_k",
        "runtime_policy",
        "tp_geometry",
        "uva",
    }
