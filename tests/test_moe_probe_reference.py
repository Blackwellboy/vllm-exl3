import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch


def test_native_reference_routes_by_expert_id_not_topk_column(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/gb10_exl3_moe_parity.py'
    spec = importlib.util.spec_from_file_location('parity_probe_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'vllm_exl3_c', SimpleNamespace(exl3_gemv=lambda x,t,*args: x*t))
    x = torch.tensor([[0.01, -0.02]], dtype=torch.float16)
    ids = torch.tensor([[7, 2]])
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float16)
    packs = [{p:{'trellis':e+1, 'suh':None, 'svh':None} for p in ('gate','up','down')} for e in range(8)]
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert, weight in [(7,0.25),(2,0.75)]:
        z=(x*(expert+1)).float()
        expected += weight * ((torch.nn.functional.silu(z)*z).half()*(expert+1)).float()
    actual=mod._reference(x,ids,weights,packs,2,torch.device('cpu'))
    torch.testing.assert_close(actual,expected)
