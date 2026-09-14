#!/usr/bin/env python3
"""GB10 parity probe: the fused exl3_moe launch against the native exl3_gemv map.

Run on the GB10 (DGX Spark) host with a CUDA-built exllamav3_ext and vllm_exl3_c:

    python tools/gb10_exl3_moe_parity.py                 # 1.4.x + 1.5.0 binding, both K
    python tools/gb10_exl3_moe_parity.py --json out.json

Scope, stated explicitly so the receipt can never be read as the other one:

  * binding arity 30 (exllamav3 1.4.7-1.4.9) -> launch is args + num_active, tail ()
  * binding arity 35 (exllamav3 >= 1.5.0)    -> launch is args + num_active +
      (output_scratch=None, fused_base=None, count_lo=1, count_hi=<temp rows>, m_tile=16)
    and the 30-argument call must be rejected (the original "incompatible function
    arguments" failure), so a green run also proves the padding is load-bearing.

Within a launch K is uniform (K_gate == K_up == K_down, one kernel instance), so this
probe runs the full launch once per K in --ks and compares each against the native
exll3_gemv reference built with that same K. It never assumes one K for the stack: a
mixed-bitrate model is per-layer K, and every layer's launch carries its own.

Inputs are the synthetic-but-real packs the existing native fixtures build
(tests/test_native_p2b_moe.py, tools/bench_moe_decode.py): int16 EXL3 trellises with
suh/svh, decoded by both kernels. Parity bar is the repo's cosine >= 0.999.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import torch

sys.path.insert(0, "src")

import vllm_exl3.exl3 as exl3  # noqa: E402

COS_MIN = 0.999
TOPK = 2
vllm_exl3_c: Any = None
exllamav3_ext: Any = None


def _stub_binding(n_args: int):
    """CPU stand-in with the docstring shape pybind11 emits for an unnamed binding."""
    params = ", ".join(f"arg{i}: int" for i in range(n_args))

    class _Stub:
        __doc__ = f"exl3_moe({params}) -> None\n\nexl3_moe"

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def __call__(self, *a):
            self.calls.append(a)

    return _Stub()


def _pack(in_f: int, out_f: int, k: int, device: torch.device):
    """One expert's gate/up/down EXL3 pack, the layout the native fixtures use."""
    return {
        "trellis": torch.randint(
            -32768, 32767, (in_f // 16, out_f // 16, 16 * k),
            dtype=torch.int16, device=device,
        ),
        "suh": (torch.randn(in_f, dtype=torch.float16, device=device) / 64.0),
        "svh": torch.randn(out_f, dtype=torch.float16, device=device),
    }


def _ptr_tensor(packs: list[dict], key: str, device: torch.device) -> torch.Tensor:
    return torch.tensor([p[key].data_ptr() for p in packs], dtype=torch.int64, device=device)


def _reference(x, ids, weights, packs, k, device):
    """Native per-expert reference: silu(gate x) * (up x), then down x, weighted."""
    acc = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=device)
    for e, pack in enumerate(packs):
        g = vllm_exl3_c.exl3_gemv(x, pack["gate"]["trellis"], pack["gate"]["suh"], pack["gate"]["svh"], k, True).float()
        u = vllm_exl3_c.exl3_gemv(x, pack["up"]["trellis"], pack["up"]["suh"], pack["up"]["svh"], k, True).float()
        h = (torch.nn.functional.silu(g) * u).half()
        d = vllm_exl3_c.exl3_gemv(h, pack["down"]["trellis"], pack["down"]["suh"], pack["down"]["svh"], k, True).float()
        acc += weights[:, e : e + 1].float() * d
    return acc


def one_k(binding, arity: int, k: int, *, hidden: int, inter: int, n_exp: int,
          tokens: int, device: torch.device, max_rows: int,
          dry: bool = False) -> dict[str, Any]:
    """Full fused launch at one K, compared to the native map at the same K."""
    packs = [
        {p: _pack(hidden if p != "down" else inter, inter if p != "down" else hidden, k, device)
         for p in ("gate", "up", "down")}
        for _ in range(n_exp)
    ]
    x = (torch.randn(tokens, hidden, dtype=torch.float16, device=device) * 0.1)
    weights = torch.softmax(torch.randn(tokens, n_exp, dtype=torch.float16, device=device), dim=-1)
    ids = weights.topk(TOPK, dim=-1).indices
    w_sel = weights.gather(1, ids)

    flat_expert = ids.reshape(-1)
    order = flat_expert.argsort()
    token_sorted = torch.arange(tokens, device=device, dtype=torch.long).repeat_interleave(TOPK)[order]
    weight_sorted = w_sel.reshape(-1).to(torch.float16)[order]
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=device)
    expert_count.scatter_add_(0, flat_expert, torch.ones_like(flat_expert, dtype=torch.long))

    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=device)
    concurrency = 1 if dry else int(exllamav3_ext.exl3_moe_max_concurrency(device.index or 0))
    temps = (
        torch.zeros(concurrency, max_rows, hidden, dtype=torch.float16, device=device),
        torch.zeros(concurrency, max_rows, hidden, dtype=torch.float16, device=device),
        torch.zeros(concurrency, max_rows, inter, dtype=torch.float16, device=device),
        torch.zeros(concurrency, max_rows, inter, dtype=torch.float16, device=device),
    )
    args = (
        x, out, expert_count, token_sorted, weight_sorted, *temps,
        exl3.MOE_ACT_SILU, k, k, k,
        _ptr_tensor([p["gate"] for p in packs], "trellis", device),
        _ptr_tensor([p["gate"] for p in packs], "suh", device),
        _ptr_tensor([p["gate"] for p in packs], "svh", device),
        _ptr_tensor([p["up"] for p in packs], "trellis", device),
        _ptr_tensor([p["up"] for p in packs], "suh", device),
        _ptr_tensor([p["up"] for p in packs], "svh", device),
        _ptr_tensor([p["down"] for p in packs], "trellis", device),
        _ptr_tensor([p["down"] for p in packs], "suh", device),
        _ptr_tensor([p["down"] for p in packs], "svh", device),
        True, False, True, False, True, False,   # mcg only, the fused kernel's contract
        0.0,
    )
    # The production call expression, with the tail the plugin derives for this binding.
    tail = exl3._exl3_moe_tail(binding, exl3._exl3_moe_temp_rows(temps))
    binding(*args, -1, *tail)
    if dry:
        return {
            "k": k,
            "arity_called": len(binding.calls[-1]),
            "tail": list(tail),
            "count_hi": tail[3] if tail else None,
            "cosine": None,
            "pass": len(binding.calls[-1]) == arity,
        }
    torch.cuda.synchronize()

    ref = _reference(x, ids, w_sel, packs, k, device)
    cos = torch.nn.functional.cosine_similarity(out.view(-1), ref.view(-1), dim=0).item()
    return {
        "k": k,
        "arity_called": len(args) + 1 + len(tail),
        "tail": list(tail),
        "count_hi": tail[3] if tail else None,
        "cosine": round(float(cos), 6),
        "pass": bool(cos >= COS_MIN),
    }


def _dry_run(args, receipt: dict[str, Any]) -> int:
    """CPU-only: the arg assembly and arity math the hardware run depends on.

    No parity claim is made here — the binding is a stub. Run without --dry-run on the
    GB10 for the real comparison against the native map.
    """
    device = torch.device("cpu")
    ok = True
    for n_args in (exl3.EXL3_MOE_ARITY_147, exl3.EXL3_MOE_ARITY_150):
        binding = _stub_binding(n_args)
        tail = exl3._exl3_moe_tail(binding, args.max_rows)
        expected = (() if n_args == exl3.EXL3_MOE_ARITY_147
                    else (None, None, 1, args.max_rows, 16))
        rows = [
            one_k(binding, n_args, k, hidden=args.hidden, inter=args.inter,
                  n_exp=args.experts, tokens=args.tokens, device=device,
                  max_rows=args.max_rows, dry=True)
            for k in args.ks
        ]
        good = tuple(tail) == expected and all(r["pass"] for r in rows)
        ok = ok and good
        print(f"dry arity={n_args} tail={tuple(tail)} expected={expected} "
              f"called={[r['arity_called'] for r in rows]} "
              f"ks={[r['k'] for r in rows]} {'ok' if good else 'MISMATCH'}")
    receipt["pass"] = ok
    receipt["dry_run"] = True
    print(("GB10_EXL3_MOE_PARITY DRYRUN-OK " if ok else "GB10_EXL3_MOE_PARITY DRYRUN-FAIL ") +
          "no hardware parity claim")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4],
                    help="per-layer K values to launch (mixed-K stack, one launch each)")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=64, help="temp buffer rows (= count_hi)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU only: stub binding, verify arg assembly/arity, no parity claim")
    args = ap.parse_args()

    receipt: dict[str, Any] = {"scope": None, "arity": None, "tail_expected": None}
    if args.dry_run:
        return _dry_run(args, receipt)
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device (run on the GB10 host)")
        return 2
    global vllm_exl3_c, exllamav3_ext
    try:
        import exllamav3_ext
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: exllamav3_ext not importable: {exc!r}")
        return 2
    try:
        import vllm_exl3_c
    except Exception as exc:  # noqa: BLE001
        print(f"SKIP: vllm_exl3_c not importable: {exc!r}")
        return 2

    device = torch.device("cuda", torch.cuda.current_device())
    binding = exllamav3_ext.exl3_moe
    arity = exl3._exl3_moe_arity(binding)
    tail = exl3._exl3_moe_tail(binding, args.max_rows)
    expected = () if arity == exl3.EXL3_MOE_ARITY_147 else (
        (None, None, 1, args.max_rows, 16) if arity == exl3.EXL3_MOE_ARITY_150 else None
    )
    receipt.update(
        arity=arity,
        scope=(
            "exllamav3 1.4.x (30-arg binding)" if arity == exl3.EXL3_MOE_ARITY_147
            else "exllamav3 >= 1.5.0 (35-arg binding)" if arity == exl3.EXL3_MOE_ARITY_150
            else "UNKNOWN — stop, arity is not a release the plugin knows"
        ),
        tail= list(tail),
        tail_expected=list(expected) if expected is not None else None,
        doc_has_arg34=("arg34" in (binding.__doc__ or "")),
    )
    print(json.dumps({k: receipt[k] for k in ("scope", "arity", "tail", "tail_expected", "doc_has_arg34")}))

    ok = True
    if expected is None:
        print("FAIL: unknown exl3_moe arity; the plugin cannot pad it")
        ok = False
    elif tuple(tail) != expected:
        print(f"FAIL: tail {tuple(tail)} != expected {expected}")
        ok = False

    if arity == exl3.EXL3_MOE_ARITY_150:
        # The 1.5.0 binding must reject the pre-1.5.0 arity, so the tail is load-bearing.
        try:
            binding(*(torch.zeros(1, dtype=torch.float16, device=device),) * 30)
            receipt["wrong_arity_rejected"] = False
            print("FAIL: 1.5.0 binding accepted the 30-argument call")
            ok = False
        except TypeError as exc:
            receipt["wrong_arity_rejected"] = True
            receipt["wrong_arity_error"] = str(exc)[:160]
            print(f"OK: 30-arg call rejected: {str(exc)[:80]}")

    if ok:
        receipt["layers"] = [
            one_k(binding, arity, k, hidden=args.hidden, inter=args.inter, n_exp=args.experts,
                  tokens=args.tokens, device=device, max_rows=args.max_rows)
            for k in args.ks
        ]
        for layer in receipt["layers"]:
            print(f"K={layer['k']}: {layer['arity_called']} args, cosine {layer['cosine']:.5f}")
            ok = ok and layer["pass"]

    receipt["pass"] = ok
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(receipt, fh, indent=2)
    print(("GB10_EXL3_MOE_PARITY PASS " if ok else "GB10_EXL3_MOE_PARITY FAIL ") +
          f"arity={receipt['arity']} ks={args.ks} cos_min={COS_MIN}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
