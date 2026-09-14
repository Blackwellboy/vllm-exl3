"""CPU tests for the exl3_moe binding-arity detection that keeps the fused launch working
across exllamav3 1.4.x (30 positional arguments) and 1.5.0 (35)."""

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


def _pybind_doc(n_args: int) -> str:
    # Shape of the docstring pybind11 generates for an unnamed-argument binding.
    params = ", ".join(
        f"arg{i}: torch.Tensor" if i < 9 else f"arg{i}: typing.SupportsInt"
        for i in range(n_args)
    )
    return f"exl3_moe({params}) -> None\n\nexl3_moe"


class _Bound:
    def __init__(self, doc: str) -> None:
        self.__doc__ = doc
        self.calls: list[tuple] = []

    def __call__(self, *args):
        self.calls.append(args)


def test_arity_from_pybind_docstring():
    assert exl3._exl3_moe_arity(_Bound(_pybind_doc(30))) == 30
    assert exl3._exl3_moe_arity(_Bound(_pybind_doc(35))) == 35


def test_arity_from_python_signature_and_unknown():
    def thirty(*a):
        return None

    assert exl3._exl3_moe_arity(thirty) is None

    def named(a, b, c):
        return None

    assert exl3._exl3_moe_arity(named) == 3


def test_tail_is_empty_for_1_4_x():
    assert exl3._exl3_moe_tail(_Bound(_pybind_doc(exl3.EXL3_MOE_ARITY_147)), 2048) == ()


def test_tail_reproduces_the_all_fused_launch_for_1_5_0():
    tail = exl3._exl3_moe_tail(_Bound(_pybind_doc(exl3.EXL3_MOE_ARITY_150)), 2048)
    assert tail == (None, None, 1, 2048, 16)


def test_unknown_arity_gets_no_tail():
    def fn(*a):
        return None

    assert exl3._exl3_moe_tail(fn, 2048) == ()


def test_launch_pads_only_the_1_5_0_binding(monkeypatch):
    # Drive the exact call expression used in apply_exl3_fused_moe.
    for n_args, expect in ((30, 30), (35, 35)):
        fn = _Bound(_pybind_doc(n_args))
        args = tuple(range(29))
        n_active_host = -1 if exl3._exl3_moe_accepts_num_active(fn) else None
        tail = exl3._exl3_moe_tail(fn, 2048)
        if tail and n_active_host is None:
            n_active_host = -1
        if n_active_host is not None:
            fn(*args, n_active_host, *tail)
        else:
            fn(*args)
        assert len(fn.calls[0]) == expect, (n_args, len(fn.calls[0]))
        if n_args == 35:
            assert fn.calls[0][29] == -1
            assert fn.calls[0][30:] == (None, None, 1, 2048, 16)


def test_temp_rows_from_buffers_or_default():
    temps = (torch.zeros(2, 512, 8), None, None, None)
    assert exl3._exl3_moe_temp_rows(temps) == 512
    assert exl3._exl3_moe_temp_rows((None, None, None, None)) == exl3.TEMP_ROWS_FUSED
    assert exl3._exl3_moe_temp_rows(None) == exl3.TEMP_ROWS_FUSED
