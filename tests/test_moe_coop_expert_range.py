"""Guard: coop path must range-filter plugin sentinels (n_exp), not min=-1."""

from pathlib import Path


def test_coop_path_passes_local_expert_range_not_unfiltered():
    src = (Path(__file__).resolve().parents[1] / "src" / "vllm_exl3" / "exl3.py").read_text(
        encoding="utf-8"
    )
    start = src.index("exl3_moe_coop fast path")
    chunk = src[start : start + 2500]
    assert "xh, sel_c, rw_c, 0, int(n_exp), hidden" in chunk
    assert "xh, sel_c, rw_c, -1, -1, hidden" not in chunk
