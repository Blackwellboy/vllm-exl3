from __future__ import annotations

from types import SimpleNamespace

from vllm_exl3.bf16_madv_compat import (
    _loaded_shard_ids,
    install_mixed_bf16_madv_compat,
)


def test_loaded_shard_ids_handles_fused_and_qkv() -> None:
    w = SimpleNamespace(shape=(30, 4))
    assert _loaded_shard_ids(w, None, 3, [10, 10, 10]) == {0, 1, 2}
    assert _loaded_shard_ids(w, "k", 3, [10, 10, 10]) == {1}
    assert _loaded_shard_ids(w, (0, 1), 3, [10, 10, 10]) == {0, 1}


def test_mixed_bf16_wrapper_syncs_and_reclaims_consumed_source() -> None:
    events: list[str] = []

    class FakeLinearMethod:
        def _make_weight_loader(
            self,
            suffix,
            n_shards,
            output_partition_sizes,
            is_row_parallel,
            bf16_shards,
            layer=None,
            is_qkv_parallel=False,
        ):
            def loader(param, loaded_weight, loaded_shard_id=None):
                events.append("copy")

            return loader

    class FakeStream:
        def synchronize(self):
            events.append("sync")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(current_stream=lambda: FakeStream())
    )
    module = SimpleNamespace(
        Exl3LinearMethod=FakeLinearMethod,
        torch=fake_torch,
        _madv_dontneed_cpu_tensor=lambda src: events.append("madv"),
    )

    install_mixed_bf16_madv_compat(module)
    method = FakeLinearMethod()
    loader = method._make_weight_loader(
        "weight", 2, [10, 10], False, [1], None, False
    )
    param = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    source = SimpleNamespace(
        device=SimpleNamespace(type="cpu"),
        shape=(10, 4),
    )

    loader(param, source, 1)

    assert events == ["copy", "sync", "madv"]
    assert module._vllm_exl3_bf16_madv_compat_installed is True


def test_mixed_bf16_wrapper_does_not_reclaim_non_bf16_shard() -> None:
    events: list[str] = []

    class FakeLinearMethod:
        def _make_weight_loader(self, *args, **kwargs):
            def loader(param, loaded_weight, loaded_shard_id=None):
                events.append("copy")

            return loader

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_stream=lambda: SimpleNamespace(
                synchronize=lambda: events.append("sync")
            )
        )
    )
    module = SimpleNamespace(
        Exl3LinearMethod=FakeLinearMethod,
        torch=fake_torch,
        _madv_dontneed_cpu_tensor=lambda src: events.append("madv"),
    )
    install_mixed_bf16_madv_compat(module)
    loader = FakeLinearMethod()._make_weight_loader(
        "weight", 2, [10, 10], False, [1], None, False
    )
    param = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    source = SimpleNamespace(device=SimpleNamespace(type="cpu"), shape=(10, 4))

    loader(param, source, 0)

    assert events == ["copy"]
