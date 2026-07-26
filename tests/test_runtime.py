"""Tests for device resolution and seeding.

These run without torch installed and without a GPU. Where torch behaviour
matters, it is simulated by patching the module's lazy importer rather than by
requiring the ``inference`` extra.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Never

import pytest

from panaf_ape_detection import runtime
from panaf_ape_detection.runtime import (
    ALLOW_CPU_ENV,
    CpuInferenceRefusedError,
    DeviceUnavailableError,
    available_devices,
    module_device,
    require_accelerator,
    resolve_device,
    set_seeds,
)
from panaf_ape_detection.types import DeviceKind


def fake_torch(*, cuda: bool = False, mps: bool = False) -> SimpleNamespace:
    """Build a stand-in torch module reporting the given accelerator support."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            manual_seed_all=lambda _seed: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        manual_seed=lambda _seed: None,
    )


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_torch", lambda: None)


# --------------------------------------------------------------------------- #
# available_devices
# --------------------------------------------------------------------------- #


def test_cpu_is_always_available(no_torch):
    assert available_devices() == {DeviceKind.CPU}


def test_reports_cuda_when_torch_says_so(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch(cuda=True))

    assert available_devices() == {DeviceKind.CPU, DeviceKind.CUDA}


def test_reports_mps_when_torch_says_so(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch(mps=True))

    assert available_devices() == {DeviceKind.CPU, DeviceKind.MPS}


def test_auto_is_never_reported_as_available():
    assert DeviceKind.AUTO not in available_devices()


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #


def test_auto_falls_back_to_cpu_without_torch(no_torch):
    assert resolve_device(DeviceKind.AUTO) is DeviceKind.CPU


def test_auto_prefers_cuda_over_mps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch(cuda=True, mps=True))

    assert resolve_device(DeviceKind.AUTO) is DeviceKind.CUDA


def test_auto_picks_mps_when_only_mps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "_torch", lambda: fake_torch(mps=True))

    assert resolve_device(DeviceKind.AUTO) is DeviceKind.MPS


def test_explicit_cpu_always_resolves(no_torch):
    assert resolve_device(DeviceKind.CPU) is DeviceKind.CPU


@pytest.mark.parametrize("requested", [DeviceKind.CUDA, DeviceKind.MPS])
def test_unavailable_device_raises_rather_than_downgrading(no_torch, requested: DeviceKind):
    """A silent CPU fallback would invalidate every timing claim in a run."""
    with pytest.raises(DeviceUnavailableError, match=requested.value):
        resolve_device(requested)


def test_unavailable_device_error_lists_alternatives(no_torch):
    with pytest.raises(DeviceUnavailableError, match="available: cpu"):
        resolve_device(DeviceKind.CUDA)


def test_resolve_never_returns_auto(monkeypatch: pytest.MonkeyPatch):
    for torch_state in (None, fake_torch(), fake_torch(cuda=True), fake_torch(mps=True)):
        monkeypatch.setattr(runtime, "_torch", lambda t=torch_state: t)
        assert resolve_device(DeviceKind.AUTO) is not DeviceKind.AUTO


def test_broken_torch_import_is_treated_as_absent(monkeypatch: pytest.MonkeyPatch):
    def explode() -> None:
        raise RuntimeError("broken install")

    monkeypatch.setattr(runtime, "_torch", explode)

    with pytest.raises(RuntimeError):
        available_devices()


# --------------------------------------------------------------------------- #
# set_seeds
# --------------------------------------------------------------------------- #


def test_seeding_makes_python_random_reproducible(no_torch):
    set_seeds(42)
    first = [random.random() for _ in range(5)]
    set_seeds(42)
    second = [random.random() for _ in range(5)]

    assert first == second


def test_seeding_works_without_torch(no_torch):
    set_seeds(0)  # must not raise


def test_seeding_calls_torch_when_present(monkeypatch: pytest.MonkeyPatch):
    seen: list[int] = []
    torch = fake_torch(cuda=True)
    torch.manual_seed = seen.append
    torch.cuda.manual_seed_all = seen.append
    monkeypatch.setattr(runtime, "_torch", lambda: torch)

    set_seeds(7)

    assert seen == [7, 7]


def test_negative_seed_is_rejected(no_torch):
    with pytest.raises(ValueError, match="non-negative"):
        set_seeds(-1)


# --------------------------------------------------------------------------- #
# module_device
#
# Regression guard for a verified upstream bug: PyTorch-Wildlife 1.3.0 accepts
# `device=` and never applies it, so a model built with device="cuda" runs on
# CPU. Detecting that means checking the tensors, not the claim -- and the real
# weights sit at `.predictor.model.model`, behind an `AutoBackend` whose own
# `parameters()` yields nothing.
# --------------------------------------------------------------------------- #


class _FakeParameter:
    def __init__(self, device: str) -> None:
        self.device = device


class _FakeModule:
    """Stand-in for a torch module holding parameters on given devices."""

    def __init__(self, *devices: str) -> None:
        self._devices = devices

    def parameters(self) -> Iterator[_FakeParameter]:
        return (_FakeParameter(device) for device in self._devices)


class _EmptyWrapper:
    """Stand-in for ultralytics' AutoBackend: has parameters(), yields none."""

    def __init__(self, inner: object) -> None:
        self.model = inner

    def parameters(self) -> Iterator[_FakeParameter]:
        return iter(())


def test_module_device_reads_a_plain_module():
    assert module_device(_FakeModule("cpu")) == "cpu"


def test_module_device_finds_nested_weights():
    detector = SimpleNamespace(predictor=SimpleNamespace(model=_FakeModule("mps:0")))

    assert module_device(detector) == "mps:0"


def test_module_device_descends_past_an_empty_wrapper():
    """The ultralytics shape: AutoBackend.parameters() is empty, weights are deeper."""
    detector = SimpleNamespace(
        predictor=SimpleNamespace(model=_EmptyWrapper(_FakeModule("cuda:0")))
    )

    assert module_device(detector) == "cuda:0"


def test_module_device_reports_mixed_placement():
    assert module_device(_FakeModule("cpu", "cuda:0")) == "mixed"


def test_module_device_returns_none_when_no_parameters_exist():
    assert module_device(SimpleNamespace(nothing="here")) is None


def test_module_device_tolerates_a_raising_parameters():
    class Hostile:
        def parameters(self) -> Never:
            raise RuntimeError("no")

    assert module_device(Hostile()) is None


def test_module_device_survives_a_reference_cycle():
    node = SimpleNamespace(model=None)
    node.model = node  # self-referential wrapper must not hang

    assert module_device(node) is None


# --------------------------------------------------------------------------- #
# CPU refusal
#
# Project policy: inference never runs on CPU. A CPU run does not crash, it just
# takes hours, so it is refused rather than warned about.
# --------------------------------------------------------------------------- #


def test_require_accelerator_refuses_cpu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ALLOW_CPU_ENV, raising=False)

    with pytest.raises(CpuInferenceRefusedError, match="Colab"):
        require_accelerator(DeviceKind.CPU)


@pytest.mark.parametrize("device", [DeviceKind.CUDA, DeviceKind.MPS])
def test_require_accelerator_passes_accelerators_through(
    device: DeviceKind, monkeypatch: pytest.MonkeyPatch
):
    """MPS counts: the policy is about CPU, not about being local."""
    monkeypatch.delenv(ALLOW_CPU_ENV, raising=False)

    assert require_accelerator(device) is device


def test_the_escape_hatch_allows_cpu_but_only_when_set_to_one(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ALLOW_CPU_ENV, "1")
    assert require_accelerator(DeviceKind.CPU) is DeviceKind.CPU

    # Anything else is not opting in -- "0" or "false" must still refuse.
    monkeypatch.setenv(ALLOW_CPU_ENV, "0")
    with pytest.raises(CpuInferenceRefusedError):
        require_accelerator(DeviceKind.CPU)
