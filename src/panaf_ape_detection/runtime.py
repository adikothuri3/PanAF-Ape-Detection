"""Device selection and seeding.

The two runtime decisions every pipeline stage needs, made in exactly one place
so that a run cannot end up half on GPU and half on CPU, or seeded twice with
different values.

Every machine-learning import in this module is **lazy** — inside the function
that needs it. Importing :mod:`panaf_ape_detection.runtime` must stay cheap and
must succeed without the ``inference`` extra installed.
"""

from __future__ import annotations

import importlib
import logging
import random
from typing import TYPE_CHECKING

from panaf_ape_detection.types import DeviceKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

__all__ = [
    "DeviceUnavailableError",
    "available_devices",
    "module_device",
    "resolve_device",
    "set_seeds",
]

logger = logging.getLogger(__name__)


class DeviceUnavailableError(RuntimeError):
    """Raised when an explicitly requested device is not available.

    Deliberately an error rather than a silent downgrade to CPU: a run that was
    asked for CUDA and quietly used CPU produces timings that mean nothing and
    a ``RunMetadata`` record that misrepresents what happened.
    """


def _torch() -> ModuleType | None:
    """Import torch if it is installed, otherwise return ``None``.

    Uses :func:`importlib.import_module` rather than a bare ``import torch`` so
    that type checking gives the same answer whether or not the optional
    ``inference`` extra happens to be installed in the environment mypy runs in.
    A direct import makes the return type ``Any`` when torch is missing and a
    concrete module when it is present, so any cast or ignore is correct in one
    environment and an error in the other.
    """
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None
    except Exception:  # pragma: no cover - broken install
        logger.warning("torch is installed but failed to import", exc_info=True)
        return None


def available_devices() -> set[DeviceKind]:
    """Return the concrete devices usable in this environment.

    ``cpu`` is always present. ``cuda`` and ``mps`` are reported only when torch
    is installed and says they are available.

    Returns:
        A set of concrete :class:`DeviceKind` values, never containing ``auto``.
    """
    devices = {DeviceKind.CPU}
    torch = _torch()
    if torch is None:
        return devices

    try:
        if torch.cuda.is_available():
            devices.add(DeviceKind.CUDA)
    except Exception:  # pragma: no cover - driver-level failure
        logger.debug("torch.cuda.is_available() raised", exc_info=True)

    try:
        if torch.backends.mps.is_available():
            devices.add(DeviceKind.MPS)
    except (AttributeError, Exception):  # pragma: no cover - older torch
        logger.debug("torch.backends.mps.is_available() raised", exc_info=True)

    return devices


def resolve_device(requested: DeviceKind) -> DeviceKind:
    """Resolve a configured device to the concrete device to run on.

    ``auto`` picks the fastest available, preferring CUDA, then Apple MPS, then
    CPU. An explicit request that is unavailable raises rather than falling back.

    Args:
        requested: The value of ``model.device`` from configuration.

    Returns:
        A concrete :class:`DeviceKind` — never ``auto``.

    Raises:
        DeviceUnavailableError: If a specific device was requested and is not
            available here.
    """
    devices = available_devices()

    if requested is DeviceKind.AUTO:
        for candidate in (DeviceKind.CUDA, DeviceKind.MPS, DeviceKind.CPU):
            if candidate in devices:
                logger.info("device 'auto' resolved to '%s'", candidate.value)
                return candidate
        raise AssertionError("cpu is always available")  # pragma: no cover

    if requested not in devices:
        available = ", ".join(sorted(device.value for device in devices))
        msg = (
            f"device {requested.value!r} was requested but is not available "
            f"(available: {available}). Set model.device to 'auto' to pick "
            f"automatically, or to one of the available devices."
        )
        raise DeviceUnavailableError(msg)

    return requested


def module_device(module: object) -> str | None:
    """Return the device a torch module's parameters actually live on.

    Asking a library which device it is using is not the same as asking where
    the weights are. PyTorch-Wildlife 1.3.0 accepts a ``device=`` argument,
    stores it, and then never applies it — the line that would is commented out
    in ``yolov8_base._load_model``. A model constructed with ``device="cuda"``
    therefore runs on CPU while every report claims CUDA.

    This function checks the tensors rather than the claim, so that discrepancy
    is detectable instead of silent.

    Args:
        module: Anything exposing ``.parameters()``, or an object wrapping such
            a module one or two attributes deep (``.model``, ``.model.model``).

    Returns:
        A device string such as ``"cpu"`` or ``"mps:0"``, ``"mixed"`` when
        parameters are split across devices, or ``None`` when no parameters
        could be found.
    """
    # Breadth-first over the usual wrapper attribute names. Depth matters:
    # ultralytics nests the real weights at `.predictor.model.model`, and the
    # intermediate `AutoBackend.parameters()` yields nothing at all — so a
    # shallow search silently reports "unknown" and hides the very mismatch
    # this function exists to detect.
    wrapper_attributes = ("predictor", "model", "net", "module")
    queue: list[tuple[object, int]] = [(module, 0)]
    seen: set[int] = set()
    max_depth = 4

    while queue:
        candidate, depth = queue.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        parameters = getattr(candidate, "parameters", None)
        if callable(parameters):
            try:
                devices = {str(parameter.device) for parameter in parameters()}
            except Exception:
                devices = set()
            # An empty set means a wrapper that owns no tensors: keep descending.
            if len(devices) == 1:
                return devices.pop()
            if devices:
                return "mixed"

        if depth < max_depth:
            for attribute in wrapper_attributes:
                queue.append((getattr(candidate, attribute, None), depth + 1))

    return None


def set_seeds(seed: int) -> None:
    """Seed every random number generator this project might use.

    Covers Python's :mod:`random`, NumPy and torch (CPU and CUDA), each guarded
    so the call works in the lightweight environment.

    Seeding is not the same as determinism: GPU kernels can still produce
    slightly different results run to run. See ``docs/obsidian/05 Technical/reproducibility.md``.

    Args:
        seed: The seed from ``project.seed``. Must be non-negative.

    Raises:
        ValueError: If *seed* is negative.
    """
    if seed < 0:
        msg = f"seed must be non-negative, got {seed}"
        raise ValueError(msg)

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:
        logger.debug("numpy not installed; skipping numpy seeding")
    else:
        np.random.seed(seed)

    torch = _torch()
    if torch is None:
        logger.debug("torch not installed; skipping torch seeding")
        return

    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - driver-level failure
        logger.debug("torch.cuda.manual_seed_all failed", exc_info=True)
