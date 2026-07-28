"""MegaDetector V6 via PyTorch-Wildlife.

Every heavyweight import is inside a method, so this module can be imported
without the ``inference`` extra installed.

**This adapter exists mostly to work around two verified upstream defects.**
PyTorch-Wildlife 1.3.0:

1. Ships default ``version=`` strings that its own validation rejects, so the
   variant must always be passed explicitly.
2. Accepts ``device=``, stores it, and **never applies it** -- the line that
   would is commented out in ``yolov8_base._load_model``. The weights load on
   CPU, nothing raises, and the object still reports the device you asked for.
   On a Colab GPU runtime that is CPU speed with CUDA in the metadata.

So this adapter forces the device after construction and then *verifies* it by
inspecting the tensors, refusing to run if they disagree. See
``docs/obsidian/05 Technical/model.md``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from panaf_ape_detection.runtime import module_device, require_accelerator, resolve_device
from panaf_ape_detection.types import BoundingBox, Detection, DeviceKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

__all__ = ["MegaDetectorV6Runner"]

logger = logging.getLogger(__name__)

CLASS_NAMES: dict[int, str] = {0: "animal", 1: "person", 2: "vehicle"}
"""MegaDetector's entire output space. Not species. Not behaviour."""


DEFAULT_DETECTION_THRESHOLD = 0.2
"""The threshold PyTorch-Wildlife applies when none is passed.

``YOLOV8Base.single_image_detection(img, det_conf_thres=0.2)``. Repeated here
only so the default is visible; real runs pass the configured value.

**This was a silent bug for one full round of experiments.** The adapter filtered
detections at ``model.confidence_threshold`` *after* inference but never told the
model, so every run inferred at 0.2. Because the configured value also happened to
be 0.2, the numbers were correct -- but a sweep down to 0.05 returned byte-identical
results, which is how it was caught. Same shape as the ``device=`` bug: a value
accepted, stored, and never applied where it mattered.
"""


class DeviceMismatchError(RuntimeError):
    """Raised when the weights are not on the device that was requested."""


class MegaDetectorV6Runner:
    """Runs MegaDetector V6 over frames, on a verified device.

    Attributes are exposed to satisfy
    :class:`~panaf_ape_detection.inference.base.Detector`.
    """

    def __init__(
        self,
        variant: str,
        *,
        device: DeviceKind = DeviceKind.AUTO,
        model_name: str = "MegaDetectorV6",
        confidence_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    ) -> None:
        """Load the detector and pin it to a verified device.

        Args:
            variant: Exact weight identifier, e.g. ``MDV6-yolov9-c``. Required --
                the library's own defaults raise ``ValueError``.
            device: Requested device; ``auto`` resolves by availability.
            model_name: Detector family, recorded in run metadata.
            confidence_threshold: Score below which the *model* discards a box.
                Passed to every inference call. **This must be set from
                configuration**: the library's own default is 0.2, so leaving it
                alone silently pins every run to 0.2 no matter what the config
                says, and a threshold sweep below that returns identical results.

        Raises:
            ValueError: If *variant* is not accepted by the installed library.
            DeviceMismatchError: If the weights cannot be placed on the resolved
                device.
        """
        self._model_name = model_name
        self._variant = variant
        self._confidence_threshold = confidence_threshold
        # Policy: inference never runs on CPU. See runtime.require_accelerator.
        self._device = require_accelerator(resolve_device(device))
        self._frames_seen = 0
        self._seconds_spent = 0.0

        from PytorchWildlife.models import detection as pw

        logger.info("loading %s (%s) on %s", model_name, variant, self._device.value)
        try:
            self._model = pw.MegaDetectorV6(
                device=self._device.value, pretrained=True, version=variant
            )
        except ValueError as exc:
            msg = (
                f"{variant!r} is not a valid MegaDetector V6 variant: {exc}. "
                "Valid variants are listed in docs/obsidian/05 Technical/model.md."
            )
            raise ValueError(msg) from exc

        self._force_device()
        self._silence_per_frame_logging()

    def _silence_per_frame_logging(self) -> None:
        """Stop Ultralytics printing one line per frame.

        Its predictor does ``if self.args.verbose: LOGGER.info(...)`` once per
        batch, which here is once per *frame*::

            0: 1280x1280 1 animal, 45.2ms

        Harmless at ten frames and a serious problem at scale: a 500-clip run is
        180,000 lines. They bury the pipeline's own progress reporting, and a
        Colab cell holding that much output slows the browser badly and is a
        plausible contributor to a dropped session. Nothing reads these lines --
        per-frame timing is recorded in run metadata instead.

        Both levers are set. ``args.verbose`` is the one that stops the line
        above, and the logger level covers anything else the library decides to
        print during a long run.
        """
        import logging as _logging

        predictor = getattr(self._model, "predictor", None)
        arguments = getattr(predictor, "args", None)
        if arguments is not None:
            arguments.verbose = False

        _logging.getLogger("ultralytics").setLevel(_logging.WARNING)

    def _force_device(self) -> None:
        """Move the weights onto the resolved device and verify they arrived.

        Three assignments are required. Moving the model alone leaves the
        *inputs* on CPU, and the next forward pass dies with
        ``input(device='cpu') and weight(device='mps:0') must be on the same
        device``.
        """
        import torch

        target = self._device.value
        # setup_model() runs lazily, so a first inference is needed before the
        # parameters exist to inspect or move.
        self._warm_up()

        actual = module_device(self._model)
        if actual is not None and actual.startswith(target):
            logger.info("weights verified on %s", actual)
            return

        logger.warning(
            "PyTorch-Wildlife ignored device=%r (weights on %r); forcing it", target, actual
        )
        torch_device = torch.device(target)
        self._model.predictor.model.to(torch_device)
        self._model.predictor.device = torch_device
        self._model.predictor.args.device = target

        moved = module_device(self._model)
        if moved is None or not moved.startswith(target):
            msg = (
                f"could not place the model on {target!r}; weights report {moved!r}. "
                "Refusing to run, because run metadata would misreport the device."
            )
            raise DeviceMismatchError(msg)
        logger.info("weights forced onto %s", moved)

    def _warm_up(self) -> None:
        """Run one tiny inference so the lazily-built predictor exists."""
        import numpy as np

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self._model.single_image_detection(blank, det_conf_thres=self._confidence_threshold)
        except Exception:
            logger.debug("warm-up inference failed; continuing", exc_info=True)

    @property
    def name(self) -> str:
        """Detector family."""
        return self._model_name

    @property
    def variant(self) -> str:
        """Exact weight identifier."""
        return self._variant

    @property
    def device(self) -> DeviceKind:
        """The device the weights were verified to be on."""
        return self._device

    @property
    def verified_device(self) -> str | None:
        """The raw device string the weights report, e.g. ``"mps:0"``."""
        return module_device(self._model)

    @property
    def seconds_per_frame(self) -> float:
        """Mean inference time per frame so far."""
        return self._seconds_spent / self._frames_seen if self._frames_seen else 0.0

    def detect(self, frame: np.ndarray[Any, Any]) -> list[Detection]:
        """Detect objects in one BGR frame.

        Args:
            frame: Height-by-width-by-3 BGR array.

        Returns:
            Unfiltered detections. Boxes are clamped to the frame, because a
            detector may predict slightly outside it and
            :class:`~panaf_ape_detection.types.FrameDetections` rejects that.
        """
        started = time.perf_counter()
        # det_conf_thres is not optional in practice: omitting it takes the
        # library's 0.2 default, which is a second threshold nothing in this
        # project controls. See DEFAULT_DETECTION_THRESHOLD.
        raw = self._model.single_image_detection(frame, det_conf_thres=self._confidence_threshold)
        self._seconds_spent += time.perf_counter() - started
        self._frames_seen += 1

        height, width = frame.shape[:2]
        return self._to_detections(raw, width=width, height=height)

    @staticmethod
    def _to_detections(raw: object, *, width: int, height: int) -> list[Detection]:
        """Convert PyTorch-Wildlife output into the project's schema.

        The library returns a dict containing a ``supervision.Detections``
        object. Only the fields this project needs are read, so an upstream
        addition cannot break the conversion.
        """
        if not isinstance(raw, dict):
            return []
        payload = raw.get("detections")
        if payload is None:
            return []

        boxes = getattr(payload, "xyxy", None)
        if boxes is None or len(boxes) == 0:
            return []
        confidences = getattr(payload, "confidence", None)
        class_ids = getattr(payload, "class_id", None)

        detections: list[Detection] = []
        for index, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = (float(v) for v in box[:4])
            x_min, y_min = max(0.0, x_min), max(0.0, y_min)
            x_max, y_max = min(float(width), x_max), min(float(height), y_max)
            if x_max <= x_min or y_max <= y_min:
                continue

            score = float(confidences[index]) if confidences is not None else 1.0
            category = int(class_ids[index]) if class_ids is not None else 0
            detections.append(
                Detection(
                    box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                    confidence=min(1.0, max(0.0, score)),
                    category_id=category,
                    category_name=CLASS_NAMES.get(category, str(category)),
                )
            )
        return detections
