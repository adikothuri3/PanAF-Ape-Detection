---
tags: [reference, model, verified]
status: verified
source: PyTorch-Wildlife 1.3.0 source, read from the installed package
updated: 2026-07-24
---

# MegaDetector Variants

Verified by reading the installed **PyTorch-Wildlife 1.3.0** source, not from documentation.

## Class vocabulary — the whole output space

```python
{0: "animal", 1: "person", 2: "vehicle"}
```

**No species. No individuals. No behaviour.** A chimpanzee, a gorilla and a duiker are all `animal`.
Behaviour labels come from the dataset ([[PanAf500 Action Labels]]), never the model.

## `MegaDetectorV6` — YOLO / RT-DETR weights

| `version=` | Weights file | Input |
|---|---|---|
| `MDV6-yolov9-c` | `MDV6b-yolov9-c.pt` | 1280 |
| `MDV6-yolov9-e` | `MDV6-yolov9-e-1280.pt` | 1280 |
| `MDV6-yolov10-c` | `MDV6-yolov10-c.pt` | 1280 |
| `MDV6-yolov10-e` | `MDV6-yolov10-e-1280.pt` | 1280 |
| `MDV6-rtdetr-c` | `MDV6b-rtdetr-c.pt` | 1280 |

## `MegaDetectorV6Apache` — Apache-licensed RT-DETR weights

| `version=` | Weights file | Input |
|---|---|---|
| `MDV6-apa-rtdetr-c` | `MDV6-apa-rtdetr-c.pth` | 640 |
| `MDV6-apa-rtdetr-e` | `MDV6-apa-rtdetr-e.pth` | 640 |

Weights download from Zenodo record `15398270` on first use. They are **never committed** —
`.gitignore` blocks `*.pt` / `*.pth`.

## ⚠️ `device=` is ignored — the model silently runs on CPU

The single most consequential trap in this stack. `MegaDetectorV6(device="cuda")` stores the value
and never applies it: in `yolov8_base._load_model` the line that would is commented out —
`# self.predictor.args.device = device # Will uncomment later`.

Nothing raises. `detector.device` still says `"cuda"`. **On a Colab GPU runtime you would get CPU
speed while every log line claimed CUDA**, and the run metadata would record a device the model
never touched.

The fix, applied *after* the model is set up — all three lines are required:

```python
torch_device = torch.device(device)
detector.predictor.model.to(torch_device)  # weights
detector.predictor.device = torch_device  # inputs — omit this and the forward pass crashes
detector.predictor.args.device = device  # args, for anything reading them later
```

Verify rather than assume: `runtime.module_device(detector)` reads the actual parameter devices.
`scripts/smoke_detect.py` is the working reference. Measured on an M1: 2.13 s CPU → 1.53 s MPS.

## ⚠️ Both upstream defaults are broken

- `MegaDetectorV6.__init__` defaults to `version='yolov9c'`
- `MegaDetectorV6Apache.__init__` defaults to `version='MDV6-rtdetr-x-apache'`

**Neither string is accepted by its own method's validation** — both fall through to
`raise ValueError`. Constructing either class without an explicit `version=` fails outright.

This is why `model.variant` is a required field in `configs/*.yaml` and is never defaulted in
application code. An upstream default is not a specification.

## Licensing differs by variant

The YOLOv9/YOLOv10-derived weights and the Apache RT-DETR weights do **not** carry the same terms —
`MegaDetectorV6Apache` exists precisely because of that. Verify the licence of the exact variant
before any deployment or commercial use. See [licensing docs](../docs/licensing.md).

## Related

[[PanAf500 Action Labels]] · [[PyTorch-Wildlife and MegaDetector]] · [model docs](../docs/model.md) · [[Phase 1 Task Spec]]
