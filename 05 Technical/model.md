# Model

## What MegaDetector is

MegaDetector is a general-purpose **animal detector** for camera-trap imagery. It draws boxes and
assigns each one to a class from a three-item vocabulary, verified from the installed
PyTorch-Wildlife 1.3.0 `MegaDetectorV6.CLASS_NAMES`:

```python
{0: "animal", 1: "person", 2: "vehicle"}
```

That is the entire output space. Everything below follows from it.

## What MegaDetector is not

**It does not identify species.** There is no chimpanzee class and no gorilla class. A chimpanzee,
a gorilla, a duiker, a bird and an elephant all come back as `animal`. If you want to know which
species is in a PanAf500 box, read the dataset's own annotation — never the detector's output.

**It does not identify individuals.** MegaDetector has no notion of identity across frames. Track
identities come from a tracker applied afterwards, and are stable only within a single clip.

**It does not recognise behaviour.** It cannot tell sitting from climbing. The behaviour labels
this project displays are read from PanAf500's annotations and rendered beside the box. In the
annotated clips, the box is a prediction and the behaviour label is ground truth — they must never
be presented as if both came from the model.

Stating this plainly matters because "MegaDetector detected 40 chimpanzees displaying tool use" is
a sentence this pipeline is structurally incapable of supporting, and it is exactly the sentence a
reader will infer from a video with boxes and behaviour labels on it. Caption accordingly.

## Why pretrained inference is the Phase 1 baseline

No fine-tuning happens in this phase. That is a deliberate choice, not a shortcut:

1. **A baseline you skip is a baseline you can never claim.** If a fine-tuned model later reaches
   some accuracy, the only way to show the fine-tuning did anything is to have measured the
   pretrained model on the same clips first.
2. **It may be sufficient.** MegaDetector was trained on a very large, diverse camera-trap corpus.
   Great apes in camera-trap footage are plausibly in-distribution. Finding out costs a day; not
   finding out costs weeks of unnecessary training.
3. **It localises the actual problem.** If pretrained detection is already reliable, the interesting
   work is tracking and behaviour, not detection. If it fails, Phase 1 tells you *how* it fails,
   which determines whether the fix is a threshold, a different variant, or real fine-tuning.
4. **It is cheap and reversible.** Inference needs no labelled training split, no GPU-hours, and no
   hyperparameter search. Nothing about it forecloses fine-tuning later.
5. **Fine-tuning without a baseline invites self-deception.** With nothing to compare against,
   every number looks like progress.

## Variants

PyTorch-Wildlife exposes two MegaDetector V6 classes. Verified from the installed 1.3.0 source:

### `MegaDetectorV6` (YOLO / RT-DETR weights)

| `version=` string | Weights file | Input size |
| --- | --- | --- |
| `MDV6-yolov9-c` | `MDV6b-yolov9-c.pt` | 1280 |
| `MDV6-yolov9-e` | `MDV6-yolov9-e-1280.pt` | 1280 |
| `MDV6-yolov10-c` | `MDV6-yolov10-c.pt` | 1280 |
| `MDV6-yolov10-e` | `MDV6-yolov10-e-1280.pt` | 1280 |
| `MDV6-rtdetr-c` | `MDV6b-rtdetr-c.pt` | 1280 |

### `MegaDetectorV6Apache` (Apache-licensed RT-DETR weights)

| `version=` string | Weights file | Input size |
| --- | --- | --- |
| `MDV6-apa-rtdetr-c` | `MDV6-apa-rtdetr-c.pth` | 640 |
| `MDV6-apa-rtdetr-e` | `MDV6-apa-rtdetr-e.pth` | 640 |

Weights are downloaded on first use from Zenodo record `15398270`. They are **not** committed here
and `.gitignore` blocks `*.pt` / `*.pth` to keep it that way.

The `-c` variants are the smaller/faster ones and `-e` the larger; the exact speed–accuracy
trade-off on this footage is an empirical question Phase 1 does not attempt to answer.

### ⚠️ The `device=` argument is ignored — the model runs on CPU

**Verified against the installed PyTorch-Wildlife 1.3.0.** This is the most consequential trap in
the stack, and it is silent.

`MegaDetectorV6(device="cuda")` accepts the argument, stores it on `self.device`, and passes it to
`_load_model` — where the only line that would apply it is commented out:

```python
# src/PytorchWildlife/models/detection/ultralytics_based/yolov8_base.py
# self.predictor.args.device = device # Will uncomment later
```

The Ultralytics predictor therefore keeps its default and **loads the weights onto CPU**. Nothing
raises. `detector.device` still reports `"cuda"`. On a Colab GPU runtime you would get CPU speed
while every log line, and the run metadata, claimed CUDA.

Setting `predictor.args.device` after construction does **not** help — `setup_model()` has already
run. The working fix is three lines, applied after the model is set up:

```python
import torch

torch_device = torch.device(device)  # "cuda" | "mps" | "cpu"
detector.predictor.model.to(torch_device)  # move the weights
detector.predictor.device = torch_device  # tell preprocessing where inputs go
detector.predictor.args.device = device  # keep args consistent
```

Omitting the second line moves the weights but not the inputs, and the next forward pass dies with
`input(device='cpu') and weight(device='mps:0') must be on the same device`.

**Do not trust the claim — check the tensors.** `panaf_ape_detection.runtime.module_device()` reports
where the parameters actually live, and `scripts/smoke_detect.py` is a working reference: it detects
the mismatch, applies the fix, and verifies it took. Measured on an M1: 2.13 s on CPU, 1.53 s after
the move to MPS.

The Phase 1c adapter must do this and record the **verified** device in `RunMetadata`, not the
requested one.

### The upstream defaults are broken — always set the variant explicitly

Verified against the installed PyTorch-Wildlife 1.3.0:

- `MegaDetectorV6.__init__` defaults to `version='yolov9c'`
- `MegaDetectorV6Apache.__init__` defaults to `version='MDV6-rtdetr-x-apache'`

Neither string appears in its own method's `if/elif` chain, so **both defaults fall through to
`raise ValueError`**. Constructing either class without an explicit `version=` fails outright.

This is worth knowing for two reasons: it means the variant must be passed explicitly for the code
to work at all, and it is a reminder that an upstream default is not a specification. `configs/*.yaml`
therefore require `model.variant`, and `panaf-phase1 validate-config` warns when the value is
outside the table above.

## Why threshold and variant must be recorded

A detection count is meaningless without both.

**Confidence threshold** trades recall against precision, continuously. The same clip at 0.1 and
0.5 will report substantially different numbers of apes, and neither is "the" answer. A result that
does not state its threshold cannot be compared with anything, including a later run of itself.
Phase 1's `configs/base.yaml` uses `0.2` as a starting point chosen to favour recall — it is a
starting point to be revised from evidence, not a recommendation.

**Variant** determines the architecture and weights entirely. `MDV6-yolov9-c` and `MDV6-yolov10-e`
are different models.

Both are fields in `ModelConfig` and both belong in `RunMetadata` (see
[`reproducibility.md`](reproducibility.md)). The failure mode this prevents is the classic one:
finding a good result months later and being unable to reproduce it because nobody wrote down which
model produced it.

## Anticipated failure conditions

These are the conditions **expected** to cause trouble, based on the nature of camera-trap footage
and of object detectors generally. They are stated in advance so Phase 1 can check them
deliberately — **none of these has been measured yet, and none should be reported as a finding
until it has been.**

| Condition | Expected effect | Why |
| --- | --- | --- |
| Night-time / infrared | Missed detections | Monochrome IR imagery differs from the RGB distribution; texture and colour cues vanish. |
| Heavy vegetation occlusion | Missed or fragmented boxes | Only parts of the animal are visible; the detector may fire on each visible part separately. |
| Distant / small subjects | Missed detections | Below a certain pixel height, there is not enough signal at any input resolution. |
| Motion blur | Low scores, unstable boxes | Fast movement smears edges the detector relies on. |
| Multiple overlapping apes | Merged or suppressed boxes | Non-maximum suppression cannot distinguish heavy overlap from duplicate detections. |
| Dense foliage, no animal | False positives | Branch and shadow structure can resemble limbs. |
| Camera interaction (very close subject) | Odd boxes | The animal fills or exceeds the frame; scale is far from typical training data. |
| Empty frames after a trigger | False positives | Camera traps record motion, not animals — wind-triggered clips test precision directly. |
| Score instability frame to frame | Tracker ID switches | A detection oscillating around the threshold appears and disappears, breaking association. |

The last row is the one that matters most for Phase 1d: a tracker cannot fix detections that do not
exist. If scores oscillate around `0.2`, lowering the threshold may help tracking more than any
tracker tuning will.

## The tracker choice (Phase 1d)

The project brief names two options: **SORT** or **ByteTrack**. The choice is between those two and
is deliberately deferred until the detection baseline exists, because the evidence that should decide
it — how stable detections actually are frame to frame — is exactly what Phase 1c produces.

Two things worth knowing in advance:

- **ByteTrack is already reachable.** `supervision` ships it and arrives as a PyTorch-Wildlife
  dependency, so choosing it adds no new dependency. SORT would.
- **ByteTrack's design targets this failure mode.** It associates low-confidence boxes in a second
  pass rather than discarding them, which is aimed squarely at detections that flicker around the
  threshold. If Phase 1c shows that flicker dominates, that is an argument for ByteTrack; if
  detections turn out to be stable and the problem is elsewhere, SORT's simplicity is worth more.

Whichever is chosen, record the reason in the experiment log. "It was the default" is not a reason.

## Future work: comparing variants

Before any fine-tuning is considered, the cheap experiment is a variant sweep — run the same clips
through each variant at a fixed threshold and compare. This is explicitly **Phase 3**, not now,
and it is only worth doing once Phase 1 has established that the harness produces trustworthy
output.

The architecture supports it without code changes: one config file per variant, each committed
alongside its run metadata. If a sweep ever requires editing a Python file, something has been
hard-coded that should not have been.

Fine-tuning is justified only if the sweep shows that no pretrained variant is adequate **and**
Phase 1 identified a systematic, learnable failure mode. "The model made mistakes" is not that.
