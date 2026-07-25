# Architecture

**Status: partly implemented.** The foundation — `config.py`, `paths.py`, `types.py`, `runtime.py`,
`provenance.py` and `cli.py` — exists and is tested. Every stage module below (`data/`,
`inference/`, `tracking/`, `visualization/`, `evaluation/`, `pipeline/`) is still a design, marked
`[planned]`, so that the next implementation task has a target rather than a blank page.

## Principle

Research code rots when the experiment logic, the model, and the plumbing become one thing. The
structure here exists to keep three questions answerable independently:

- *What data went in?* → `data/`
- *What model produced these boxes?* → `inference/`
- *How was it drawn?* → `visualization/`

If changing the tracker requires editing the video writer, the boundaries have failed.

## Intended modules

```text
src/panaf_ape_detection/
├── __init__.py         Lightweight namespace. No ML imports, ever.
├── paths.py            Repository-root discovery and the canonical layout.
├── types.py            Shared schemas: Detection, TrackedDetection, RunMetadata.
├── config.py           Typed YAML configuration.
├── runtime.py          Device resolution, seeding, and verifying where weights
│                       actually live. Lazy torch imports only.
├── provenance.py       Checksums, git state, dependency versions, and the one
│                       producer of RunMetadata.
├── cli.py              Typer entry point. Thin: parses, delegates, prints.
│
├── data/               [planned]
│   ├── manifest.py     Load and validate the clip manifest; verify checksums.
│   ├── video.py        Decode clips, iterate frames, honour frame_stride.
│   └── annotations.py  Read dataset-provided behaviour labels and boxes.
│
├── inference/          [planned]
│   ├── base.py         Detector protocol: frames in, Detection objects out.
│   ├── megadetector.py PyTorch-Wildlife MegaDetector V6 adapter.
│   └── filtering.py    Confidence thresholding and box post-processing.
│
├── tracking/           [planned]
│   ├── base.py         Tracker protocol: Detection sequence -> TrackedDetection.
│   └── bytetrack.py    Concrete backend (chosen after the detection baseline).
│
├── visualization/      [planned]
│   ├── overlays.py     Draw boxes, scores, track ids, behaviour labels.
│   └── video.py        Encode annotated clips and GIFs.
│
├── evaluation/         [planned]
│   └── qualitative.py  Failure-case tabulation and review helpers.
│
└── pipeline/           [planned]
    ├── runner.py       Orchestration: wires stages together per config.
    └── metadata.py     Build and persist RunMetadata.
```

## Dependency direction

Dependencies point one way. Nothing lower imports something higher.

```mermaid
flowchart TD
    CLI["cli.py"] --> PIPE["pipeline/"]
    PIPE --> DATA["data/"]
    PIPE --> INF["inference/"]
    PIPE --> TRK["tracking/"]
    PIPE --> VIS["visualization/"]
    PIPE --> EVAL["evaluation/"]
    DATA --> FOUND
    INF --> FOUND
    TRK --> FOUND
    VIS --> FOUND
    EVAL --> FOUND
    FOUND["types.py · config.py · paths.py<br/><i>foundation — no ML imports</i>"]
```

Rules this encodes:

1. **The foundation imports nothing from the project.** `types.py`, `config.py` and `paths.py` are
   leaves. This is why they can be tested without a GPU and imported in a bare Colab runtime.
2. **Stage modules do not import each other.** `tracking/` never imports `inference/`; it consumes
   `Detection` objects from `types.py`. Swapping the detector cannot break the tracker.
3. **Only `pipeline/` knows the order of stages.** It is the one place that understands that
   detection precedes tracking.
4. **`cli.py` stays thin.** It parses arguments, loads config, calls one pipeline function, and
   formats output. No experiment logic lives in the CLI, and none lives in a notebook.
5. **Heavy imports are lazy.** `import torch` happens inside a function in `inference/`, never at
   module scope in a package that the CLI imports eagerly. This is what makes
   `panaf-phase1 doctor` work after a bare `uv sync`, and it is enforced by
   `scripts/verify_repository.py`.

## Why stages are separated

| Boundary | What it buys |
| --- | --- |
| `data/` vs `inference/` | Frame extraction is expensive and deterministic; detection is expensive and model-dependent. Separating them lets frames be cached in `data/interim/` and reused across every threshold and variant sweep. |
| `inference/` vs `tracking/` | The tracker consumes a generic `Detection`. ByteTrack can be swapped for SORT, or the detector for a fine-tuned one, without either knowing about the other. |
| `tracking/` vs `visualization/` | Drawing is a presentation concern. A bug in the overlay must never be able to change a recorded detection. |
| `visualization/` vs `evaluation/` | Judgements about failure cases are data, not pixels. Evaluation reads detection records, not rendered video. |
| everything vs `pipeline/` | Stage code stays unit-testable in isolation; orchestration is the only part that needs an end-to-end fixture. |

## Why model and tracker are configuration-driven

The MegaDetector variant and the tracker backend are fields in `configs/*.yaml`, resolved through
`ModelConfig` and `TrackingConfig`. They are deliberately **not** constants in application code.

1. **A result without its exact variant is not a result.** `MDV6-yolov9-c` and `MDV6-yolov10-e` are
   different models with different speed/accuracy trade-offs. "MegaDetector V6 found 82% of apes"
   means nothing without saying which one. `RunMetadata` records the variant with every run.
2. **Comparing variants must not require a code change.** Phase 3 anticipates a variant sweep. If
   the variant lives in code, each comparison is a diff; if it lives in config, each comparison is
   a file that can be committed alongside its results.
3. **Upstream defaults are not trustworthy.** Verified against the installed PyTorch-Wildlife
   1.3.0: `MegaDetectorV6.__init__` defaults to `version='yolov9c'`, and `MegaDetectorV6Apache`
   to `version='MDV6-rtdetr-x-apache'` — **neither string is accepted by that same method's own
   validation**, so relying on the default raises `ValueError`. An explicit, configured variant is
   not pedantry here; it is the only thing that works.
4. **Licensing differs by variant.** The YOLOv9/YOLOv10-derived weights and the Apache RT-DETR
   weights do not carry the same terms. A variant buried in code is a licence obligation nobody can
   audit. See [`licensing.md`](licensing.md).

The tracker backend is configuration for the same reason, plus one more: the choice is genuinely
undecided. `TrackingBackend` enumerates `none`, `bytetrack` and `sort`, and the default config sets
`enabled: false`. The decision will be made from Phase 1c evidence about how stable detections
actually are — not guessed now.

## Adding deployment adapters later

This phase is offline batch research. Later phases may need live inference on a robot. The
separation above is what makes that an addition rather than a rewrite:

- **The detector protocol is the seam.** `inference/base.py` defines frames in, `Detection` out.
  A `TensorRTMegaDetector` or an ONNX Runtime adapter implements the same protocol and drops into
  the same config field. Research code never learns it changed.
- **`data/` is the input seam.** A camera source that yields frames satisfies the same iterator
  contract as a decoded file. `pipeline/` does not care which it got.
- **Presentation is already separate.** A deployment target that streams boxes over a socket
  instead of writing MP4 files replaces `visualization/`, and touches nothing else.
- **Config is already the switchboard.** Deployment settings become new sections validated the same
  strict way, rather than environment-variable sprawl.

The rule to preserve: **deployment concerns must not leak into stage modules.** No `if deployed:`
branches in `inference/megadetector.py`. If a deployment need cannot be expressed as a new adapter
behind an existing protocol, the protocol is wrong and should be changed deliberately.

## Testing strategy

- Foundation modules: pure unit tests, no fixtures. Already at high coverage.
- `data/`: tested against small synthetic videos generated in the test, never against real
  PanAf clips — the suite must run in CI with no dataset present.
- `inference/`: the adapter is tested with a stub detector satisfying the protocol. **Tests never
  download weights.** A real-weights test, if ever added, is marked and excluded from default CI.
- `tracking/`, `visualization/`: tested on synthetic `Detection` sequences with known answers.
- `pipeline/`: one end-to-end test with a stub detector and a two-frame synthetic clip.
