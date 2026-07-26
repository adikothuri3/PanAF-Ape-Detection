# Dataset

Operational instructions for obtaining and organising the data are in
[`../../../data/README.md`](../../../data/README.md). This document covers what the dataset *is* and how to
think about it.

## PanAf20K

PanAf20K is a large video dataset for wild ape detection and behaviour recognition, released by
Brookes et al. (2024). It comprises **more than 7 million frames across roughly 20,000 camera-trap
videos** of chimpanzees and gorillas, collected at **14 field sites in tropical Africa** as part of
the Pan African Programme: The Cultured Chimpanzee.

The footage is genuine field data: fixed camera traps, triggered by motion, recording whatever
passes. That is what makes it valuable and what makes it hard. It is not curated wildlife
photography.

The full dataset carries coarse, video-level annotation.

## PanAf500

PanAf500 is the **densely annotated subset**: 500 videos (~180,000 frames) from the same corpus,
annotated per frame with

- bounding boxes for visible apes,
- identity tracks linking the same ape across frames within a video,
- frame-wise behavioural action labels.

**PanAf500 is the only subset Phase 1 uses**, because it is the only one carrying the per-frame
boxes and behaviour labels that this phase needs in order to judge detections and caption them.

### The nine behaviour labels

PanAf500 carries **nine action labels**, listed here verbatim from the project's onboarding
documentation:

| | | |
| --- | --- | --- |
| sitting | standing | walking |
| running | climbing up | climbing down |
| hanging | sitting on back | camera interaction |

Several of these name exactly the conditions expected to degrade a detector: `climbing up` /
`climbing down` and `hanging` imply occlusion and unusual poses, and `camera interaction` means a
subject filling or exceeding the frame. That makes the label a useful axis for clip selection as
well as the target of the Phase 1 comparison.

Phase 1 step 5 displays these labels **beside** detections. They are **dataset ground truth, never
model predictions** — MegaDetector cannot predict behaviour (see [`model.md`](model.md)) — and any
annotated clip must make that distinction visible to a viewer.

### Annotation format — verified

Read from real deposit files, not assumed. One JSON per clip, named after the video, under
`PanAf500/annotations/{train,validation,test}/`:

```json
{"video": "035FQHMNRk",
 "annotations": [{"frame_id": 1,
                  "detections": [{"bbox": [144.0, 0.0, 491.0, 397.0],
                                  "ape_id": 0,
                                  "species": "chimpanzee",
                                  "behaviour": "climbing_up"}]}]}
```

| Field | Meaning |
| --- | --- |
| `frame_id` | **1-based** frame number (1..360 for a 360-frame clip) |
| `bbox` | `[x1, y1, x2, y2]` in absolute pixels — the same `xyxy` convention as `BoundingBox` |
| `ape_id` | Individual identity **within this video only**; ground truth for tracking |
| `species` | `chimpanzee` or `gorilla` |
| `behaviour` | One of the nine, in **underscore form** (`climbing_up`, `camera_interaction`) |

The file carries **no frame dimensions**; they come from decoding the video, which is what makes the
loaded records self-describing.

> **Two traps, both silent if missed.**
>
> 1. `frame_id` is **1-based** while everything else here is 0-based. An unconverted join shifts
>    every box by one frame — which reads as a mediocre detector, not a bug.
> 2. Behaviour labels use **underscores on disk** but prose in the onboarding document. A lookup
>    against `"climbing up"` matches nothing, silently.
>
> Both are handled in exactly one place — `panaf_ape_detection.data.annotations` — and covered by
> regression tests. Do not re-implement either conversion elsewhere.

Frames with **no** detections are common and meaningful: they are what make false-positive rate
measurable. Roughly 44% of the clips profiled had at least one.

## Annotations relevant to this project

| Annotation | Used for | Phase |
| --- | --- | --- |
| Bounding boxes | Eventual comparison against MegaDetector output | 2 (not Phase 1) |
| Identity tracks | Sanity-checking tracker behaviour and ID switches | 1d |
| Behaviour labels | Overlaying beside detections in annotated clips | 1e |
| Species / site metadata | Recording clip provenance in the manifest | 1b |

In Phase 1 the ground-truth boxes are used **qualitatively** — to look at, to notice what the
detector missed — not to compute a score. A scored comparison requires matching criteria, an IoU
threshold, and care about frames where the detector is right but the ground truth is sparse. That
is Phase 2 work, and doing it carelessly now would produce a number worse than no number.

## Sample-selection strategy

Phase 1 uses approximately **5–10 clips**. The sample is small and purposive, not random, and it is
chosen to span the conditions expected to break a detector:

| Axis | Sample should include |
| --- | --- |
| Lighting | Daylight **and** night-time / infrared |
| Subject scale | Close subjects **and** small distant ones |
| Subject count | Single ape **and** multiple overlapping apes |
| Motion | Static or slow **and** fast movement with blur |
| Occlusion | Clear view **and** heavy vegetation |
| Species | Both chimpanzee and gorilla footage, if available |
| Behaviour | Beyond `sitting`/`standing`, at least one clip labelled `climbing up`/`climbing down`, `hanging`, or `camera interaction` |
| Negative case | At least one clip where apes are absent or barely visible |

That last row is easy to skip and worth keeping. A detector's false-positive behaviour is invisible
if every clip contains an animal.

Every choice goes in the manifest's `selected_reason` column. The distinction between "I chose this
clip to test infrared performance" and "this clip happened to be first alphabetically" is the
difference between an experiment and an anecdote.

## Limitations

**Of the data itself**

- Camera-trap footage is low-resolution, often infrared at night, frequently occluded, and
  triggered by motion rather than by the presence of an ape — so some clips contain no animal.
- Field sites are unevenly represented, and site conditions (vegetation, camera placement, hardware)
  vary substantially.
- Behaviour classes are inherently imbalanced: common postures vastly outnumber rare behaviours.
- Annotation is human-produced. Boxes and behaviour labels carry judgement calls, particularly for
  partially visible animals.

**Of this project's use of it**

- 5–10 clips cannot support quantitative claims about the dataset. Any Phase 1 statement should be
  phrased as an observation about specific clips, not a property of PanAf500.
- Purposive selection introduces bias by design. That is acceptable for finding failure modes and
  unacceptable for estimating accuracy.
- Frames within a clip are highly correlated; 180,000 frames is nowhere near 180,000 independent
  observations.

## Ethical and conservation context

This dataset exists because great apes are endangered and because monitoring wild populations by
hand does not scale. The Pan African Programme collected this footage to study chimpanzee behaviour
and culture across Africa; the machine-learning framing is downstream of a conservation purpose.

Practical implications for this work:

- **Location data is sensitive.** Precise locations of endangered animals can assist poaching. Field
  sites are identified in the dataset at the granularity the depositors chose — do not augment that
  with finer location detail, and do not publish georeferenced material.
- **Do not redistribute footage.** Beyond the licence obligation, this is footage of endangered
  animals in specific, identifiable places. Annotated clips are derived works; check the licence
  before putting them in a public repository, a talk, or a blog post.
- **Overstated capability has a cost.** Conservation practitioners may act on claims about automated
  monitoring. Reporting a detector as "identifying chimpanzees" when it emits a generic `animal`
  class is not a harmless simplification.
- **Credit the field work.** The dataset represents years of effort by field teams across 14 sites.
  Citing the paper is the minimum.

## Citation requirements

If you use this data, cite the PanAf20K paper **and** the data deposit. BibTeX entries for both are
in [`../../../references.bib`](../../../references.bib).

> Brookes, O., Mirmehdi, M., Stephens, C., Angedakin, S., Corogenes, K., Dowd, D., Dieguez, P.,
> Hicks, T. C., Jones, S., Lee, K., Leinert, V., Lapuente, J., McCarthy, M. S., Meier, A., Murai, M.,
> Normand, E., Vergnes, V., Wessling, E. G., Wittig, R. M., Langergraber, K., Maldonado, N., Yang, X.,
> Zuberbühler, K., Boesch, C., Arandjelovic, M., Kühl, H., and Burghardt, T. (2024).
> *PanAf20K: A Large Video Dataset for Wild Ape Detection and Behaviour Recognition.*
> arXiv:2401.13554.

Check the deposit page for its own preferred citation form and DOI, which may differ from the
paper's, and cite whichever it specifies in addition to the paper.

Cite MegaDetector and PyTorch-Wildlife separately if you report detection results — they are
distinct contributions from distinct teams. Do not cite this repository as a source of methods; it
is a scaffold that runs other people's models.
