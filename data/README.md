# Data

**No dataset files are stored in this repository, and none may ever be committed to it.**

Everything under `data/raw/`, `data/interim/` and `data/processed/` is git-ignored except the
`.gitkeep` markers, this README and the example manifest. If you find yourself running
`git add -f` on something in here, stop.

## PanAf20K vs PanAf500

| | PanAf20K | PanAf500 |
| --- | --- | --- |
| Size | ~20,000 camera-trap videos, >7 million frames | 500 videos (a subset of PanAf20K) |
| Subjects | Chimpanzees and gorillas, 14 African field sites | Same footage, densely annotated |
| Annotations | Coarse, video-level | Dense: per-frame bounding boxes, ape identity tracks, and frame-wise behaviour labels |
| Use here | **Not used in Phase 1** | **The only subset Phase 1 uses** |

PanAf500 is the densely annotated subset, and it is the right starting point precisely because it
carries per-frame boxes and behaviour labels — the two things Phase 1 needs in order to sanity-check
detections and to display behaviour labels beside them.

**Do not download the full PanAf20K dataset for this phase.** It is orders of magnitude larger than
anything Phase 1 requires, and downloading it wastes the deposit's bandwidth as well as yours.

Source of these figures: the PanAf20K paper (Brookes et al., 2024), cited in
[`../references.bib`](../references.bib). Details of the annotation file formats are deliberately
**not** documented here — write them down only after you have inspected the actual files or the
deposit's own documentation. Do not guess a schema.

## Phase 1 sample size

Select **approximately 5–10 clips**. Not 50, not "all of PanAf500".

The point of Phase 1 is a fast, honest qualitative read on how a pretrained detector behaves on this
footage. A small sample you have actually watched frame by frame is worth far more than a large
sample you have only summarised. Choose clips that deliberately span the hard cases:

- daylight vs. night-time / infrared
- close subjects vs. small distant ones
- a single ape vs. several overlapping apes
- static subjects vs. fast movement and motion blur
- clear view vs. heavy vegetation occlusion

Because the sample is small and hand-picked, it cannot support statistical claims about the dataset.
Say so in the write-up.

## Obtaining the data

The dataset is deposited at the University of Bristol Research Data Repository:

<https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt>

Acquisition is **manual and deliberate**:

1. Open the deposit page and read the licence and access conditions in full.
2. Complete whatever access or agreement process the deposit requires.
3. Download only the PanAf500 subset, and only the clips you have selected.
4. Place the files under `data/raw/` (see layout below).
5. Compute checksums and record your selection in a manifest (see below).

There is **no download script in this repository, on purpose.** An unverified scraper against a
research deposit is a good way to violate terms of use, hammer someone's server, or silently fetch
the wrong thing. A download utility can be added later once the exact endpoints and terms have been
confirmed — with a rate limit and a resume, not a `for` loop over `wget`.

### Licensing constraints

- The deposit is released under a **non-commercial** data licence. Read it yourself; do not rely on
  this summary.
- Dataset access and use remain governed by that licence regardless of anything in this repository.
- **Dataset files must never be redistributed through this repository** — not in git, not in a
  release artefact, not in a notebook output cell, not in a GIF that happens to contain frames.
- The annotated clips you produce are derived works of the dataset. Check the licence before
  publishing them anywhere, including in a report, a slide deck, or a GitHub README.
- The repository code licence does **not** cover the data. See
  [`../docs/licensing.md`](../docs/licensing.md).

## Expected local layout

```text
data/
├── raw/                     # immutable, as-downloaded. Never write here.
│   └── panaf500/
│       ├── videos/          # e.g. <clip_id>.mp4
│       └── annotations/     # dataset-provided annotation files
├── interim/                 # derived but reusable (extracted frames, decoded metadata)
│   └── frames/
│       └── <clip_id>/
├── processed/               # analysis-ready derived data
├── README.md                # this file (tracked)
├── sample_manifest.example.csv   # template (tracked)
└── sample_manifest.csv      # your actual selection (git-ignored)
```

Adjust the `panaf500/` subtree to match whatever structure the deposit actually ships; record the
real layout in your experiment log entry rather than reshuffling files to match this diagram.

### `data/raw/` is immutable

Treat it as read-only:

- Nothing in this codebase will ever write to `data/raw/`.
- Do not rename, re-encode, trim, or "clean up" files in place. If you need a modified version, it
  is a derived artefact and belongs in `data/interim/`.
- **Extracted frames go in `data/interim/` or `artifacts/frames/` — never in `data/raw/`.** Mixing
  derived frames into the raw tree destroys the one guarantee that makes checksums meaningful.

If a raw file's checksum ever stops matching the manifest, something has modified your inputs and
every result derived from them is suspect.

## The clip manifest

Clip selection is data, not a decision buried in a shell history. Copy the template and fill it in:

```bash
cp data/sample_manifest.example.csv data/sample_manifest.csv
```

`data/sample_manifest.csv` is git-ignored, because its `video_filename` column reveals dataset
contents. Only the example template is tracked.

### Columns

| Column | Meaning |
| --- | --- |
| `clip_id` | Stable identifier used in output filenames and log entries. |
| `split` | Dataset split the clip belongs to, as the deposit defines it. |
| `video_filename` | Filename only, relative to the videos directory. Not an absolute path. |
| `annotation_filename` | Corresponding annotation filename, if present. |
| `species` | Species as recorded by the dataset — never as guessed by the detector. |
| `site` | Field site as recorded by the dataset. |
| `selected_reason` | **Why you chose this clip.** e.g. "night-time IR, tests low-light failure". |
| `video_sha256` | SHA-256 of the video file. |
| `annotation_sha256` | SHA-256 of the annotation file. |
| `notes` | Anything else worth knowing: unusual framing, corrupt frames, ambiguity. |

`selected_reason` is the column that stops this being a random sample. Fill it in properly — future
you will need to know whether a failure mode was hunted for or stumbled upon.

### Checksums

Record a SHA-256 for every input file:

```bash
# macOS
shasum -a 256 data/raw/panaf500/videos/<file>

# Linux
sha256sum data/raw/panaf500/videos/<file>
```

Checksums are what make a result re-runnable rather than merely re-described. They let you prove
that the file you ran inference over today is the same file you ran it over last month, that a
collaborator has the same bytes you do, and that a partial or corrupted download did not quietly
change your results. Once inference is implemented, every run's metadata will embed these digests
(see [`../docs/reproducibility.md`](../docs/reproducibility.md)).

Verify before a run, not after a surprising result.

## Checklist before running anything

- [ ] I have read the deposit's licence and complied with its access process.
- [ ] I downloaded only PanAf500, and only the clips I need.
- [ ] Files are under `data/raw/` and I have not modified them.
- [ ] `data/sample_manifest.csv` exists, with a filled-in `selected_reason` for every clip.
- [ ] Every checksum in the manifest matches the file on disk.
- [ ] `git status` shows no dataset files staged or untracked-but-tempting.
