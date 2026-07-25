# Licensing

**This is not boilerplate.** Three separately licensed things meet in this project — the code, the
dataset, and the model weights — and they are governed by three different sets of terms. Assuming
one licence covers all three is the most likely way to get this wrong.

Nothing here is legal advice. Where it matters, read the primary sources and ask someone qualified.

## 1. This repository's code

**No code licence has been selected. There is no `LICENSE` file.**

That is a deliberate state, not an oversight. Under default copyright, absence of a licence means
**no rights are granted to anyone** to use, copy, modify or redistribute this code. It is not
public-domain and it is not open source, regardless of where it is hosted.

Practical consequences while this remains true:

- Nobody else may reuse this code, even if the repository is public.
- Do not copy code *out* of this repository into a differently licensed project.
- Contributions are accepted on the understanding that the licence is still to be decided.

**Before publishing this repository**, choose a licence deliberately — MIT and Apache-2.0 are the
usual candidates for research tooling, with Apache-2.0 additionally granting patent rights — and:

1. Add a root `LICENSE` file.
2. Update this section and the README's licensing table.
3. Check with anyone whose institutional policy applies. Internship and university work often has
   IP terms that constrain the choice. **Ask before publishing, not after.**

Choosing a code licence changes nothing about the two sections below.

## 2. The PanAf20K / PanAf500 dataset

**The repository code licence does not cover the data. It never will.** Licensing the code under
MIT would grant no one any right to the dataset.

The dataset is deposited at the University of Bristol Research Data Repository:

<https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt>

Key points:

- The deposit is released under a **non-commercial** data licence. **Read the exact terms on the
  deposit page** — this file is a pointer, not a substitute, and licence terms are revised.
- Access and use are governed by that licence and by whatever access process the deposit requires.
- **Dataset files must never be redistributed through this repository.** Not committed to git, not
  attached to a release, not embedded in notebook output, not included in a zip.
- **Annotated clips and GIFs are derived works of the dataset.** The Phase 1 deliverable is 2–3
  annotated clips — check the licence before putting them in a public README, a poster, a talk, or
  a paper. "It has boxes drawn on it now" does not make it your footage.
- Extracted frames are also derived works. Same rules.
- Cite the dataset (see [`../references.bib`](../references.bib)).

If the licence is non-commercial, then commercial use — including use by a company, or work whose
purpose is a commercial product — is outside it, no matter how the code is licensed.

## 3. MegaDetector V6 model weights

**The repository code licence does not cover model weights either**, and the weights' licence is
not necessarily the same as the licence of the code that loads them.

Three distinct things, three distinct licences:

| Thing | Licence |
| --- | --- |
| PyTorch-Wildlife (the library) | MIT |
| MegaDetector repository (the code) | MIT |
| MegaDetector V6 **weights** | **Varies by variant — verify per variant** |

### Verify the licence of the exact variant you use

This is the part most likely to be got wrong, because "MegaDetector is MIT" is true of the code and
not a statement about every set of weights.

The V6 variants derive from different upstream architectures with different terms. In particular,
PyTorch-Wildlife ships a separate `MegaDetectorV6Apache` class with `MDV6-apa-rtdetr-*` weights —
that class exists **precisely because** the licensing of the other variants differs. YOLOv9- and
YOLOv10-derived weights are commonly distributed under **AGPL-3.0**, which is a copyleft licence
with network-use obligations, whereas the Apache-licensed RT-DETR variants are permissive.

Before selecting a variant, and especially before any deployment or commercial use:

1. Look up the specific variant string you have configured (e.g. `MDV6-yolov9-c`).
2. Find that variant's licence at its source — the PyTorch-Wildlife model zoo, the Zenodo record
   the weights download from (record `15398270`), and the upstream architecture's licence.
3. Record the answer in your experiment log alongside the variant.
4. If the project may ever be commercial or deployed, **prefer the Apache variants** and confirm
   that choice rather than assuming it.

The variant is a configuration field partly for this reason: a variant buried in application code
is a licence obligation nobody can audit. See [`model.md`](model.md).

### Weights are not committed

`.gitignore` blocks `*.pt`, `*.pth`, `*.onnx`, `*.safetensors` and similar. Weights are downloaded
on first use to a local cache. Do not commit them, do not attach them to a release, and do not
mirror them — redistribution is a separate act from use, and may be separately restricted.

## 4. Redistribution

Do not casually redistribute data or weights. Concretely, do not:

- commit dataset files, frames, or weights to git;
- attach them to a GitHub release or a Drive folder shared beyond the permitted users;
- embed frames in committed notebook output;
- publish annotated clips without checking the dataset licence;
- mirror weights "for convenience".

Sharing a *link* to the official source is almost always fine. Sharing *the bytes* is what the
licences constrain.

## 5. Deployment or commercial use requires a separate review

If this work moves toward deployment, a product, or anything commercial, stop and review all three
layers together:

- [ ] Has a code licence been chosen, and does institutional/internship IP policy permit the
      intended use?
- [ ] Does the dataset licence permit it? A non-commercial licence generally does **not** permit
      commercial use, and this may extend to a model whose development depended on that data.
- [ ] Does the exact weight variant's licence permit it? AGPL-3.0 imposes source-disclosure
      obligations that reach network-served applications.
- [ ] Are the outputs (annotated clips, extracted frames) themselves redistributable?
- [ ] Are there ethical constraints beyond licensing — location sensitivity for endangered species,
      overstated capability claims? See [`dataset.md`](dataset.md).

Each checkbox can independently block the use. Answering three of four is not a pass.

## Summary

| Component | Licence | Covered by repo code licence? |
| --- | --- | --- |
| This repository's code | **Not yet selected** | — |
| PanAf20K / PanAf500 data | Bristol deposit, non-commercial | **No** |
| MegaDetector V6 weights | **Varies by variant** | **No** |
| PyTorch-Wildlife library | MIT | No (its own) |
| MegaDetector code | MIT | No (its own) |
