#!/usr/bin/env python3
"""Fetch a small, purposive sample of PanAf500 clips.

The dataset is deposited at the University of Bristol Research Data Repository
under the **Non-Commercial Government Licence v2**. The deposit exposes a
browsable file tree, so individual clips can be fetched rather than the 42.2 GiB
complete archive -- which is what makes the onboarding's "5 to 10 clips, not all
7 million frames" practical.

``data/README.md`` previously said a download utility could be added "once the
exact dataset endpoint and terms have been validated". Both are now verified,
which is why this script exists.

**Selection is purposive, not random.** It runs in two passes:

1. Download *annotations only* for a candidate pool (~50 KB each, no video) and
   score every clip against the axes in ``05 Technical/dataset.md``: behaviour
   variety, subject count, subject size, species, and frames containing no ape.
2. Greedily pick the clips that together cover the most axes, then download only
   those videos.

The reasoning for each pick is written into the manifest's ``selected_reason``
column. That column is what separates a purposive sample from an arbitrary one.

**Downloads dataset files. Never run this in CI.** Nothing it writes may be
committed -- ``.gitignore`` and the pre-commit hooks enforce that.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BASE_URL = "https://data.bris.ac.uk/datasets/1h73erszj3ckn2qjwm4sqmr2wt/PanAf500"
SPLITS: tuple[str, ...] = ("train", "validation", "test")
LICENCE = "Non-Commercial Government Licence v2 (see the deposit page)"
DEPOSIT = "https://data.bris.ac.uk/data/dataset/1h73erszj3ckn2qjwm4sqmr2wt"

# Behaviours that imply occlusion, unusual pose or extreme scale -- the
# conditions docs/obsidian/05 Technical/model.md expects to degrade a detector.
HARD_BEHAVIOURS = frozenset(
    {"climbing_up", "climbing_down", "hanging", "camera_interaction", "sitting_on_back", "running"}
)
TIMEOUT = 60
RETRIES = 4
BACKOFF_SECONDS = 2.0


@dataclass
class ClipProfile:
    """What the annotations alone reveal about one clip."""

    clip_id: str
    split: str
    frames: int
    boxes: int
    empty_frames: int
    max_apes_in_frame: int
    distinct_apes: int
    behaviours: Counter[str] = field(default_factory=Counter)
    species: Counter[str] = field(default_factory=Counter)
    median_relative_area: float = 0.0

    def features(self) -> set[str]:
        """Return the selection axes this clip covers."""
        found: set[str] = set()
        for behaviour, count in self.behaviours.items():
            if count >= 5:
                found.add(f"behaviour:{behaviour}")
        for name in self.species:
            found.add(f"species:{name}")
        if self.empty_frames >= 10:
            found.add("axis:empty-frames")
        if self.max_apes_in_frame >= 3:
            found.add("axis:crowded")
        if self.distinct_apes >= 3:
            found.add("axis:multiple-individuals")
        if self.median_relative_area <= 0.05:
            found.add("axis:small-subject")
        if self.median_relative_area >= 0.35:
            found.add("axis:large-subject")
        return found

    def reason(self, covered: set[str]) -> str:
        """Explain, in one line, why this clip earns its place in the sample."""
        parts: list[str] = []
        hard = sorted(b for b in self.behaviours if b in HARD_BEHAVIOURS)
        if hard:
            parts.append("hard behaviours: " + ", ".join(hard))
        if "axis:empty-frames" in covered:
            parts.append(f"{self.empty_frames} frames with no ape (false-positive check)")
        if "axis:crowded" in covered:
            parts.append(f"up to {self.max_apes_in_frame} apes per frame (occlusion, NMS)")
        if "axis:small-subject" in covered:
            parts.append(f"small subjects (median {self.median_relative_area:.1%} of frame)")
        if "axis:large-subject" in covered:
            parts.append(f"large subjects (median {self.median_relative_area:.1%} of frame)")
        species = "/".join(sorted(self.species))
        parts.append(f"species: {species}")
        return "; ".join(parts)


def fetch(url: str, *, retries: int = RETRIES) -> bytes:
    """GET *url*, retrying transient failures.

    The deposit is a public research server reached over the open internet, and
    a single dropped connection should not end a ten-minute run. Retries cover
    timeouts, resets and 5xx responses; a 404 is not retried, because it will
    never succeed.

    Args:
        url: The address to fetch.
        retries: Attempts before giving up.

    Returns:
        The response body.

    Raises:
        RuntimeError: With an actionable message, after the last attempt.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "panaf-ape-detection/0.1"})
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return bytes(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                msg = (
                    f"HTTP {exc.code} for {url}\n"
                    "The deposit layout may have changed, or that clip may not exist in this split."
                )
                raise RuntimeError(msg) from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc

        if attempt < retries:
            delay = BACKOFF_SECONDS * attempt
            print(f"    retry {attempt}/{retries - 1} in {delay:.0f}s ({last}) ...", flush=True)
            time.sleep(delay)

    msg = (
        f"could not fetch {url} after {retries} attempts: {last}\n"
        "Check network access to data.bris.ac.uk. On Colab this is usually transient -- "
        "re-run the cell. If it persists, the deposit may be down; try the URL in a browser."
    )
    raise RuntimeError(msg)


def list_annotations(split: str) -> list[str]:
    """Return the clip ids that have an annotation file in *split*.

    Raises:
        RuntimeError: If the listing cannot be fetched, or parses to nothing --
            which would mean the deposit's page format has changed and the
            scrape needs updating, rather than that the split is empty.
    """
    url = f"{BASE_URL}/annotations/{split}/"
    listing = fetch(url).decode("utf-8", "replace")
    found = sorted({m.group(1) for m in re.finditer(r'href="([^"]+)\.json"', listing)})
    if not found:
        msg = (
            f"no annotation files found at {url}. The deposit's directory listing format has "
            "probably changed, so the link scrape in list_annotations() needs updating."
        )
        raise RuntimeError(msg)
    return found


def profile_clip(split: str, clip_id: str, cache: Path) -> ClipProfile | None:
    """Download (or reuse) one annotation file and summarise it."""
    target = cache / f"{clip_id}.json"
    if not target.is_file():
        try:
            target.write_bytes(fetch(f"{BASE_URL}/annotations/{split}/{clip_id}.json"))
        except RuntimeError:
            return None

    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    annotations = document.get("annotations", [])
    boxes = [box for frame in annotations for box in frame.get("detections", [])]
    if not annotations:
        return None

    # The deposit does not state frame dimensions, so estimate them from the
    # furthest box edge seen in the clip. Good enough to *rank* subject size;
    # true dimensions come from decoding the video later.
    width = max((box["bbox"][2] for box in boxes), default=1.0) or 1.0
    height = max((box["bbox"][3] for box in boxes), default=1.0) or 1.0
    areas = [
        ((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])) / (width * height)
        for b in boxes
    ]

    per_frame = [len(frame.get("detections", [])) for frame in annotations]
    return ClipProfile(
        clip_id=clip_id,
        split=split,
        frames=len(annotations),
        boxes=len(boxes),
        empty_frames=sum(1 for n in per_frame if n == 0),
        max_apes_in_frame=max(per_frame, default=0),
        distinct_apes=len({b["ape_id"] for b in boxes}),
        behaviours=Counter(b["behaviour"] for b in boxes),
        species=Counter(b["species"] for b in boxes),
        median_relative_area=statistics.median(areas) if areas else 0.0,
    )


def select(profiles: list[ClipProfile], count: int) -> list[tuple[ClipProfile, str]]:
    """Greedily choose *count* clips covering the most selection axes.

    Set cover, not sampling: each pick is the clip adding the most axes not yet
    represented. Once every axis is covered, remaining picks favour clips with
    hard behaviours, so the sample keeps getting harder rather than merely bigger.
    """
    chosen: list[tuple[ClipProfile, str]] = []
    covered: set[str] = set()
    remaining = list(profiles)

    while remaining and len(chosen) < count:

        def gain(
            profile: ClipProfile, seen: frozenset[str] = frozenset(covered)
        ) -> tuple[int, int, int]:
            new = profile.features() - seen
            hard = sum(profile.behaviours[b] for b in HARD_BEHAVIOURS)
            return (len(new), hard, profile.boxes)

        best = max(remaining, key=gain)
        added = best.features() - covered
        covered |= best.features()
        remaining.remove(best)
        chosen.append((best, best.reason(added or best.features())))

    return chosen


def download_clip(profile: ClipProfile, raw_root: Path) -> tuple[Path, Path]:
    """Download one clip's video and annotation into ``data/raw/panaf500/``.

    Refuses to overwrite: raw data is immutable, and a silent re-download would
    invalidate any checksum already recorded against it.
    """
    videos = raw_root / "videos"
    annotations = raw_root / "annotations"
    videos.mkdir(parents=True, exist_ok=True)
    annotations.mkdir(parents=True, exist_ok=True)

    video_path = videos / f"{profile.clip_id}.mp4"
    annotation_path = annotations / f"{profile.clip_id}.json"

    if not video_path.exists():
        video_path.write_bytes(fetch(f"{BASE_URL}/videos/{profile.clip_id}.mp4"))
    if not annotation_path.exists():
        annotation_path.write_bytes(
            fetch(f"{BASE_URL}/annotations/{profile.split}/{profile.clip_id}.json")
        )
    return video_path, annotation_path


def main(argv: list[str] | None = None) -> int:
    """Select and download a purposive PanAf500 sample, then write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--count", type=int, default=10, help="Clips to download (onboarding: 5-10)."
    )
    parser.add_argument(
        "--pool",
        type=int,
        default=150,
        help="Candidate annotations to profile before selecting.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data" / "sample_manifest.csv",
        help="Manifest to write.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Select and report; download nothing."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Take every annotated clip in the deposit instead of a purposive sample. "
            "Roughly 500 clips and ~1.1 GB of video; --count and --pool are ignored."
        ),
    )
    arguments = parser.parse_args(argv)

    from panaf_ape_detection.manifest import MANIFEST_COLUMNS
    from panaf_ape_detection.provenance import file_sha256

    print("PanAf500 clip fetcher")
    print(f"  deposit: {DEPOSIT}")
    print(f"  licence: {LICENCE}")
    print("  Downloaded files must never be committed to this repository.\n")

    candidates: list[tuple[str, str]] = []
    for split in SPLITS:
        for clip_id in list_annotations(split):
            candidates.append((split, clip_id))
    print(f"{len(candidates)} annotated clips in the deposit")

    if arguments.all:
        # The whole deposit. Purposive selection exists to make a *small* sample
        # representative; taking everything makes it moot, and the dataset's own
        # train/validation/test split -- recorded per row -- is what a tuning
        # experiment should be divided by instead.
        pool = candidates
    else:
        # Even stride across the listing keeps the pool spread over all three splits.
        step = max(1, len(candidates) // arguments.pool)
        pool = candidates[::step][: arguments.pool]

    cache = REPO_ROOT / "artifacts" / "panaf500_annotation_cache"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"profiling {len(pool)} candidates (annotations only, no video)...")

    # Modest concurrency: this is a public research server, not a CDN.
    with ThreadPoolExecutor(max_workers=6) as pool_executor:
        profiles = list(pool_executor.map(lambda item: profile_clip(item[0], item[1], cache), pool))
    usable = [p for p in profiles if p is not None]
    print(f"  profiled {len(usable)}")

    if arguments.all:
        selected = [(profile, "whole deposit (--all)") for profile in usable]
    else:
        selected = select(usable, arguments.count)

    print(f"\nSelected {len(selected)} clips:\n")
    # The per-clip reasons are the point of a purposive sample and noise for the
    # whole deposit, where the split counts are what matters instead.
    if arguments.all:
        by_split = Counter(profile.split for profile, _ in selected)
        for split in SPLITS:
            print(f"  {split:<12} {by_split.get(split, 0)} clips")
    else:
        for profile, reason in selected:
            print(
                f"  {profile.clip_id}  [{profile.split}]  "
                f"{profile.frames} frames, {profile.boxes} boxes"
            )
            print(f"      {reason}")

    covered: set[str] = set()
    for profile, _ in selected:
        covered |= profile.features()
    print("\nAxes covered:")
    for axis in sorted(covered):
        print(f"  - {axis}")

    if arguments.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    raw_root = REPO_ROOT / "data" / "raw" / "panaf500"
    print(f"\nDownloading into {raw_root} ...")
    rows: list[dict[str, str]] = []
    failed: list[str] = []
    total_clips = len(selected)
    for number, (profile, reason) in enumerate(selected, start=1):
        try:
            video_path, annotation_path = download_clip(profile, raw_root)
        except Exception as exc:
            # Over 500 clips from a public research server, an occasional
            # failure is expected. Aborting would discard every clip already
            # fetched and write no manifest at all, so the run could not even
            # start. Skip it, keep going, and say so at the end.
            failed.append(profile.clip_id)
            print(f"  [{number}/{total_clips}] {profile.clip_id}: FAILED ({exc})")
            continue
        rows.append(
            {
                "clip_id": profile.clip_id,
                "split": profile.split,
                "video_filename": video_path.name,
                "annotation_filename": annotation_path.name,
                "species": "/".join(sorted(profile.species)),
                "site": "",  # not published per clip in this deposit
                "selected_reason": reason,
                "video_sha256": file_sha256(video_path),
                "annotation_sha256": file_sha256(annotation_path),
                "notes": (
                    f"{profile.frames} frames; {profile.boxes} boxes; "
                    f"{profile.empty_frames} empty frames; "
                    f"max {profile.max_apes_in_frame} apes/frame"
                ),
            }
        )
        size = video_path.stat().st_size
        print(f"  [{number}/{total_clips}] {profile.clip_id}: {size:,} bytes", flush=True)

    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    with arguments.manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    total = sum((raw_root / "videos" / r["video_filename"]).stat().st_size for r in rows)
    print(f"\nWrote {arguments.manifest} ({len(rows)} clips, {total / 1e6:.1f} MB of video)")
    if failed:
        # Named, not just counted: the manifest is the record of what a run
        # covered, and a silently smaller dataset is a silently different result.
        print(f"\n{len(failed)} clip(s) could not be downloaded and are NOT in the manifest:")
        print("  " + ", ".join(failed))
        print("Re-running this script retries them; clips already on disk are skipped.")
    print("Reminder: data/ is git-ignored. Do not commit these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
