"""Duplicate detection and grouping for leakage-safe validation.

Two distinct problems are handled here.

**Contradictory labels.** The legacy dataset contains five pairs of byte-identical
images filed under both classes (docs/AUDIT.md 3.5). No model can be right about
both members of such a pair, so they place a hard ceiling on measurable accuracy
and must be found and quarantined rather than silently trained on.

**Grouping for cross-validation.** Near-duplicates that straddle a fold boundary
let a model recognise a specific photograph rather than a tread condition. The
audit measured this on the legacy dataset and found no inflation - group-aware
folds scored 0.733 against 0.732 for plain folds - but that was a measurement, not
a guarantee, and it has to be re-checked on any new dataset. Grouping is therefore
computed always and used always; it costs nothing when there is nothing to catch.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["DuplicateReport", "content_hash", "perceptual_hash", "hamming",
           "find_duplicates", "assign_groups"]


def content_hash(path: str | Path) -> str:
    """MD5 of the file bytes - catches exact duplicates only."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: str | Path, size: int = 8) -> np.ndarray:
    """Difference hash: catches re-encodes, crops and rescales of one photograph.

    Each bit records whether a pixel is brighter than its right-hand neighbour in a
    heavily downsampled greyscale version, so the hash survives compression and
    resizing while still differing between genuinely different images.
    """
    with Image.open(path) as pil:
        small = pil.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    return np.packbits((arr[:, 1:] > arr[:, :-1]).ravel())


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.unpackbits(a ^ b).sum())


@dataclass
class DuplicateReport:
    exact_groups: list[list[str]] = field(default_factory=list)
    contradictory_groups: list[list[str]] = field(default_factory=list)
    near_pairs: list[tuple[str, str, int]] = field(default_factory=list)
    cross_label_near_pairs: list[tuple[str, str, int]] = field(default_factory=list)

    @property
    def contradictory_files(self) -> set[str]:
        return {p for group in self.contradictory_groups for p in group}

    def as_dict(self) -> dict[str, object]:
        return {
            "n_exact_groups": len(self.exact_groups),
            "n_contradictory_groups": len(self.contradictory_groups),
            "n_near_pairs": len(self.near_pairs),
            "n_cross_label_near_pairs": len(self.cross_label_near_pairs),
            "contradictory_groups": self.contradictory_groups,
        }


def find_duplicates(
    paths: list[str], labels: list[str], *, near_threshold: int = 6
) -> DuplicateReport:
    """Find exact duplicates, label contradictions and near-duplicates."""
    report = DuplicateReport()

    by_hash: dict[str, list[int]] = defaultdict(list)
    for i, path in enumerate(paths):
        by_hash[content_hash(path)].append(i)
    for members in by_hash.values():
        if len(members) < 2:
            continue
        group = [paths[i] for i in members]
        report.exact_groups.append(group)
        if len({labels[i] for i in members}) > 1:
            report.contradictory_groups.append(group)

    hashes = [perceptual_hash(p) for p in paths]
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            distance = hamming(hashes[i], hashes[j])
            if distance <= near_threshold:
                pair = (paths[i], paths[j], distance)
                report.near_pairs.append(pair)
                if labels[i] != labels[j]:
                    report.cross_label_near_pairs.append(pair)
    return report


def assign_groups(paths: list[str], *, near_threshold: int = 6) -> np.ndarray:
    """Group index per image; near-duplicates share a group.

    Union-find over the near-duplicate graph, so a chain of mutually similar
    photographs ends up in one group rather than several overlapping pairs.
    """
    n = len(paths)
    hashes = [perceptual_hash(p) for p in paths]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashes[i], hashes[j]) <= near_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots])
