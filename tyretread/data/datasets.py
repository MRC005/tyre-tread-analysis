"""Dataset discovery.

A dataset is a directory of class subdirectories. Keeping discovery in one place
means an experiment can name a dataset by identifier and the rest of the pipeline
does not care where the files came from, which is what makes cross-dataset
evaluation straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import CONFIG, DATA_DIR, EXTERNAL_DATA_DIR

__all__ = ["DatasetSpec", "ImageRecord", "LEGACY_SCRAPED", "MENDELEY_TYRES",
           "DATASETS", "list_images"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DatasetSpec:
    """Where a dataset lives and how its folder names map to canonical labels."""

    identifier: str
    root: Path
    #: directory name -> canonical label ("worn" or "serviceable")
    label_map: dict[str, str]
    citation: str = ""
    licence: str = ""
    notes: str = ""


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: str
    source_dir: str
    dataset: str


#: The original scraped dataset. Kept for reproducing the published result and for
#: cross-domain evaluation; not a domain match for phone photographs.
LEGACY_SCRAPED = DatasetSpec(
    identifier="legacy_scraped",
    root=DATA_DIR,
    label_map={"good": "serviceable", "bad": "worn"},
    citation="Kaggle: numberfive/tire-tread-photos",
    licence="Unknown - not redistributable",
    notes=(
        "369 web-scraped images, median 284x204 px. 266 of 369 filenames are "
        "Google Images defaults. Contains five byte-identical cross-label pairs. "
        "See docs/AUDIT.md sections 3.5-3.8."
    ),
)


def list_images(spec: DatasetSpec) -> list[ImageRecord]:
    """Every labelled image in ``spec``, sorted for reproducibility."""
    records: list[ImageRecord] = []
    for directory, label in sorted(spec.label_map.items()):
        folder = spec.root / directory
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(ImageRecord(path, label, directory, spec.identifier))
    return records


#: Pathmanaban et al. 2023, CC BY 4.0. Investigated as a replacement for the legacy
#: dataset; the audit found its labels describe tyre *damage* (sidewall cracking,
#: splits, perishing) rather than tread depth, and that image metadata alone predicts
#: its labels at 77% balanced accuracy. See docs/DATA.md and exp006.
MENDELEY_TYRES = DatasetSpec(
    identifier="mendeley_tyres",
    root=EXTERNAL_DATA_DIR / "mendeley_tyres",
    label_map={"serviceable": "serviceable", "worn": "worn"},
    citation=(
        "P, PATHMANABAN; C, Abishek; Sai, Kousik muthayala; S, Karthick; S, Aakash "
        '(2023), "Digital images of defective and good condition tyres", Mendeley '
        "Data, V1, doi: 10.17632/bn7ch8tvyp.1"
    ),
    licence="CC BY 4.0",
    notes=(
        "1856 images (1028 labelled defective, 828 good) - the dataset page states "
        "1854. The page also states a uniform 3000x3000 phone-camera acquisition; "
        "measured, there are 666 distinct resolutions, none of them 3000x3000, and "
        "17% of the images are small squares consistent with web thumbnails. The "
        "'defective' class is dominated by sidewall cracking and splits rather than "
        "worn tread. See docs/DATA.md."
    ),
)

DATASETS = {spec.identifier: spec for spec in (LEGACY_SCRAPED, MENDELEY_TYRES)}
