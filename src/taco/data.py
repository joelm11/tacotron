"""Download and pre-process the LJSpeech dataset."""

import os
import tarfile
from pathlib import Path
from urllib import request

DATASET_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
DATASET_ARCHIVE = "ljspeech.tar.bz2"


def setup_dir_structure(project_root: Path):
    """Create a data directory with `raw/` and `features/` subfolders."""
    raw = project_root / "data" / "raw"
    features = project_root / "data" / "features"
    raw.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    return raw, features


def download_extract_dataset(output_dir: Path):
    archive_path = output_dir / DATASET_ARCHIVE
    if not archive_path.exists():
        print(f"Downloading dataset to {archive_path} ...")
        request.urlretrieve(DATASET_URL, archive_path)
    else:
        print(f"Archive already exists at {archive_path}, skipping download.")

    extracted_dir = output_dir / "LJSpeech-1.1"
    if not extracted_dir.exists():
        print("Extracting archive ...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=output_dir, filter="data")
        os.remove(archive_path)
    else:
        print("Dataset already extracted, skipping extraction.")


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_dataset_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "LJSpeech-1.1"


def get_metadata_path(project_root: Path) -> Path:
    return get_dataset_dir(project_root) / "metadata.csv"


def get_wav_path(project_root: Path, label: str) -> Path:
    return get_dataset_dir(project_root) / "wavs" / f"{label}.wav"


def get_feat_dir(project_root: Path) -> Path:
    return project_root / "data" / "features"


def main():
    project_root = get_project_root()
    raw_dir, _ = setup_dir_structure(project_root)
    download_extract_dataset(raw_dir)


if __name__ == "__main__":
    main()
