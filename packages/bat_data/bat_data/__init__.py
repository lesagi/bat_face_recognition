"""bat_data — manifest, dataset, splitter, miners, profiler."""

from bat_data.manifest import (
    FILENAME_PATTERN,
    ParsedFilename,
    build_manifest,
    compute_quality,
    group_files_by_class,
    is_augmented_file,
    manifest_from_csv,
    manifest_to_csv,
    parse_filename,
)
from bat_data.miners import HardNegativeMiner, MinedTriplets, SemiHardMiner
from bat_data.profiler import profile, save_profile
from bat_data.splitter import IdentitySplitter, SplitCounts

__all__ = [
    "FILENAME_PATTERN",
    "ParsedFilename",
    "parse_filename",
    "group_files_by_class",
    "is_augmented_file",
    "compute_quality",
    "build_manifest",
    "manifest_to_csv",
    "manifest_from_csv",
    "IdentitySplitter",
    "SplitCounts",
    "HardNegativeMiner",
    "SemiHardMiner",
    "MinedTriplets",
    "profile",
    "save_profile",
    # Lazy (torch-dependent):
    "BatDataset",
    "default_image_loader",
]


def __getattr__(name: str):  # PEP 562 lazy attr loading for torch deps
    if name in {"BatDataset", "default_image_loader"}:
        from bat_data import dataset as _dataset

        return getattr(_dataset, name)
    raise AttributeError(f"module 'bat_data' has no attribute {name!r}")
