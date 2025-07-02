from __future__ import annotations

import shutil
import json
import pandas as pd
import argparse
import logging

from pathlib import Path
from typing import Sequence

from halfpipe2bids import __version__
from halfpipe2bids import utils as hp2b_utils

hp2b_log = logging.getLogger("halfpipe2bids")


timeseries_json_extra_keys = [
    "ConfoundRegressors",
    "NumberOfVolumesDiscardedByMotionScrubbing",
    "MeanFramewiseDisplacement",
    "SamplingFrequency",
    "ROICentroids",
]


def global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Convert neuroimaging data from the HalfPipe format to the "
            "standardized BIDS (Brain Imaging Data Structure) format."
        ),
    )
    parser.add_argument(
        "halfpipe_dir",
        action="store",
        type=Path,
        help="The directory with the HALFPipe output.",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="The directory where the output files should be stored.",
    )
    parser.add_argument(
        "analysis_level",
        help="Level of the analysis that will be performed. Only group"
        " level is available.",
        choices=["group"],
    )
    parser.add_argument(
        "--denoise-meta-data",
        help="Additional metadata for denoising.",
        action="store_true",
    )
    parser.add_argument(
        "--impute-nan",
        help="Imputation and bad ROI removal.",
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--verbosity",
        help="Verbosity level.",
        required=False,
        choices=[0, 1, 2, 3],
        default=2,
        type=int,
        nargs=1,
    )
    return parser


def workflow(args: argparse.Namespace) -> None:
    hp2b_log.info(vars(args))
    output_dir = args.output_dir
    halfpipe_dir = args.halfpipe_dir

    # path_atlas = halfpipe_dir / "atlas"
    path_derivatives = halfpipe_dir / "derivatives"
    path_halfpipe_timeseries = path_derivatives / "halfpipe"
    # path_fmriprep = path_derivatives / "fmriprep"
    # path_label_nii = path_atlas / "atlas-Schaefer2018Combined_dseg.tsv"
    path_halfpipe_spec = halfpipe_dir / "spec.json"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset-level metadata
    hp2b_utils.create_dataset_metadata_json(output_dir)
    all_files = path_halfpipe_timeseries.glob("sub-*/**/*.*")

    # copy all files to the output directory
    for src in all_files:
        dst = hp2b_utils.get_bids_filename(src, output_dir)
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        hp2b_log.info(f"Renaming {src} to {dst}")
        shutil.copy2(src, dst)  # copy2 to preserve metadata

    # populate timeseries.json with extra information
    with open(path_halfpipe_spec, "r") as f:
        halfpipe_spec = json.load(f)
    print(halfpipe_spec.keys())


def populate_timeseries_json(
    path_timeseries_json, fmriprep_dir, halfpipe_spec
):
    sub = path_timeseries_json.stem.split("sub-")[-1].split("_")[0]
    task = path_timeseries_json.stem.split("task-")[-1].split("_")[0]
    confound_file = (
        fmriprep_dir
        / f"sub-{sub}"
        / "func"
        / f"sub-{sub}_task-{task}_desc-confounds_timeseries.tsv"
    )
    confounds = pd.read_csv(confound_file, sep="\t")
    print(confounds.columns)
    with open(path_timeseries_json, "r") as f:
        timeseries_meta = json.load(f)

    sampling_freq = timeseries_meta.get("SamplingFrequency", None)

    # convert sampling_freq from sec to Hz
    # TODO: this is an upstream issue that should be reported
    if sampling_freq is not None:
        sampling_freq = 1.0 / sampling_freq

    # timeseries_data.update(fmriprep_info)
    # timeseries_data.update(halfpipe_info)

    with open(path_timeseries_json, "w") as f:
        json.dump(timeseries_meta, f, indent=4)


def main(argv: None | Sequence[str] = None) -> None:
    """Entry point."""
    parser = global_parser()
    args = parser.parse_args(argv)
    workflow(args)
