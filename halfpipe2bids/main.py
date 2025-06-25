from __future__ import annotations

import os
import json
import pandas as pd
import argparse
import logging

from pathlib import Path
from typing import Sequence

from halfpipe2bids import __version__
from halfpipe2bids import utils as hp2b_utils

hp2b_log = logging.getLogger("halfpipe2bids")


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
        "-v", "--version", action="version", version=__version__
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

    parser.add_argument(
        "--NaN_Handling",
        help="Enable NaN handling (imputation and bad ROI removal).",
        action="store_true",
    )

    return parser


def workflow(args: argparse.Namespace) -> None:
    hp2b_log.info(vars(args))
    output_dir = args.output_dir
    halfpipe_dir = args.halfpipe_dir

    path_atlas = halfpipe_dir / "atlas"
    path_derivatives = halfpipe_dir / "derivatives"
    path_halfpipe_timeseries = path_derivatives / "halfpipe"
    path_fmriprep = path_derivatives / "fmriprep"
    path_label_nii = path_atlas / "atlas-Schaefer2018Combined_dseg.tsv"
    path_halfpipe_spec = halfpipe_dir / "spec.json"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset-level metadata
    hp2b_utils.crearte_dataset_metadata_json(output_dir)

    # Load atlas ROI labels
    label_atlas = hp2b_utils.load_label_schaefer(path_label_nii)

    # Get denoising strategies and associated confounds from spec
    strategy_confounds = hp2b_utils.get_strategy_confounds(path_halfpipe_spec)

    task = "task-rest"  # TODO: make this dynamic
    atlas_name = "schaefer400"  # TODO: make this dynamic

    subjects = hp2b_utils.get_subjects(path_halfpipe_timeseries)

    for strategy in strategy_confounds:
        hp2b_log.info(f"Processing strategy: {strategy}")

        # --- Phase 1: Load all raw data ---
        dict_timeseries = {}
        dict_mean_framewise = {}
        dict_scrubvolume = {}
        dict_samplingfrequency = {}
        missing_list = []
        base_names = {}

        for subject in subjects:
            hp_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}_"
                "timeseries.tsv"
            )
            json_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}_"
                "timeseries.json"
            )
            conf_path = (
                f"{path_fmriprep}/{subject}/func/"
                f"{subject}_{task}_desc-confounds_timeseries.tsv"
            )

            if Path(hp_path).exists():
                df_ts = pd.read_csv(hp_path, sep="\t", header=None)

                if df_ts.shape[1] != len(label_atlas):
                    hp2b_log.warning(
                        f"{hp_path} has unexpected number of columns. Skipped."
                    )
                    continue

                df_ts.columns = label_atlas
                dict_timeseries[subject] = df_ts

                df_conf = pd.read_csv(conf_path, sep="\t")
                dict_mean_framewise[subject] = df_conf[
                    "framewise_displacement"
                ].mean()
                dict_scrubvolume[subject] = df_conf.filter(
                    like="motion_outlier"
                ).shape[1]

                with open(json_path) as f:
                    meta = json.load(f)
                dict_samplingfrequency[subject] = meta.get(
                    "SamplingFrequency", None
                )
            else:
                missing_list.append(subject)

        # --- Phase 2: Prepare BIDS-compatible filenames ---
        for subject, df in dict_timeseries.items():
            nroi = df.shape[1]  # initial number of ROIs
            base_name = f"{subject}_{task}_seg-{atlas_name}{nroi}_desc-denoise{strategy}"
            base_names[subject] = base_name

        # --- Phase 3: Optional NaN Handling ---
        if args.NaN_Handling:
            labels_to_drop = hp2b_utils.remove_bad_rois(
                dict_timeseries, label_atlas
            )
            remaining_labels = [
                label for label in label_atlas if label not in labels_to_drop
            ]

            for subject in dict_timeseries:
                df_clean = hp2b_utils.impute_and_clean(
                    dict_timeseries[subject]
                )
                df_clean = df_clean[remaining_labels]
                dict_timeseries[subject] = df_clean

                # Update base name with new nroi count after cleaning
                nroi = len(remaining_labels)
                base_name = f"{subject}_{task}_seg-{atlas_name}{nroi}_desc-denoise{strategy}"
                base_names[subject] = base_name

        # --- Phase 4: Write final BIDS-formatted outputs ---
        for subject, df in dict_timeseries.items():
            nroi = df.shape[1]
            subject_output = output_dir / subject / "func"
            os.makedirs(subject_output, exist_ok=True)
            base_name = base_names[subject]

            # Time series
            ts_path = subject_output / f"{base_name}_timeseries.tsv"
            df.columns = range(nroi)  # Reset columns to 0...nroi-1
            df.to_csv(ts_path, sep="\t", index=False)

            # Correlation matrix
            corr = df.corr(method="pearson")
            conn_path = (
                subject_output
                / f"{base_name}_meas-PearsonCorrelation_relmat.tsv"
            )
            corr.columns = range(nroi)
            corr.to_csv(conn_path, sep="\t", index=False)

            # Metadata JSON
            json_data = {
                "ConfoundRegressors": strategy_confounds[strategy],
                "NumberOfVolumesDiscardedByMotionScrubbing": dict_scrubvolume[
                    subject
                ],
                "MeanFramewiseDisplacement": dict_mean_framewise[subject],
                "SamplingFrequency": dict_samplingfrequency[subject],
            }
            json_out_path = subject_output / f"{base_name}_timeseries.json"
            with open(json_out_path, "w") as f:
                json.dump(json_data, f, indent=4)


def main(argv: None | Sequence[str] = None) -> None:
    """Entry point."""
    parser = global_parser()
    args = parser.parse_args(argv)
    workflow(args)
