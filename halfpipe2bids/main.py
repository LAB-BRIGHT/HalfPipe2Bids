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

        # --- Phase 2: Handle NaNs and identify bad ROIs ---
        labels_to_drop = hp2b_utils.remove_bad_rois(
            dict_timeseries, label_atlas
        )
        remaining_labels = list(set(label_atlas) - set(labels_to_drop))

        dict_clean_timeseries = {}
        for subject in dict_timeseries:
            df_clean = hp2b_utils.impute_and_clean(dict_timeseries[subject])
            df_clean = df_clean[remaining_labels]
            dict_clean_timeseries[subject] = df_clean

        # Compute Pearson correlation matrix per subject
        dict_corr = {
            subject: df.corr(method="pearson")
            for subject, df in dict_clean_timeseries.items()
        }

        # --- Phase 3: Write output to BIDS format ---
        nroi = len(remaining_labels)
        regressors = strategy_confounds[strategy]

        for subject in dict_clean_timeseries:
            subject_output = output_dir / subject / "func"
            os.makedirs(subject_output, exist_ok=True)
            base_name = (
                f"{subject}_{task}_seg-{atlas_name}{nroi}_"
                f"desc-denoise{strategy}"
            )

            # Time series
            ts_path = subject_output / f"{base_name}_timeseries.tsv"
            dict_clean_timeseries[subject].columns = range(nroi)
            dict_clean_timeseries[subject].to_csv(
                ts_path, sep="\t", index=False
            )

            # Connectivity matrix
            conn_path = (
                subject_output
                / f"{base_name}_meas-PearsonCorrelation_relmat.tsv"
            )
            dict_corr[subject].columns = range(nroi)
            dict_corr[subject].to_csv(conn_path, sep="\t", index=False)

            # JSON sidecar
            json_data = {
                "ConfoundRegressors": regressors,
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
