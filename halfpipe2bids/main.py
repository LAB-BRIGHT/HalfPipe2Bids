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

    label_atlas = hp2b_utils.load_label_schaefer(path_label_nii)
    strategy_confounds = hp2b_utils.get_strategy_confounds(path_halfpipe_spec)
    subjects = hp2b_utils.get_subjects(path_halfpipe_timeseries)

    task = "task-rest"  # TODO: make this dynamic
    atlas_name = "schaefer400"  # TODO: make this dynamic

    # --- Phase 1: Load raw time series ---

    final_timeseries = {}  # final_timeseries[strategy][subject] = DataFrame
    raw_data_by_subject = (
        {}
    )  # raw_data_by_subject[subject][strategy] = DataFrame

    for subject in subjects:
        raw_data_by_subject[subject] = {}

        for strategy in strategy_confounds:
            hp_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}"
                "_timeseries.tsv"
            )
            if not Path(hp_path).exists():
                continue

            df = pd.read_csv(hp_path, sep="\t", header=None)
            if df.shape[1] != len(label_atlas):
                continue

            df.columns = label_atlas
            raw_data_by_subject[subject][strategy] = df

    # --- Phase 2: Optional NaN handling (imputation and ROI filtering) ---

    if args.NaN_Handling:
        for strategy in strategy_confounds:
            # Build subject-wise dict for each strategy
            data_for_strategy = {
                subject: raw_data_by_subject[subject][strategy]
                for subject in raw_data_by_subject
                if strategy in raw_data_by_subject[subject]
            }

            labels_to_drop = hp2b_utils.remove_bad_rois(
                data_for_strategy, label_atlas
            )
            remaining_labels = [
                label for label in label_atlas if label not in labels_to_drop
            ]

            for subject in data_for_strategy:
                df_clean = hp2b_utils.impute_and_clean(
                    data_for_strategy[subject]
                )
                df_clean = df_clean[remaining_labels]

                if strategy not in final_timeseries:
                    final_timeseries[strategy] = {}
                final_timeseries[strategy][subject] = df_clean
    else:
        for subject in raw_data_by_subject:
            for strategy in raw_data_by_subject[subject]:
                if strategy not in final_timeseries:
                    final_timeseries[strategy] = {}
                final_timeseries[strategy][subject] = raw_data_by_subject[
                    subject
                ][strategy]

    # --- Phase 3: Renaming and BIDS export ---

    for subject in subjects:
        for strategy in strategy_confounds:
            hp2b_log.info(f"Processing {subject} | strategy: {strategy}")

            if subject not in final_timeseries.get(strategy, {}):
                continue

            df_ts = final_timeseries[strategy][subject]
            nroi = df_ts.shape[1]
            base_name = f"{subject}_{task}_seg-{atlas_name}"
            f"{nroi}_desc-denoise{strategy}"
            subject_output = output_dir / subject / "func"
            os.makedirs(subject_output, exist_ok=True)

            # Save time series TSV
            ts_path = subject_output / f"{base_name}_timeseries.tsv"
            df_ts.columns = range(nroi)
            df_ts.to_csv(ts_path, sep="\t", index=False)

            # Save correlation matrix TSV
            corr = df_ts.corr(method="pearson")
            conn_path = subject_output / f"{base_name}"
            "_meas-PearsonCorrelation_relmat.tsv"
            corr.columns = range(nroi)
            corr.to_csv(conn_path, sep="\t", index=False)

            # Load and extract metadata
            json_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}"
                "_timeseries.json"
            )
            with open(json_path) as f:
                meta = json.load(f)
            sampling_freq = meta.get("SamplingFrequency", None)

            conf_path = (
                f"{path_fmriprep}/{subject}/func/"
                f"{subject}_{task}_desc-confounds_timeseries.tsv"
            )
            df_conf = pd.read_csv(conf_path, sep="\t")
            mean_fd = df_conf["framewise_displacement"].mean()
            scrub_vols = df_conf.filter(like="motion_outlier").shape[1]

            json_data = {
                "ConfoundRegressors": strategy_confounds[strategy],
                "NumberOfVolumesDiscardedByMotionScrubbing": scrub_vols,
                "MeanFramewiseDisplacement": mean_fd,
                "SamplingFrequency": sampling_freq,
            }

            json_out_path = subject_output / f"{base_name}_timeseries.json"
            with open(json_out_path, "w") as f:
                json.dump(json_data, f, indent=4)


def main(argv: None | Sequence[str] = None) -> None:
    """Entry point."""
    parser = global_parser()
    args = parser.parse_args(argv)
    workflow(args)
