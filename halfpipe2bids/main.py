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
        "level is available.",
        choices=["group"],
    )
    parser.add_argument(
        "-v", "--version", action="version", version=__version__
    )
    parser.add_argument(
        "--verbosity",
        help="""
        Verbosity level.
        """,
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
    # atlas_nii_path = path_atlas / "atlas-Schaefer2018Combined_dseg.nii.gz"

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    # dataset level metadata
    hp2b_utils.crearte_dataset_metadata_json(output_dir)
    # strategy_confounds = get_strategy_confounds(path_halfpipe_spec)
    label_atlas = hp2b_utils.load_label_schaefer(path_label_nii)

    # get denosing strategies and labels from the HalfPipe spec file
    strategy_confounds = hp2b_utils.get_strategy_confounds(path_halfpipe_spec)

    # Look for a subject's timeseries in the HalfPipe directory
    # and extract task, atlas
    task = "task-rest"  # TODO: make this dynamic
    atlas_name = "schaefer400"  # TODO: make this dynamic

    # nan_info = {
    #   key-> filename,
    #   value -> number of NaN values, or other information}
    # for ts_file in path_all_timeseries_files:
    #     nan_info[str(ts_file)] = load_timeseries_get_nan_info(
    #                           ts_file,
    #                           atlas_name, task)
    # remaining_labels= clean_timeseries(nan_info)

    # for ts_file in path_all_timeseries_files:
    #     metadata_path, bids_ts_path, bids_conn_path = \
    #           create_bids_connectome_filenames()
    #     create_metadata_json(metadata_path)
    #     clean_timeseries(bids_ts_path, remaining_labels, bids_conn_path)

    # original code; need to further refacot
    subjects = hp2b_utils.get_subjects(path_halfpipe_timeseries)
    for strategy in strategy_confounds.keys():
        hp2b_log.info(f"Processing strategy: {strategy}")
        missing_list = []
        dict_mean_framewise = {}
        dict_scrubvolume = {}
        dict_samplingfrequency = {}
        dict_timeseries = {}
        for subject in subjects:
            halfpipe_timeseries_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}"
                "_timeseries.tsv"
            )
            halfpipe_json_path = (
                f"{path_halfpipe_timeseries}/{subject}/func/{task}/"
                f"{subject}_{task}_feature-{strategy}_atlas-{atlas_name}"
                "_timeseries.json"
            )
            fmriprep_confounds_path = (
                f"{path_fmriprep}/{subject}/func/"
                f"{subject}_{task}_desc-confounds_timeseries.tsv"
            )
            if Path(halfpipe_timeseries_path).exists():
                df_confounds = pd.read_csv(fmriprep_confounds_path, sep="\t")
                dict_mean_framewise[subject] = df_confounds[
                    "framewise_displacement"
                ].mean()
                dict_scrubvolume[subject] = df_confounds.filter(
                    like="motion_outlier"
                ).shape[1]

                with open(halfpipe_json_path) as f:
                    meta = json.load(f)
                dict_samplingfrequency[subject] = meta.get(
                    "SamplingFrequency", None
                )

                df_timeseries = pd.read_csv(
                    halfpipe_timeseries_path, sep="\t", header=None
                )
                df_timeseries.columns = label_atlas
                dict_timeseries[subject] = df_timeseries
            else:
                missing_list.append(subject)

        # remove bad ROIs per denosing strategy
        labels_to_drop = hp2b_utils.remove_bad_rois(
            dict_timeseries, label_atlas
        )
        remaining_labels = list(set(label_atlas) - set(labels_to_drop))

        dict_clean_timeseries = {}
        for subject in dict_timeseries:
            dict_clean_timeseries[subject] = hp2b_utils.impute_and_clean(
                dict_timeseries[subject]
            )

        # calculate Pearson correlation for each subject
        # TODO: use nilearn connectivity module for this
        dict_corr = {
            sub: dict_clean_timeseries[sub].corr(method="pearson")
            for sub in dict_clean_timeseries
        }

        nroi = len(remaining_labels)
        regressors = strategy_confounds[strategy]

        for subject in dict_clean_timeseries:
            subject_output = output_dir / subject / "func"
            os.makedirs(subject_output, exist_ok=True)
            base_name = (
                f"{subject}_{task}_"
                f"seg-{atlas_name}{nroi}_desc-denoise{strategy}"
            )

            # Time series
            ts_path = subject_output / f"{base_name}_timeseries.tsv"
            dict_clean_timeseries[subject].columns = range(nroi)
            dict_clean_timeseries[subject].to_csv(
                ts_path, sep="\t", index=False
            )

            # Connectivity
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
            json_path = subject_output / f"{base_name}_timeseries.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=4)


def main(argv: None | Sequence[str] = None) -> None:
    """Entry point."""
    parser = global_parser()

    args = parser.parse_args(argv)

    workflow(args)
